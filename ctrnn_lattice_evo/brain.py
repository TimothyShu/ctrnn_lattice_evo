"""
brain.py — Interface layer between the CTRNN and the world simulator.

Public API
----------
run_brain_episode(key, genome, cfg, wcfg) -> (final_state, steps_survived)
    One full episode with the CTRNN driving the agent.  JIT-compilable and
    vmap-safe.

run_brain_episode_full(key, genome, cfg, wcfg)
        -> (final_state, steps_survived, mean_c_act, total_raw_food)
    As above, plus the activation cost and cumulative raw food score.  This is
    what the evolutionary evaluator uses — c_act feeds the activation penalty
    and raw_food feeds fitness_mode="food".

batch_run_brain_episode(keys, genomes, cfg, wcfg)
batch_run_brain_episode_full(keys, genomes, cfg, wcfg)
    vmapped over a population.

ctrnn_evo's make_ctrnn_controller is gone: it carried voltage state in a
mutable Python closure, which breaks jit and vmap, and existed only for
interactive poking.  Use run_brain_episode.

Lattice note: sensors occupy slots [:n_in] and motors slots [-n_out:], which
under topology.py's row-major indexing puts them at opposite corners of the
grid.  Signal therefore has to traverse the sheet — at 8x8 with r=2 that is a
minimum of 4 hops, each one a tau of low-pass filtering.  Nothing here
enforces that; it falls out of the slot ordering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome
from .forward import forward_pass
from .world import (
    WorldConfig, WorldState,
    food_at, sensor_readout, step_world, reset_world,
)

__all__ = [
    "run_brain_episode",
    "run_brain_episode_full",
    "batch_run_brain_episode",
    "batch_run_brain_episode_full",
]


# ── Internal scan body ────────────────────────────────────────────────────────

def _brain_world_step(
    carry: tuple[WorldState, jnp.ndarray],   # (world_state, voltage_vector)
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
):
    """One combined brain+world step, for use inside jax.lax.scan.

    carry:  (WorldState, v [N_max] float32)
    returns: (new carry, (alive, c_act, raw_food))
    """
    world_state, v = carry

    # Sensors -> CTRNN input, padded to [N_max].  Input neurons occupy the
    # first n_in slots; every other slot receives zero external drive.
    sensors = sensor_readout(world_state, wcfg)                       # [n_in]
    input_vec = jnp.zeros(cfg.N_max, dtype=jnp.float32).at[:cfg.n_in].set(sensors)

    # K inner CTRNN ticks per world step.
    v_new, output, c_act = forward_pass(genome, v, input_vec, cfg)    # output: [n_out]

    # Motor neurons emit tanh values in [-1, 1]; step_world scales by
    # max_speed internally.  This requires n_out == 2 (vx, vy) — a Config with
    # n_out != 2 is fine for cost or topology work but cannot run an episode.
    new_world = step_world(world_state, output, wcfg)
    alive = jnp.all(new_world.agent_energy > 0.0)

    # Raw food score: uncapped food_at summed over food types, credited only
    # while alive.  Read-only — it does not affect world energy.
    raw_food = (
        jnp.sum(jax.vmap(
            lambda hpos: food_at(new_world.agent_pos, hpos, wcfg)
        )(new_world.hotspot_pos))
        * alive.astype(jnp.float32)
    )

    return (new_world, v_new), (alive, c_act, raw_food)


def _episode_scan(key: jax.Array, genome: Genome, cfg: Config, wcfg: WorldConfig):
    """Run the full episode scan and return the raw per-step stacks.

    Voltage is initialised to zero at episode start, so episodes are
    independent — no state leaks between them.

    No per-step keys: WorldState carries its own rng_key and step_world splits
    from that, so all episode stochasticity (hotspot drift) enters via
    reset_world.  ctrnn_evo pre-split one key per step and threaded them into
    the scan body, which never read them — that allocated
    episode_steps x 2 uint32 per genome per evaluation for nothing.
    """
    k_world, _k_unused = jax.random.split(key)

    world_state = reset_world(k_world, wcfg)
    v0 = jnp.zeros(cfg.N_max, dtype=jnp.float32)

    def body(carry, _):
        return _brain_world_step(carry, genome, cfg, wcfg)

    return jax.lax.scan(body, (world_state, v0), None, length=wcfg.episode_steps)


# ── Public: single-genome episodes ────────────────────────────────────────────

def run_brain_episode(
    key: jax.Array,
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray]:
    """Run one episode.

    Returns
    -------
    final_state    : WorldState after the last step
    steps_survived : int32 scalar — steps on which all energies stayed > 0
    """
    (final_world, _v), (alive_mask, _c_act, _raw_food) = _episode_scan(
        key, genome, cfg, wcfg
    )
    return final_world, jnp.sum(alive_mask.astype(jnp.int32))


def run_brain_episode_full(
    key: jax.Array,
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one episode, also returning activation cost and raw food score.

    Returns
    -------
    final_state    : WorldState after the last step
    steps_survived : int32 scalar
    mean_c_act     : float32 in [0, 1] — mean |tanh(v)| over active neurons,
                     averaged across ticks and steps.  Size-independent, which
                     is why C0_act is exactly 1.0 and needs no calibration.
    total_raw_food : float32 >= 0 — uncapped food_at summed over steps and food
                     types, accumulated only while alive.  May exceed
                     episode_steps * n_food_types for an agent sitting on a
                     hotspot; divide by that product for a comparable score.

    Both scores are non-negative by construction, which the multiplicative
    penalty relies on: if f_raw could go negative, a penalty multiplier below 1
    would IMPROVE it.
    """
    (final_world, _v), (alive_mask, c_act_steps, raw_food_steps) = _episode_scan(
        key, genome, cfg, wcfg
    )
    return (
        final_world,
        jnp.sum(alive_mask.astype(jnp.int32)),
        jnp.mean(c_act_steps),
        jnp.sum(raw_food_steps),
    )


# ── Public: population batches ────────────────────────────────────────────────

batch_run_brain_episode = jax.vmap(
    run_brain_episode,
    in_axes=(0, 0, None, None),
)
"""vmapped run_brain_episode.

keys    : [pop_size, 2]  — one PRNGKey per genome
genomes : batched Genome pytree, leading pop_size dimension
cfg, wcfg : shared (static)

-> (final_states with leading pop_size dim, steps_survived int32 [pop_size])
"""

batch_run_brain_episode_full = jax.vmap(
    run_brain_episode_full,
    in_axes=(0, 0, None, None),
)
"""vmapped run_brain_episode_full — what eval_population calls.

-> (final_states, steps [P], mean_c_act [P], total_raw_food [P])
"""