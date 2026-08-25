"""
brain.py — Interface layer between CTRNN and the world simulator.

Public API
----------
make_ctrnn_controller(genome, cfg) -> controller_fn
    Returns a stateful-closure controller compatible with run_episode.
    **Note**: this closure carries a mutable Python list for voltage state,
    which breaks jit/vmap.  Use run_brain_episode for compiled evaluation.

run_brain_episode(key, genome, cfg, wcfg) -> (final_state, steps_survived)
    Runs a full episode with the CTRNN brain driving the agent.
    JIT-compilable and vmap-safe.

run_brain_episode_full(key, genome, cfg, wcfg) -> (final_state, steps_survived, mean_c_act, total_raw_food)
    Same as run_brain_episode but also returns mean activation cost and cumulative
    raw food score (uncapped, summed over all alive steps and food types).
    Used by the evolutionary evaluator.

batch_run_brain_episode(keys, genomes, cfg, wcfg) -> (final_states, steps)
    vmapped version over a population of genomes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome
from .forward import forward_pass
from .world import WorldConfig, WorldState, food_at, sensor_readout, step_world, reset_world


# ── Internal scan body ────────────────────────────────────────────────────────

def _brain_world_step(
    carry: tuple[WorldState, jnp.ndarray],   # (world_state, voltage_vector)
    rng_key: jax.Array,
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[tuple[WorldState, jnp.ndarray], jnp.ndarray]:
    """
    One combined brain+world step for use inside jax.lax.scan.

    carry:
        world_state  — current WorldState
        v            — CTRNN voltage vector [N_max], float32
    rng_key:
        per-step key (pre-split outside the scan)
    returns:
        new carry, alive flag (bool scalar)
    """
    world_state, v = carry

    # Sensors → CTRNN input padded to [N_max]
    # Input neurons occupy the first cfg.n_in slots of the voltage vector.
    # All other slots receive zero external drive.
    sensors = sensor_readout(world_state, wcfg)          # [n_in]
    input_vec = jnp.zeros(cfg.N_max, dtype=jnp.float32).at[:cfg.n_in].set(sensors)

    # CTRNN forward pass (K inner ticks)
    v_new, output, _c_act = forward_pass(genome, v, input_vec, cfg)  # output: [n_out]

    # output neurons fire tanh values in [-1, 1] → action
    # step_world scales by max_speed internally
    action = output                                       # [2]

    # Advance world
    new_world = step_world(world_state, action, wcfg)
    alive = jnp.all(new_world.agent_energy > 0.0)

    # Raw food score: sum of uncapped food_at across all food types.
    # Only credited while alive — dead agents stop accumulating score.
    # This is read-only and does NOT affect world energy or the world model.
    raw_food = (
        jnp.sum(jax.vmap(
            lambda hpos: food_at(new_world.agent_pos, hpos, wcfg)
        )(new_world.hotspot_pos))
        * alive.astype(jnp.float32)
    )

    return (new_world, v_new), (alive, _c_act, raw_food)


# ── Public: single-genome episode ────────────────────────────────────────────

def run_brain_episode(
    key: jax.Array,
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray]:
    """
    Run a full episode (wcfg.episode_steps steps) with CTRNN brain.

    The CTRNN voltage vector is initialised to zero at episode start.
    Per-step RNG keys are pre-split before the scan so the function is
    fully jit-compilable.

    Returns
    -------
    final_state    : WorldState after the last step
    steps_survived : int32 scalar — number of steps with energy > 0
    """
    k_world, k_steps = jax.random.split(key)

    # Initialise world and CTRNN voltage
    world_state = reset_world(k_world, wcfg)
    v0 = jnp.zeros(cfg.N_max, dtype=jnp.float32)

    # Pre-split one key per step (avoids dynamic splitting inside scan)
    step_keys = jax.random.split(k_steps, wcfg.episode_steps)

    def body(carry, rng_key):
        new_carry, (alive, _c_act, _raw_food) = _brain_world_step(carry, rng_key, genome, cfg, wcfg)
        return new_carry, alive

    (final_world, _v_final), alive_mask = jax.lax.scan(
        body, (world_state, v0), step_keys
    )

    steps_survived = jnp.sum(alive_mask.astype(jnp.int32))
    return final_world, steps_survived


# ── Public: single-genome episode + activation cost ──────────────────────────

def run_brain_episode_full(
    key: jax.Array,
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Same as run_brain_episode but also accumulates mean activation cost and
    total raw food score.

    Returns
    -------
    final_state     : WorldState after the last step
    steps_survived  : int32 scalar
    mean_c_act      : float32 scalar — mean activation cost over all episode steps
    total_raw_food  : float32 scalar — sum of uncapped food_at across all steps and
                      food types (only accumulated while alive).  Can exceed
                      episode_steps * n_food_types for an agent perfectly centred
                      on every hotspot.  Normalise by (episode_steps * n_food_types)
                      to get a score in [0, ∞); well-foraging agents typically reach
                      values around 0.5–2.0 depending on hotspot_sigma.
    """
    k_world, k_steps = jax.random.split(key)

    world_state = reset_world(k_world, wcfg)
    v0          = jnp.zeros(cfg.N_max, dtype=jnp.float32)
    step_keys   = jax.random.split(k_steps, wcfg.episode_steps)

    def body(carry, rng_key):
        new_carry, (alive, c_act, raw_food) = _brain_world_step(carry, rng_key, genome, cfg, wcfg)
        return new_carry, (alive, c_act, raw_food)

    (final_world, _v_final), (alive_mask, c_act_steps, raw_food_steps) = jax.lax.scan(
        body, (world_state, v0), step_keys
    )

    steps_survived = jnp.sum(alive_mask.astype(jnp.int32))
    mean_c_act     = jnp.mean(c_act_steps)
    total_raw_food = jnp.sum(raw_food_steps)
    return final_world, steps_survived, mean_c_act, total_raw_food


# ── Public: population-level batch ───────────────────────────────────────────

batch_run_brain_episode = jax.vmap(
    run_brain_episode,
    in_axes=(0, 0, None, None),
)
"""
vmapped run_brain_episode over a population.

Arguments
---------
keys    : [pop_size, 2]   — one PRNGKey per genome
genomes : Genome pytree with leading pop_size batch dimension
cfg     : Config (shared)
wcfg    : WorldConfig (shared)

Returns
-------
final_states  : WorldState with leading pop_size dimension
steps_survived: int32 [pop_size]
"""


# ── Public: stateless controller adaptor ─────────────────────────────────────

def make_ctrnn_controller(genome: Genome, cfg: Config):
    """
    Wrap a genome as a run_episode-compatible controller function.

    The returned function carries CTRNN voltage state in a Python closure
    list so that successive calls (within one episode) are stateful.

    **Warning**: this closure is NOT jit/vmap-safe.  Use run_brain_episode
    for compiled evaluation.  This function exists for quick interactive
    testing with run_episode and controllers from controllers.py.

    Signature: controller(key, sensors, state, wcfg) -> action [2]
    """
    # Mutable voltage container (reset once per object creation → one episode)
    _v = [jnp.zeros(cfg.N_max, dtype=jnp.float32)]

    def controller(
        key: jax.Array,
        sensors: jnp.ndarray,
        state: WorldState,
        wcfg: WorldConfig,
    ) -> jnp.ndarray:
        input_vec = jnp.zeros(cfg.N_max, dtype=jnp.float32).at[:cfg.n_in].set(sensors)
        v_new, output, _ = forward_pass(genome, _v[0], input_vec, cfg)
        _v[0] = v_new
        return output   # [n_out], tanh values in [-1, 1]

    return controller
