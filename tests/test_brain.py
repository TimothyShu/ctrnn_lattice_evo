"""
Tests — Brain integration layer.

Ported from ctrnn_evo.  Changes:
  * `position` gone from the genome rebuild
  * make_ctrnn_controller tests deleted (jit-unsafe interactive helper, cut)
  * N_max=16 so the 4x4 lattice fits
  * new: signal must traverse the lattice from sensor corner to motor corner
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config, WorldConfig, run_episode
from ctrnn_lattice_evo.genome import grid_genome, uniform_genome
from ctrnn_lattice_evo.brain import (
    run_brain_episode,
    run_brain_episode_full,
    batch_run_brain_episode,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1, K=4)


@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=200)


@pytest.fixture(scope="module")
def genome(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def pop_genomes(cfg):
    keys = jax.random.split(jax.random.PRNGKey(42), 8)
    return jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)


# ── 1. Return types and shapes ────────────────────────────────────────────────

def test_run_brain_episode_return_types(genome, cfg, wcfg):
    final_state, steps = run_brain_episode(jax.random.PRNGKey(1), genome, cfg, wcfg)
    assert hasattr(final_state, "agent_energy")
    assert steps.shape == ()
    assert steps.dtype in (jnp.int32, jnp.int64)


def test_final_state_shapes(genome, cfg, wcfg):
    final_state, _ = run_brain_episode(jax.random.PRNGKey(2), genome, cfg, wcfg)
    assert final_state.agent_pos.shape == (2,)
    assert final_state.hotspot_pos.shape == (wcfg.n_food_types, wcfg.n_food, 2)
    assert final_state.agent_energy.shape == (wcfg.n_food_types,)


# ── 2. steps_survived bounds ──────────────────────────────────────────────────

def test_steps_survived_bounded(genome, cfg, wcfg):
    _, steps = run_brain_episode(jax.random.PRNGKey(3), genome, cfg, wcfg)
    assert 0 <= int(steps) <= wcfg.episode_steps


def test_steps_survived_full_episode_possible(wcfg):
    """The solvability ceiling.  When the lattice arm underperforms, this is
    what tells you whether the world moved or the network did — which is why
    controllers.py is kept even though evolution never calls it."""
    from ctrnn_lattice_evo.controllers import nearest_hotspot
    _, steps = run_episode(jax.random.PRNGKey(99), nearest_hotspot, wcfg)
    assert int(steps) == wcfg.episode_steps, (
        f"nearest_hotspot survived only {int(steps)}/{wcfg.episode_steps} steps — "
        "world may not be survivable at current parameters."
    )


# ── 3. Degenerate genomes ─────────────────────────────────────────────────────

def test_zero_weight_genome_runs(cfg, wcfg):
    g = grid_genome(jax.random.PRNGKey(5), cfg)
    g = dataclasses.replace(
        g, bias=jnp.zeros_like(g.bias), weight_matrix=jnp.zeros_like(g.weight_matrix)
    )
    _, steps = run_brain_episode(jax.random.PRNGKey(6), g, cfg, wcfg)
    assert 0 <= int(steps) <= wcfg.episode_steps


def test_fully_pruned_genome_runs(cfg, wcfg):
    """The end state of aggressive pruning: no edges at all.  Must not crash —
    a lambda sweep will produce these at the high end."""
    g = grid_genome(jax.random.PRNGKey(5), cfg)
    g = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    _, steps = run_brain_episode(jax.random.PRNGKey(6), g, cfg, wcfg)
    assert 0 <= int(steps) <= wcfg.episode_steps


# ── 4. Determinism ────────────────────────────────────────────────────────────

def test_determinism(genome, cfg, wcfg):
    key = jax.random.PRNGKey(7)
    _, s1 = run_brain_episode(key, genome, cfg, wcfg)
    _, s2 = run_brain_episode(key, genome, cfg, wcfg)
    assert int(s1) == int(s2)


def test_different_keys_differ(genome, cfg, wcfg):
    """A lattice genome is far more strongly connected than ctrnn_evo's
    15%-dense random init, and a saturated network can drive the agent into a
    wall identically regardless of world seed.  If this goes flaky, that is a
    signal about the initialisation, not a reason to loosen the threshold."""
    results = {int(run_brain_episode(jax.random.PRNGKey(100 + i), genome, cfg, wcfg)[1])
               for i in range(6)}
    assert len(results) >= 2, \
        "All 6 world seeds gave identical survival — network may be saturated"


# ── 5. Batching ───────────────────────────────────────────────────────────────

def test_batch_run_brain_episode_shapes(pop_genomes, cfg, wcfg):
    keys = jax.random.split(jax.random.PRNGKey(10), 8)
    final_states, steps = batch_run_brain_episode(keys, pop_genomes, cfg, wcfg)
    assert steps.shape == (8,)
    assert final_states.agent_pos.shape == (8, 2)
    assert final_states.agent_energy.shape == (8, wcfg.n_food_types)


def test_batch_run_bounded(pop_genomes, cfg, wcfg):
    keys = jax.random.split(jax.random.PRNGKey(11), 8)
    _, steps = batch_run_brain_episode(keys, pop_genomes, cfg, wcfg)
    assert jnp.all(steps >= 0) and jnp.all(steps <= wcfg.episode_steps)


# ── 6. run_brain_episode_full ─────────────────────────────────────────────────

def test_run_brain_episode_full_returns_four(genome, cfg, wcfg):
    assert len(run_brain_episode_full(jax.random.PRNGKey(20), genome, cfg, wcfg)) == 4


def test_run_brain_episode_full_c_act_bounded(genome, cfg, wcfg):
    """c_act in [0,1] is the precondition for act_frac using C0_act=1.0."""
    _, steps, c_act, raw_food = run_brain_episode_full(
        jax.random.PRNGKey(21), genome, cfg, wcfg)
    assert 0.0 <= float(c_act) <= 1.0 + 1e-6
    assert float(raw_food) >= 0.0
    assert 0 <= int(steps) <= wcfg.episode_steps


def test_raw_food_nonnegative_for_all_seeds(genome, cfg, wcfg):
    """f_raw >= 0 is what the clamped multiplicative penalty relies on: if
    raw fitness could go negative, the penalty would IMPROVE it."""
    for i in range(8):
        _, _, _, raw_food = run_brain_episode_full(
            jax.random.PRNGKey(200 + i), genome, cfg, wcfg)
        assert float(raw_food) >= 0.0


# ── 7. Voltage reset ──────────────────────────────────────────────────────────

def test_voltage_reset_between_episodes(genome, cfg, wcfg):
    """Strengthened from ctrnn_evo, where this only asserted two scalars had
    shape ().  Same world key twice must give the same result, which can only
    hold if voltage is zeroed at episode start."""
    key = jax.random.PRNGKey(30)
    _, s1 = run_brain_episode(key, genome, cfg, wcfg)
    _, s2 = run_brain_episode(key, genome, cfg, wcfg)
    assert int(s1) == int(s2), "Voltage state leaked between episodes"


# ── 8. Lattice traversal ──────────────────────────────────────────────────────

def test_sensors_and_motors_are_far_apart(cfg):
    """Sensors occupy slots [:n_in] and motors [-n_out:].  Row-major on a 4x4
    lattice puts them at opposite corners, so signal must cross the sheet."""
    from ctrnn_lattice_evo.topology import dist_matrix
    d = dist_matrix(cfg.grid_W, cfg.grid_H)
    assert float(d[0, cfg.N_max - 1]) == float(max(cfg.grid_W, cfg.grid_H) - 1)


def test_signal_traverses_lattice_to_output(cfg, wcfg):
    """Each hop is a tau of low-pass filtering.  If the sheet cannot transmit
    at all, no penalty setting rescues the arm and the experiment is
    unrunnable — worth knowing before queueing 30 hours."""
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    from ctrnn_lattice_evo.forward import forward_pass

    v = jnp.zeros(cfg.N_max)
    drive = jnp.zeros(cfg.N_max).at[:cfg.n_in].set(5.0)
    for _ in range(20):
        v, out, _ = forward_pass(g, v, drive, cfg)

    assert float(jnp.max(jnp.abs(out))) > 1e-3, \
        "No signal reached the motor slots across the lattice"


def test_uniform_arm_also_runs(cfg, wcfg):
    """The locality control must be executable on the same path as the grid
    arm, or the comparison cannot be made."""
    g = uniform_genome(jax.random.PRNGKey(0), cfg)
    _, steps = run_brain_episode(jax.random.PRNGKey(1), g, cfg, wcfg)
    assert 0 <= int(steps) <= wcfg.episode_steps