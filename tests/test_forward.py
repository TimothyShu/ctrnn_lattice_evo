"""
Tests for forward.py — the CTRNN integrator.

Topology-agnostic: these test the ODE, not the graph, so they port almost
unchanged.  Two substantive additions at the end cover the fact that a
lattice genome is denser and fully active, which drives W_eff @ y harder
than ctrnn_lattice_evo's 15%-dense random init ever did.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_lattice_evo import Config, Genome, E, FSI
from ctrnn_lattice_evo.forward import forward_pass, batch_forward
from ctrnn_lattice_evo.genome import grid_genome


@pytest.fixture
def cfg():
    """N_max must be a perfect square now — 8 has no lattice."""
    return Config(N_max=16, n_out=1, grid_W=4, grid_H=4, grid_r=1, dt=0.5, K=20)


def minimal_genome(cfg: Config, **overrides) -> Genome:
    """Blank genome (all inactive, no edges) with overrides applied.

    Six fields, not seven — `position` is gone.
    """
    N = cfg.N_max
    g = Genome(
        active_mask=jnp.zeros(N, dtype=bool)
                       .at[:cfg.n_in].set(True)
                       .at[-cfg.n_out:].set(True),
        neuron_type=jnp.zeros(N, dtype=jnp.uint8),
        tau=jnp.full(N, 10.0),
        bias=jnp.zeros(N),
        weight_matrix=jnp.zeros((N, N)),
        edge_mask=jnp.zeros((N, N), dtype=bool),
    )
    return dataclasses.replace(g, **overrides)


# ── Integrator behaviour ─────────────────────────────────────────────────────

def test_isolated_excitatory_neuron_decays(cfg):
    """v(t) ~ v0 * exp(-K*dt/tau).  K=20, dt=0.5, tau=10 -> exp(-1) ~ 0.368."""
    N, tau_val, idx = cfg.N_max, 10.0, 3

    active = jnp.zeros(N, dtype=bool).at[0].set(True).at[-1].set(True).at[idx].set(True)
    tau = jnp.full(N, 1.0).at[idx].set(tau_val)
    g = minimal_genome(cfg, active_mask=active, tau=tau)

    v0 = jnp.zeros(N).at[idx].set(1.0)
    v_final, _, _ = forward_pass(g, v0, jnp.zeros(N), cfg)

    expected = np.exp(-cfg.K * cfg.dt / tau_val)
    assert abs(float(v_final[idx]) - expected) < 0.05, \
        f"Expected ~{expected:.3f}, got {float(v_final[idx]):.3f}"


def test_inactive_neuron_does_not_propagate(cfg):
    """A masked neuron contributes zero regardless of weight magnitude.

    Now load-bearing: prune_isolated is the ONLY node-death path on a
    lattice, so silenced slots must be genuinely inert.
    """
    N, idx_masked, idx_target = cfg.N_max, 4, 5

    active = (jnp.zeros(N, dtype=bool)
              .at[0].set(True).at[-1].set(True).at[idx_target].set(True))
    W = jnp.zeros((N, N)).at[idx_target, idx_masked].set(10.0)
    edge = jnp.zeros((N, N), dtype=bool).at[idx_target, idx_masked].set(True)

    g = minimal_genome(cfg, active_mask=active, weight_matrix=W, edge_mask=edge)
    v0 = jnp.zeros(N).at[idx_masked].set(5.0)

    v_final, _, _ = forward_pass(g, v0, jnp.zeros(N), cfg)
    assert float(v_final[idx_target]) == pytest.approx(0.0, abs=1e-5)


def test_dales_law_inhibitory_suppresses(cfg):
    """An FS-I source must pull its target down — sign comes from the source."""
    N, fsi_idx, e_idx = cfg.N_max, 3, 4

    active = (jnp.zeros(N, dtype=bool)
              .at[0].set(True).at[-1].set(True).at[fsi_idx].set(True).at[e_idx].set(True))
    ntype = jnp.zeros(N, dtype=jnp.uint8).at[fsi_idx].set(FSI)
    W = jnp.zeros((N, N)).at[e_idx, fsi_idx].set(1.0)
    edge = jnp.zeros((N, N), dtype=bool).at[e_idx, fsi_idx].set(True)

    g = minimal_genome(cfg, active_mask=active, neuron_type=ntype,
                       weight_matrix=W, edge_mask=edge)
    v0 = jnp.zeros(N).at[fsi_idx].set(3.0).at[e_idx].set(0.5)

    v_final, _, _ = forward_pass(g, v0, jnp.zeros(N), cfg)
    assert float(v_final[e_idx]) < 0.5


def test_batch_forward_matches_single(cfg):
    P = 16
    keys = jax.random.split(jax.random.PRNGKey(42), P)
    pop = jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)

    v0s = jnp.zeros((P, cfg.N_max))
    ins = jnp.zeros((P, cfg.N_max))
    v_batch, out_batch, _ = batch_forward(pop, v0s, ins, cfg)

    for i in range(P):
        g_i = jax.tree_util.tree_map(lambda x: x[i], pop)
        v_i, out_i, _ = forward_pass(g_i, v0s[i], ins[i], cfg)
        np.testing.assert_allclose(v_batch[i], v_i, rtol=1e-4, atol=1e-6,
                           err_msg=f"v mismatch at organism {i}")
        np.testing.assert_allclose(out_batch[i], out_i, rtol=1e-4, atol=1e-6,
                           err_msg=f"output mismatch at organism {i}")


# ── Numerical stability at lattice density ───────────────────────────────────

def test_no_nans_on_lattice_population(cfg):
    """A 4x4 r=1 lattice is 35% dense with all 16 nodes live — a far stronger
    recurrent drive than ctrnn_lattice_evo's 15%-dense, half-active random init."""
    keys = jax.random.split(jax.random.PRNGKey(99), 64)
    pop = jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)
    v0s = jnp.zeros((64, cfg.N_max))
    ins = jnp.zeros((64, cfg.N_max))

    v_final, output, c_act = batch_forward(pop, v0s, ins, cfg)

    assert not jnp.any(jnp.isnan(v_final)), "NaN in v_final"
    assert not jnp.any(jnp.isinf(v_final)), "Inf in v_final"
    assert not jnp.any(jnp.isnan(output)), "NaN in output"
    assert not jnp.any(jnp.isnan(c_act)), "NaN in c_act"


def test_no_nans_on_production_lattice():
    """8x8 at r=2: 1092 edges, 64 live nodes.  Euler at dt=0.5 against a
    tau_fsi floor of 1.0 is already at the stability edge (dt <= tau_min);
    this is where it would break if it breaks."""
    cfg = Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2, dt=0.5, K=20)
    keys = jax.random.split(jax.random.PRNGKey(7), 32)
    pop = jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)

    v_final, output, c_act = batch_forward(
        pop, jnp.zeros((32, 64)), jnp.zeros((32, 64)), cfg
    )
    assert not jnp.any(jnp.isnan(v_final))
    assert not jnp.any(jnp.isinf(v_final))
    assert jnp.all(jnp.abs(output) <= 1.0 + 1e-5)


def test_c_act_bounded(cfg):
    """c_act is mean |tanh(v)| over active neurons, so it is in [0, 1]
    regardless of network size — the property that lets act_frac use a fixed
    C0 of 1.0 with no calibration."""
    keys = jax.random.split(jax.random.PRNGKey(11), 16)
    pop = jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)
    _, _, c_act = batch_forward(pop, jnp.zeros((16, cfg.N_max)),
                                jnp.zeros((16, cfg.N_max)), cfg)
    assert jnp.all(c_act >= 0.0)
    assert jnp.all(c_act <= 1.0 + 1e-6)


def test_input_drive_reaches_output_across_lattice():
    """Sensors sit at slots [:n_in] (grid corner (0,0)) and motors at
    [-n_out:] (corner (7,7)), 7 hops apart at Chebyshev distance.  With r=2
    that is a minimum of 4 hops, each one a tau of low-pass filtering.  If
    this fails, the lattice cannot transmit and no penalty setting will
    rescue the arm.
    """
    cfg = Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2, dt=0.5, K=20)
    g = grid_genome(jax.random.PRNGKey(0), cfg)

    v = jnp.zeros(cfg.N_max)
    drive = jnp.zeros(cfg.N_max).at[:cfg.n_in].set(5.0)

    # Several world steps: one forward_pass is K ticks, not K hops.
    for _ in range(20):
        v, out, _ = forward_pass(g, v, drive, cfg)

    assert float(jnp.max(jnp.abs(out))) > 1e-3, \
        "No signal reached the output slots across the lattice"