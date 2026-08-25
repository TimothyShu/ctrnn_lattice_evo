"""
Tests — Mutation operators.

Ported from ctrnn_evo.  What changed:

  DELETED   perturb_position  — positions are static, in topology.py
  DELETED   add_node          — no-op with every lattice slot already live
  DELETED   remove_node       — would punch holes in the substrate
  REPLACED  remove_edge       -> remove_edges (per-edge Bernoulli)
  KEPT      perturb_weights, perturb_tau, perturb_bias, type_flip, add_edge

remove_edges throughput is tested in test_pruning.py; this file covers the
surviving operators and the combined `mutate`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config, E, FSI, SII
from ctrnn_lattice_evo.genome import grid_genome, validate_genome
from ctrnn_lattice_evo.topology import local_mask
from ctrnn_lattice_evo import mutation as M
from ctrnn_lattice_evo.mutation import (
    MutationRates,
    perturb_weights,
    perturb_tau,
    perturb_bias,
    type_flip,
    add_edge,
    remove_edges,
    mutate,
)


@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture(scope="module")
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def rates():
    return MutationRates(add_node_prob=0.0, remove_node_prob=0.0)


# ── Deleted operators ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["perturb_position", "add_node", "remove_node", "remove_edge"])
def test_deleted_operators_are_gone(name):
    """Left in place they are worse than useless: remove_node deactivates
    lattice slots, and remove_edge's single-edge argmax caps pruning at one
    edge per generation."""
    assert not hasattr(M, name), f"{name} should have been removed"


@pytest.mark.parametrize("field", ["position_sigma", "add_node_prob_unused"])
def test_position_sigma_removed(field):
    assert not hasattr(MutationRates(), "position_sigma")


def test_node_probs_default_to_zero():
    """If these fields survive for config compatibility, they must default off."""
    r = MutationRates()
    assert getattr(r, "add_node_prob", 0.0) == 0.0
    assert getattr(r, "remove_node_prob", 0.0) == 0.0


# ── perturb_weights ───────────────────────────────────────────────────────────

def test_perturb_weights_changes_weights(g, cfg):
    out = perturb_weights(jax.random.PRNGKey(1), g, cfg, sigma=0.1)
    assert not jnp.allclose(out.weight_matrix, g.weight_matrix)


def test_perturb_weights_stays_nonnegative(g, cfg):
    """Magnitudes are non-negative; Dale sign comes from neuron_type."""
    out = perturb_weights(jax.random.PRNGKey(2), g, cfg, sigma=5.0)
    assert jnp.all(out.weight_matrix >= 0.0)


def test_perturb_weights_zero_sigma_is_noop(g, cfg):
    out = perturb_weights(jax.random.PRNGKey(3), g, cfg, sigma=0.0)
    assert jnp.array_equal(out.weight_matrix, g.weight_matrix)


def test_perturb_weights_leaves_topology_alone(g, cfg):
    out = perturb_weights(jax.random.PRNGKey(4), g, cfg, sigma=0.1)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)
    assert jnp.array_equal(out.active_mask, g.active_mask)


def test_perturb_weights_scales_with_sigma(g, cfg):
    small = perturb_weights(jax.random.PRNGKey(5), g, cfg, sigma=0.01)
    large = perturb_weights(jax.random.PRNGKey(5), g, cfg, sigma=1.0)
    d_small = float(jnp.mean(jnp.abs(small.weight_matrix - g.weight_matrix)))
    d_large = float(jnp.mean(jnp.abs(large.weight_matrix - g.weight_matrix)))
    assert d_large > d_small


# ── perturb_tau ───────────────────────────────────────────────────────────────

def test_perturb_tau_stays_in_type_range(g, cfg):
    """tau outside its type range breaks validate_genome and, at the FS-I
    floor of 1.0, threatens Euler stability at dt=0.5."""
    out = perturb_tau(jax.random.PRNGKey(6), g, cfg, sigma=100.0)
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    assert jnp.all(out.tau >= lo[out.neuron_type] - 1e-4)
    assert jnp.all(out.tau <= hi[out.neuron_type] + 1e-4)


def test_perturb_tau_stays_positive(g, cfg):
    out = perturb_tau(jax.random.PRNGKey(7), g, cfg, sigma=1000.0)
    assert jnp.all(out.tau > 0)


def test_perturb_tau_never_below_dt(g, cfg):
    """dt <= tau is the integrator's stability condition."""
    out = perturb_tau(jax.random.PRNGKey(8), g, cfg, sigma=100.0)
    assert jnp.all(out.tau >= cfg.dt)


def test_perturb_tau_zero_sigma_is_noop(g, cfg):
    out = perturb_tau(jax.random.PRNGKey(9), g, cfg, sigma=0.0)
    assert jnp.allclose(out.tau, g.tau)


# ── perturb_bias ──────────────────────────────────────────────────────────────

def test_perturb_bias_changes_bias(g, cfg):
    out = perturb_bias(jax.random.PRNGKey(10), g, cfg, sigma=0.1)
    assert not jnp.allclose(out.bias, g.bias)


def test_perturb_bias_may_be_negative(g, cfg):
    """Unlike weights, bias is signed — clamping it would be a bug."""
    out = perturb_bias(jax.random.PRNGKey(11), g, cfg, sigma=1.0)
    assert jnp.any(out.bias < 0.0)


def test_perturb_bias_zero_sigma_is_noop(g, cfg):
    out = perturb_bias(jax.random.PRNGKey(12), g, cfg, sigma=0.0)
    assert jnp.allclose(out.bias, g.bias)


# ── type_flip ─────────────────────────────────────────────────────────────────

def test_type_flip_changes_some_types(g, cfg):
    out = type_flip(jax.random.PRNGKey(13), g, cfg, flip_prob=0.9)
    assert not jnp.array_equal(out.neuron_type, g.neuron_type)


def test_type_flip_protects_io_slots(g, cfg):
    """I/O neurons must stay excitatory — an inhibitory sensor would invert
    the whole input signal."""
    out = type_flip(jax.random.PRNGKey(14), g, cfg, flip_prob=1.0)
    assert jnp.all(out.neuron_type[:cfg.n_in] == E)
    assert jnp.all(out.neuron_type[-cfg.n_out:] == E)


def test_type_flip_produces_valid_types(g, cfg):
    out = type_flip(jax.random.PRNGKey(15), g, cfg, flip_prob=1.0)
    assert jnp.all(out.neuron_type <= SII)


def test_type_flip_reclamps_tau(g, cfg):
    """A neuron flipped E -> FSI keeps a tau from the E range unless it is
    re-clamped, which would silently violate the type invariant."""
    out = type_flip(jax.random.PRNGKey(16), g, cfg, flip_prob=1.0)
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    assert jnp.all(out.tau >= lo[out.neuron_type] - 1e-4)
    assert jnp.all(out.tau <= hi[out.neuron_type] + 1e-4)


def test_type_flip_zero_prob_is_noop(g, cfg):
    out = type_flip(jax.random.PRNGKey(17), g, cfg, flip_prob=0.0)
    assert jnp.array_equal(out.neuron_type, g.neuron_type)


def test_type_flip_leaves_topology_alone(g, cfg):
    out = type_flip(jax.random.PRNGKey(18), g, cfg, flip_prob=0.5)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)


# ── add_edge ──────────────────────────────────────────────────────────────────

def test_add_edge_adds_exactly_one(g, cfg):
    out = add_edge(jax.random.PRNGKey(19), g, cfg)
    assert int(out.edge_mask.sum()) == int(g.edge_mask.sum()) + 1


def test_add_edge_never_removes(g, cfg):
    out = add_edge(jax.random.PRNGKey(20), g, cfg)
    assert jnp.all(g.edge_mask <= out.edge_mask)


def test_add_edge_is_unmasked(g, cfg):
    """Deliberate: locality is the initialisation, not a hard ceiling, so
    evolution can buy a long-range shortcut if it earns one against the
    distance penalty."""
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = g
    for i in range(100):
        gg = add_edge(jax.random.fold_in(jax.random.PRNGKey(21), i), gg, cfg)
    assert int((gg.edge_mask & ~m).sum()) > 0


def test_add_edge_only_between_active_neurons(cfg):
    g2 = grid_genome(jax.random.PRNGKey(22), cfg)
    g2 = dataclasses.replace(g2, active_mask=g2.active_mask.at[7].set(False))
    g2 = dataclasses.replace(
        g2, edge_mask=g2.edge_mask.at[7, :].set(False).at[:, 7].set(False))

    out = add_edge(jax.random.PRNGKey(23), g2, cfg)
    assert not bool(out.edge_mask[7].any())
    assert not bool(out.edge_mask[:, 7].any())


def test_add_edge_initialises_positive_weight(g, cfg):
    """A new edge with weight 0 is invisible to selection and would never be
    evaluated before the cost penalty removes it."""
    out = add_edge(jax.random.PRNGKey(24), g, cfg)
    new = out.edge_mask & ~g.edge_mask
    assert float(jnp.sum(jnp.where(new, out.weight_matrix, 0.0))) > 0.0


def test_add_edge_saturates_gracefully(cfg):
    """No-op when every active pair already has an edge — must not corrupt."""
    g2 = grid_genome(jax.random.PRNGKey(25), cfg)
    full = (g2.active_mask[:, None] & g2.active_mask[None, :])
    full = full & ~jnp.eye(cfg.N_max, dtype=bool)
    g2 = dataclasses.replace(g2, edge_mask=full)
    out = add_edge(jax.random.PRNGKey(26), g2, cfg)
    assert jnp.array_equal(out.edge_mask, g2.edge_mask)


# ── mutate (combined) ─────────────────────────────────────────────────────────

def test_mutate_returns_valid_genome(g, cfg, rates):
    assert validate_genome(mutate(jax.random.PRNGKey(27), g, cfg, rates), cfg)


def test_mutate_is_deterministic(g, cfg, rates):
    key = jax.random.PRNGKey(28)
    a = mutate(key, g, cfg, rates)
    b = mutate(key, g, cfg, rates)
    assert jnp.allclose(a.weight_matrix, b.weight_matrix)
    assert jnp.array_equal(a.edge_mask, b.edge_mask)


def test_mutate_preserves_io_slots(g, cfg, rates):
    out = mutate(jax.random.PRNGKey(29), g, cfg, rates)
    assert jnp.all(out.active_mask[:cfg.n_in])
    assert jnp.all(out.active_mask[-cfg.n_out:])


def test_mutate_preserves_lattice_size_without_pruning(g, cfg):
    """With no edge removal, prune_isolated has nothing to strand, so every
    lattice slot must survive."""
    r = MutationRates(add_node_prob=0.0, remove_node_prob=0.0,
                      add_edge_prob=0.0, remove_edge_prob=0.0)
    out = mutate(jax.random.PRNGKey(30), g, cfg, r)
    assert int(out.active_mask.sum()) == cfg.N_max


def test_mutate_edge_mask_stays_within_active_pairs(g, cfg, rates):
    """The invariant validate_genome now asserts.  A violation is silent:
    effective_weights masks it, but edge_count_cost overcounts and every
    penalty is miscalibrated."""
    gg = g
    for i in range(50):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(31), i), gg, cfg, rates)
        pairs = gg.active_mask[:, None] & gg.active_mask[None, :]
        assert jnp.all(gg.edge_mask <= pairs)


def test_mutate_never_produces_nan(g, cfg, rates):
    gg = g
    for i in range(50):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(32), i), gg, cfg, rates)
    assert not jnp.any(jnp.isnan(gg.weight_matrix))
    assert not jnp.any(jnp.isnan(gg.tau))
    assert not jnp.any(jnp.isnan(gg.bias))


def test_mutate_is_vmappable(g, cfg, rates):
    keys = jax.random.split(jax.random.PRNGKey(33), 8)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 8), g)
    out = jax.vmap(mutate, in_axes=(0, 0, None, None))(keys, batch, cfg, rates)
    assert out.weight_matrix.shape == (8, cfg.N_max, cfg.N_max)


def test_mutate_diversifies_identical_genomes(g, cfg, rates):
    """Grid genomes start structurally identical, so mutate is the only
    source of topological diversity in the population."""
    outs = [mutate(jax.random.PRNGKey(40 + i), g, cfg, rates) for i in range(20)]
    assert any(not jnp.array_equal(outs[0].edge_mask, o.edge_mask) for o in outs[1:])


def test_mutate_zero_rates_is_near_identity(g, cfg):
    r = MutationRates(weight_sigma=0.0, tau_sigma=0.0, bias_sigma=0.0,
                      type_flip_prob=0.0, add_node_prob=0.0,
                      remove_node_prob=0.0, add_edge_prob=0.0,
                      remove_edge_prob=0.0)
    out = mutate(jax.random.PRNGKey(34), g, cfg, r)
    assert jnp.allclose(out.weight_matrix, g.weight_matrix)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)


# ── Mutation warm-up scaling ──────────────────────────────────────────────────

def test_sigma_scaling_increases_spread(g, cfg):
    """mutation_warmup_scale multiplies the continuous sigmas during warm-up:
    wide exploration while the penalty is still ramping from zero."""
    a = perturb_weights(jax.random.PRNGKey(35), g, cfg, sigma=0.1)
    b = perturb_weights(jax.random.PRNGKey(35), g, cfg, sigma=0.5)
    assert float(jnp.std(b.weight_matrix - g.weight_matrix)) > \
           float(jnp.std(a.weight_matrix - g.weight_matrix))