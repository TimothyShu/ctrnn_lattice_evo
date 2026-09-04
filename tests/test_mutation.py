"""
Tests — Mutation operators.

Changes from ctrnn_evo:

  DELETED   perturb_position   — positions are static, in topology.py
  REPLACED  remove_edge        -> remove_edges (count from pre-mutation E)
  REPLACED  add_edge           -> add_edges    (same, so the pair is neutral)
  GATED     add_node / remove_node — kept, but only run when
            cfg.node_ops_enabled, which Config restricts to the sparse arm

MutationRates now carries ONE structural rate, edge_churn, driving both edge
operators.  Two separate rates would let them diverge again, which is what
produced the drift collapse in the first four-arm gate.

Operator BALANCE is tested in test_pruning.py; this file covers the operators
individually and the combined mutate().
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config, E, SII
from ctrnn_lattice_evo.genome import grid_genome, sparse_genome, validate_genome
from ctrnn_lattice_evo.topology import local_mask, dist_matrix, distance_kernel
from ctrnn_lattice_evo import mutation as M
from ctrnn_lattice_evo.mutation import (
    MutationRates,
    perturb_weights, perturb_tau, perturb_bias,
    type_flip, add_edges, remove_edges, count_active_edges,
    add_node, remove_node, mutate,
)


@pytest.fixture(scope="module")
def cfg():
    """Grid arm — node operators off."""
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture(scope="module")
def cfg_sparse():
    """Sparse arm — the only arm where node operators are permitted."""
    return Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2,
                  init_mode="sparse", node_ops_enabled=True)


@pytest.fixture(scope="module")
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def rates():
    return MutationRates()


# ── Removed / renamed API ─────────────────────────────────────────────────────

def test_perturb_position_is_gone():
    """Positions live on the lattice, not the genome."""
    assert not hasattr(M, "perturb_position")
    assert not hasattr(MutationRates(), "position_sigma")


def test_single_edge_operators_are_gone():
    """The singular forms were the unbalanced pair: remove_edge deleted one
    edge per generation, add_edge added one.  Keeping either available would
    let the imbalance back in by accident."""
    assert not hasattr(M, "remove_edge")
    assert not hasattr(M, "add_edge")


def test_one_shared_structural_rate():
    """Two separate rates would let the operators diverge again.  edge_churn
    drives both, so E[dE] = p*E - p*E = 0 by construction rather than by
    tuning."""
    r = MutationRates()
    assert hasattr(r, "edge_churn")
    assert not hasattr(r, "add_edge_prob")
    assert not hasattr(r, "remove_edge_p_per_edge")


# ── count_active_edges ────────────────────────────────────────────────────────

def test_count_matches_mask(g):
    assert int(count_active_edges(g)) == int(g.edge_mask.sum()) == 84


def test_count_ignores_inactive_pairs(g, cfg):
    """Edges touching a dead neuron must not count, or both operators would be
    driven by an inflated E."""
    g2 = dataclasses.replace(g, active_mask=g.active_mask.at[5].set(False))
    assert int(count_active_edges(g2)) < int(count_active_edges(g))


# ── perturb_weights ───────────────────────────────────────────────────────────

def test_perturb_weights_changes_weights(g, cfg):
    out = perturb_weights(jax.random.PRNGKey(1), g, cfg, sigma=0.1)
    assert not jnp.allclose(out.weight_matrix, g.weight_matrix)


def test_perturb_weights_stays_nonnegative(g, cfg):
    """Magnitudes only — Dale sign comes from neuron_type."""
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
    d_s = float(jnp.mean(jnp.abs(small.weight_matrix - g.weight_matrix)))
    d_l = float(jnp.mean(jnp.abs(large.weight_matrix - g.weight_matrix)))
    assert d_l > d_s


# ── perturb_tau ───────────────────────────────────────────────────────────────

def test_perturb_tau_stays_in_type_range(g, cfg):
    out = perturb_tau(jax.random.PRNGKey(6), g, cfg, sigma=100.0)
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    assert jnp.all(out.tau >= lo[out.neuron_type] - 1e-4)
    assert jnp.all(out.tau <= hi[out.neuron_type] + 1e-4)


def test_perturb_tau_never_below_dt(g, cfg):
    """dt <= tau is the Euler stability condition."""
    out = perturb_tau(jax.random.PRNGKey(7), g, cfg, sigma=1000.0)
    assert jnp.all(out.tau >= cfg.dt)


def test_perturb_tau_zero_sigma_is_noop(g, cfg):
    out = perturb_tau(jax.random.PRNGKey(8), g, cfg, sigma=0.0)
    assert jnp.allclose(out.tau, g.tau)


# ── perturb_bias ──────────────────────────────────────────────────────────────

def test_perturb_bias_changes_bias(g, cfg):
    out = perturb_bias(jax.random.PRNGKey(9), g, cfg, sigma=0.1)
    assert not jnp.allclose(out.bias, g.bias)


def test_perturb_bias_may_be_negative(g, cfg):
    """Unlike weights, bias is signed — clamping it would be a bug."""
    out = perturb_bias(jax.random.PRNGKey(10), g, cfg, sigma=1.0)
    assert jnp.any(out.bias < 0.0)


def test_perturb_bias_zero_sigma_is_noop(g, cfg):
    out = perturb_bias(jax.random.PRNGKey(11), g, cfg, sigma=0.0)
    assert jnp.allclose(out.bias, g.bias)


# ── type_flip ─────────────────────────────────────────────────────────────────

def test_type_flip_changes_some_types(g, cfg):
    out = type_flip(jax.random.PRNGKey(12), g, cfg, flip_prob=0.9)
    assert not jnp.array_equal(out.neuron_type, g.neuron_type)


def test_type_flip_protects_io_slots(g, cfg):
    """An inhibitory sensor would invert the whole input signal."""
    out = type_flip(jax.random.PRNGKey(13), g, cfg, flip_prob=1.0)
    assert jnp.all(out.neuron_type[:cfg.n_in] == E)
    assert jnp.all(out.neuron_type[-cfg.n_out:] == E)


def test_type_flip_produces_valid_types(g, cfg):
    out = type_flip(jax.random.PRNGKey(14), g, cfg, flip_prob=1.0)
    assert jnp.all(out.neuron_type <= SII)


def test_type_flip_reclamps_tau(g, cfg):
    """A neuron flipped E -> FSI keeps a tau from the E range unless it is
    re-clamped, silently violating the type invariant."""
    out = type_flip(jax.random.PRNGKey(15), g, cfg, flip_prob=1.0)
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    assert jnp.all(out.tau >= lo[out.neuron_type] - 1e-4)
    assert jnp.all(out.tau <= hi[out.neuron_type] + 1e-4)


def test_type_flip_zero_prob_is_noop(g, cfg):
    out = type_flip(jax.random.PRNGKey(16), g, cfg, flip_prob=0.0)
    assert jnp.array_equal(out.neuron_type, g.neuron_type)


# ── Edge operators (mechanics; balance lives in test_pruning.py) ──────────────

def test_add_edges_never_creates_self_edge(g, cfg):
    gg = g
    for i in range(30):
        gg = add_edges(jax.random.fold_in(jax.random.PRNGKey(17), i), gg, cfg,
                       p_per_edge=0.1)
    assert not jnp.any(jnp.diag(gg.edge_mask))


def test_add_edges_is_unmasked(g, cfg):
    """Deliberate: locality is the initialisation, not a hard ceiling."""
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = add_edges(jax.random.PRNGKey(18), g, cfg, p_per_edge=0.2)
    assert int((gg.edge_mask & ~m).sum()) > 0


def test_add_edges_only_between_active_neurons(cfg_sparse):
    gs = sparse_genome(jax.random.PRNGKey(19), cfg_sparse)
    inactive = jnp.where(~gs.active_mask)[0]
    out = add_edges(jax.random.PRNGKey(20), gs, cfg_sparse, p_per_edge=0.3)
    for idx in inactive[:5]:
        assert not bool(out.edge_mask[idx].any())
        assert not bool(out.edge_mask[:, idx].any())


def test_add_edges_log_kernel_none_matches_default(g, cfg):
    """Passing log_kernel=None explicitly must be identical to omitting it —
    both take the plain _pick_n path."""
    a = add_edges(jax.random.PRNGKey(60), g, cfg, p_per_edge=0.2)
    b = add_edges(jax.random.PRNGKey(60), g, cfg, p_per_edge=0.2, log_kernel=None)
    assert jnp.array_equal(a.edge_mask, b.edge_mask)


def test_add_edges_log_kernel_biases_toward_local(g, cfg):
    """With a tight kernel, edges added over many trials should land closer
    (in Chebyshev distance) on average than with uniform proposal."""
    d = dist_matrix(cfg.grid_W, cfg.grid_H)
    kernel = distance_kernel(cfg.grid_W, 0.5, cfg.grid_H)
    log_kernel = jnp.where(kernel > 0, jnp.log(kernel), -jnp.inf)

    def mean_new_edge_distance(log_k):
        total_d, total_n = 0.0, 0
        for i in range(40):
            key = jax.random.fold_in(jax.random.PRNGKey(61), i)
            out = add_edges(key, g, cfg, p_per_edge=0.3, log_kernel=log_k)
            new = out.edge_mask & ~g.edge_mask
            total_d += float(jnp.sum(jnp.where(new, d, 0.0)))
            total_n += int(new.sum())
        return total_d / total_n

    biased = mean_new_edge_distance(log_kernel)
    uniform = mean_new_edge_distance(None)
    assert biased < uniform


def test_add_edges_log_kernel_never_creates_self_edge(g, cfg):
    kernel = distance_kernel(cfg.grid_W, 1.0, cfg.grid_H)
    log_kernel = jnp.where(kernel > 0, jnp.log(kernel), -jnp.inf)
    gg = g
    for i in range(30):
        gg = add_edges(jax.random.fold_in(jax.random.PRNGKey(62), i), gg, cfg,
                       p_per_edge=0.1, log_kernel=log_kernel)
    assert not jnp.any(jnp.diag(gg.edge_mask))


def test_mutate_accepts_log_kernel(g, cfg, rates):
    kernel = distance_kernel(cfg.grid_W, 1.0, cfg.grid_H)
    log_kernel = jnp.where(kernel > 0, jnp.log(kernel), -jnp.inf)
    out = mutate(jax.random.PRNGKey(63), g, cfg, rates, log_kernel=log_kernel)
    assert validate_genome(out, cfg)


def test_mutate_is_vmappable_with_log_kernel(g, cfg, rates):
    kernel = distance_kernel(cfg.grid_W, 1.0, cfg.grid_H)
    log_kernel = jnp.where(kernel > 0, jnp.log(kernel), -jnp.inf)
    keys = jax.random.split(jax.random.PRNGKey(64), 8)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 8), g)
    out = jax.vmap(mutate, in_axes=(0, 0, None, None, None))(
        keys, batch, cfg, rates, log_kernel)
    assert out.weight_matrix.shape == (8, cfg.N_max, cfg.N_max)


def test_explicit_n_edges_overrides_measurement(g, cfg):
    """Both operators must accept an externally supplied E — that is how
    mutate() drives them from one shared pre-mutation count."""
    big = add_edges(jax.random.PRNGKey(21), g, cfg, p_per_edge=0.1,
                    n_edges=jnp.asarray(1000))
    small = add_edges(jax.random.PRNGKey(21), g, cfg, p_per_edge=0.1,
                      n_edges=jnp.asarray(10))
    assert int(count_active_edges(big)) > int(count_active_edges(small))


# ── Node operator gating ──────────────────────────────────────────────────────

def test_node_ops_default_off():
    assert Config(N_max=16, grid_W=4, grid_H=4).node_ops_enabled is False


@pytest.mark.parametrize("mode", ["grid", "uniform"])
def test_node_ops_rejected_on_fixed_lattice_arms(mode):
    """Enforced in Config rather than left to the runner: remove_node would
    punch holes in the substrate under study."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, init_mode=mode, node_ops_enabled=True)


def test_node_ops_allowed_on_sparse_arm(cfg_sparse):
    assert cfg_sparse.node_ops_enabled is True


def test_add_node_self_disables_on_full_lattice(g, cfg):
    """No free hidden slot, so the update is a no-op even called directly."""
    out = add_node(jax.random.PRNGKey(22), g, cfg)
    assert int(out.active_mask.sum()) == int(g.active_mask.sum())
    assert jnp.array_equal(out.edge_mask, g.edge_mask)


# ── Node operators on the sparse arm ──────────────────────────────────────────

def test_add_node_activates_a_slot(cfg_sparse):
    gs = sparse_genome(jax.random.PRNGKey(23), cfg_sparse)
    out = add_node(jax.random.PRNGKey(24), gs, cfg_sparse)
    assert int(out.active_mask.sum()) == int(gs.active_mask.sum()) + 1


def test_add_node_wires_in_and_out(cfg_sparse):
    """Both directions, so prune_isolated does not immediately undo it."""
    gs = sparse_genome(jax.random.PRNGKey(25), cfg_sparse)
    out = add_node(jax.random.PRNGKey(26), gs, cfg_sparse)
    new = jnp.where(out.active_mask & ~gs.active_mask)[0]
    assert len(new) == 1
    slot = int(new[0])
    assert bool(out.edge_mask[:, slot].any())
    assert bool(out.edge_mask[slot, :].any())


def test_remove_node_deactivates_and_clears_edges(cfg_sparse):
    gs = sparse_genome(jax.random.PRNGKey(27), cfg_sparse)
    out = remove_node(jax.random.PRNGKey(28), gs, cfg_sparse)
    assert int(out.active_mask.sum()) == int(gs.active_mask.sum()) - 1
    slot = int(jnp.where(gs.active_mask & ~out.active_mask)[0][0])
    assert not bool(out.edge_mask[slot].any())
    assert not bool(out.edge_mask[:, slot].any())


def test_remove_node_protects_io(cfg_sparse):
    gs = sparse_genome(jax.random.PRNGKey(29), cfg_sparse)
    out = gs
    for i in range(40):
        out = remove_node(jax.random.fold_in(jax.random.PRNGKey(30), i), out, cfg_sparse)
    assert jnp.all(out.active_mask[:cfg_sparse.n_in])
    assert jnp.all(out.active_mask[-cfg_sparse.n_out:])


def test_sparse_arm_can_grow(cfg_sparse):
    """The sparse arm's identity is 'start small and grow'.  Without node ops
    it could only shrink, and comparing the lattice against a handicapped
    baseline would not be a fair test."""
    r = MutationRates(add_node_prob=1.0, remove_node_prob=0.0, edge_churn=0.0)
    gs = sparse_genome(jax.random.PRNGKey(31), cfg_sparse)
    start = int(gs.active_mask.sum())
    for i in range(10):
        gs = mutate(jax.random.fold_in(jax.random.PRNGKey(32), i), gs, cfg_sparse, r)
    assert int(gs.active_mask.sum()) > start


# ── mutate (combined) ─────────────────────────────────────────────────────────

def test_mutate_returns_valid_genome(g, cfg, rates):
    assert validate_genome(mutate(jax.random.PRNGKey(33), g, cfg, rates), cfg)


def test_mutate_valid_on_sparse_arm(cfg_sparse, rates):
    gs = sparse_genome(jax.random.PRNGKey(34), cfg_sparse)
    assert validate_genome(mutate(jax.random.PRNGKey(35), gs, cfg_sparse, rates),
                           cfg_sparse)


def test_mutate_is_deterministic(g, cfg, rates):
    key = jax.random.PRNGKey(36)
    a = mutate(key, g, cfg, rates)
    b = mutate(key, g, cfg, rates)
    assert jnp.allclose(a.weight_matrix, b.weight_matrix)
    assert jnp.array_equal(a.edge_mask, b.edge_mask)


def test_mutate_preserves_io_slots(g, cfg, rates):
    out = mutate(jax.random.PRNGKey(37), g, cfg, rates)
    assert jnp.all(out.active_mask[:cfg.n_in])
    assert jnp.all(out.active_mask[-cfg.n_out:])


def test_mutate_edge_mask_stays_within_active_pairs(g, cfg, rates):
    """A violation is silent: effective_weights masks it, but edge_count_cost
    overcounts and every penalty is miscalibrated."""
    gg = g
    for i in range(50):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(38), i), gg, cfg, rates)
        pairs = gg.active_mask[:, None] & gg.active_mask[None, :]
        assert jnp.all(gg.edge_mask <= pairs)


def test_mutate_never_produces_nan(g, cfg, rates):
    gg = g
    for i in range(50):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(39), i), gg, cfg, rates)
    assert not jnp.any(jnp.isnan(gg.weight_matrix))
    assert not jnp.any(jnp.isnan(gg.tau))
    assert not jnp.any(jnp.isnan(gg.bias))


def test_mutate_is_vmappable(g, cfg, rates):
    keys = jax.random.split(jax.random.PRNGKey(40), 8)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 8), g)
    out = jax.vmap(mutate, in_axes=(0, 0, None, None))(keys, batch, cfg, rates)
    assert out.weight_matrix.shape == (8, cfg.N_max, cfg.N_max)


def test_mutate_diversifies_identical_genomes(g, cfg, rates):
    """Grid genomes start structurally identical, so mutate is the ONLY source
    of topological diversity in the population."""
    outs = [mutate(jax.random.PRNGKey(50 + i), g, cfg, rates) for i in range(20)]
    assert any(not jnp.array_equal(outs[0].edge_mask, o.edge_mask) for o in outs[1:])


def test_mutate_zero_rates_is_identity(g, cfg):
    r = MutationRates(weight_sigma=0.0, tau_sigma=0.0, bias_sigma=0.0,
                      type_flip_prob=0.0, edge_churn=0.0)
    out = mutate(jax.random.PRNGKey(41), g, cfg, r)
    assert jnp.allclose(out.weight_matrix, g.weight_matrix)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)