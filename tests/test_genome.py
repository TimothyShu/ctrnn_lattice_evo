"""
Tests for genome.py under the lattice substrate.

Changes from ctrnn_lattice_evo:
  - the `position` field is gone; geometry lives in topology.py
  - three constructors instead of one: grid / uniform / sparse
  - grid genomes are structurally identical at generation 0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config, E
from ctrnn_lattice_evo.genome import (
    grid_genome,
    uniform_genome,
    sparse_genome,
    prune_isolated,
    effective_weights,
    validate_genome,
)
from ctrnn_lattice_evo.topology import local_mask, expected_edges


@pytest.fixture
def cfg():
    """4x4 lattice at r=1 — 84 directed edges, 35% of the 240 possible."""
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


# ── Config consistency ───────────────────────────────────────────────────────

def test_config_rejects_non_matching_grid():
    """N_max must equal grid_W * grid_H — a mismatch corrupts every mask."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=8, grid_H=8, grid_r=1)


def test_config_grid_defaults_are_square():
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.grid_W * cfg.grid_H == cfg.N_max


# ── Invariants (all three modes) ─────────────────────────────────────────────

@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_genome_is_valid(ctor, cfg):
    assert validate_genome(ctor(jax.random.PRNGKey(0), cfg), cfg)


@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_io_slots_always_active(ctor, cfg):
    gg = ctor(jax.random.PRNGKey(1), cfg)
    assert jnp.all(gg.active_mask[:cfg.n_in]), "Input slots must be active"
    assert jnp.all(gg.active_mask[-cfg.n_out:]), "Output slots must be active"


@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_io_slots_are_excitatory(ctor, cfg):
    gg = ctor(jax.random.PRNGKey(2), cfg)
    assert jnp.all(gg.neuron_type[:cfg.n_in] == E)
    assert jnp.all(gg.neuron_type[-cfg.n_out:] == E)


@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_weight_magnitudes_nonnegative(ctor, cfg):
    gg = ctor(jax.random.PRNGKey(3), cfg)
    assert jnp.all(gg.weight_matrix >= 0)


@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_tau_within_type_ranges(ctor, cfg):
    gg = ctor(jax.random.PRNGKey(4), cfg)
    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    lo, hi = tau_lo[gg.neuron_type], tau_hi[gg.neuron_type]
    assert jnp.all(jnp.where(gg.active_mask, gg.tau >= lo - 1e-4, True))
    assert jnp.all(jnp.where(gg.active_mask, gg.tau <= hi + 1e-4, True))


@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_no_edges_touch_inactive_neurons(ctor, cfg):
    """edge_mask <= active_pairs.  A violation is silent: effective_weights
    masks it out, but edge_count_cost overcounts and miscalibrates penalties."""
    gg = ctor(jax.random.PRNGKey(5), cfg)
    active_pairs = gg.active_mask[:, None] & gg.active_mask[None, :]
    assert jnp.all(gg.edge_mask <= active_pairs)


# ── position field is gone ───────────────────────────────────────────────────

def test_genome_has_no_position_field(g):
    assert not hasattr(g, "position"), "position must live in topology.py, not the genome"


def test_genome_has_six_leaves(g):
    """Six fields, not seven — logger's _GENOME_FIELDS must match."""
    leaves, _ = jax.tree_util.tree_flatten(g)
    assert len(leaves) == 6


def test_genome_is_jax_pytree(g):
    leaves, treedef = jax.tree_util.tree_flatten(g)
    g2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(g.tau, g2.tau)
    assert jnp.array_equal(g.edge_mask, g2.edge_mask)


# ── Grid mode ────────────────────────────────────────────────────────────────

def test_grid_genome_is_fully_active(g, cfg):
    """Every lattice slot is occupied — no node growth, only node death."""
    assert int(g.active_mask.sum()) == cfg.N_max


def test_grid_genome_matches_local_mask(g, cfg):
    assert jnp.array_equal(g.edge_mask, local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H))


def test_grid_genome_edge_count(g, cfg):
    assert int(g.edge_mask.sum()) == expected_edges(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    assert int(g.edge_mask.sum()) == 84


def test_grid_genomes_structurally_identical(cfg):
    """Generation-0 diversity is weights/tau/bias/type only — NOT topology.

    This is a deliberate departure from ctrnn_lattice_evo, where every individual
    started from its own random edge set.  The population now searches from a
    single topology, which narrows initial structural diversity.  Asserted
    here so the change is a decision rather than a surprise in the pilot.
    """
    a = grid_genome(jax.random.PRNGKey(0), cfg)
    b = grid_genome(jax.random.PRNGKey(99), cfg)
    assert jnp.array_equal(a.edge_mask, b.edge_mask)
    assert jnp.array_equal(a.active_mask, b.active_mask)
    assert not jnp.allclose(a.weight_matrix, b.weight_matrix)
    assert not jnp.allclose(a.tau, b.tau)


def test_grid_genome_survives_prune_isolated(g, cfg):
    """A fresh lattice must be a fixed point — every slot has in- and out-edges."""
    pruned = prune_isolated(g, cfg)
    assert jnp.array_equal(pruned.active_mask, g.active_mask)
    assert jnp.array_equal(pruned.edge_mask, g.edge_mask)


# ── Uniform mode — the locality control ──────────────────────────────────────

def test_uniform_genome_matches_grid_edge_count(cfg):
    """Same density as grid, no spatial structure.  If these diverge, the arm
    that isolates locality from density is confounded."""
    e_grid = int(grid_genome(jax.random.PRNGKey(0), cfg).edge_mask.sum())
    e_unif = int(uniform_genome(jax.random.PRNGKey(0), cfg).edge_mask.sum())
    assert e_unif == pytest.approx(e_grid, rel=0.10)


def test_uniform_genome_is_not_local(cfg):
    """Most uniform edges must fall outside the lattice mask, or the control
    is not controlling for anything."""
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    local_frac = float((gu.edge_mask & m).sum()) / float(gu.edge_mask.sum())
    assert local_frac < 0.60


def test_uniform_genome_is_fully_active(cfg):
    """Matched to grid on node count so N^2 compute is identical across arms."""
    assert int(uniform_genome(jax.random.PRNGKey(0), cfg).active_mask.sum()) == cfg.N_max


def test_uniform_genomes_differ_between_seeds(cfg):
    a = uniform_genome(jax.random.PRNGKey(0), cfg)
    b = uniform_genome(jax.random.PRNGKey(99), cfg)
    assert not jnp.array_equal(a.edge_mask, b.edge_mask)


# ── Sparse mode — the legacy regime ──────────────────────────────────────────

def test_sparse_genome_is_sparser_than_grid(cfg):
    e_grid = int(grid_genome(jax.random.PRNGKey(0), cfg).edge_mask.sum())
    e_sparse = int(sparse_genome(jax.random.PRNGKey(0), cfg).edge_mask.sum())
    assert e_sparse < e_grid


def test_sparse_genome_has_inactive_slots(cfg):
    """Starts small and grows — the regime the lattice is being compared against."""
    gs = sparse_genome(jax.random.PRNGKey(0), cfg)
    assert int(gs.active_mask.sum()) < cfg.N_max


# ── Batching ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctor", [grid_genome, uniform_genome, sparse_genome])
def test_constructor_is_vmappable(ctor, cfg):
    """Each constructor must vmap over keys directly.  Mode is chosen by
    picking the function, not by passing a string — a Python string cannot be
    a traced argument."""
    keys = jax.random.split(jax.random.PRNGKey(7), 8)
    batch = jax.vmap(ctor, in_axes=(0, None))(keys, cfg)
    assert batch.active_mask.shape == (8, cfg.N_max)
    assert batch.weight_matrix.shape == (8, cfg.N_max, cfg.N_max)
    assert batch.edge_mask.shape == (8, cfg.N_max, cfg.N_max)


def test_batched_grid_shares_one_topology(cfg):
    keys = jax.random.split(jax.random.PRNGKey(7), 8)
    batch = jax.vmap(grid_genome, in_axes=(0, None))(keys, cfg)
    for i in range(1, 8):
        assert jnp.array_equal(batch.edge_mask[0], batch.edge_mask[i])


# ── effective_weights ────────────────────────────────────────────────────────

def test_effective_weights_respects_edge_mask(g):
    W = effective_weights(g)
    assert jnp.all(jnp.where(g.edge_mask, True, W == 0.0))


def test_effective_weights_dale_sign(g):
    """Sign is set by the source neuron's type, not the target's."""
    W = effective_weights(g)
    excit = (g.neuron_type == E)[None, :] & g.edge_mask
    assert jnp.all(jnp.where(excit, W >= 0.0, True))