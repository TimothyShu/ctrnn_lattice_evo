"""
Tests for the edge-removal operator and pruning throughput.

Why this file exists: ctrnn_lattice_evo's remove_edge deletes exactly ONE edge per
genome per generation (argmax over noise), fired at remove_edge_prob=0.1.
That caps the lineage removal rate at 1 edge/generation regardless of
population size.  An 8x8 r=2 lattice has 1092 edges, so a 50-70% prune needs
550-760 removal events — unreachable inside 500 generations, and the failure
mode is a run that simply never prunes rather than an error.

remove_edges replaces it with a per-edge Bernoulli draw, so throughput scales
with edge count automatically.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config
from ctrnn_lattice_evo.genome import grid_genome, prune_isolated
from ctrnn_lattice_evo.mutation import MutationRates, remove_edges, add_edge, mutate
from ctrnn_lattice_evo.topology import local_mask


@pytest.fixture
def cfg():
    return Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2)


@pytest.fixture
def small_cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture
def rates():
    """Structural node operators are off — lattice slots are fixed."""
    return MutationRates(
        add_node_prob=0.0,
        remove_node_prob=0.0,
        add_edge_prob=0.1,
        remove_edge_prob=1.0,
    )


# ── remove_edges basics ──────────────────────────────────────────────────────

def test_remove_edges_never_adds(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=0.01)
    assert jnp.all(out.edge_mask <= g.edge_mask)


def test_remove_edges_zero_probability_is_noop(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=0.0)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)


def test_remove_edges_unit_probability_clears_all(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=1.0)
    assert int(out.edge_mask.sum()) == 0


def test_remove_edges_leaves_other_fields_untouched(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=0.05)
    assert jnp.array_equal(out.weight_matrix, g.weight_matrix)
    assert jnp.array_equal(out.tau, g.tau)
    assert jnp.array_equal(out.neuron_type, g.neuron_type)


# ── Throughput — the actual blocker ──────────────────────────────────────────

def test_removal_count_scales_with_edge_count(g, cfg):
    """~p*E removals per call, not ~1.  This is the property ctrnn_lattice_evo's
    single-edge argmax operator lacked."""
    p = 0.05
    start = int(g.edge_mask.sum())
    out = remove_edges(jax.random.PRNGKey(2), g, cfg, p_per_edge=p)
    removed = start - int(out.edge_mask.sum())
    assert removed == pytest.approx(p * start, rel=0.35), \
        f"removed {removed}, expected ~{p * start:.0f}"
    assert removed > 10, "removal rate does not scale with edge count"


def test_removal_reaches_target_density_in_budget(g, cfg):
    """50% pruning must be reachable inside a realistic generation budget.

    Fails outright against a one-edge-per-call operator: 300 calls can remove
    at most 300 of 1092 edges (27%), and that is the optimistic ceiling
    assuming every removal is also selected for.
    """
    start = int(g.edge_mask.sum())
    gg = g
    for i in range(300):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(3), i),
                          gg, cfg, p_per_edge=0.002)
    assert int(gg.edge_mask.sum()) <= 0.5 * start


def test_removal_can_reach_seventy_percent(g, cfg):
    """The upper end of the anticipated 50-70% band."""
    start = int(g.edge_mask.sum())
    gg = g
    for i in range(500):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(4), i),
                          gg, cfg, p_per_edge=0.002)
    assert int(gg.edge_mask.sum()) <= 0.30 * start


def test_removal_rate_is_roughly_geometric(g, cfg):
    """E_n ~ E_0 * (1-p)^n — confirms independent per-edge draws rather than
    a fixed-count removal, which would decay linearly."""
    p, n = 0.01, 50
    gg = g
    for i in range(n):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(5), i),
                          gg, cfg, p_per_edge=p)
    expected = int(g.edge_mask.sum()) * (1 - p) ** n
    assert int(gg.edge_mask.sum()) == pytest.approx(expected, rel=0.20)


# ── Interaction with prune_isolated ──────────────────────────────────────────

def test_stranded_nodes_are_deactivated(g, cfg):
    """remove_edges drops the old stranding guard (invalid under simultaneous
    removal).  prune_isolated is now the sole node-death path — and it is
    one-way: a node that dies cannot come back."""
    gg = remove_edges(jax.random.PRNGKey(6), g, cfg, p_per_edge=0.9)
    gg = prune_isolated(gg, cfg)
    assert int(gg.active_mask.sum()) < cfg.N_max
    assert jnp.all(gg.active_mask[:cfg.n_in])
    assert jnp.all(gg.active_mask[-cfg.n_out:])


def test_node_count_does_not_collapse_at_working_rate(g, cfg):
    """At the intended p, the lattice should thin its edges without losing
    most of its nodes.  If this fails, node death is outrunning edge pruning
    and the experiment measures the wrong thing."""
    gg = g
    for i in range(300):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(7), i),
                          gg, cfg, p_per_edge=0.002)
        gg = prune_isolated(gg, cfg)
    assert int(gg.active_mask.sum()) >= 0.5 * cfg.N_max, \
        f"only {int(gg.active_mask.sum())}/{cfg.N_max} nodes survived"


def test_io_slots_never_deactivated(g, cfg):
    gg = remove_edges(jax.random.PRNGKey(8), g, cfg, p_per_edge=1.0)
    gg = prune_isolated(gg, cfg)
    assert jnp.all(gg.active_mask[:cfg.n_in])
    assert jnp.all(gg.active_mask[-cfg.n_out:])


# ── Locality retention under mutation ────────────────────────────────────────

def test_add_edge_can_leave_the_lattice(g, cfg):
    """add_edge is deliberately unmasked: locality is the initialisation, not
    a hard ceiling, so evolution can buy a long-range shortcut if it earns one."""
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = g
    for i in range(200):
        gg = add_edge(jax.random.fold_in(jax.random.PRNGKey(9), i), gg, cfg)
    assert int((gg.edge_mask & ~m).sum()) > 0


def test_local_fraction_stable_under_mutation(g, cfg, rates):
    """At add_edge_prob=0.1 a lineage sees ~50 additions over 500 generations
    against 1092 lattice edges, so locality should erode by only a few
    percent.  Guards the erosion question empirically instead of by argument.
    """
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    r = dataclasses.replace(rates, remove_edge_prob=0.0)
    gg = g
    for i in range(500):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(10), i), gg, cfg, r)

    frac = float((gg.edge_mask & m).sum()) / float(gg.edge_mask.sum())
    assert frac > 0.85, f"locality eroded to {frac:.2f}"


def test_locality_erodes_faster_without_the_lattice_prior(cfg):
    """Sanity check on the metric: a genome that never started local should
    score far lower on the same measure."""
    from ctrnn_lattice_evo.genome import uniform_genome
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    frac = float((gu.edge_mask & m).sum()) / float(gu.edge_mask.sum())
    assert frac < 0.60


# ── Node operators are disabled ──────────────────────────────────────────────

def test_node_operators_are_off(g, cfg, rates):
    """Lattice slots are fixed: add_node is a no-op with all slots live, and
    remove_node would punch holes in the substrate."""
    assert rates.add_node_prob == 0.0
    assert rates.remove_node_prob == 0.0


def test_mutate_preserves_lattice_size(g, cfg, rates):
    r = dataclasses.replace(rates, remove_edge_prob=0.0, add_edge_prob=0.0)
    gg = mutate(jax.random.PRNGKey(11), g, cfg, r)
    assert int(gg.active_mask.sum()) == cfg.N_max