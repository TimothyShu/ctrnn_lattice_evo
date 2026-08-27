"""
Tests — edge-removal throughput.

Why this file exists: ctrnn_evo's remove_edge deleted exactly ONE edge per
genome per generation (an argmax over noise), fired at remove_edge_prob=0.1.
That capped the lineage removal rate at 1 edge/generation regardless of
population size.  An 8x8 r=2 lattice has 1092 edges, so 50-70% pruning needs
550-760 removal events — unreachable inside 500 generations, and the failure
mode is a run that simply never prunes rather than an error.

remove_edges replaces it with an independent per-edge Bernoulli draw, so
throughput scales with edge count automatically.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config
from ctrnn_lattice_evo.genome import grid_genome, uniform_genome, prune_isolated
from ctrnn_lattice_evo.mutation import MutationRates, remove_edges, add_edge, mutate
from ctrnn_lattice_evo.topology import local_mask


@pytest.fixture(scope="module")
def cfg():
    """The production lattice — 1092 edges is the number that matters here."""
    return Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2)


@pytest.fixture(scope="module")
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def rates():
    return MutationRates()


# ── Basics ────────────────────────────────────────────────────────────────────

def test_never_adds(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=0.01)
    assert jnp.all(out.edge_mask <= g.edge_mask)


def test_zero_probability_is_noop(g, cfg):
    out = remove_edges(jax.random.PRNGKey(2), g, cfg, p_per_edge=0.0)
    assert jnp.array_equal(out.edge_mask, g.edge_mask)


def test_unit_probability_clears_all(g, cfg):
    out = remove_edges(jax.random.PRNGKey(3), g, cfg, p_per_edge=1.0)
    assert int(out.edge_mask.sum()) == 0


def test_leaves_other_fields_untouched(g, cfg):
    out = remove_edges(jax.random.PRNGKey(4), g, cfg, p_per_edge=0.05)
    assert jnp.array_equal(out.weight_matrix, g.weight_matrix)
    assert jnp.array_equal(out.tau, g.tau)
    assert jnp.array_equal(out.neuron_type, g.neuron_type)
    assert jnp.array_equal(out.active_mask, g.active_mask)


# ── Throughput — the actual blocker ───────────────────────────────────────────

def test_removal_count_scales_with_edge_count(g, cfg):
    """~p*E removals per call, not ~1.  This is the property the single-edge
    argmax operator lacked."""
    p = 0.05
    start = int(g.edge_mask.sum())
    out = remove_edges(jax.random.PRNGKey(5), g, cfg, p_per_edge=p)
    removed = start - int(out.edge_mask.sum())
    assert removed == pytest.approx(p * start, rel=0.35), \
        f"removed {removed}, expected ~{p * start:.0f}"
    assert removed > 10, "removal rate does not scale with edge count"


def test_removal_reaches_target_density_in_budget(g, cfg, rates):
    """50% pruning must be reachable inside a realistic generation budget.

    Fails outright against a one-edge-per-call operator: 300 calls can remove
    at most 300 of 1092 edges (27%), and that is the optimistic ceiling
    assuming every removal is also selected for.
    """
    start = int(g.edge_mask.sum())
    gg = g
    for i in range(300):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(6), i),
                          gg, cfg, p_per_edge=rates.remove_edge_p_per_edge)
    assert int(gg.edge_mask.sum()) <= 0.5 * start


def test_removal_can_reach_seventy_percent(g, cfg, rates):
    """The upper end of the anticipated 50-70% band."""
    start = int(g.edge_mask.sum())
    gg = g
    for i in range(500):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(7), i),
                          gg, cfg, p_per_edge=rates.remove_edge_p_per_edge)
    assert int(gg.edge_mask.sum()) <= 0.30 * start


def test_decay_is_geometric(g, cfg):
    """E_n ~ E_0 * (1-p)^n confirms independent per-edge draws; a fixed-count
    removal would decay linearly instead."""
    p, n = 0.01, 50
    gg = g
    for i in range(n):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(8), i),
                          gg, cfg, p_per_edge=p)
    expected = int(g.edge_mask.sum()) * (1 - p) ** n
    assert int(gg.edge_mask.sum()) == pytest.approx(expected, rel=0.20)


def test_default_rate_is_in_the_useful_range(g, cfg, rates):
    """The default must clear ~50% within a few hundred generations — too low
    and nothing prunes, too high and the network is stripped before any
    strategy evolves."""
    start = int(g.edge_mask.sum())
    gg = g
    for i in range(300):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(9), i), gg, cfg,
                          p_per_edge=rates.remove_edge_p_per_edge)
    frac = int(gg.edge_mask.sum()) / start
    assert 0.2 <= frac <= 0.7, f"default rate leaves {frac:.2f} of edges"


# ── Interaction with prune_isolated ───────────────────────────────────────────

def test_stranded_nodes_are_deactivated(g, cfg):
    """remove_edges drops the old stranding guard, which precomputed
    out_degree <= 1 and is invalid under simultaneous removal.  prune_isolated
    is now the sole node-death path — and it is one-way."""
    gg = remove_edges(jax.random.PRNGKey(10), g, cfg, p_per_edge=0.9)
    gg = prune_isolated(gg, cfg)
    assert int(gg.active_mask.sum()) < cfg.N_max
    assert jnp.all(gg.active_mask[:cfg.n_in])
    assert jnp.all(gg.active_mask[-cfg.n_out:])


def test_io_slots_never_deactivated(g, cfg):
    gg = remove_edges(jax.random.PRNGKey(11), g, cfg, p_per_edge=1.0)
    gg = prune_isolated(gg, cfg)
    assert jnp.all(gg.active_mask[:cfg.n_in])
    assert jnp.all(gg.active_mask[-cfg.n_out:])


def test_node_count_does_not_collapse_at_working_rate(g, cfg, rates):
    """At the intended rate the lattice should thin its edges without losing
    most of its nodes.  If this fails, node death is outrunning edge pruning
    and the experiment is measuring the wrong thing.

    The 0.5 threshold is a judgement call, not a derived bound — a failure
    here is a finding about prune_isolated being one-way, not necessarily a
    bug.
    """
    gg = g
    for i in range(300):
        gg = remove_edges(jax.random.fold_in(jax.random.PRNGKey(12), i), gg, cfg,
                          p_per_edge=rates.remove_edge_p_per_edge)
        gg = prune_isolated(gg, cfg)
    assert int(gg.active_mask.sum()) >= 0.5 * cfg.N_max, \
        f"only {int(gg.active_mask.sum())}/{cfg.N_max} nodes survived"


def test_node_death_is_one_way(g, cfg):
    """No node operator on the grid arm means a slot that dies cannot return."""
    r = MutationRates(add_edge_prob=1.0, remove_edge_p_per_edge=0.5)
    gg = prune_isolated(remove_edges(jax.random.PRNGKey(13), g, cfg, p_per_edge=0.8), cfg)
    after = int(gg.active_mask.sum())
    for i in range(20):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(14), i), gg, cfg, r)
    assert int(gg.active_mask.sum()) <= after


# ── Locality retention ────────────────────────────────────────────────────────

def test_local_fraction_stable_under_mutation(g, cfg, rates):
    """At add_edge_prob=0.1 a lineage sees ~50 additions over 500 generations
    against 1092 lattice edges, so locality should erode by only a few
    percent.  Guards the erosion question empirically rather than by argument.
    """
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    r = dataclasses.replace(rates, remove_edge_p_per_edge=0.0)
    gg = g
    for i in range(500):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(15), i), gg, cfg, r)
    frac = float((gg.edge_mask & m).sum()) / float(gg.edge_mask.sum())
    assert frac > 0.85, f"locality eroded to {frac:.2f}"


def test_uniform_arm_local_fraction_is_the_floor(cfg):
    """A random digraph at the same density lands ~n_edges/(N^2-N) inside the
    lattice ball by chance — about 0.27 here, NOT 0.  That is the baseline the
    grid arm's local_fraction must be read against."""
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    frac = float((gu.edge_mask & m).sum()) / float(gu.edge_mask.sum())
    assert 0.15 < frac < 0.45


def test_add_edge_can_leave_the_lattice(g, cfg):
    """add_edge is deliberately unmasked: evolution may buy a long-range
    shortcut, and the distance penalty decides whether it keeps it."""
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = g
    for i in range(200):
        gg = add_edge(jax.random.fold_in(jax.random.PRNGKey(16), i), gg, cfg)
    assert int((gg.edge_mask & ~m).sum()) > 0