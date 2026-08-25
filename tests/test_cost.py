"""
Tests for cost.py under proportional-only penalties.

Two failure modes drive this file:

  1. Sign flip.  `f_raw * (1 - frac*C/C0)` goes negative once the bracket
     exceeds 1, and because f_raw >= 0 always, tournament selection then
     prefers the WORSE network.  It does not crash; it silently inverts
     evolution.  test_ordering_preserved_under_extreme_penalty is the guard.

  2. C0 miscalibration.  The ctrnn_lattice_evo references (C0_edge=154,
     C0_wiring=77) were measured on a sparse random init.  A lattice is ~7x
     and ~3x those values, so an uncalibrated frac collapses the population
     at generation 0 and reads as "locality fails".
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config
from ctrnn_lattice_evo.genome import grid_genome, uniform_genome
from ctrnn_lattice_evo.cost import (
    edge_count_cost,
    dist_cost,
    adjusted_fitness,
    reference_costs,
)
from ctrnn_lattice_evo.topology import dist_matrix, local_mask


@pytest.fixture
def cfg():
    return Config(N_max=16, n_out=1, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture
def dist(cfg):
    return dist_matrix(cfg.grid_W, cfg.grid_H)


@pytest.fixture
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


# ── edge_count_cost ──────────────────────────────────────────────────────────

def test_edge_count_cost_zero_no_edges(g):
    g0 = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    assert float(edge_count_cost(g0)) == pytest.approx(0.0)


def test_edge_count_cost_nonnegative(g):
    assert float(edge_count_cost(g)) >= 0.0


def test_edge_count_cost_equals_mask_sum(g):
    assert float(edge_count_cost(g)) == pytest.approx(float(g.edge_mask.sum()))


def test_edge_count_cost_grid_known_value(g):
    """4x4 at r=1 — the number every other figure in this file derives from."""
    assert int(edge_count_cost(g)) == 84


def test_edge_count_cost_ignores_inactive_pairs(g):
    """Edges to a deactivated neuron must not be counted."""
    g2 = dataclasses.replace(g, active_mask=g.active_mask.at[5].set(False))
    assert float(edge_count_cost(g2)) < float(edge_count_cost(g))


# ── dist_cost ────────────────────────────────────────────────────────────────

def test_dist_cost_zero_no_edges(g, dist):
    g0 = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    assert float(dist_cost(g0, dist)) == pytest.approx(0.0)


def test_dist_cost_nonnegative(g, dist):
    assert float(dist_cost(g, dist)) >= 0.0


def test_dist_cost_scales_with_distance(cfg, dist):
    """Same edge COUNT, different reach: the long-range set must cost more.

    Replaces ctrnn_lattice_evo's test that zeroed the position array — positions are
    no longer a genome field, so the comparison has to be between two edge
    sets on one fixed lattice.
    """
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    n = int(g.edge_mask.sum())

    # Long-range set of the same cardinality: take the n most distant pairs.
    flat = dist.reshape(-1)
    order = jnp.argsort(-flat)               # descending distance
    far = jnp.zeros_like(flat, dtype=bool).at[order[:n]].set(True).reshape(dist.shape)
    far = far & (dist > 0)

    g_far = dataclasses.replace(g, edge_mask=far)
    assert int(far.sum()) == pytest.approx(n, rel=0.05)
    assert float(dist_cost(g_far, dist)) > float(dist_cost(g, dist))


def test_dist_cost_equals_edge_count_at_radius_one(cfg, dist):
    """At r=1 every lattice edge has length 1, so dist_cost degenerates to
    edge_count_cost.  The two penalty axes are collinear here — meaningful
    only at r >= 2, where lengths take more than one value."""
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    assert float(dist_cost(g, dist)) == pytest.approx(float(edge_count_cost(g)))


def test_dist_cost_separates_from_edge_count_at_radius_two():
    """The precondition for treating length as an independent penalty axis."""
    cfg2 = Config(N_max=64, n_out=1, grid_W=8, grid_H=8, grid_r=2)
    d2 = dist_matrix(8, 8)
    g2 = grid_genome(jax.random.PRNGKey(0), cfg2)
    assert float(dist_cost(g2, d2)) > float(edge_count_cost(g2))


# ── reference_costs — the calibration guard ──────────────────────────────────

def test_reference_costs_match_grid_init(cfg, dist):
    """C0 must be measured from the actual lattice, not inherited from
    ctrnn_lattice_evo's sparse-init constants."""
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    C0_edge, C0_dist = reference_costs(cfg)
    assert float(edge_count_cost(g)) == pytest.approx(C0_edge, rel=0.05)
    assert float(dist_cost(g, dist)) == pytest.approx(C0_dist, rel=0.05)


def test_reference_costs_are_not_legacy_constants(cfg):
    """Explicitly reject the ctrnn_lattice_evo values — inheriting them is a ~7x
    over-penalty at generation 0 and collapses the run."""
    C0_edge, C0_dist = reference_costs(cfg)
    assert C0_edge != pytest.approx(154.0, rel=0.01)
    assert C0_dist != pytest.approx(77.0, rel=0.01)


def test_reference_costs_scale_with_lattice_size():
    small = reference_costs(Config(N_max=16, n_out=1, grid_W=4, grid_H=4, grid_r=1))
    large = reference_costs(Config(N_max=64, n_out=1, grid_W=8, grid_H=8, grid_r=2))
    assert large[0] > small[0]


# ── adjusted_fitness — the sign-flip guards ──────────────────────────────────

def test_no_penalty_returns_raw(g, cfg, dist):
    assert float(adjusted_fitness(1.0, g, 0.0, cfg, dist)) == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["edge_frac", "dist_frac", "act_frac"])
def test_each_frac_reduces_fitness(field, g, cfg, dist):
    c_act = 0.5
    c2 = dataclasses.replace(cfg, **{field: 0.2})
    assert float(adjusted_fitness(1.0, g, c_act, c2, dist)) < 1.0


def test_multiplier_never_negative(g, cfg, dist):
    """The clamp.  Without it the bracket goes negative and selection inverts."""
    c2 = dataclasses.replace(cfg, edge_frac=5.0)
    assert float(adjusted_fitness(1.0, g, 0.0, c2, dist)) >= 0.0


def test_ordering_preserved_under_extreme_penalty(g, cfg, dist):
    """THE test.  With an unclamped multiplier of -4, f_raw=0.9 maps to -3.6
    and f_raw=0.1 to -0.4, so tournament selection picks the worse network.
    Fails against an unclamped implementation; passes once clamped at 0."""
    c2 = dataclasses.replace(cfg, edge_frac=5.0)
    hi = float(adjusted_fitness(0.9, g, 0.0, c2, dist))
    lo = float(adjusted_fitness(0.1, g, 0.0, c2, dist))
    assert hi >= lo


def test_ordering_preserved_across_frac_sweep(g, cfg, dist):
    """Monotonicity must hold at every penalty strength, not just the extreme."""
    for frac in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0]:
        c2 = dataclasses.replace(cfg, edge_frac=frac)
        hi = float(adjusted_fitness(0.9, g, 0.0, c2, dist))
        lo = float(adjusted_fitness(0.1, g, 0.0, c2, dist))
        assert hi >= lo, f"ordering inverted at edge_frac={frac}"


def test_penalty_at_init_equals_frac(g, cfg, dist):
    """frac is defined as the fraction of fitness surrendered at reference
    cost, so a genome AT reference cost must lose exactly that fraction."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    assert float(adjusted_fitness(1.0, g, 0.0, c2, dist)) == pytest.approx(0.8, rel=0.05)


def test_penalty_is_proportional_to_raw_fitness(g, cfg, dist):
    """The point of the proportional form: the same frac works whether
    f_raw ~ 0.009 or ~ 0.9, with no per-experiment recalibration."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    for f_raw in [0.01, 0.1, 0.5, 0.9]:
        got = float(adjusted_fitness(f_raw, g, 0.0, c2, dist))
        assert got == pytest.approx(0.8 * f_raw, rel=0.05)


def test_fewer_edges_scores_higher(g, cfg, dist):
    """Pruning must actually pay under a live penalty."""
    c2 = dataclasses.replace(cfg, edge_frac=0.3)
    half = g.edge_mask & (jnp.arange(g.edge_mask.size).reshape(g.edge_mask.shape) % 2 == 0)
    g_half = dataclasses.replace(g, edge_mask=half)
    assert float(adjusted_fitness(1.0, g_half, 0.0, c2, dist)) > \
           float(adjusted_fitness(1.0, g, 0.0, c2, dist))


def test_zero_raw_fitness_stays_zero(g, cfg, dist):
    """Multiplicative penalties cannot punish a network that scored nothing —
    this is what removes the early-generation over-pruning pressure."""
    c2 = dataclasses.replace(cfg, edge_frac=0.5)
    assert float(adjusted_fitness(0.0, g, 0.0, c2, dist)) == pytest.approx(0.0)


def test_combined_fracs_accumulate(g, cfg, dist):
    single = dataclasses.replace(cfg, edge_frac=0.2)
    both = dataclasses.replace(cfg, edge_frac=0.2, act_frac=0.2)
    assert float(adjusted_fitness(1.0, g, 0.5, both, dist)) < \
           float(adjusted_fitness(1.0, g, 0.5, single, dist))


# ── Cross-arm comparability ──────────────────────────────────────────────────

def test_grid_and_uniform_pay_equal_edge_penalty(cfg, dist):
    """Matched edge count means the edge penalty cannot explain a difference
    between the two arms — only locality can."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(adjusted_fitness(1.0, gg, 0.0, c2, dist)) == \
           pytest.approx(float(adjusted_fitness(1.0, gu, 0.0, c2, dist)), rel=0.10)


def test_grid_pays_less_dist_penalty_than_uniform(cfg, dist):
    """At equal edge count the lattice is cheaper in wire length — the
    mechanism by which a distance penalty preserves locality."""
    c2 = dataclasses.replace(cfg, dist_frac=0.2)
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(adjusted_fitness(1.0, gg, 0.0, c2, dist)) > \
           float(adjusted_fitness(1.0, gu, 0.0, c2, dist))