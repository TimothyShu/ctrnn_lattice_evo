"""
Tests for cost.py — proportional penalties only.

The failure this file exists to catch is a sign flip, not a crash.

    f = f_raw * (1 - frac*C/C0)

goes negative once the bracket does, and because f_raw >= 0 always, a BETTER
network then maps to a MORE negative adjusted fitness.  Tournament selection
then prefers the worse individual: evolution runs backwards, silently, with no
error and no NaN.  penalty_scale clamps at 0 to prevent it, and
test_ordering_preserved_* is the guard.

Secondary concern: C0 miscalibration.  ctrnn_evo's references (C0_edge=154,
C0_wiring=77) were measured on a sparse random init and are several times off
for a lattice, which drives the bracket negative at generation 0.  Config now
derives them from the lattice; test_config covers that.
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
    penalty_scale,
    adjusted_fitness,
    cfg_dist_matrix,
)
from ctrnn_lattice_evo.topology import dist_matrix, local_mask


@pytest.fixture
def cfg():
    """4x4 r=1 — 84 edges, C0_edge = C0_dist = 84 (collinear at r=1)."""
    return Config(N_max=16, n_out=1, grid_W=4, grid_H=4, grid_r=1)


@pytest.fixture
def cfg2():
    """8x8 r=2 — the production lattice; 1092 edges, C0_dist = 1764."""
    return Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2)


@pytest.fixture
def dist(cfg):
    return dist_matrix(cfg.grid_W, cfg.grid_H)


@pytest.fixture
def g(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


# ── edge_count_cost ───────────────────────────────────────────────────────────

def test_edge_count_cost_zero_no_edges(g):
    g0 = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    assert float(edge_count_cost(g0)) == pytest.approx(0.0)


def test_edge_count_cost_nonnegative(g):
    assert float(edge_count_cost(g)) >= 0.0


def test_edge_count_cost_equals_mask_sum(g):
    assert float(edge_count_cost(g)) == pytest.approx(float(g.edge_mask.sum()))


def test_edge_count_cost_grid_known_value(g):
    """4x4 at r=1 — the number every other figure here derives from."""
    assert int(edge_count_cost(g)) == 84


def test_edge_count_cost_production_known_value(cfg2):
    g2 = grid_genome(jax.random.PRNGKey(0), cfg2)
    assert int(edge_count_cost(g2)) == 1092


def test_edge_count_cost_ignores_inactive_pairs(g):
    """Edges touching a deactivated neuron must not be counted — otherwise
    every penalty is miscalibrated while the forward pass looks fine."""
    g2 = dataclasses.replace(g, active_mask=g.active_mask.at[5].set(False))
    assert float(edge_count_cost(g2)) < float(edge_count_cost(g))


def test_edge_count_cost_matches_c0_at_init(g, cfg):
    """A fresh lattice sits exactly AT reference cost, by construction."""
    assert float(edge_count_cost(g)) == pytest.approx(cfg.C0_edge)


# ── dist_cost ─────────────────────────────────────────────────────────────────

def test_dist_cost_zero_no_edges(g, dist):
    g0 = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    assert float(dist_cost(g0, dist)) == pytest.approx(0.0)


def test_dist_cost_nonnegative(g, dist):
    assert float(dist_cost(g, dist)) >= 0.0


def test_dist_cost_matches_c0_at_init(g, cfg, dist):
    assert float(dist_cost(g, dist)) == pytest.approx(cfg.C0_dist)


def test_dist_cost_scales_with_distance(cfg, dist):
    """Same edge COUNT, different reach: the long-range set must cost more.

    Replaces ctrnn_evo's version, which zeroed the position array — positions
    are no longer a genome field, so the comparison is between two edge sets on
    one fixed lattice.
    """
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    n = int(g.edge_mask.sum())

    flat = dist.reshape(-1)
    order = jnp.argsort(-flat)                       # most distant pairs first
    far = jnp.zeros_like(flat, dtype=bool).at[order[:n]].set(True)
    far = far.reshape(dist.shape) & (dist > 0)

    g_far = dataclasses.replace(g, edge_mask=far)
    assert int(far.sum()) == pytest.approx(n, rel=0.05)
    assert float(dist_cost(g_far, dist)) > float(dist_cost(g, dist))


def test_dist_cost_equals_edge_count_at_radius_one(g, dist):
    """At r=1 every lattice edge has length 1, so dist_cost degenerates to
    edge_count_cost.  The two penalty axes are COLLINEAR here — reporting them
    as independent regressors at this radius would be an artifact."""
    assert float(dist_cost(g, dist)) == pytest.approx(float(edge_count_cost(g)))


def test_dist_cost_separates_from_edge_count_at_radius_two(cfg2):
    """The precondition for treating length as its own penalty axis."""
    d2 = dist_matrix(cfg2.grid_W, cfg2.grid_H)
    g2 = grid_genome(jax.random.PRNGKey(0), cfg2)
    assert float(dist_cost(g2, d2)) > float(edge_count_cost(g2))
    assert float(dist_cost(g2, d2)) == pytest.approx(1764.0)


def test_cfg_dist_matrix_matches_topology(cfg):
    assert jnp.array_equal(cfg_dist_matrix(cfg),
                           dist_matrix(cfg.grid_W, cfg.grid_H))


# ── penalty_scale — the clamp ─────────────────────────────────────────────────

def test_scale_is_one_with_no_penalty(g, cfg, dist):
    assert float(penalty_scale(g, 0.0, cfg, dist)) == pytest.approx(1.0)


def test_scale_equals_one_minus_frac_at_reference(g, cfg, dist):
    """frac is DEFINED as the fraction surrendered at reference cost, so a
    genome sitting at C0 must lose exactly that fraction."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    assert float(penalty_scale(g, 0.0, c2, dist)) == pytest.approx(0.8, rel=1e-4)


def test_scale_never_negative(g, cfg, dist):
    for frac in [1.5, 2.0, 5.0, 50.0]:
        c2 = dataclasses.replace(cfg, edge_frac=frac)
        assert float(penalty_scale(g, 0.0, c2, dist)) >= 0.0


def test_scale_never_exceeds_one(g, cfg, dist):
    for frac in [0.0, 0.1, 0.5, 1.0]:
        c2 = dataclasses.replace(cfg, edge_frac=frac)
        assert float(penalty_scale(g, 0.5, c2, dist)) <= 1.0 + 1e-6


def test_scale_monotone_decreasing_in_frac(g, cfg, dist):
    prev = 1.1
    for frac in [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]:
        c2 = dataclasses.replace(cfg, edge_frac=frac)
        s = float(penalty_scale(g, 0.0, c2, dist))
        assert s <= prev + 1e-6
        prev = s


def test_act_penalty_uses_c0_act_of_one(g, cfg, dist):
    """c_act is mean |tanh(v)| in [0,1], so C0_act=1.0 needs no calibration."""
    c2 = dataclasses.replace(cfg, act_frac=0.3)
    assert float(penalty_scale(g, 1.0, c2, dist)) == pytest.approx(0.7, rel=1e-4)
    assert float(penalty_scale(g, 0.5, c2, dist)) == pytest.approx(0.85, rel=1e-4)


# ── adjusted_fitness ──────────────────────────────────────────────────────────

def test_no_penalty_returns_raw(g, cfg, dist):
    for f_raw in [0.0, 0.01, 0.5, 1.0, 3.0]:
        assert float(adjusted_fitness(f_raw, g, 0.0, cfg, dist)) == pytest.approx(f_raw)


@pytest.mark.parametrize("field", ["edge_frac", "dist_frac", "act_frac"])
def test_each_frac_reduces_fitness(field, g, cfg, dist):
    c2 = dataclasses.replace(cfg, **{field: 0.2})
    assert float(adjusted_fitness(1.0, g, 0.5, c2, dist)) < 1.0


def test_never_negative(g, cfg, dist):
    """The clamp, at the fitness level."""
    c2 = dataclasses.replace(cfg, edge_frac=5.0)
    assert float(adjusted_fitness(1.0, g, 0.0, c2, dist)) >= 0.0


def test_ordering_preserved_under_extreme_penalty(g, cfg, dist):
    """THE test.  With an unclamped multiplier of -4, f_raw=0.9 maps to -3.6
    and f_raw=0.1 to -0.4, so tournament selection picks the WORSE network.
    Fails against an unclamped implementation; passes once clamped at 0."""
    c2 = dataclasses.replace(cfg, edge_frac=5.0)
    hi = float(adjusted_fitness(0.9, g, 0.0, c2, dist))
    lo = float(adjusted_fitness(0.1, g, 0.0, c2, dist))
    assert hi >= lo


def test_ordering_preserved_across_frac_sweep(g, cfg, dist):
    """Monotonicity must hold at every point in the sweep, not just the
    extreme — including values where the bracket crosses zero."""
    for frac in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 5.0]:
        c2 = dataclasses.replace(cfg, edge_frac=frac)
        hi = float(adjusted_fitness(0.9, g, 0.0, c2, dist))
        lo = float(adjusted_fitness(0.1, g, 0.0, c2, dist))
        assert hi >= lo, f"ordering inverted at edge_frac={frac}"


def test_ordering_preserved_with_all_three_penalties(g, cfg, dist):
    """Burdens add, so three moderate fracs can cross zero where none would
    alone — the combination is where an unclamped version would first bite in
    a real run."""
    c2 = dataclasses.replace(cfg, edge_frac=0.5, dist_frac=0.5, act_frac=0.5)
    hi = float(adjusted_fitness(0.9, g, 1.0, c2, dist))
    lo = float(adjusted_fitness(0.1, g, 1.0, c2, dist))
    assert hi >= lo
    assert hi >= 0.0 and lo >= 0.0


def test_penalty_at_init_equals_frac(g, cfg, dist):
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    assert float(adjusted_fitness(1.0, g, 0.0, c2, dist)) == pytest.approx(0.8, rel=1e-4)


def test_penalty_is_proportional_to_raw_fitness(g, cfg, dist):
    """The point of the multiplicative form: the same frac works whether
    f_raw ~ 0.009 or ~ 0.9, with no per-experiment recalibration."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    for f_raw in [0.01, 0.1, 0.5, 0.9]:
        got = float(adjusted_fitness(f_raw, g, 0.0, c2, dist))
        assert got == pytest.approx(0.8 * f_raw, rel=1e-4)


def test_zero_raw_fitness_stays_zero(g, cfg, dist):
    """Multiplicative penalties cannot punish a network that scored nothing.
    This is what removes ctrnn_evo's early-generation over-pruning pressure:
    cost only bites once the network works."""
    c2 = dataclasses.replace(cfg, edge_frac=0.5, dist_frac=0.5, act_frac=0.5)
    assert float(adjusted_fitness(0.0, g, 1.0, c2, dist)) == pytest.approx(0.0)


def test_fewer_edges_scores_higher(g, cfg, dist):
    """Pruning must actually pay under a live penalty, or the sweep does
    nothing."""
    c2 = dataclasses.replace(cfg, edge_frac=0.3)
    half = g.edge_mask & (
        jnp.arange(g.edge_mask.size).reshape(g.edge_mask.shape) % 2 == 0
    )
    g_half = dataclasses.replace(g, edge_mask=half)
    assert float(adjusted_fitness(1.0, g_half, 0.0, c2, dist)) > \
           float(adjusted_fitness(1.0, g, 0.0, c2, dist))


def test_combined_fracs_accumulate(g, cfg, dist):
    single = dataclasses.replace(cfg, edge_frac=0.2)
    both = dataclasses.replace(cfg, edge_frac=0.2, act_frac=0.2)
    assert float(adjusted_fitness(1.0, g, 0.5, both, dist)) < \
           float(adjusted_fitness(1.0, g, 0.5, single, dist))


def test_absolute_lambda_mode_is_gone():
    """One code path.  ctrnn_evo's dual absolute/proportional branch is where
    the sign flip hid."""
    import ctrnn_lattice_evo.cost as cost_module
    src = open(cost_module.__file__).read()
    assert "lambda_edge" not in src
    assert "lambda_dist" not in src
    assert "lambda_act" not in src


# ── Cross-arm comparability ───────────────────────────────────────────────────

def test_grid_and_uniform_pay_equal_edge_penalty(cfg, dist):
    """Matched edge count means the edge penalty cannot explain a difference
    between the two arms — only locality can.  If this fails, the locality
    control has stopped controlling."""
    c2 = dataclasses.replace(cfg, edge_frac=0.2)
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(adjusted_fitness(1.0, gg, 0.0, c2, dist)) == \
           pytest.approx(float(adjusted_fitness(1.0, gu, 0.0, c2, dist)), rel=0.10)


def test_grid_pays_less_dist_penalty_than_uniform(cfg, dist):
    """At equal edge count the lattice is cheaper in wire length — the
    mechanism by which a distance penalty would preserve locality."""
    c2 = dataclasses.replace(cfg, dist_frac=0.2)
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(adjusted_fitness(1.0, gg, 0.0, c2, dist)) > \
           float(adjusted_fitness(1.0, gu, 0.0, c2, dist))


def test_uniform_dist_cost_exceeds_lattice(cfg, dist):
    """The same fact stated directly on the cost rather than through fitness."""
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(dist_cost(gu, dist)) > float(dist_cost(gg, dist))


def test_uniform_edge_cost_matches_lattice(cfg):
    gg = grid_genome(jax.random.PRNGKey(0), cfg)
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    assert float(edge_count_cost(gu)) == pytest.approx(float(edge_count_cost(gg)))