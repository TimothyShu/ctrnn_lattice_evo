"""
Tests — edge operator balance.

This file previously asserted the OPPOSITE of what it now asserts, and the
reason is worth recording.

ctrnn_evo's remove_edge deleted exactly one edge per genome per generation,
which capped lineage pruning at 1 edge/generation and made a 50-70% prune of a
1092-edge lattice unreachable in 500 generations.  The fix was to make removal
per-edge Bernoulli, and this file's job was to prove that pruning could now
reach the target band.

That overcorrected.  Removal then scaled with edge count (p*E deletions) while
addition stayed a single coin flip (<=1 per generation), so the pair had an
attractor at p_add_fires/p_rem ~= 33 edges.  The first four-arm gate ran at
edge_frac=0 — NO cost pressure whatsoever — and every arm collapsed to 27-41
edges and 10-24 nodes regardless of whether it started at 150, 1092 or 4032.
The fitness ranking simply tracked how far each arm had to fall.  Pure drift,
and it dominated the entire result.

Both operators now scale with E and share one rate, so their expectations
cancel identically at every density:

    E[dE] = p*E (added) - p*E (removed) = 0

Edge count is a driftless random walk that preserves whatever density an arm
started at.  These tests now guard THAT, because it is the property the
experiment depends on: with drift neutral, any systematic movement in edge
count under a live penalty is selection rather than operator bias.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config
from ctrnn_lattice_evo.genome import (
    grid_genome, uniform_genome, sparse_genome, prune_isolated,
)
from ctrnn_lattice_evo.mutation import (
    MutationRates, add_edges, remove_edges, count_active_edges, mutate,
)
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


def _churn(key, genome, cfg, p, n_steps):
    """Apply the balanced pair n_steps times, sharing the pre-mutation E."""
    gg = genome
    for i in range(n_steps):
        k = jax.random.fold_in(key, i)
        ka, kr = jax.random.split(k)
        n = count_active_edges(gg)
        gg = add_edges(ka, gg, cfg, p_per_edge=p, n_edges=n)
        gg = remove_edges(kr, gg, cfg, p_per_edge=p, n_edges=n)
    return gg


# ── Single-operator behaviour ─────────────────────────────────────────────────

def test_remove_never_adds(g, cfg):
    out = remove_edges(jax.random.PRNGKey(1), g, cfg, p_per_edge=0.01)
    assert jnp.all(out.edge_mask <= g.edge_mask)


def test_add_never_removes(g, cfg):
    out = add_edges(jax.random.PRNGKey(2), g, cfg, p_per_edge=0.01)
    assert jnp.all(g.edge_mask <= out.edge_mask)


def test_zero_rate_is_noop(g, cfg):
    assert jnp.array_equal(
        remove_edges(jax.random.PRNGKey(3), g, cfg, p_per_edge=0.0).edge_mask,
        g.edge_mask)
    assert jnp.array_equal(
        add_edges(jax.random.PRNGKey(3), g, cfg, p_per_edge=0.0).edge_mask,
        g.edge_mask)


def test_removal_count_scales_with_edge_count(g, cfg):
    """~p*E removals, not ~1.  ctrnn_evo's single-edge argmax could not do
    this, which is what made the lattice unprunable in a realistic budget."""
    p = 0.05
    start = int(count_active_edges(g))
    out = remove_edges(jax.random.PRNGKey(4), g, cfg, p_per_edge=p)
    removed = start - int(count_active_edges(out))
    assert removed == pytest.approx(p * start, rel=0.4)
    assert removed > 10


def test_addition_count_scales_with_edge_count(g, cfg):
    """The half that was missing.  Addition must scale too, or the pair has an
    attractor and every arm is dragged to it."""
    p = 0.05
    start = int(count_active_edges(g))
    out = add_edges(jax.random.PRNGKey(5), g, cfg, p_per_edge=p)
    added = int(count_active_edges(out)) - start
    assert added == pytest.approx(p * start, rel=0.4)
    assert added > 10


def test_add_respects_active_pairs_and_diagonal(cfg):
    gs = sparse_genome(jax.random.PRNGKey(6), cfg)
    out = add_edges(jax.random.PRNGKey(7), gs, cfg, p_per_edge=0.5)
    pairs = out.active_mask[:, None] & out.active_mask[None, :]
    assert jnp.all(out.edge_mask <= pairs)
    assert not jnp.any(jnp.diag(out.edge_mask))


def test_add_initialises_positive_weights(g, cfg):
    """A zero-weight edge is invisible to selection and would be removed again
    before it could ever be evaluated."""
    out = add_edges(jax.random.PRNGKey(8), g, cfg, p_per_edge=0.05)
    new = out.edge_mask & ~g.edge_mask
    assert int(new.sum()) > 0
    assert float(jnp.min(jnp.where(new, out.weight_matrix, jnp.inf))) > 0.0


def test_add_saturates_gracefully(cfg):
    """No-op when every active pair already has an edge — must not corrupt."""
    gg = grid_genome(jax.random.PRNGKey(9), cfg)
    full = (gg.active_mask[:, None] & gg.active_mask[None, :]) & ~jnp.eye(cfg.N_max, dtype=bool)
    gg = dataclasses.replace(gg, edge_mask=full)
    out = add_edges(jax.random.PRNGKey(10), gg, cfg, p_per_edge=0.1)
    assert jnp.array_equal(out.edge_mask, gg.edge_mask)


def test_remove_empties_gracefully(g, cfg):
    """Zero is absorbing: nothing spawns from an empty graph.  Far from the
    lattice's 1092, but the sparse arm at ~150 is closer to it."""
    empty = dataclasses.replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
    out = remove_edges(jax.random.PRNGKey(11), empty, cfg, p_per_edge=0.5)
    assert int(count_active_edges(out)) == 0
    out = add_edges(jax.random.PRNGKey(12), empty, cfg, p_per_edge=0.5)
    assert int(count_active_edges(out)) == 0


# ── THE property: drift neutrality ────────────────────────────────────────────

def test_single_step_is_neutral_in_expectation(g, cfg, rates):
    """Averaged over many genomes, one balanced step should not move E."""
    start = int(count_active_edges(g))
    deltas = []
    for i in range(40):
        gg = _churn(jax.random.fold_in(jax.random.PRNGKey(13), i), g, cfg,
                    rates.edge_churn, 1)
        deltas.append(int(count_active_edges(gg)) - start)
    mean = sum(deltas) / len(deltas)
    # ~3.3 each way at E=1092, so a single step has sd ~2.6; the mean of 40
    # draws should sit well inside +/-1.5.
    assert abs(mean) < 1.5, f"single-step drift {mean:+.2f} edges/generation"


def test_edge_count_holds_over_a_full_run(g, cfg, rates):
    """1000 generations of pure drift must NOT move the lattice off 1092.

    This is the test that would have caught the gate failure.  Under the old
    operators the lattice fell to ~33 edges here; the run reported 40.8.
    """
    start = int(count_active_edges(g))
    gg = _churn(jax.random.PRNGKey(14), g, cfg, rates.edge_churn, 1000)
    end = int(count_active_edges(gg))
    assert end == pytest.approx(start, rel=0.25), \
        f"edge count drifted {start} -> {end} with no selection pressure"


def test_density_is_preserved_from_any_start(cfg, rates):
    """No attractor: each arm stays near ITS OWN density.

    A per-slot addition rate would instead give an equilibrium of
    M*p_add/(p_add+p_rem) — depending only on the rates — so one global rate
    would drag every arm to the same place regardless of initialisation, which
    destroys the arm distinction just as thoroughly as an unbalanced operator.
    """
    for name, genome in [
        ("grid",    grid_genome(jax.random.PRNGKey(15), cfg)),
        ("sparse",  sparse_genome(jax.random.PRNGKey(15), cfg)),
    ]:
        start = int(count_active_edges(genome))
        gg = _churn(jax.random.PRNGKey(16), genome, cfg, rates.edge_churn, 500)
        end = int(count_active_edges(gg))
        assert end == pytest.approx(start, rel=0.35), \
            f"{name}: {start} -> {end}"


def test_arms_do_not_converge_on_each_other(cfg, rates):
    """The sparse and grid arms must stay an order of magnitude apart."""
    gg = _churn(jax.random.PRNGKey(17), grid_genome(jax.random.PRNGKey(18), cfg),
                cfg, rates.edge_churn, 500)
    gs = _churn(jax.random.PRNGKey(17), sparse_genome(jax.random.PRNGKey(18), cfg),
                cfg, rates.edge_churn, 500)
    assert int(count_active_edges(gg)) > 3 * int(count_active_edges(gs))


def test_node_count_holds_when_edges_hold(g, cfg, rates):
    """The node collapse was downstream of the edge collapse.

    prune_isolated is one-way and fires when a hidden neuron loses all its in-
    or out-edges.  At 1092 edges over 64 slots that never happens; at 40 edges
    most nodes are isolated by arithmetic, which is how the gate run went from
    64 nodes to 23.6.  With edge count held, node count should hold too — no
    node operator required.
    """
    gg = g
    for i in range(500):
        k = jax.random.fold_in(jax.random.PRNGKey(19), i)
        ka, kr = jax.random.split(k)
        n = count_active_edges(gg)
        gg = add_edges(ka, gg, cfg, p_per_edge=rates.edge_churn, n_edges=n)
        gg = remove_edges(kr, gg, cfg, p_per_edge=rates.edge_churn, n_edges=n)
        gg = prune_isolated(gg, cfg)
    assert int(gg.active_mask.sum()) >= 0.9 * cfg.N_max, \
        f"only {int(gg.active_mask.sum())}/{cfg.N_max} nodes survived"


def test_shared_n_edges_removes_ordering_bias(g, cfg, rates):
    """Both operators must be driven by the SAME pre-mutation E.

    Letting each recompute it means the second sees the first's effect, a
    systematic drift of order p^2 * E per generation.  Small — about 10 edges
    over 1000 generations against a random-walk sd of ~80 — but it is free to
    remove and it makes the neutrality exact rather than approximate.
    """
    p = 0.2                                   # exaggerated to make it visible
    shared, independent = [], []
    for i in range(30):
        k = jax.random.fold_in(jax.random.PRNGKey(20), i)
        ka, kr = jax.random.split(k)

        n = count_active_edges(g)
        a = remove_edges(kr, add_edges(ka, g, cfg, p_per_edge=p, n_edges=n),
                         cfg, p_per_edge=p, n_edges=n)
        shared.append(int(count_active_edges(a)))

        b = remove_edges(kr, add_edges(ka, g, cfg, p_per_edge=p), cfg, p_per_edge=p)
        independent.append(int(count_active_edges(b)))

    start = int(count_active_edges(g))
    assert abs(sum(shared) / len(shared) - start) <= \
           abs(sum(independent) / len(independent) - start) + 2.0


# ── Selection, not drift ──────────────────────────────────────────────────────

def test_pruning_requires_selection(g, cfg, rates):
    """With drift neutral, edge count only falls if something selects for it.

    That is the whole point of the rebalance: the four-arm gate ran at
    edge_frac=0 and pruned 94-97% anyway, so nothing it measured could be
    attributed to the cost term.  Now a flat trace at edge_frac=0 is the
    correct null, and any decline under edge_frac>0 is a real effect.
    """
    start = int(count_active_edges(g))
    gg = _churn(jax.random.PRNGKey(21), g, cfg, rates.edge_churn, 300)
    assert int(count_active_edges(gg)) > 0.5 * start


def test_higher_churn_does_not_bias_direction(g, cfg):
    """Turnover rate sets variance, not direction — E[dE] = 0 at any p."""
    start = int(count_active_edges(g))
    for p in [0.001, 0.01, 0.05]:
        gg = _churn(jax.random.PRNGKey(22), g, cfg, p, 100)
        assert int(count_active_edges(gg)) == pytest.approx(start, rel=0.3), \
            f"p={p} moved E systematically"


# ── Locality retention ────────────────────────────────────────────────────────

def test_local_fraction_erodes_slowly(g, cfg, rates):
    """Removals come out of the current topology, additions land anywhere, so
    a lattice loses local edges and gains mostly non-local ones — composition
    shifts even with edge count flat.  At ~3.3 additions/generation against
    1092 edges this is slow, but over 1000 generations it is not negligible,
    which is why local_fraction is logged per generation.
    """
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = _churn(jax.random.PRNGKey(23), g, cfg, rates.edge_churn, 200)
    frac = float((gg.edge_mask & m).sum()) / float(gg.edge_mask.sum())
    assert 0.5 < frac < 1.0, f"local fraction {frac:.3f} after 200 generations"


def test_uniform_arm_local_fraction_is_the_floor(cfg):
    """A random digraph at the same density lands ~n_edges/(N^2-N) inside the
    lattice ball by chance — about 0.27 here, NOT 0.  That is the baseline the
    grid arm's local_fraction is read against."""
    gu = uniform_genome(jax.random.PRNGKey(0), cfg)
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    frac = float((gu.edge_mask & m).sum()) / float(gu.edge_mask.sum())
    assert 0.15 < frac < 0.45


def test_add_edges_can_leave_the_lattice(g, cfg):
    """Deliberately unmasked: evolution may buy a long-range shortcut, and the
    distance penalty decides whether it keeps it."""
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    gg = add_edges(jax.random.PRNGKey(24), g, cfg, p_per_edge=0.1)
    assert int((gg.edge_mask & ~m).sum()) > 0


# ── mutate() integration ──────────────────────────────────────────────────────

def test_mutate_is_edge_neutral(g, cfg, rates):
    """The property must survive the full operator stack, not just the pair."""
    r = dataclasses.replace(rates, weight_sigma=0.0, tau_sigma=0.0,
                            bias_sigma=0.0, type_flip_prob=0.0)
    start = int(count_active_edges(g))
    gg = g
    for i in range(300):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(25), i), gg, cfg, r)
    assert int(count_active_edges(gg)) == pytest.approx(start, rel=0.3)


def test_mutate_holds_node_count(g, cfg, rates):
    gg = g
    for i in range(300):
        gg = mutate(jax.random.fold_in(jax.random.PRNGKey(26), i), gg, cfg, rates)
    assert int(gg.active_mask.sum()) >= 0.9 * cfg.N_max