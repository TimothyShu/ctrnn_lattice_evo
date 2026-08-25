"""
Tests for topology.py — the lattice substrate.

These are the arithmetic gate for the whole experiment: every cost
normalisation, every C0 reference and every density figure is derived from
local_mask, so a wrong mask corrupts the results silently rather than
crashing.  Nothing downstream is worth running until these pass.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo.topology import (
    grid_coords,
    dist_matrix,
    local_mask,
    expected_edges,
)


# ── Reference implementation (independent of topology.py) ────────────────────

def _A(L: int, r: int) -> int:
    """Ordered 1-D pairs (i, j) with |i - j| <= r on a line of length L."""
    return L + 2 * sum(L - d for d in range(1, min(r, L - 1) + 1))


def _brute_force_edges(W: int, H: int, r: int) -> int:
    """Count directed Chebyshev-r pairs by explicit enumeration."""
    coords = [(i, j) for i in range(W) for j in range(H)]
    n = 0
    for (a, b) in coords:
        for (c, d) in coords:
            if (a, b) == (c, d):
                continue
            if max(abs(a - c), abs(b - d)) <= r:
                n += 1
    return n


# ── grid_coords ──────────────────────────────────────────────────────────────

def test_grid_coords_shape():
    assert grid_coords(4).shape == (16, 2)
    assert grid_coords(4, 8).shape == (32, 2)


def test_grid_coords_row_major():
    """Slot k must sit at (k // H, k % H) — row-major, matching I/O slot order."""
    c = grid_coords(4, 4)
    for k in range(16):
        assert int(c[k, 0]) == k // 4
        assert int(c[k, 1]) == k % 4


def test_grid_coords_all_distinct():
    c = grid_coords(4, 4)
    seen = {(int(r), int(col)) for r, col in c}
    assert len(seen) == 16


# ── dist_matrix ──────────────────────────────────────────────────────────────

def test_dist_matrix_shape_and_dtype():
    d = dist_matrix(4)
    assert d.shape == (16, 16)
    assert d.dtype == jnp.float32


def test_dist_matrix_symmetric():
    d = dist_matrix(4)
    assert jnp.array_equal(d, d.T)


def test_dist_matrix_zero_diagonal():
    d = dist_matrix(4)
    assert jnp.all(jnp.diag(d) == 0.0)


def test_dist_matrix_is_chebyshev():
    """Adjacent-diagonal neighbours are distance 1, not sqrt(2) or 2."""
    d = dist_matrix(4, 4)
    assert float(d[0, 5]) == 1.0     # (0,0) -> (1,1), diagonal
    assert float(d[0, 1]) == 1.0     # (0,0) -> (0,1), orthogonal
    assert float(d[0, 2]) == 2.0     # (0,0) -> (0,2)


def test_dist_matrix_corner_to_corner():
    """Opposite corners of an 8x8 lattice are 7 apart under Chebyshev."""
    d = dist_matrix(8, 8)
    assert float(d[0, 63]) == 7.0


def test_dist_matrix_triangle_inequality():
    d = dist_matrix(4, 4)
    assert jnp.all(d[:, None, :] + d[None, :, :].transpose(1, 0, 2) >= 0)
    # explicit spot check
    for a, b, c in [(0, 5, 10), (3, 7, 12), (1, 2, 15)]:
        assert float(d[a, c]) <= float(d[a, b]) + float(d[b, c]) + 1e-6


# ── local_mask ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("W,H,r", [(4, 4, 1), (4, 4, 2), (8, 8, 1), (8, 8, 2), (4, 16, 2)])
def test_local_mask_edge_count_matches_closed_form(W, H, r):
    m = local_mask(W, r, H)
    assert int(m.sum()) == _A(W, r) * _A(H, r) - W * H


@pytest.mark.parametrize("W,H,r", [(4, 4, 1), (4, 4, 2), (5, 3, 1)])
def test_local_mask_edge_count_matches_brute_force(W, H, r):
    """Guards the closed form itself, not just its use."""
    m = local_mask(W, r, H)
    assert int(m.sum()) == _brute_force_edges(W, H, r)


def test_local_mask_known_values():
    """The two configurations the experiment is calibrated against."""
    assert int(local_mask(4, 1).sum()) == 84       # test-scale lattice
    assert int(local_mask(8, 2).sum()) == 1092     # production lattice


def test_expected_edges_agrees_with_mask():
    for W, H, r in [(4, 4, 1), (8, 8, 2), (4, 16, 2)]:
        assert expected_edges(W, r, H) == int(local_mask(W, r, H).sum())


def test_local_mask_no_self_edges():
    assert not jnp.any(jnp.diag(local_mask(4, 1)))
    assert not jnp.any(jnp.diag(local_mask(8, 2)))


def test_local_mask_symmetric():
    """Directed, but the lattice relation itself is symmetric."""
    m = local_mask(4, 2)
    assert jnp.array_equal(m, m.T)


def test_local_mask_is_boolean():
    assert local_mask(4, 1).dtype == jnp.bool_


def test_all_masked_pairs_within_radius():
    """No edge in the mask exceeds the radius — the defining property."""
    for W, r in [(4, 1), (4, 2), (8, 2)]:
        d = dist_matrix(W)
        m = local_mask(W, r)
        assert float(jnp.max(jnp.where(m, d, 0.0))) <= float(r)


def test_all_pairs_within_radius_are_masked():
    """Converse of the above — the mask omits nothing it should contain."""
    for W, r in [(4, 1), (4, 2), (8, 2)]:
        d = dist_matrix(W)
        m = local_mask(W, r)
        should_be = (d <= r) & (d > 0)
        assert jnp.array_equal(m, should_be)


def test_local_mask_monotone_in_radius():
    """A larger radius is a strict superset — no edge is ever lost."""
    m1, m2 = local_mask(8, 1), local_mask(8, 2)
    assert jnp.all(m2 | ~m1)               # m1 implies m2
    assert int(m2.sum()) > int(m1.sum())


def test_full_radius_is_all_pairs():
    """At r >= W-1 on a square lattice the mask is complete (minus diagonal)."""
    W = 4
    m = local_mask(W, W - 1)
    assert int(m.sum()) == W * W * (W * W - 1) // 1 - 0 if False else int(m.sum())
    assert int(m.sum()) == (W * W) * (W * W - 1)


# ── Degree structure ─────────────────────────────────────────────────────────

def test_interior_node_has_full_neighbourhood():
    """An interior node on 8x8 at r=2 reaches (2r+1)^2 - 1 = 24 neighbours."""
    m = local_mask(8, 2)
    interior = 3 * 8 + 3          # (row 3, col 3), far from every boundary
    assert int(m[interior].sum()) == 24


def test_corner_node_has_reduced_degree():
    """Corners are clipped — this is why I/O placement affects degree."""
    m = local_mask(8, 2)
    assert int(m[0].sum()) == 8   # (r+1)^2 - 1 = 8
    assert int(m[0].sum()) < int(m[3 * 8 + 3].sum())


def test_every_node_has_at_least_one_edge():
    """No isolated slot at init — prune_isolated must be a no-op on a fresh grid."""
    for W, r in [(4, 1), (8, 2)]:
        m = local_mask(W, r)
        assert jnp.all(m.sum(axis=1) > 0)
        assert jnp.all(m.sum(axis=0) > 0)


# ── Rectangular lattices ─────────────────────────────────────────────────────

def test_rectangular_lattice_supported():
    m = local_mask(4, 2, 16)
    assert m.shape == (64, 64)
    assert int(m.sum()) == 972


def test_rectangular_traversal_is_longer_than_square():
    """4x16 doubles the I/O hop count relative to 8x8 at equal N."""
    assert float(dist_matrix(8, 8)[0, 63]) == 7.0
    assert float(dist_matrix(4, 16)[0, 63]) == 15.0