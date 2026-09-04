# ctrnn_lattice_evo/topology.py
from __future__ import annotations
import jax.numpy as jnp


def grid_coords(W: int, H: int | None = None) -> jnp.ndarray:
    """Row-major grid coordinates. Returns [W*H, 2] int32 as (row, col)."""
    H = W if H is None else H
    rows, cols = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing="ij")
    return jnp.stack([rows.ravel(), cols.ravel()], axis=-1).astype(jnp.int32)


def dist_matrix(W: int, H: int | None = None) -> jnp.ndarray:
    """Pairwise Chebyshev distance between grid slots. Returns [N, N] float32."""
    c = grid_coords(W, H)
    d = jnp.abs(c[:, None, :] - c[None, :, :])      # [N, N, 2]
    return jnp.max(d, axis=-1).astype(jnp.float32)


def local_mask(W: int, r: int, H: int | None = None) -> jnp.ndarray:
    """Directed adjacency for a Chebyshev ball of radius r. No self-edges."""
    d = dist_matrix(W, H)
    return (d <= r) & (d > 0)

def expected_edges(W: int, r: int, H: int | None = None) -> int:
    """Closed-form directed edge count for local_mask(W, r, H).

    Chebyshev distance is the max over axes, so a pair is within radius r
    exactly when it is within r on both axes independently — the count
    factorises. (This is why the same form does NOT hold for Manhattan.)
    """
    H = W if H is None else H

    def _A(L: int) -> int:
        """Ordered 1-D pairs (i, j) with |i - j| <= r on a line of length L."""
        return L + 2 * sum(L - d for d in range(1, min(r, L - 1) + 1))

    return _A(W) * _A(H) - W * H

def distance_kernel(W: int, r_lambda: float | None, H: int | None = None) -> jnp.ndarray:
    """Unnormalised per-slot addition weight: exp(-d / r_lambda), 0 on the
    diagonal.

    r_lambda is in LATTICE UNITS, so it means the same thing across grid
    sizes: the e-folding reach of a new edge.  r_lambda = inf (or <= 0 by
    convention) recovers uniform addition, which is the current behaviour and
    the natural control point.

    Computed ONCE per run — it depends only on the lattice, never on a genome.
    """
    d = dist_matrix(W, H)
    if r_lambda is None or r_lambda <= 0 or not jnp.isfinite(jnp.asarray(r_lambda)):
        w = jnp.ones_like(d)
    else:
        w = jnp.exp(-d / r_lambda)
    return jnp.where(d > 0, w, 0.0)     # no self-edges


def reference_costs(W: int, r: int, H: int | None = None) -> tuple[float, float]:
    m = local_mask(W, r, H)
    d = dist_matrix(W, H)
    return float(m.sum()), float(jnp.sum(jnp.where(m, d, 0.0)))