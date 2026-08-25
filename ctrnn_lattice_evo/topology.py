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