from __future__ import annotations

import jax.numpy as jnp

from .config import Config
from .genome import Genome
from .topology import dist_matrix

__all__ = [
    "edge_count_cost",
    "dist_cost",
    "penalty_scale",
    "adjusted_fitness",
    "cfg_dist_matrix",
]


def edge_count_cost(genome: Genome) -> jnp.ndarray:
    """Number of active edges between active neurons — no spatial weighting.

    The pure connection-count penalty of Clune et al. (2013): every edge costs
    the same regardless of length.
    """
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(genome.edge_mask * active_pair).astype(jnp.float32)


def dist_cost(genome: Genome, dist: jnp.ndarray) -> jnp.ndarray:
    """Summed Chebyshev wire length over all active edges.

    `dist` is passed in rather than derived from the genome: geometry is a
    property of the lattice, shared by every individual, so it is computed once
    per run (topology.dist_matrix) instead of stored per genome.

    Penalises long axons more than short ones.  Note that at grid_r=1 every
    lattice edge has length 1, so this is EXACTLY a scalar multiple of
    edge_count_cost — the two penalty axes are collinear and dist_frac buys
    nothing.  They separate only at r >= 2.
    """
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(dist * genome.edge_mask * active_pair).astype(jnp.float32)


def cfg_dist_matrix(cfg: Config) -> jnp.ndarray:
    """The lattice distance matrix for this config.

    Convenience for callers outside the hot loop.  The evolutionary loop should
    build this ONCE per run and thread it through, not call this per genome.
    """
    return dist_matrix(cfg.grid_W, cfg.grid_H)


def penalty_scale(
    genome: Genome,
    c_act: jnp.ndarray | float,
    cfg: Config,
    dist: jnp.ndarray,
) -> jnp.ndarray:
    """The multiplier applied to raw fitness, in [0, 1].

        scale = max(0, 1 - edge_frac*C_edge/C0_edge
                        - dist_frac*C_dist/C0_dist
                        - act_frac *C_act /C0_act)

    Each frac reads directly as "fraction of fitness surrendered at reference
    cost", so the same value means the same thing across lattice sizes, radii
    and fitness regimes with no recalibration.

    THE CLAMP IS LOAD-BEARING.  Without jnp.maximum the bracket goes negative
    once the weighted cost ratio exceeds 1, and because f_raw >= 0 always, a
    BETTER network then maps to a MORE negative adjusted fitness — tournament
    selection silently prefers the worse individual and evolution runs
    backwards with no error, no NaN, and no crash.  Clamping at 0 flattens the
    tail instead, which loses gradient among already-hopeless genomes but never
    inverts the ordering.  Guarded by
    test_ordering_preserved_under_extreme_penalty.
    """
    c_edge = edge_count_cost(genome)
    c_dist = dist_cost(genome, dist)

    burden = (
        cfg.edge_frac * c_edge / cfg.C0_edge
        + cfg.dist_frac * c_dist / cfg.C0_dist
        + cfg.act_frac * c_act / cfg.C0_act
    )
    return jnp.maximum(0.0, 1.0 - burden)


def adjusted_fitness(
    f_raw: jnp.ndarray | float,
    genome: Genome,
    c_act: jnp.ndarray | float,
    cfg: Config,
    dist: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the proportional cost penalties to raw task fitness.

        f = f_raw * penalty_scale(...)

    Multiplicative, not subtractive.  This is what removes the early-generation
    over-pruning pressure seen in ctrnn_evo: when f_raw is near zero the
    absolute penalty is near zero too, so a network is not punished for
    carrying edges before those edges have had any chance to become useful.
    Cost only bites once the network works.

    The flip side is that pruning pressure is strongest at the TOP of the
    fitness distribution, which can stall a leading lineage — worth logging the
    penalty magnitude for the best individual per generation.

    ctrnn_evo's absolute lambda_* mode is deliberately gone.  One code path
    means there is nowhere for the sign flip described in penalty_scale to
    hide.

    Setting every frac to 0 returns f_raw unchanged.
    """
    return f_raw * penalty_scale(genome, c_act, cfg, dist)