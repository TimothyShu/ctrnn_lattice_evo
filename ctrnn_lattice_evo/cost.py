from __future__ import annotations

import jax.numpy as jnp

from .config import Config
from .genome import Genome


def edge_count_cost(genome: Genome) -> float:
    """
    Count of active edges between active neurons — no spatial weighting.

    Implements the pure connection-count penalty of Clune et al. (2013):
    every edge costs the same regardless of how long it is.
    """
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(genome.edge_mask * active_pair).astype(jnp.float32)


def dist_cost(genome: Genome) -> float:
    """
    Sum of Euclidean wire lengths over all active edges — distance-weighted.

    Penalises long axons more than short ones, reflecting the metabolic
    reality that long-range connections are expensive to build and maintain.
    Generalises edge_count_cost to spatially embedded networks.
    """
    diff = genome.position[:, None, :] - genome.position[None, :, :]  # [N, N, 2]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))                       # [N, N]
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(dist * genome.edge_mask * active_pair)


def adjusted_fitness(
    f_raw: float,
    genome: Genome,
    c_act: float,
    cfg: Config,
) -> float:
    """
    Apply cost penalties to raw task fitness.

    Two modes per penalty term (proportional takes priority when frac > 0):

    Absolute mode (legacy):
        penalty = lambda_* × C_*
        Fixed coefficient — must be recalibrated whenever the fitness scale
        changes (e.g. switching from survival to food-score fitness).

    Proportional mode (recommended):
        penalty = frac_* × f_raw × (C_* / C0_*)
        Keeps the penalty at a fixed percentage of raw fitness when the
        network is at reference cost C0_*, so the same frac works across
        different fitness regimes without recalibration.

    Any combination of the six parameters can be active simultaneously.
    Setting all to 0.0 returns f_raw unchanged.
    """
    c_edge = edge_count_cost(genome)
    c_dist = dist_cost(genome)

    # Wiring-length penalty
    if cfg.dist_frac > 0.0:
        penalty_dist = cfg.dist_frac * f_raw * (c_dist / cfg.C0_wiring)
    else:
        penalty_dist = cfg.lambda_dist * c_dist

    # Activation penalty
    if cfg.act_frac > 0.0:
        penalty_act = cfg.act_frac * f_raw * (c_act / cfg.C0_act)
    else:
        penalty_act = cfg.lambda_act * c_act

    # Edge-count penalty
    if cfg.edge_frac > 0.0:
        penalty_edge = cfg.edge_frac * f_raw * (c_edge / cfg.C0_edge)
    else:
        penalty_edge = cfg.lambda_edge * c_edge

    return f_raw - penalty_dist - penalty_act - penalty_edge
