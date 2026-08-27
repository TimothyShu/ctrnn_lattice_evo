"""
analysis.py — Post-hoc analysis metrics for evolved CTRNN genomes.

Everything here runs on numpy/networkx rather than JAX, so it is for
end-of-run analysis, not the hot evolutionary loop.  analyse_population in
particular loops in Python over the whole population calling networkx — fine
once per run, prohibitive per generation.

Public API
----------
local_fraction(genome, cfg) -> float
    Share of active edges lying inside the lattice neighbourhood.  The metric
    the locality claim rests on.

modularity_q(genome, cfg) -> float
    Newman-Girvan Q of the absolute effective-weight network.

network_stats(genome, cfg) -> dict
    n_active, n_edges, density, mean_weight, wiring_cost, local_fraction.

analyse_genome / analyse_population / summarise_run
    Aggregations of the above.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from networkx.algorithms.community import (
    greedy_modularity_communities,
    modularity as nx_modularity,
)

from .config import Config
from .genome import Genome, effective_weights
from .cost import dist_cost, edge_count_cost
from .topology import dist_matrix, local_mask


# ── Locality ──────────────────────────────────────────────────────────────────

def local_fraction(genome: Genome, cfg: Config) -> float:
    """Fraction of active edges that lie within the lattice neighbourhood.

    This is the metric the whole locality argument rests on, and it had no home
    in ctrnn_evo.  Logged per generation it answers, empirically, how fast the
    lattice prior erodes: add_edge is deliberately unmasked, so evolution can
    place edges anywhere, and this measures whether it does.

    Read it against the right floor.  A uniform random digraph at the same
    density does NOT score 0 — roughly n_edges/(N^2 - N) of its edges land
    inside the ball by chance, about 0.27 at 8x8 r=2.  A grid genome starts at
    1.0.  The interesting quantity is where the grid arm settles relative to
    that 0.27 floor, not its absolute value.

    Returns 0.0 for a genome with no active edges, which an aggressive prune
    can reach.
    """
    active_mask = np.array(genome.active_mask)
    edge_mask = np.array(genome.edge_mask)
    active_edges = edge_mask & active_mask[:, None] & active_mask[None, :]

    n_edges = int(active_edges.sum())
    if n_edges == 0:
        return 0.0

    m = np.array(local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H))
    return float((active_edges & m).sum()) / float(n_edges)


def mean_edge_length(genome: Genome, cfg: Config) -> float:
    """Mean Chebyshev length over active edges; 0.0 when there are none.

    A blunter companion to local_fraction: local_fraction says how many edges
    left the ball, this says how far they went.  Bounded below by 1.0 for any
    genome with edges.
    """
    n_edges = float(edge_count_cost(genome))
    if n_edges == 0.0:
        return 0.0
    d = dist_matrix(cfg.grid_W, cfg.grid_H)
    return float(dist_cost(genome, d)) / n_edges


# ── Modularity Q ──────────────────────────────────────────────────────────────

def modularity_q(genome: Genome, cfg: Config) -> float:
    """Newman-Girvan modularity Q for one genome.

    Pipeline: effective_weights -> absolute value -> symmetrise -> restrict to
    active neurons -> weighted undirected graph -> greedy community detection
    (deterministic) -> Q.

    Returns 0.0 for degenerate cases (fewer than 2 active neurons, or no
    edges).  Q is in (-0.5, 1.0]; higher means more modular.

    Interpret with care on a lattice.  A lattice is locally clustered but NOT
    block-structured — there are no separated modules, just a band — so Q
    lands somewhere in the middle and moves for reasons that have little to do
    with locality.  If Q comes out near zero on grid genomes it is not
    detecting lattice structure at all and should not be cited as evidence for
    it; local_fraction is the metric with a clean interpretation here.
    """
    W_eff = np.array(effective_weights(genome))
    W_abs = np.abs(W_eff)
    W_sym = (W_abs + W_abs.T) / 2.0

    active_mask = np.array(genome.active_mask)
    active_idxs = np.where(active_mask)[0]
    n_active = len(active_idxs)

    if n_active < 2:
        return 0.0

    W_sub = W_sym[np.ix_(active_idxs, active_idxs)]
    if W_sub.sum() == 0.0:
        return 0.0

    G = nx.Graph()
    G.add_nodes_from(range(n_active))
    for i in range(n_active):
        for j in range(i + 1, n_active):
            w = float(W_sub[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        return 0.0

    communities = greedy_modularity_communities(G, weight="weight")
    return float(nx_modularity(G, communities, weight="weight"))


# ── Network statistics ────────────────────────────────────────────────────────

def network_stats(genome: Genome, cfg: Config) -> dict:
    """Structural statistics for one genome.

    Returns
    -------
    n_active       : int   — active neurons
    n_edges        : int   — active edges between active neuron pairs
    density        : float — n_edges / (n_active*(n_active-1)), 0.0 if degenerate
    mean_weight    : float — mean |W_eff| over active edges
    wiring_cost    : float — summed Chebyshev wire length
    mean_edge_len  : float — wiring_cost / n_edges
    local_fraction : float — share of active edges inside the lattice ball

    Note the density denominator is n_active*(n_active-1), NOT the lattice mask
    size.  A fresh 8x8 r=2 lattice therefore reads as 27% dense, not 100% — and
    since add_edge is unmasked, a genome's active edges can exceed the mask.
    """
    active_mask = np.array(genome.active_mask)
    edge_mask = np.array(genome.edge_mask)

    n_active = int(active_mask.sum())
    active_edge_mask = edge_mask & active_mask[:, None] & active_mask[None, :]
    n_edges = int(active_edge_mask.sum())

    possible = n_active * (n_active - 1)
    density = float(n_edges / possible) if possible > 0 else 0.0

    W_abs = np.abs(np.array(effective_weights(genome)))
    mean_weight = float(W_abs[active_edge_mask].mean()) if n_edges > 0 else 0.0

    # dist_cost now takes the lattice distance matrix explicitly — geometry is
    # a property of the grid, not of the genome.  Building it here is fine
    # because this is not the hot loop.
    d = dist_matrix(cfg.grid_W, cfg.grid_H)
    wiring_cost = float(dist_cost(genome, d))

    return {
        "n_active":       n_active,
        "n_edges":        n_edges,
        "density":        density,
        "mean_weight":    mean_weight,
        "wiring_cost":    wiring_cost,
        "mean_edge_len":  (wiring_cost / n_edges) if n_edges > 0 else 0.0,
        "local_fraction": local_fraction(genome, cfg),
    }


# ── Single-genome analysis ────────────────────────────────────────────────────

def analyse_genome(genome: Genome, cfg: Config) -> dict:
    """All analysis metrics for one genome, as a flat dict."""
    return {"q": modularity_q(genome, cfg), **network_stats(genome, cfg)}


# ── Population analysis ───────────────────────────────────────────────────────

_METRIC_KEYS = (
    "q", "n_active", "n_edges", "density",
    "mean_weight", "wiring_cost", "mean_edge_len", "local_fraction",
)


def analyse_population(pop_genomes: Genome, cfg: Config) -> dict[str, list]:
    """analyse_genome for every genome in a batched population.

    pop_genomes carries a leading population_size dimension on each field.
    Returns a dict of lists, each of length population_size.

    This is a Python loop calling networkx once per individual, so it is slow
    — at population_size=1000 and 64 neurons expect it to take a while.  Run it
    at the end of a run, never per generation; collect_stats in evolution.py
    computes the cheap subset that is needed every generation.
    """
    import jax.tree_util as jtu

    pop_size = pop_genomes.active_mask.shape[0]
    results: dict[str, list] = {k: [] for k in _METRIC_KEYS}

    for i in range(pop_size):
        g = jtu.tree_map(lambda x: x[i], pop_genomes)
        metrics = analyse_genome(g, cfg)
        for k in results:
            results[k].append(metrics[k])

    return results


# ── Run summary ───────────────────────────────────────────────────────────────

def summarise_run(
    history: list[dict],
    pop_genomes: Genome,
    best_genome: Genome,
    cfg: Config,
) -> dict:
    """Summary of a completed run.

    Parameters
    ----------
    history     : per-generation stats dicts from run_evolution
    pop_genomes : final batched population
    best_genome : unbatched best genome from the final generation
    cfg         : Config

    Returns fitness trajectories, final population structure, and best-genome
    metrics.  final_local_fraction_mean is the one to compare across arms: the
    grid arm starts at 1.0, the uniform arm near n_edges/(N^2-N), and where
    the grid arm ends says how much of the lattice prior survived selection.
    """
    fitness_max = [h["max_fitness"] for h in history]
    fitness_mean = [h["mean_fitness"] for h in history]
    steps_mean = [h["mean_steps"] for h in history]

    pop_metrics = analyse_population(pop_genomes, cfg)
    q_vals = pop_metrics["q"]

    best_metrics = analyse_genome(best_genome, cfg)

    return {
        # Fitness trajectory (one value per generation)
        "fitness_max":  fitness_max,
        "fitness_mean": fitness_mean,
        "steps_mean":   steps_mean,

        # Final population structure
        "final_q_mean":              float(np.mean(q_vals)),
        "final_q_max":               float(np.max(q_vals)),
        "final_n_active_mean":       float(np.mean(pop_metrics["n_active"])),
        "final_n_edges_mean":        float(np.mean(pop_metrics["n_edges"])),
        "final_conn_cost_mean":      float(np.mean(pop_metrics["wiring_cost"])),
        "final_local_fraction_mean": float(np.mean(pop_metrics["local_fraction"])),
        "final_mean_edge_len_mean":  float(np.mean(pop_metrics["mean_edge_len"])),

        # Best genome
        "best_q":              best_metrics["q"],
        "best_n_active":       best_metrics["n_active"],
        "best_n_edges":        best_metrics["n_edges"],
        "best_local_fraction": best_metrics["local_fraction"],
    }