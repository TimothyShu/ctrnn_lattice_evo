"""
analysis.py — Post-hoc analysis metrics for evolved CTRNN genomes.

All functions operate on numpy/networkx — not JAX-compiled — so they are
intended for end-of-run analysis, not the hot evolutionary loop.

Public API
----------
modularity_q(genome, cfg) -> float
    Newman-Girvan Q score of the absolute effective-weight network.

network_stats(genome, cfg) -> dict
    n_active, n_edges, density, mean_weight, wiring_cost.

analyse_genome(genome, cfg) -> dict
    Flat dict combining modularity_q + network_stats.

analyse_population(pop_genomes, cfg) -> dict[str, list]
    analyse_genome for every genome in a batched population.

summarise_run(history, pop_genomes, best_genome, cfg) -> dict
    Fitness trajectory + final population structure + best-genome metrics.
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
from .cost import dist_cost


# ── Modularity Q ──────────────────────────────────────────────────────────────

def modularity_q(genome: Genome, cfg: Config) -> float:
    """
    Compute Newman-Girvan modularity Q for one genome.

    Pipeline:
      1. W_eff  = effective_weights(genome)  — applies Dale's law, edge mask,
                                               activity mask
      2. W_abs  = |W_eff|                    — absolute values (non-negative)
      3. W_sym  = (W_abs + W_abs.T) / 2     — symmetrise for undirected Q
      4. Extract the n_active × n_active submatrix of active neurons
      5. Build a weighted undirected networkx graph
      6. Community detection: greedy_modularity_communities (deterministic)
      7. Q = networkx.algorithms.community.modularity(...)

    Returns 0.0 for degenerate cases: fewer than 2 active neurons, or no edges.
    Q ∈ (-0.5, 1.0]; higher values indicate more modular structure.
    """
    # --- Compute absolute, symmetrised weight matrix (numpy) -----------------
    W_eff = np.array(effective_weights(genome))   # [N_max, N_max]
    W_abs = np.abs(W_eff)
    W_sym = (W_abs + W_abs.T) / 2.0

    # --- Restrict to active neurons ------------------------------------------
    active_mask = np.array(genome.active_mask)    # [N_max] bool
    active_idxs = np.where(active_mask)[0]        # indices of live neurons
    n_active    = len(active_idxs)

    if n_active < 2:
        return 0.0

    W_sub = W_sym[np.ix_(active_idxs, active_idxs)]  # [n_active, n_active]

    if W_sub.sum() == 0.0:
        return 0.0

    # --- Build networkx weighted undirected graph ----------------------------
    G = nx.Graph()
    G.add_nodes_from(range(n_active))
    for i in range(n_active):
        for j in range(i + 1, n_active):
            w = float(W_sub[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        return 0.0

    # --- Community detection and Q -------------------------------------------
    communities = greedy_modularity_communities(G, weight="weight")
    q = nx_modularity(G, communities, weight="weight")
    return float(q)


# ── Network statistics ────────────────────────────────────────────────────────

def network_stats(genome: Genome, cfg: Config) -> dict:
    """
    Compute structural statistics for one genome.

    Returns
    -------
    n_active        : int   — number of active neurons
    n_edges         : int   — active edges between active neuron pairs
    density         : float — n_edges / (n_active*(n_active-1)), or 0.0
    mean_weight     : float — mean |W_eff| over active edges; 0.0 if no edges
    wiring_cost : float — sum of wire lengths (from cost.py)
    """
    active_mask = np.array(genome.active_mask)
    edge_mask   = np.array(genome.edge_mask)

    n_active = int(active_mask.sum())

    # Active edges: edge_mask restricted to active neuron pairs
    active_edge_mask = (
        edge_mask
        & active_mask[:, None]
        & active_mask[None, :]
    )
    n_edges = int(active_edge_mask.sum())

    # Density over ordered pairs (i,j), i != j
    possible = n_active * (n_active - 1)
    density  = float(n_edges / possible) if possible > 0 else 0.0

    # Mean absolute effective weight over active edges
    W_eff = np.array(effective_weights(genome))
    W_abs = np.abs(W_eff)
    mean_weight = float(W_abs[active_edge_mask].mean()) if n_edges > 0 else 0.0

    c_conn = float(dist_cost(genome))

    return {
        "n_active":        n_active,
        "n_edges":         n_edges,
        "density":         density,
        "mean_weight":     mean_weight,
        "wiring_cost": c_conn,
    }


# ── Single-genome analysis ────────────────────────────────────────────────────

def analyse_genome(genome: Genome, cfg: Config) -> dict:
    """
    Compute all analysis metrics for one genome.

    Returns a flat dict with keys:
        q, n_active, n_edges, density, mean_weight, wiring_cost
    """
    q     = modularity_q(genome, cfg)
    stats = network_stats(genome, cfg)
    return {"q": q, **stats}


# ── Population analysis ───────────────────────────────────────────────────────

def analyse_population(pop_genomes: Genome, cfg: Config) -> dict[str, list]:
    """
    Run analyse_genome for every genome in a batched population.

    pop_genomes has a leading population_size dimension on each field
    (as returned by init_population / run_evolution).

    Returns a dict of lists, each of length population_size:
        {"q": [...], "n_active": [...], "n_edges": [...],
         "density": [...], "mean_weight": [...], "wiring_cost": [...]}
    """
    import jax.tree_util as jtu

    pop_size = pop_genomes.active_mask.shape[0]
    results: dict[str, list] = {
        "q": [], "n_active": [], "n_edges": [],
        "density": [], "mean_weight": [], "wiring_cost": [],
    }

    for i in range(pop_size):
        # Slice out the i-th genome (unbatch)
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
    """
    Produce a comprehensive summary of a completed evolutionary run.

    Parameters
    ----------
    history     : list of stats dicts from run_evolution (one per generation)
    pop_genomes : final batched population (leading pop_size dimension)
    best_genome : unbatched best genome (argmax fitness from final generation)
    cfg         : Config

    Returns
    -------
    dict with keys:

    Fitness trajectory (one value per generation):
        fitness_max   : list[float]
        fitness_mean  : list[float]
        steps_mean    : list[float]

    Final population structure (scalars):
        final_q_mean          : float
        final_q_max           : float
        final_n_active_mean   : float
        final_conn_cost_mean  : float

    Best genome metrics (scalars):
        best_q        : float
        best_n_active : int
    """
    # --- Fitness trajectory --------------------------------------------------
    fitness_max  = [h["max_fitness"]  for h in history]
    fitness_mean = [h["mean_fitness"] for h in history]
    steps_mean   = [h["mean_steps"]   for h in history]

    # --- Final population structure ------------------------------------------
    pop_metrics = analyse_population(pop_genomes, cfg)
    q_vals      = pop_metrics["q"]

    final_q_mean         = float(np.mean(q_vals))
    final_q_max          = float(np.max(q_vals))
    final_n_active_mean  = float(np.mean(pop_metrics["n_active"]))
    final_conn_cost_mean = float(np.mean(pop_metrics["wiring_cost"]))

    # --- Best genome ---------------------------------------------------------
    best_metrics  = analyse_genome(best_genome, cfg)
    best_q        = best_metrics["q"]
    best_n_active = best_metrics["n_active"]

    return {
        "fitness_max":           fitness_max,
        "fitness_mean":          fitness_mean,
        "steps_mean":            steps_mean,
        "final_q_mean":          final_q_mean,
        "final_q_max":           final_q_max,
        "final_n_active_mean":   final_n_active_mean,
        "final_conn_cost_mean":  final_conn_cost_mean,
        "best_q":                best_q,
        "best_n_active":         best_n_active,
    }
