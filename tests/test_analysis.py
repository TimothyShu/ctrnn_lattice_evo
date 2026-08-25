"""
Tests — Analysis layer.

Ported from ctrnn_evo with `position` removed from both hand-built genomes.
network_stats now derives the distance matrix from cfg rather than reading a
genome field.

New section at the end: local_fraction.  That metric is what the locality
claim rests on and it had no home in ctrnn_evo.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_lattice_evo import Config, WorldConfig, Genome
from ctrnn_lattice_evo.genome import grid_genome, uniform_genome
from ctrnn_lattice_evo.evolution import init_population
from ctrnn_lattice_evo.analysis import (
    modularity_q,
    network_stats,
    analyse_genome,
    analyse_population,
    summarise_run,
    local_fraction,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1,
                  K=4, population_size=8, tournament_size=3)


@pytest.fixture(scope="module")
def genome(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def pop(cfg):
    return init_population(jax.random.PRNGKey(1), cfg)


@pytest.fixture(scope="module")
def history():
    return [
        {
            "generation": i,
            "max_fitness": 0.5 + i * 0.05,
            "mean_fitness": 0.3 + i * 0.04,
            "max_steps": 50 + i * 2,
            "mean_steps": 30.0 + i * 1.5,
            "mean_n_active": 16.0,
            "mean_edge_cost": 84.0,
            "mean_wiring_cost": 84.0,
        }
        for i in range(5)
    ]


def _blank(cfg: Config, **overrides) -> Genome:
    """Six-field blank genome — no `position`."""
    N = cfg.N_max
    g = Genome(
        active_mask=jnp.zeros(N, dtype=bool),
        neuron_type=jnp.zeros(N, dtype=jnp.uint8),
        tau=jnp.full(N, (cfg.tau_e_range[0] + cfg.tau_e_range[1]) / 2.0),
        bias=jnp.zeros(N),
        weight_matrix=jnp.zeros((N, N)),
        edge_mask=jnp.zeros((N, N), dtype=bool),
    )
    return dataclasses.replace(g, **overrides)


def _make_two_cluster_genome(cfg: Config) -> Genome:
    """Block-diagonal: two internally dense clusters, no cross edges."""
    N, n_in, n_out = cfg.N_max, cfg.n_in, cfg.n_out
    cluster_A = list(range(n_in, 6))
    cluster_B = list(range(6, 10))
    active_idxs = list(range(n_in)) + cluster_A + cluster_B + list(range(N - n_out, N))
    active_mask = jnp.zeros(N, dtype=bool).at[jnp.array(active_idxs)].set(True)

    wm = np.zeros((N, N), dtype=np.float32)
    em = np.zeros((N, N), dtype=bool)
    for cluster in (cluster_A, cluster_B):
        for i in cluster:
            for j in cluster:
                wm[i, j] = 1.0
                if i != j:
                    em[i, j] = True

    return _blank(cfg, active_mask=active_mask,
                  weight_matrix=jnp.array(wm), edge_mask=jnp.array(em))


def _make_fully_connected_genome(cfg: Config) -> Genome:
    N, n_in, n_out = cfg.N_max, cfg.n_in, cfg.n_out
    active_idxs = list(range(n_in)) + list(range(n_in, 10)) + list(range(N - n_out, N))
    active_mask = jnp.zeros(N, dtype=bool).at[jnp.array(active_idxs)].set(True)

    am = np.array(active_mask)
    wm = np.outer(am.astype(float), am.astype(float)).astype(np.float32)
    np.fill_diagonal(wm, 0)

    return _blank(cfg, active_mask=active_mask,
                  weight_matrix=jnp.array(wm), edge_mask=jnp.array(wm > 0))


# ── 1. modularity_q ───────────────────────────────────────────────────────────

def test_modularity_q_returns_float(genome, cfg):
    assert isinstance(modularity_q(genome, cfg), float)


def test_modularity_q_in_valid_range(cfg):
    for seed in range(10):
        q = modularity_q(uniform_genome(jax.random.PRNGKey(seed), cfg), cfg)
        assert -0.5 - 1e-6 <= q <= 1.0 + 1e-6, f"Q={q:.4f} out of range at seed {seed}"


def test_modularity_q_no_edges_returns_zero(cfg, genome):
    g = dataclasses.replace(genome, edge_mask=jnp.zeros_like(genome.edge_mask))
    assert modularity_q(g, cfg) == 0.0


def test_modularity_q_single_active_returns_zero(cfg, genome):
    g = dataclasses.replace(
        genome,
        active_mask=jnp.zeros(cfg.N_max, dtype=bool).at[0].set(True),
        edge_mask=jnp.zeros_like(genome.edge_mask),
    )
    assert modularity_q(g, cfg) == 0.0


def test_modularity_q_two_clusters_higher_than_fully_connected(cfg):
    q_mod = modularity_q(_make_two_cluster_genome(cfg), cfg)
    q_dense = modularity_q(_make_fully_connected_genome(cfg), cfg)
    assert q_mod > q_dense


def test_modularity_q_two_clusters_positive(cfg):
    assert modularity_q(_make_two_cluster_genome(cfg), cfg) > 0.0


def test_modularity_q_deterministic(genome, cfg):
    assert modularity_q(genome, cfg) == modularity_q(genome, cfg)


def test_lattice_q_is_intermediate(cfg):
    """A lattice is locally clustered without being block-structured, so its Q
    should sit between a fully-connected network and a clean two-module one.
    Informative rather than load-bearing — if it comes out near zero, Q is not
    picking up lattice structure and should not be used as evidence for it."""
    q_lattice = modularity_q(grid_genome(jax.random.PRNGKey(0), cfg), cfg)
    q_dense = modularity_q(_make_fully_connected_genome(cfg), cfg)
    assert q_lattice > q_dense


# ── 2. network_stats ──────────────────────────────────────────────────────────

def test_network_stats_keys(genome, cfg):
    expected = {"n_active", "n_edges", "density", "mean_weight",
                "wiring_cost", "local_fraction"}
    assert expected <= set(network_stats(genome, cfg))


def test_network_stats_n_active_matches_mask(genome, cfg):
    assert network_stats(genome, cfg)["n_active"] == int(genome.active_mask.sum())


def test_network_stats_n_edges_matches_mask(genome, cfg):
    expected = int((genome.edge_mask
                    & genome.active_mask[:, None]
                    & genome.active_mask[None, :]).sum())
    assert network_stats(genome, cfg)["n_edges"] == expected


def test_network_stats_grid_known_values(genome, cfg):
    """4x4 at r=1: 16 nodes, 84 edges, 84/240 = 35% density."""
    stats = network_stats(genome, cfg)
    assert stats["n_active"] == 16
    assert stats["n_edges"] == 84
    assert stats["density"] == pytest.approx(84 / 240, rel=1e-3)


def test_network_stats_density_denominator_is_all_pairs(genome, cfg):
    """Density is out of n_active*(n_active-1), NOT out of the lattice mask.
    A full lattice therefore reads as 35%, not 100% — this is the denominator
    the cost normalisation uses."""
    stats = network_stats(genome, cfg)
    assert stats["density"] < 1.0


def test_network_stats_density_range(cfg):
    for seed in range(5):
        stats = network_stats(uniform_genome(jax.random.PRNGKey(seed), cfg), cfg)
        assert 0.0 <= stats["density"] <= 1.0 + 1e-6


def test_network_stats_wiring_cost_nonneg(genome, cfg):
    assert network_stats(genome, cfg)["wiring_cost"] >= 0.0


def test_network_stats_no_edges_is_degenerate_safe(cfg, genome):
    g = dataclasses.replace(genome, edge_mask=jnp.zeros_like(genome.edge_mask))
    stats = network_stats(g, cfg)
    assert stats["mean_weight"] == 0.0
    assert stats["n_edges"] == 0
    assert stats["density"] == 0.0


# ── 3. analyse_genome ─────────────────────────────────────────────────────────

def test_analyse_genome_all_scalars(genome, cfg):
    for k, v in analyse_genome(genome, cfg).items():
        assert isinstance(v, (int, float)), f"'{k}' is {type(v)}, expected scalar"


def test_analyse_genome_q_consistent(genome, cfg):
    assert abs(analyse_genome(genome, cfg)["q"] - modularity_q(genome, cfg)) < 1e-9


def test_analyse_genome_n_active_consistent(genome, cfg):
    assert analyse_genome(genome, cfg)["n_active"] == network_stats(genome, cfg)["n_active"]


# ── 4. analyse_population ─────────────────────────────────────────────────────

def test_analyse_population_list_lengths(pop, cfg):
    for k, v in analyse_population(pop, cfg).items():
        assert len(v) == cfg.population_size, f"'{k}' has length {len(v)}"


def test_analyse_population_q_values_valid(pop, cfg):
    for i, q in enumerate(analyse_population(pop, cfg)["q"]):
        assert -0.5 - 1e-6 <= q <= 1.0 + 1e-6, f"Q={q:.4f} out of range at genome {i}"


def test_analyse_population_n_active_positive(pop, cfg):
    for n in analyse_population(pop, cfg)["n_active"]:
        assert n >= cfg.n_in + cfg.n_out


# ── 5. summarise_run ──────────────────────────────────────────────────────────

def test_summarise_run_expected_keys(history, pop, genome, cfg):
    expected = {"fitness_max", "fitness_mean", "steps_mean",
                "final_q_mean", "final_q_max",
                "final_n_active_mean", "final_conn_cost_mean",
                "best_q", "best_n_active"}
    assert expected <= set(summarise_run(history, pop, genome, cfg))


def test_summarise_run_fitness_list_length(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert len(result["fitness_max"]) == len(history)
    assert len(result["fitness_mean"]) == len(history)


def test_summarise_run_fitness_values_from_history(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    for i, h in enumerate(history):
        assert result["fitness_max"][i] == pytest.approx(h["max_fitness"])
        assert result["fitness_mean"][i] == pytest.approx(h["mean_fitness"])


def test_summarise_run_best_n_active_valid(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert cfg.n_in + cfg.n_out <= result["best_n_active"] <= cfg.N_max


def test_summarise_run_final_q_max_ge_mean(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert result["final_q_max"] >= result["final_q_mean"] - 1e-6


# ── 6. local_fraction — the locality metric ───────────────────────────────────

def test_local_fraction_is_one_on_fresh_lattice(genome, cfg):
    assert local_fraction(genome, cfg) == pytest.approx(1.0)


def test_local_fraction_low_on_uniform_init(cfg):
    """The control arm never started local, so it should score far lower on
    the same measure — the contrast that makes the metric meaningful."""
    assert local_fraction(uniform_genome(jax.random.PRNGKey(0), cfg), cfg) < 0.60


def test_local_fraction_in_unit_interval(cfg):
    for seed in range(5):
        f = local_fraction(uniform_genome(jax.random.PRNGKey(seed), cfg), cfg)
        assert 0.0 <= f <= 1.0


def test_local_fraction_no_edges_is_safe(cfg, genome):
    """Division by zero at the end of an aggressive prune."""
    g = dataclasses.replace(genome, edge_mask=jnp.zeros_like(genome.edge_mask))
    f = local_fraction(g, cfg)
    assert f == 0.0 or f == pytest.approx(1.0)


def test_local_fraction_drops_when_distal_edges_added(genome, cfg):
    """Adding an out-of-lattice edge must move the metric — otherwise erosion
    would be invisible in the per-generation trace."""
    from ctrnn_lattice_evo.topology import dist_matrix
    d = dist_matrix(cfg.grid_W, cfg.grid_H)
    far = jnp.argmax(d[0])
    g2 = dataclasses.replace(genome, edge_mask=genome.edge_mask.at[0, far].set(True))
    assert local_fraction(g2, cfg) < local_fraction(genome, cfg)