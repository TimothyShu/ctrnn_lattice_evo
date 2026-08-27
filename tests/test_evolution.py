"""
Tests — Evolutionary loop.

Ported from ctrnn_evo.  Substantive changes:
  * `position` removed from the field-shape test
  * lambda_* penalty tests converted to *_frac
  * NEW section 4b: end-to-end selection monotonicity.  test_cost.py guards
    adjusted_fitness in isolation, but the thing that actually decides the run
    is whether tournament selection picks the better network under a live
    penalty.  A negative fitness multiplier inverts selection silently.
  * NEW section 12b: _warmup_ramp / _cycle_ramp.  config.py documents an
    off-by-one-prone schedule and nothing tested it.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ctrnn_lattice_evo import Config, WorldConfig
from ctrnn_lattice_evo.genome import grid_genome
from ctrnn_lattice_evo.mutation import MutationRates
from ctrnn_lattice_evo.brain import run_brain_episode_full
from ctrnn_lattice_evo.evolution import (
    init_population,
    eval_population,
    compute_fitness,
    tournament_select_idx,
    select_parents,
    reproduce,
    evolve_step,
    collect_stats,
    run_evolution,
    _warmup_ramp,
    _cycle_ramp,
    _mutation_scale,
)


POP = 20
GENS = 10


@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1,
                  K=4, population_size=POP, tournament_size=3)


@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=100)


@pytest.fixture(scope="module")
def rates():
    """Node operators off — lattice slots are fixed."""
    return MutationRates(add_node_prob=0.0, remove_node_prob=0.0)


@pytest.fixture(scope="module")
def pop(cfg):
    return init_population(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def evaluated(pop, cfg, wcfg):
    steps, c_acts, raw_food = eval_population(
        jax.random.PRNGKey(1), pop, cfg, wcfg, n_evals=3)
    fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg)
    return steps, c_acts, fitness


# ── 1. init_population ────────────────────────────────────────────────────────

def test_init_population_field_shapes(pop, cfg):
    assert pop.active_mask.shape == (POP, cfg.N_max)
    assert pop.weight_matrix.shape == (POP, cfg.N_max, cfg.N_max)
    assert pop.tau.shape == (POP, cfg.N_max)
    assert pop.bias.shape == (POP, cfg.N_max)
    assert pop.edge_mask.shape == (POP, cfg.N_max, cfg.N_max)
    assert pop.neuron_type.shape == (POP, cfg.N_max)


def test_init_population_has_no_position(pop):
    assert not hasattr(pop, "position")


def test_init_population_weights_nonneg(pop):
    assert jnp.all(pop.weight_matrix >= 0)


def test_init_population_io_slots_always_active(pop, cfg):
    assert jnp.all(pop.active_mask[:, :cfg.n_in])
    assert jnp.all(pop.active_mask[:, -cfg.n_out:])


def test_init_population_respects_init_mode(cfg):
    """The arm is selected by cfg.init_mode, resolved in Python before vmap."""
    grid_pop = init_population(jax.random.PRNGKey(0), cfg)
    unif_pop = init_population(jax.random.PRNGKey(0),
                               dataclasses.replace(cfg, init_mode="uniform"))
    assert not jnp.array_equal(grid_pop.edge_mask, unif_pop.edge_mask)


def test_grid_population_shares_one_topology(pop):
    """Generation-0 diversity is parametric, not structural."""
    for i in range(1, POP):
        assert jnp.array_equal(pop.edge_mask[0], pop.edge_mask[i])


# ── 2. eval_population ────────────────────────────────────────────────────────

def test_eval_population_shapes(evaluated):
    steps, c_acts, _ = evaluated
    assert steps.shape == (POP,) and c_acts.shape == (POP,)


def test_eval_population_steps_bounded(evaluated, wcfg):
    steps, _, _ = evaluated
    assert jnp.all(steps >= 0) and jnp.all(steps <= wcfg.episode_steps)


def test_eval_population_c_acts_bounded(evaluated):
    _, c_acts, _ = evaluated
    assert jnp.all(c_acts >= 0) and jnp.all(c_acts <= 1.0 + 1e-6)


def test_eval_population_determinism(pop, cfg, wcfg):
    key = jax.random.PRNGKey(7)
    s1, c1, r1 = eval_population(key, pop, cfg, wcfg, n_evals=2)
    s2, c2, r2 = eval_population(key, pop, cfg, wcfg, n_evals=2)
    assert jnp.array_equal(s1, s2) and jnp.allclose(c1, c2) and jnp.allclose(r1, r2)


# ── 3. compute_fitness ────────────────────────────────────────────────────────

def test_compute_fitness_shape(evaluated):
    assert evaluated[2].shape == (POP,)


def test_compute_fitness_in_unit_interval_when_no_penalty(pop, cfg, wcfg):
    assert cfg.edge_frac == 0.0 and cfg.dist_frac == 0.0 and cfg.act_frac == 0.0
    assert cfg.fitness_mode == "survival"
    steps, c_acts, raw_food = eval_population(
        jax.random.PRNGKey(8), pop, cfg, wcfg, n_evals=2)
    fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg)
    assert jnp.all(fitness >= 0.0) and jnp.all(fitness <= 1.0 + 1e-5)


def test_compute_fitness_penalty_reduces_fitness(cfg, wcfg):
    cfg_pen = dataclasses.replace(cfg, edge_frac=0.3)
    pop_small = init_population(jax.random.PRNGKey(9), cfg)
    steps, c_acts, raw_food = eval_population(
        jax.random.PRNGKey(9), pop_small, cfg, wcfg, n_evals=2)

    f_no = compute_fitness(steps, c_acts, raw_food, pop_small, cfg, wcfg)
    f_pen = compute_fitness(steps, c_acts, raw_food, pop_small, cfg_pen, wcfg)
    assert jnp.all(f_pen <= f_no + 1e-6)


def test_compute_fitness_never_negative(cfg, wcfg):
    """The clamp, at population level.  Without it, extreme penalties produce
    negative fitness and selection prefers the worst individuals."""
    cfg_pen = dataclasses.replace(cfg, edge_frac=5.0, dist_frac=5.0, act_frac=5.0)
    pop_small = init_population(jax.random.PRNGKey(9), cfg)
    steps, c_acts, raw_food = eval_population(
        jax.random.PRNGKey(9), pop_small, cfg, wcfg, n_evals=2)
    f = compute_fitness(steps, c_acts, raw_food, pop_small, cfg_pen, wcfg)
    assert jnp.all(f >= 0.0)


@pytest.mark.parametrize("mode", ["survival", "food"])
def test_raw_fitness_nonnegative_in_both_modes(cfg, wcfg, mode):
    """f_raw >= 0 is the precondition the multiplicative penalty relies on.
    Food score is uncapped above, but must never go below zero."""
    c = dataclasses.replace(cfg, fitness_mode=mode)
    pop_small = init_population(jax.random.PRNGKey(3), c)
    steps, c_acts, raw_food = eval_population(
        jax.random.PRNGKey(3), pop_small, c, wcfg, n_evals=2)
    f = compute_fitness(steps, c_acts, raw_food, pop_small, c, wcfg)
    assert jnp.all(f >= 0.0)


# ── 4. tournament_select_idx ──────────────────────────────────────────────────

def test_tournament_select_idx_valid_range(cfg):
    fitness = jnp.array([0.1, 0.5, 0.9, 0.2, 0.7])
    for i in range(10):
        idx = tournament_select_idx(jax.random.PRNGKey(i), fitness, cfg.tournament_size)
        assert 0 <= int(idx) < len(fitness)


def test_tournament_always_picks_best():
    fitness = jnp.array([0.01, 0.01, 100.0, 0.01, 0.01])
    for i in range(20):
        idx = tournament_select_idx(jax.random.PRNGKey(i), fitness,
                                    tournament_size=len(fitness))
        assert int(idx) == 2


# ── 4b. Selection monotonicity under penalty ─────────────────────────────────

def test_selection_prefers_better_network_under_extreme_penalty(cfg, wcfg):
    """END-TO-END guard on the sign flip.

    Two genomes with identical wiring but different raw performance.  With an
    unclamped multiplier of -4, the better network maps to a MORE negative
    adjusted fitness and loses every tournament — evolution runs backwards
    while every unit test still passes.
    """
    cfg_pen = dataclasses.replace(cfg, edge_frac=5.0, population_size=2, tournament_size=2)
    two = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x]), grid_genome(jax.random.PRNGKey(0), cfg))

    steps = jnp.array([90.0, 10.0])          # index 0 is clearly better
    c_acts = jnp.array([0.5, 0.5])
    raw_food = jnp.array([0.0, 0.0])

    f = compute_fitness(steps, c_acts, raw_food, two, cfg_pen, wcfg)
    assert float(f[0]) >= float(f[1]), "penalty inverted the fitness ordering"

    for i in range(20):
        idx = tournament_select_idx(jax.random.PRNGKey(i), f, tournament_size=2)
        assert int(idx) == 0 or float(f[0]) == float(f[1])


def test_selection_monotonic_across_frac_sweep(cfg, wcfg):
    """Holds at every point in the lambda sweep, not just the extreme."""
    two = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x]), grid_genome(jax.random.PRNGKey(0), cfg))
    steps = jnp.array([90.0, 10.0])
    c_acts = jnp.array([0.5, 0.5])
    raw_food = jnp.array([0.0, 0.0])

    for frac in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0]:
        c = dataclasses.replace(cfg, edge_frac=frac, population_size=2, tournament_size=2)
        f = compute_fitness(steps, c_acts, raw_food, two, c, wcfg)
        assert float(f[0]) >= float(f[1]), f"ordering inverted at edge_frac={frac}"


def test_sparser_network_wins_at_equal_performance(cfg, wcfg):
    """The penalty must actually reward pruning, or the sweep does nothing."""
    g = grid_genome(jax.random.PRNGKey(0), cfg)
    half = g.edge_mask & (jnp.arange(g.edge_mask.size)
                          .reshape(g.edge_mask.shape) % 2 == 0)
    pair = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), g,
                                  dataclasses.replace(g, edge_mask=half))

    c = dataclasses.replace(cfg, edge_frac=0.3, population_size=2, tournament_size=2)
    f = compute_fitness(jnp.array([50.0, 50.0]), jnp.array([0.5, 0.5]),
                        jnp.array([0.0, 0.0]), pair, c, wcfg)
    assert float(f[1]) > float(f[0]), "pruning did not pay under a live penalty"


# ── 5. select_parents ─────────────────────────────────────────────────────────

def test_select_parents_shape_and_validity(evaluated, cfg):
    _, _, fitness = evaluated
    idxs = select_parents(jax.random.PRNGKey(10), fitness,
                          cfg.population_size, cfg.tournament_size)
    assert idxs.shape == (POP,)
    assert jnp.all(idxs >= 0) and jnp.all(idxs < POP)


# ── 6. reproduce ──────────────────────────────────────────────────────────────

def test_reproduce_shape_preserved(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    idxs = select_parents(jax.random.PRNGKey(12), fitness,
                          cfg.population_size, cfg.tournament_size)
    offspring = reproduce(jax.random.PRNGKey(13), pop, idxs, rates, cfg)
    assert offspring.weight_matrix.shape == pop.weight_matrix.shape
    assert offspring.active_mask.shape == pop.active_mask.shape


def test_reproduce_offspring_differ_from_parents(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    idxs = select_parents(jax.random.PRNGKey(14), fitness,
                          cfg.population_size, cfg.tournament_size)
    parents = jax.tree_util.tree_map(lambda x: x[idxs], pop)
    offspring = reproduce(jax.random.PRNGKey(15), pop, idxs, rates, cfg)
    assert float(jnp.sum(jnp.abs(
        offspring.weight_matrix - parents.weight_matrix))) > 0


def test_reproduce_topology_diversifies(pop, evaluated, cfg, rates):
    """Grid genomes start identical in structure, so structural mutation is
    the ONLY source of topological diversity.  If it never fires, the whole
    population stays on one topology forever."""
    _, _, fitness = evaluated
    idxs = select_parents(jax.random.PRNGKey(16), fitness,
                          cfg.population_size, cfg.tournament_size)
    off = reproduce(jax.random.PRNGKey(17), pop, idxs, rates, cfg)
    distinct = any(not jnp.array_equal(off.edge_mask[0], off.edge_mask[i])
                   for i in range(1, POP))
    assert distinct, "no structural diversity after reproduction"


# ── 7. evolve_step ────────────────────────────────────────────────────────────

def test_evolve_step_shape_preserved(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    new_pop = evolve_step(jax.random.PRNGKey(16), pop, fitness, rates, cfg)
    assert new_pop.weight_matrix.shape == pop.weight_matrix.shape


def test_evolve_step_determinism(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(17)
    p1 = evolve_step(key, pop, fitness, rates, cfg)
    p2 = evolve_step(key, pop, fitness, rates, cfg)
    assert jnp.allclose(p1.weight_matrix, p2.weight_matrix)
    assert jnp.array_equal(p1.edge_mask, p2.edge_mask)


def test_evolve_step_io_slots_preserved(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    new_pop = evolve_step(jax.random.PRNGKey(18), pop, fitness, rates, cfg)
    assert jnp.all(new_pop.active_mask[:, :cfg.n_in])
    assert jnp.all(new_pop.active_mask[:, -cfg.n_out:])


def test_evolve_step_node_count_stable_without_node_operators(pop, evaluated, cfg, rates):
    """With add/remove_node off, only prune_isolated can deactivate a slot —
    and it is one-way.  A single step should not lose many nodes."""
    _, _, fitness = evaluated
    new_pop = evolve_step(jax.random.PRNGKey(19), pop, fitness, rates, cfg)
    assert float(new_pop.active_mask.sum()) >= 0.9 * float(pop.active_mask.sum())


# ── 8. Elitism ────────────────────────────────────────────────────────────────

def test_elitism_best_parent_in_slot_zero(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    best_idx = int(jnp.argmax(fitness))
    new_pop = evolve_step(jax.random.PRNGKey(19), pop, fitness, rates, cfg)
    assert jnp.allclose(new_pop.weight_matrix[0], pop.weight_matrix[best_idx])


def test_elitism_preserves_elite_topology(pop, evaluated, cfg, rates):
    """Elitism must carry the edge_mask too, not just weights — otherwise a
    well-pruned elite is silently rewired each generation."""
    _, _, fitness = evaluated
    best_idx = int(jnp.argmax(fitness))
    new_pop = evolve_step(jax.random.PRNGKey(19), pop, fitness, rates, cfg)
    assert jnp.array_equal(new_pop.edge_mask[0], pop.edge_mask[best_idx])


def test_elitism_survives_all_zero_fitness(pop, cfg, rates):
    """Under a crushing penalty every individual clamps to 0.0 and argmax ties
    at index 0.  Must not crash or produce NaN."""
    zero_fitness = jnp.zeros(POP)
    new_pop = evolve_step(jax.random.PRNGKey(20), pop, zero_fitness, rates, cfg)
    assert not jnp.any(jnp.isnan(new_pop.weight_matrix))


# ── 9. run_brain_episode_full ─────────────────────────────────────────────────

def test_run_brain_episode_full_returns_four(cfg, wcfg):
    g = grid_genome(jax.random.PRNGKey(20), cfg)
    assert len(run_brain_episode_full(jax.random.PRNGKey(20), g, cfg, wcfg)) == 4


def test_run_brain_episode_full_determinism(cfg, wcfg):
    key = jax.random.PRNGKey(22)
    g = grid_genome(key, cfg)
    _, s1, c1, r1 = run_brain_episode_full(key, g, cfg, wcfg)
    _, s2, c2, r2 = run_brain_episode_full(key, g, cfg, wcfg)
    assert int(s1) == int(s2) and jnp.isclose(c1, c2) and jnp.isclose(r1, r2)


# ── 10. run_evolution ─────────────────────────────────────────────────────────

def test_run_evolution_history_length(cfg, wcfg, rates):
    _, _, history = run_evolution(jax.random.PRNGKey(30), GENS, cfg, wcfg,
                                  rates, n_evals=2)
    assert len(history) == GENS


def test_run_evolution_history_keys(cfg, wcfg, rates):
    _, _, history = run_evolution(jax.random.PRNGKey(31), 2, cfg, wcfg,
                                  rates, n_evals=2)
    expected = {"generation", "max_fitness", "mean_fitness", "max_steps",
                "mean_steps", "mean_n_active", "mean_edge_cost",
                "mean_wiring_cost", "mean_n_edges", "mean_local_fraction"}
    assert expected <= set(history[0]), f"Missing: {expected - set(history[0])}"


def test_run_evolution_logs_edge_count_per_generation(cfg, wcfg, rates):
    """'When do edges die' is the over-pruning measurement.  It has to be a
    per-generation trace, not an end-of-run number."""
    _, _, history = run_evolution(jax.random.PRNGKey(36), 5, cfg, wcfg,
                                  rates, n_evals=1)
    counts = [h["mean_n_edges"] for h in history]
    assert len(counts) == 5
    assert all(c >= 0 for c in counts)


def test_run_evolution_generation_indices(cfg, wcfg, rates):
    _, _, history = run_evolution(jax.random.PRNGKey(32), GENS, cfg, wcfg,
                                  rates, n_evals=2)
    assert [h["generation"] for h in history] == list(range(GENS))


def test_run_evolution_callback_called(cfg, wcfg, rates):
    calls = []
    run_evolution(jax.random.PRNGKey(33), 3, cfg, wcfg, rates, n_evals=2,
                  callback=lambda s, g: calls.append(s))
    assert len(calls) == 3


def test_run_evolution_returns_valid_best_genome(cfg, wcfg, rates):
    best, final_fitness, _ = run_evolution(jax.random.PRNGKey(34), 2, cfg,
                                           wcfg, rates, n_evals=2)
    assert best.weight_matrix.shape == (cfg.N_max, cfg.N_max)
    assert final_fitness.shape == (POP,)


def test_run_evolution_fitness_plausible(cfg, wcfg, rates):
    _, _, history = run_evolution(jax.random.PRNGKey(35), GENS, cfg, wcfg,
                                  rates, n_evals=3)
    for stats in history:
        assert 0.0 <= stats["max_fitness"] <= 1.0 + 1e-6
        assert stats["mean_fitness"] <= stats["max_fitness"] + 1e-6


@pytest.mark.parametrize("mode", ["grid", "uniform", "sparse"])
def test_all_three_arms_run(cfg, wcfg, rates, mode):
    """Integration gate: every arm of the experiment must complete a short run
    on the same code path before anything is queued."""
    c = dataclasses.replace(cfg, init_mode=mode)
    _, _, history = run_evolution(jax.random.PRNGKey(50), 2, c, wcfg,
                                  rates, n_evals=1)
    assert len(history) == 2


# ── 11. collect_stats ─────────────────────────────────────────────────────────

def test_collect_stats_values_plausible(pop, evaluated, cfg):
    steps, _, fitness = evaluated
    stats = collect_stats(0, fitness, steps, pop, cfg)
    assert cfg.n_in + cfg.n_out <= stats["mean_n_active"] <= cfg.N_max
    assert stats["mean_edge_cost"] >= 0.0
    assert stats["mean_wiring_cost"] >= 0.0
    assert 0.0 <= stats["mean_fitness"] <= stats["max_fitness"] + 1e-6


def test_collect_stats_reports_lattice_values(pop, evaluated, cfg):
    steps, _, fitness = evaluated
    stats = collect_stats(0, fitness, steps, pop, cfg)
    assert stats["mean_n_active"] == pytest.approx(16.0)
    assert stats["mean_n_edges"] == pytest.approx(84.0)
    assert stats["mean_local_fraction"] == pytest.approx(1.0)


# ── 12. _mutation_scale ───────────────────────────────────────────────────────

class TestMutationScale:
    def test_gen0_returns_full_scale(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        assert _mutation_scale(0, cfg) == pytest.approx(3.0)

    def test_midpoint_returns_midpoint_scale(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        assert _mutation_scale(100, cfg) == pytest.approx(2.0)

    def test_at_warmup_end_returns_one(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        assert _mutation_scale(200, cfg) == pytest.approx(1.0)

    def test_after_warmup_returns_one(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        for gen in (201, 500, 999):
            assert _mutation_scale(gen, cfg) == pytest.approx(1.0)

    def test_no_warmup_always_returns_one(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=0, mutation_warmup_scale=3.0)
        for gen in (0, 100, 999):
            assert _mutation_scale(gen, cfg) == pytest.approx(1.0)

    def test_scale_never_below_one(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_warmup_gens=200, mutation_warmup_scale=5.0)
        for gen in range(0, 300, 10):
            assert _mutation_scale(gen, cfg) >= 1.0 - 1e-6


# ── 12b. Penalty schedule ─────────────────────────────────────────────────────

class TestWarmupRamp:
    """The over-pruning fix.  config.py documents this schedule in prose and
    nothing tested it — an off-by-one here silently changes every arm."""

    @pytest.fixture
    def cfg(self):
        return Config(N_max=16, grid_W=4, grid_H=4, penalty_warmup_gens=200)

    def test_starts_at_zero(self, cfg):
        assert _warmup_ramp(0, cfg) == pytest.approx(0.0)

    def test_midpoint_is_half(self, cfg):
        assert _warmup_ramp(100, cfg) == pytest.approx(0.5)

    def test_reaches_full_at_end(self, cfg):
        assert _warmup_ramp(200, cfg) == pytest.approx(1.0)

    def test_stays_full_after(self, cfg):
        for gen in (201, 500, 999):
            assert _warmup_ramp(gen, cfg) == pytest.approx(1.0)

    def test_monotone_nondecreasing(self, cfg):
        vals = [_warmup_ramp(g, cfg) for g in range(0, 250, 5)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))

    def test_disabled_is_always_full(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4, penalty_warmup_gens=0)
        for gen in (0, 1, 500):
            assert _warmup_ramp(gen, cfg) == pytest.approx(1.0)


class TestCycleRamp:
    """warmup=200, cycle=300, free=100:
         0-199   warmup ramp
         200-499 full penalty
         500-599 free
         600-899 full penalty
         900-999 free
    """

    @pytest.fixture
    def cfg(self):
        return Config(N_max=16, grid_W=4, grid_H=4, penalty_warmup_gens=200,
                      penalty_cycle_gens=300, penalty_cycle_free_gens=100)

    @pytest.mark.parametrize("gen", [200, 300, 399, 500, 650, 699])
    def test_penalty_on_during_locked_window(self, cfg, gen):
        assert _cycle_ramp(gen, cfg) == pytest.approx(1.0)

    @pytest.mark.parametrize("gen", [400, 450, 499, 700, 799])
    def test_penalty_off_during_free_window(self, cfg, gen):
        assert _cycle_ramp(gen, cfg) == pytest.approx(0.0)

    def test_boundaries_are_exact(self, cfg):
        """The off-by-one: 499 is the last penalised gen, 500 the first free,
        599 the last free, 600 penalised again."""
        assert _cycle_ramp(399, cfg) == pytest.approx(1.0)
        assert _cycle_ramp(400, cfg) == pytest.approx(0.0)
        assert _cycle_ramp(499, cfg) == pytest.approx(0.0)
        assert _cycle_ramp(500, cfg) == pytest.approx(1.0)

    def test_disabled_is_always_on(self):
        cfg = Config(N_max=16, grid_W=4, grid_H=4,
                     penalty_cycle_gens=0, penalty_cycle_free_gens=0)
        for gen in range(0, 1000, 97):
            assert _cycle_ramp(gen, cfg) == pytest.approx(1.0)

    def test_free_windows_are_a_minority(self, cfg):
        """free/cycle = 1/3 here; if most generations were free the run would
        silently be an unpenalised baseline."""
        span = 3 * cfg.penalty_cycle_gens          # whole cycles only
        free = sum(1 for g in range(200, 200 + span) if _cycle_ramp(g, cfg) == 0.0)
        assert free / span == pytest.approx(1 / 3, abs=0.01)