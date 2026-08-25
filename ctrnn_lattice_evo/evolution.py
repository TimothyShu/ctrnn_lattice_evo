"""
evolution.py — Evolutionary loop for CTRNN neuroevolution.

Public API
----------
init_population(key, cfg)
    Initialise a random population of genomes.

eval_population(key, pop_genomes, cfg, wcfg, n_evals=5)
    Evaluate each genome over n_evals independent episodes.
    Returns (mean_steps[pop], mean_c_act[pop], mean_raw_food[pop]).

compute_fitness(steps, c_acts, raw_food, pop_genomes, cfg, wcfg)
    Normalise performance metric to f_raw and apply cost penalties.
    fitness_mode="survival": f_raw = steps / episode_steps  → [0, 1]
    fitness_mode="food":     f_raw = raw_food / (episode_steps * n_food_types)  → [0, ∞)
    Returns fitness[pop].

tournament_select_idx(key, fitness, tournament_size)
    Sample tournament_size candidates, return index of the best.

select_parents(key, fitness, pop_size, tournament_size)
    Run tournament selection pop_size times.
    Returns parent_idxs[pop].

reproduce(key, pop_genomes, parent_idxs, rates, cfg)
    Gather parents by index, mutate each, return offspring population.

evolve_step(key, pop_genomes, fitness, rates, cfg)
    One full generation: select → reproduce → elitism.
    Returns new (unevaluated) population.

collect_stats(generation, fitness, steps, pop_genomes, cfg)
    Compute per-generation statistics dict (Python floats, no JAX arrays).

run_evolution(key, n_generations, cfg, wcfg, rates, n_evals, callback)
    Drive the full evolutionary loop; return (best_genome, final_fitness, history).
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp

from pathlib import Path

from .config import Config
from .genome import Genome, random_genome
from .mutation import MutationRates, mutate
from .cost import edge_count_cost, dist_cost, adjusted_fitness
from .world import WorldConfig
from .brain import run_brain_episode_full
from .logger import save_training_state, load_training_state


# ── Population initialisation ─────────────────────────────────────────────────

def init_population(key: jax.Array, cfg: Config) -> Genome:
    """
    Initialise a population of cfg.population_size random genomes.

    Returns a batched Genome with a leading [population_size] dimension
    on every field.
    """
    keys = jax.random.split(key, cfg.population_size)
    return jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)


# ── Fitness evaluation ────────────────────────────────────────────────────────

def _eval_genome(
    keys: jax.Array,       # [n_evals, 2]
    genome: Genome,
    cfg: Config,
    wcfg: WorldConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate a single genome over n_evals independent episodes.

    Returns (mean_steps, mean_c_act, mean_raw_food) averaged across episodes.
    """
    _, steps_all, c_acts_all, raw_food_all = jax.vmap(
        run_brain_episode_full, in_axes=(0, None, None, None)
    )(keys, genome, cfg, wcfg)
    return (
        jnp.mean(steps_all.astype(jnp.float32)),
        jnp.mean(c_acts_all),
        jnp.mean(raw_food_all),
    )


def eval_population(
    key: jax.Array,
    pop_genomes: Genome,
    cfg: Config,
    wcfg: WorldConfig,
    n_evals: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the full population.

    Each genome is assessed across n_evals episodes with independent seeds.
    All evaluations are parallelised via nested vmap (population x episodes).

    Returns
    -------
    mean_steps     : float32 [pop_size] — average steps survived
    mean_c_acts    : float32 [pop_size] — average activation cost
    mean_raw_food  : float32 [pop_size] — average cumulative raw food score
                     (sum of uncapped food_at over alive steps, averaged across evals)
    """
    # Split into [pop_size, n_evals, 2] keys
    flat_keys  = jax.random.split(key, cfg.population_size * n_evals)
    epoch_keys = flat_keys.reshape(cfg.population_size, n_evals, -1)

    return jax.vmap(_eval_genome, in_axes=(0, 0, None, None))(
        epoch_keys, pop_genomes, cfg, wcfg
    )


def _mutation_scale(generation: int, cfg: Config) -> float:
    """Scale factor for continuous mutation sigmas during warmup.

    Mirrors the penalty ramp in reverse: starts at mutation_warmup_scale at
    gen 0, decays linearly to 1.0 at penalty_warmup_gens, stays 1.0 after.
    Returns 1.0 when mutation_warmup_scale==1.0 or penalty_warmup_gens==0.
    """
    if cfg.mutation_warmup_scale > 1.0 and cfg.penalty_warmup_gens > 0:
        ramp = min(generation / cfg.penalty_warmup_gens, 1.0)
        return cfg.mutation_warmup_scale * (1.0 - ramp) + 1.0 * ramp
    return 1.0


def _warmup_ramp(generation: int, cfg: Config) -> float:
    """Linear ramp from 0 to 1 over penalty_warmup_gens; 1.0 thereafter."""
    if cfg.penalty_warmup_gens > 0:
        return min(generation / cfg.penalty_warmup_gens, 1.0)
    return 1.0


def _cycle_ramp(generation: int, cfg: Config) -> float:
    """1.0 during penalty phases; 0.0 during the free window at end of each cycle.
    Always returns 1.0 during the warmup period so the two functions compose cleanly."""
    if cfg.penalty_cycle_gens > 0 and cfg.penalty_cycle_free_gens > 0:
        if generation < cfg.penalty_warmup_gens:
            return 1.0  # warmup handles this range; don't interfere
        pos = (generation - cfg.penalty_warmup_gens) % cfg.penalty_cycle_gens
        if pos >= cfg.penalty_cycle_gens - cfg.penalty_cycle_free_gens:
            return 0.0
    return 1.0


def compute_fitness(
    steps: jnp.ndarray,        # [pop_size] float32 mean steps survived
    c_acts: jnp.ndarray,       # [pop_size] float32 mean activation cost
    raw_food: jnp.ndarray,     # [pop_size] float32 mean cumulative raw food score
    pop_genomes: Genome,
    cfg: Config,
    wcfg: WorldConfig,
    generation: int = 0,
) -> jnp.ndarray:
    """
    Convert raw evaluation metrics to adjusted fitness scores.

    fitness_mode="survival" (default):
        f_raw = mean_steps / episode_steps  → [0, 1]

    fitness_mode="food":
        f_raw = mean_raw_food / (episode_steps * n_food_types)
        Can exceed 1.0 for agents that actively forage near hotspot centres.

    Penalty schedule:
        ramp = _warmup_ramp(gen, cfg) × _cycle_ramp(gen, cfg)

        _warmup_ramp: linearly scales penalties from 0→1 over penalty_warmup_gens,
            then stays at 1.0 — protects early exploration from over-pruning.

        _cycle_ramp: drops to 0.0 during the free window at the end of every
            penalty_cycle_gens block after warmup — periodic loosening that lets
            cold reps escape local optima before pressure resumes.

        The product composes cleanly: during warmup _cycle_ramp=1 so warmup
        controls; after warmup _warmup_ramp=1 so cycling controls.

    Returns fitness [pop_size].
    """
    ramp = _warmup_ramp(generation, cfg) * _cycle_ramp(generation, cfg)
    if ramp != 1.0:
        cfg = dataclasses.replace(
            cfg,
            lambda_edge=cfg.lambda_edge * ramp,
            lambda_dist=cfg.lambda_dist * ramp,
            lambda_act=cfg.lambda_act  * ramp,
            dist_frac=cfg.dist_frac * ramp,
            act_frac=cfg.act_frac   * ramp,
            edge_frac=cfg.edge_frac * ramp,
        )

    if cfg.fitness_mode == "food":
        f_raw = raw_food / float(wcfg.episode_steps * wcfg.n_food_types)
    else:  # "survival" (default)
        f_raw = steps / float(wcfg.episode_steps)
    return jax.vmap(adjusted_fitness, in_axes=(0, 0, 0, None))(
        f_raw, pop_genomes, c_acts, cfg
    )


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select_idx(
    key: jax.Array,
    fitness: jnp.ndarray,
    tournament_size: int,
) -> jnp.ndarray:
    """
    Sample tournament_size candidates (with replacement) and return the
    index of the one with the highest fitness.
    """
    candidate_idxs = jax.random.choice(
        key, fitness.shape[0], shape=(tournament_size,), replace=False
    )
    best_in_tourney = jnp.argmax(fitness[candidate_idxs])
    return candidate_idxs[best_in_tourney]


def select_parents(
    key: jax.Array,
    fitness: jnp.ndarray,
    pop_size: int,
    tournament_size: int,
) -> jnp.ndarray:
    """
    Run pop_size independent tournaments and return winner indices [pop_size].
    """
    keys = jax.random.split(key, pop_size)
    return jax.vmap(tournament_select_idx, in_axes=(0, None, None))(
        keys, fitness, tournament_size
    )


# ── Reproduction ──────────────────────────────────────────────────────────────

def reproduce(
    key: jax.Array,
    pop_genomes: Genome,
    parent_idxs: jnp.ndarray,   # [pop_size] int
    rates: MutationRates,
    cfg: Config,
) -> Genome:
    """
    Build an offspring population by gathering selected parents and mutating each.

    Returns a new batched Genome [pop_size].
    """
    # Gather parents (advanced indexing — safe inside jit/vmap)
    offspring = jax.tree_util.tree_map(lambda x: x[parent_idxs], pop_genomes)

    # Mutate every offspring with an independent key
    mut_keys = jax.random.split(key, cfg.population_size)
    return jax.vmap(mutate, in_axes=(0, 0, None, None))(mut_keys, offspring, cfg, rates)


# ── One generation ────────────────────────────────────────────────────────────

def evolve_step(
    key: jax.Array,
    pop_genomes: Genome,
    fitness: jnp.ndarray,   # [pop_size] — fitness of current population
    rates: MutationRates,
    cfg: Config,
    generation: int = 0,
) -> Genome:
    """
    Produce the next generation via tournament selection + mutation + elitism.

    The best genome from the current population is copied unchanged into
    slot 0 of the offspring (elitism = 1), preventing fitness regression.

    When cfg.mutation_warmup_scale > 1.0, continuous mutation sigmas are
    scaled down from mutation_warmup_scale at gen 0 to 1.0 at
    penalty_warmup_gens — mirroring the penalty ramp to encourage
    exploration when penalty pressure is absent.

    Returns the new (unevaluated) offspring population.
    """
    k_sel, k_mut = jax.random.split(key)

    # Scale continuous sigmas during warmup
    scale = _mutation_scale(generation, cfg)
    if scale != 1.0:
        rates = dataclasses.replace(
            rates,
            weight_sigma=rates.weight_sigma   * scale,
            tau_sigma=rates.tau_sigma         * scale,
            bias_sigma=rates.bias_sigma       * scale,
            position_sigma=rates.position_sigma * scale,
        )

    # Selection + mutation
    parent_idxs = select_parents(k_sel, fitness, cfg.population_size, cfg.tournament_size)
    offspring   = reproduce(k_mut, pop_genomes, parent_idxs, rates, cfg)

    # Elitism: force the current best into slot 0 unchanged
    best_idx  = jnp.argmax(fitness)
    elite     = jax.tree_util.tree_map(lambda x: x[best_idx], pop_genomes)
    offspring = jax.tree_util.tree_map(
        lambda e, o: o.at[0].set(e), elite, offspring
    )

    return offspring


# ── Per-generation statistics ─────────────────────────────────────────────────

def collect_stats(
    generation: int,
    fitness: jnp.ndarray,           # [pop_size]
    steps: jnp.ndarray,             # [pop_size]
    pop_genomes: Genome,
    cfg: Config,
    raw_food: "jnp.ndarray | None" = None,   # [pop_size] — optional
    wcfg: "WorldConfig | None"     = None,   # needed to normalise raw_food
) -> dict:
    """
    Compute summary statistics for the current generation.

    All values are plain Python floats/ints for easy serialisation.
    If raw_food and wcfg are provided, mean_food_score is added to the dict
    (normalised by episode_steps * n_food_types so it is comparable to f_raw
    under fitness_mode="food").
    """
    edge_costs   = jax.vmap(edge_count_cost)(pop_genomes)          # [pop_size]
    wiring_costs = jax.vmap(dist_cost)(pop_genomes)                # [pop_size]
    n_active     = jnp.sum(pop_genomes.active_mask, axis=-1)       # [pop_size]

    stats = {
        "generation":       generation,
        "max_fitness":      float(jnp.max(fitness)),
        "mean_fitness":     float(jnp.mean(fitness)),
        "max_steps":        int(jnp.max(steps)),
        "mean_steps":       float(jnp.mean(steps)),
        "mean_n_active":    float(jnp.mean(n_active.astype(jnp.float32))),
        "mean_edge_cost":   float(jnp.mean(edge_costs)),
        "mean_wiring_cost": float(jnp.mean(wiring_costs)),
    }

    if raw_food is not None and wcfg is not None:
        norm = float(wcfg.episode_steps * wcfg.n_food_types)
        stats["mean_food_score"] = float(jnp.mean(raw_food) / norm)

    return stats


# ── Full evolutionary run ─────────────────────────────────────────────────────

def run_evolution(
    key: jax.Array,
    n_generations: int,
    cfg: Config,
    wcfg: WorldConfig,
    rates: MutationRates,
    n_evals: int = 5,
    callback=None,
    early_stop_fn=None,
    resume_from: "str | Path | None" = None,
    state_checkpoint_dir: "str | Path | None" = None,
    state_checkpoint_every: int = 100,
) -> tuple[Genome, jnp.ndarray, list[dict]]:
    """
    Drive the full evolutionary loop.

    Each generation:
      1. Collect stats on the current evaluated population.
      2. (Optional) call callback(stats, best_genome).
      3. (Optional) call early_stop_fn(stats) — halt if it returns True.
      4. evolve_step  -> new unevaluated offspring.
      5. eval_population + compute_fitness on offspring.
      6. (Optional) save full training state for resume capability.

    Parameters
    ----------
    key            : JAX PRNGKey
    n_generations  : maximum number of generations to run
    cfg            : network / evolution hyperparameters
    wcfg           : world parameters
    rates          : mutation operator intensities
    n_evals        : episodes per fitness estimate (averaged for stability)
    callback       : optional callable(stats, best_genome) fired each generation
    early_stop_fn  : optional callable(stats) -> bool; return True to stop early.
                     The run exits cleanly after the current generation's stats
                     and callback have fired — best_genome and history up to that
                     point are returned as normal.

                     Built-in helpers (importable from ctrnn_lattice_evo.evolution):
                       fitness_threshold(min_fitness)   — stop when max_fitness >= value
                       convergence_stop(window, tol)    — stop when max_fitness hasn't
                                                          improved by tol in last window gens

    resume_from    : path to a state_gen_*.npz written by a previous run.
                     When provided the ``key`` argument is ignored — the saved
                     RNG state is used instead.  The loop resumes from the
                     saved generation number; history returned covers only the
                     newly-completed generations.  Use load_history(run_dir) to
                     read the full history including pre-resume generations.
                     Shortcut: pass latest_state_checkpoint(run_dir) directly.

    state_checkpoint_dir  : directory to write state_gen_*.npz snapshots into
                            (typically run_dir / "checkpoints").  If None,
                            state snapshots are not saved.

    state_checkpoint_every : save a state snapshot every this many generations
                             (default 100).  Only used when state_checkpoint_dir
                             is set.

    Returns
    -------
    best_genome    : single Genome (unbatched) with highest final fitness
    final_fitness  : float32 [pop_size] fitness of the last completed generation
    history        : list of stats dicts, one per completed generation
                     (from start_gen to the last completed generation)
    """
    # ── Initialise or resume ─────────────────────────────────────────────────
    if resume_from is not None:
        pop, fitness, steps, key, start_gen, raw_food = load_training_state(resume_from)
        print(f"  Resuming from generation {start_gen} "
              f"(loaded {Path(resume_from).name})")
    else:
        start_gen = 0
        key, k_init, k_eval = jax.random.split(key, 3)
        pop                   = init_population(k_init, cfg)
        steps, c_acts, raw_food = eval_population(k_eval, pop, cfg, wcfg, n_evals)
        fitness               = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg,
                                                generation=0)

    history: list[dict] = []

    if state_checkpoint_dir is not None:
        state_checkpoint_dir = Path(state_checkpoint_dir)
        state_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Generation loop ───────────────────────────────────────────────────────
    for gen in range(start_gen, n_generations):
        # Stats on current (already-evaluated) population
        stats = collect_stats(gen, fitness, steps, pop, cfg, raw_food=raw_food, wcfg=wcfg)
        history.append(stats)

        if callback is not None:
            best_idx_cb = int(jnp.argmax(fitness))
            best_cb     = jax.tree_util.tree_map(lambda x: x[best_idx_cb], pop)
            callback(stats, best_cb)

        # Early exit — check after callback so the final state is fully logged
        if early_stop_fn is not None and early_stop_fn(stats):
            break

        # Evolve → evaluate
        key, k_step, k_eval = jax.random.split(key, 3)
        pop                     = evolve_step(k_step, pop, fitness, rates, cfg, generation=gen)
        steps, c_acts, raw_food = eval_population(k_eval, pop, cfg, wcfg, n_evals)
        fitness                 = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg,
                                                  generation=gen + 1)

        # State snapshot — labelled with the generation about to be collected
        next_gen = gen + 1
        if (
            state_checkpoint_dir is not None
            and next_gen % state_checkpoint_every == 0
        ):
            snap_path = state_checkpoint_dir / f"state_gen_{next_gen:06d}.npz"
            save_training_state(snap_path, pop, fitness, steps, key, next_gen, raw_food=raw_food)

    # ── Extract best genome (unbatched) ──────────────────────────────────────
    best_idx    = int(jnp.argmax(fitness))
    best_genome = jax.tree_util.tree_map(lambda x: x[best_idx], pop)

    return best_genome, fitness, history


# ── Built-in early-stop helpers ───────────────────────────────────────────────

def fitness_threshold(min_fitness: float):
    """
    Stop when max_fitness reaches or exceeds min_fitness.

    Example — stop once the best genome survives 95% of the episode:
        early_stop_fn=fitness_threshold(0.95)
    """
    def _check(stats: dict) -> bool:
        return stats["max_fitness"] >= min_fitness
    return _check


def convergence_stop(window: int = 50, tol: float = 1e-3):
    """
    Stop when max_fitness has not improved by more than tol over the last
    window generations.  Requires at least window generations to have run.

    Example — stop if fitness plateaus for 100 generations:
        early_stop_fn=convergence_stop(window=100, tol=1e-3)
    """
    recent: list[float] = []

    def _check(stats: dict) -> bool:
        recent.append(stats["max_fitness"])
        if len(recent) < window:
            return False
        improvement = recent[-1] - recent[-window]
        return improvement < tol

    return _check
