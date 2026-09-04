"""
evolution.py — Evolutionary loop for CTRNN neuroevolution on a lattice.

Public API
----------
init_population(key, cfg)
    Population of cfg.population_size genomes, built by the constructor for
    cfg.init_mode ("grid" / "uniform" / "sparse").

eval_population(key, pop, cfg, wcfg, n_evals=5)
    -> (mean_steps[P], mean_c_act[P], mean_raw_food[P])

compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg, generation=0)
    Normalise to f_raw, apply the ramped proportional penalties.

tournament_select_idx / select_parents / reproduce / evolve_step
    One generation: select -> mutate -> elitism.

collect_stats(generation, fitness, steps, pop, cfg, ...)
    Per-generation statistics, including the edge count and local fraction
    traces the pruning and locality claims are read from.

run_evolution(...)
    -> (best_genome, final_fitness, history)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome, constructor_for
from .mutation import MutationRates, mutate
from .cost import edge_count_cost, dist_cost, adjusted_fitness
from .topology import dist_matrix, local_mask, distance_kernel
from .world import WorldConfig
from .brain import run_brain_episode_full
from .logger import save_training_state, load_training_state


# ── Population initialisation ─────────────────────────────────────────────────

def init_population(key: jax.Array, cfg: Config) -> Genome:
    """Initialise cfg.population_size genomes for this run's arm.

    The arm is chosen by looking up the constructor in Python, BEFORE the
    vmap: cfg.init_mode is a string and a string cannot be a traced argument.

    Note that in the "grid" arm every individual gets the SAME edge_mask, so
    generation-0 diversity is parametric only (weights, tau, bias, type).
    That is a deliberate departure from ctrnn_evo, where each individual
    carried its own random topology, and it makes structural mutation the sole
    source of topological diversity thereafter.
    """
    ctor = constructor_for(cfg)
    keys = jax.random.split(key, cfg.population_size)
    return jax.vmap(ctor, in_axes=(0, None))(keys, cfg)


# ── Fitness evaluation ────────────────────────────────────────────────────────

def _eval_genome(keys: jax.Array, genome: Genome, cfg: Config, wcfg: WorldConfig):
    """Evaluate one genome over n_evals independent episodes.

    Returns (mean_steps, mean_c_act, mean_raw_food) across episodes.
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
):
    """Evaluate the whole population, n_evals episodes each.

    Nested vmap over (population x episodes).  Every genome-episode pair gets
    an independent key, so the fitness estimate averages out world randomness
    rather than sharing it across the population.

    Returns
    -------
    mean_steps    : float32 [P] — steps survived
    mean_c_acts   : float32 [P] — activation cost, in [0, 1]
    mean_raw_food : float32 [P] — cumulative uncapped food score, >= 0
    """
    flat_keys = jax.random.split(key, cfg.population_size * n_evals)
    epoch_keys = flat_keys.reshape(cfg.population_size, n_evals, -1)

    return jax.vmap(_eval_genome, in_axes=(0, 0, None, None))(
        epoch_keys, pop_genomes, cfg, wcfg
    )


# ── Penalty schedule ──────────────────────────────────────────────────────────

def _mutation_scale(generation: int, cfg: Config) -> float:
    """Scale for continuous mutation sigmas during warm-up.

    Mirrors the penalty ramp in reverse: mutation_warmup_scale at gen 0,
    decaying linearly to 1.0 at penalty_warmup_gens, 1.0 thereafter.  Wide
    exploration while the penalty is still near zero.
    """
    if cfg.mutation_warmup_scale > 1.0 and cfg.penalty_warmup_gens > 0:
        ramp = min(generation / cfg.penalty_warmup_gens, 1.0)
        return cfg.mutation_warmup_scale * (1.0 - ramp) + 1.0 * ramp
    return 1.0


def _warmup_ramp(generation: int, cfg: Config) -> float:
    """Linear 0 -> 1 over penalty_warmup_gens; 1.0 thereafter.

    This is half of the over-pruning fix: networks are not pruned before any
    foraging strategy has had a chance to evolve.
    """
    if cfg.penalty_warmup_gens > 0:
        return min(generation / cfg.penalty_warmup_gens, 1.0)
    return 1.0


def _cycle_ramp(generation: int, cfg: Config) -> float:
    """1.0 during penalty phases, 0.0 during the free window of each cycle.

    Returns 1.0 throughout the warm-up range so the two ramps compose by
    multiplication: warm-up controls early, cycling controls later.
    """
    if cfg.penalty_cycle_gens > 0 and cfg.penalty_cycle_free_gens > 0:
        if generation < cfg.penalty_warmup_gens:
            return 1.0
        pos = (generation - cfg.penalty_warmup_gens) % cfg.penalty_cycle_gens
        if pos >= cfg.penalty_cycle_gens - cfg.penalty_cycle_free_gens:
            return 0.0
    return 1.0


def compute_fitness(
    steps: jnp.ndarray,      # [P] mean steps survived
    c_acts: jnp.ndarray,     # [P] mean activation cost
    raw_food: jnp.ndarray,   # [P] mean cumulative raw food
    pop_genomes: Genome,
    cfg: Config,
    wcfg: WorldConfig,
    generation: int = 0,
) -> jnp.ndarray:
    """Convert raw evaluation metrics to adjusted fitness.

    fitness_mode="survival": f_raw = mean_steps / episode_steps, in [0, 1]
    fitness_mode="food":     f_raw = mean_raw_food / (episode_steps * n_food_types)
                             which may exceed 1.0 but is never negative.

    Both are non-negative, which the multiplicative penalty depends on: with
    f_raw < 0 a penalty multiplier below 1 would IMPROVE the score.

    The penalty is scaled by _warmup_ramp * _cycle_ramp.  Only the three
    proportional fracs are ramped — ctrnn_evo's absolute lambda_* mode is gone.

    Returns fitness [P], each >= 0 (adjusted_fitness clamps the multiplier).
    """
    ramp = _warmup_ramp(generation, cfg) * _cycle_ramp(generation, cfg)
    if ramp != 1.0:
        cfg = dataclasses.replace(
            cfg,
            edge_frac=cfg.edge_frac * ramp,
            dist_frac=cfg.dist_frac * ramp,
            act_frac=cfg.act_frac * ramp,
        )

    if cfg.fitness_mode == "food":
        f_raw = raw_food / float(wcfg.episode_steps * wcfg.n_food_types)
    else:
        f_raw = steps / float(wcfg.episode_steps)

    # Built once per generation, not per genome — geometry is shared.
    dist = dist_matrix(cfg.grid_W, cfg.grid_H)

    return jax.vmap(adjusted_fitness, in_axes=(0, 0, 0, None, None))(
        f_raw, pop_genomes, c_acts, cfg, dist
    )


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select_idx(
    key: jax.Array,
    fitness: jnp.ndarray,
    tournament_size: int,
) -> jnp.ndarray:
    """Sample tournament_size candidates and return the index of the best.

    Selection acts on adjusted fitness, which is clamped at 0.  Under a heavy
    penalty many individuals tie at exactly 0 and argmax breaks ties by lowest
    index — a real loss of gradient, but strictly better than the unclamped
    alternative, where a negative multiplier would make the BEST network score
    most negative and invert selection entirely.
    """
    candidate_idxs = jax.random.choice(
        key, fitness.shape[0], shape=(tournament_size,), replace=False
    )
    return candidate_idxs[jnp.argmax(fitness[candidate_idxs])]


def select_parents(
    key: jax.Array,
    fitness: jnp.ndarray,
    pop_size: int,
    tournament_size: int,
) -> jnp.ndarray:
    """pop_size independent tournaments; returns winner indices [pop_size]."""
    keys = jax.random.split(key, pop_size)
    return jax.vmap(tournament_select_idx, in_axes=(0, None, None))(
        keys, fitness, tournament_size
    )


# ── Reproduction ──────────────────────────────────────────────────────────────

def reproduce(
    key: jax.Array,
    pop_genomes: Genome,
    parent_idxs: jnp.ndarray,
    rates: MutationRates,
    cfg: Config,
    log_kernel: jnp.ndarray | None = None,
) -> Genome:
    """Gather the selected parents and mutate each into an offspring."""
    offspring = jax.tree_util.tree_map(lambda x: x[parent_idxs], pop_genomes)
    mut_keys = jax.random.split(key, cfg.population_size)
    return jax.vmap(mutate, in_axes=(0, 0, None, None, None))(
        mut_keys, offspring, cfg, rates, log_kernel)


# ── One generation ────────────────────────────────────────────────────────────

def evolve_step(
    key: jax.Array,
    pop_genomes: Genome,
    fitness: jnp.ndarray,
    rates: MutationRates,
    cfg: Config,
    generation: int = 0,
) -> Genome:
    """One generation: tournament selection, mutation, elitism.

    The best genome is copied unchanged into slot 0.  Because the copy is a
    whole-pytree assignment it carries edge_mask and active_mask as well as
    the weights — a well-pruned elite must not be silently rewired each
    generation.

    The addition kernel (Config.add_kernel_lambda) depends only on the
    lattice, never on a genome, so it is built once here — like dist/local
    are for collect_stats — rather than inside the population vmap in
    reproduce, which would recompute it once per genome instead of once per
    generation. 0 and inf both mean uniform addition and take the cheap
    log_kernel=None path through add_edges (see distance_kernel).

    Returns the new, unevaluated offspring population.
    """
    k_sel, k_mut = jax.random.split(key)

    scale = _mutation_scale(generation, cfg)
    if scale != 1.0:
        rates = dataclasses.replace(
            rates,
            weight_sigma=rates.weight_sigma * scale,
            tau_sigma=rates.tau_sigma * scale,
            bias_sigma=rates.bias_sigma * scale,
        )

    log_kernel = None
    if 0.0 < cfg.add_kernel_lambda < float("inf"):
        kernel = distance_kernel(cfg.grid_W, cfg.add_kernel_lambda, cfg.grid_H)
        log_kernel = jnp.where(kernel > 0, jnp.log(kernel), -jnp.inf)

    parent_idxs = select_parents(k_sel, fitness, cfg.population_size, cfg.tournament_size)
    offspring = reproduce(k_mut, pop_genomes, parent_idxs, rates, cfg, log_kernel=log_kernel)

    best_idx = jnp.argmax(fitness)
    elite = jax.tree_util.tree_map(lambda x: x[best_idx], pop_genomes)
    return jax.tree_util.tree_map(lambda e, o: o.at[0].set(e), elite, offspring)


# ── Per-generation statistics ─────────────────────────────────────────────────

def collect_stats(
    generation: int,
    fitness: jnp.ndarray,
    steps: jnp.ndarray,
    pop_genomes: Genome,
    cfg: Config,
    raw_food: "jnp.ndarray | None" = None,
    wcfg: "WorldConfig | None" = None,
) -> dict:
    """Summary statistics for the current generation, as plain Python scalars.

    Two of these are the experiment's primary traces:

    mean_n_edges — "when do edges die" is the actual over-pruning measurement,
        and it has to be a per-generation trace rather than an end-of-run
        number.  If it flattens well above the target band, raise
        remove_edge_p_per_edge rather than edge_frac: one sets how fast edges
        CAN die, the other how much dying is worth.

    mean_local_fraction — how much of the lattice prior survives.  The grid arm
        starts at 1.0; a uniform random digraph at the same density sits near
        n_edges/(N^2-N), about 0.27 at 8x8 r=2.  That is the floor to read
        against, not zero.

    Everything here is vectorised over the population; the expensive networkx
    metrics live in analysis.py and run once at the end.
    """
    dist = dist_matrix(cfg.grid_W, cfg.grid_H)
    m = local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H)

    edge_costs = jax.vmap(edge_count_cost)(pop_genomes)                # [P]
    wiring_costs = jax.vmap(dist_cost, in_axes=(0, None))(pop_genomes, dist)
    n_active = jnp.sum(pop_genomes.active_mask, axis=-1)               # [P]

    active_pairs = (
        pop_genomes.active_mask[:, :, None] & pop_genomes.active_mask[:, None, :]
    )
    active_edges = pop_genomes.edge_mask & active_pairs                # [P, N, N]
    n_edges = jnp.sum(active_edges, axis=(1, 2))                       # [P]
    n_local = jnp.sum(active_edges & m[None, :, :], axis=(1, 2))       # [P]
    local_frac = n_local / jnp.maximum(n_edges, 1)

    stats = {
        "generation":          generation,
        "max_fitness":         float(jnp.max(fitness)),
        "mean_fitness":        float(jnp.mean(fitness)),
        "max_steps":           int(jnp.max(steps)),
        "mean_steps":          float(jnp.mean(steps)),
        "mean_n_active":       float(jnp.mean(n_active.astype(jnp.float32))),
        "mean_n_edges":        float(jnp.mean(n_edges.astype(jnp.float32))),
        "mean_edge_cost":      float(jnp.mean(edge_costs)),
        "mean_wiring_cost":    float(jnp.mean(wiring_costs)),
        "mean_local_fraction": float(jnp.mean(local_frac)),
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
    """Drive the full evolutionary loop.

    Each generation: collect stats on the evaluated population, fire the
    callback, check early stop, evolve, evaluate, optionally snapshot.

    Parameters
    ----------
    n_evals       : episodes per fitness estimate, averaged for stability
    callback      : callable(stats, best_genome), fired once per generation
    early_stop_fn : callable(stats) -> bool.  Checked AFTER the callback so the
                    final generation is fully logged before exiting.  See
                    fitness_threshold and convergence_stop below.
    resume_from   : a state_gen_*.npz from a previous run.  `key` is IGNORED
                    when set — the saved RNG state is restored instead, which
                    is what makes a resumed run reproduce the sequence an
                    uninterrupted one would have followed.  The returned
                    history covers only the newly completed generations; use
                    load_history(run_dir) for the full record.
    state_checkpoint_dir / _every : where and how often to snapshot.

    Returns (best_genome unbatched, final_fitness [P], history).
    """
    if resume_from is not None:
        pop, fitness, steps, key, start_gen, raw_food = load_training_state(resume_from)
        print(f"  Resuming from generation {start_gen} "
              f"(loaded {Path(resume_from).name})")
    else:
        start_gen = 0
        key, k_init, k_eval = jax.random.split(key, 3)
        pop = init_population(k_init, cfg)
        steps, c_acts, raw_food = eval_population(k_eval, pop, cfg, wcfg, n_evals)
        fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg, generation=0)

    history: list[dict] = []

    if state_checkpoint_dir is not None:
        state_checkpoint_dir = Path(state_checkpoint_dir)
        state_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for gen in range(start_gen, n_generations):
        stats = collect_stats(gen, fitness, steps, pop, cfg, raw_food=raw_food, wcfg=wcfg)
        history.append(stats)

        if callback is not None:
            best_idx_cb = int(jnp.argmax(fitness))
            best_cb = jax.tree_util.tree_map(lambda x: x[best_idx_cb], pop)
            callback(stats, best_cb)

        if early_stop_fn is not None and early_stop_fn(stats):
            break

        key, k_step, k_eval = jax.random.split(key, 3)
        pop = evolve_step(k_step, pop, fitness, rates, cfg, generation=gen)
        steps, c_acts, raw_food = eval_population(k_eval, pop, cfg, wcfg, n_evals)
        fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg,
                                  generation=gen + 1)

        next_gen = gen + 1
        if state_checkpoint_dir is not None and next_gen % state_checkpoint_every == 0:
            snap_path = state_checkpoint_dir / f"state_gen_{next_gen:06d}.npz"
            save_training_state(snap_path, pop, fitness, steps, key, next_gen,
                                raw_food=raw_food, cfg=cfg)

    best_idx = int(jnp.argmax(fitness))
    best_genome = jax.tree_util.tree_map(lambda x: x[best_idx], pop)

    return best_genome, fitness, history


# ── Built-in early-stop helpers ───────────────────────────────────────────────

def fitness_threshold(min_fitness: float):
    """Stop when max_fitness reaches min_fitness.

    Note this tests ADJUSTED fitness, which under a live penalty is strictly
    below the raw survival fraction — a threshold of 0.95 may be unreachable
    at edge_frac=0.2 even for a perfect forager.
    """
    def _check(stats: dict) -> bool:
        return stats["max_fitness"] >= min_fitness
    return _check


def convergence_stop(window: int = 50, tol: float = 1e-3):
    """Stop when max_fitness has not improved by more than tol over `window`
    generations.  Requires at least `window` generations to have run.

    Careful with the cyclic penalty schedule: a free window drops the penalty
    to zero and lifts adjusted fitness for reasons unrelated to progress, so a
    convergence check spanning a cycle boundary can read as improvement.  Set
    window shorter than penalty_cycle_free_gens, or leave cycling off when
    using this.
    """
    recent: list[float] = []

    def _check(stats: dict) -> bool:
        recent.append(stats["max_fitness"])
        if len(recent) < window:
            return False
        return recent[-1] - recent[-window] < tol

    return _check