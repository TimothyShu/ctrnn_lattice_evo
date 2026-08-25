"""
logger.py — Logging and checkpointing for evolutionary runs.

Public API
----------
make_run_dir(base_dir, run_id) -> Path
    Create a timestamped run directory with a checkpoints/ subdirectory.

save_config(run_dir, cfg, wcfg, rates) -> None
    Serialise all three hyperparameter dataclasses to config.json.

load_config(run_dir) -> (Config, WorldConfig, MutationRates)
    Reconstruct dataclasses from config.json.

save_genome(path, genome) -> None
    Persist all 7 genome fields to a .npz archive.

load_genome(path) -> Genome
    Reconstruct a Genome from a .npz archive.

append_history(run_dir, stats) -> None
    Append one stats dict as a line to history.jsonl (safe for partial runs).

load_history(run_dir) -> list[dict]
    Read all lines from history.jsonl; returns [] if file is absent or empty.

make_logger(run_dir, checkpoint_every, verbose) -> callback
    Return callback(stats, best_genome) that logs, saves, and checkpoints.

save_training_state(path, pop, fitness, steps, key, generation) -> None
    Snapshot the full training state needed to resume an interrupted run.

load_training_state(path) -> (pop, fitness, steps, key, generation)
    Reconstruct training state from a snapshot written by save_training_state.

latest_state_checkpoint(run_dir) -> Path | None
    Return the most recent state_gen_*.npz in the checkpoints/ directory,
    or None if no state snapshots exist.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .config import Config
from .genome import Genome
from .world import WorldConfig
from .mutation import MutationRates


# Genome field order must match the pytree registration in genome.py:
# lambda g: [g.active_mask, g.neuron_type, g.tau, g.bias,
#            g.position, g.weight_matrix, g.edge_mask]
_GENOME_FIELDS = [
    "active_mask",
    "neuron_type",
    "tau",
    "bias",
    "position",
    "weight_matrix",
    "edge_mask",
]

# Prefix used when packing a batched (population) Genome into a state archive.
# Distinguishes population fields from scalar arrays (fitness, steps, rng_key).
_POP_PREFIX = "pop_"


# ── Directory management ──────────────────────────────────────────────────────

def make_run_dir(base_dir: str | Path = "runs", run_id: str | None = None) -> Path:
    """
    Create a fresh run directory under base_dir.

    Name format: run_{YYYYMMDD_HHMMSS_ffffff}_{run_id}
    If run_id is None, a 4-char random hex suffix is generated so that
    two calls in the same second still produce distinct paths.

    Creates base_dir and the checkpoints/ subdirectory if they don't exist.
    Returns the Path to the new run directory.
    """
    base_dir = Path(base_dir)
    if run_id is None:
        run_id = os.urandom(2).hex()
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"run_{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


# ── Config serialisation ──────────────────────────────────────────────────────

def save_config(
    run_dir: Path,
    cfg: Config,
    wcfg: WorldConfig,
    rates: MutationRates,
) -> None:
    """Write all three hyperparameter dataclasses to config.json."""
    data = {
        "config":         dataclasses.asdict(cfg),
        "world_config":   dataclasses.asdict(wcfg),
        "mutation_rates": dataclasses.asdict(rates),
    }
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(data, f, indent=2)


def load_config(run_dir: Path) -> tuple[Config, WorldConfig, MutationRates]:
    """Reconstruct Config, WorldConfig, MutationRates from config.json.

    JSON encodes tuples as lists; we convert any list field values back to
    tuples so that the roundtrip is exact (Config stores tau ranges as tuples).
    """
    def _fix_tuples(d: dict) -> dict:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}

    def _migrate(d: dict) -> dict:
        d = dict(d)
        # lambda_conn was split into lambda_edge + lambda_dist; treat old value as lambda_dist
        if "lambda_conn" in d and "lambda_dist" not in d:
            d["lambda_dist"] = d.pop("lambda_conn")
        # n_in is now derived from n_food_types via __post_init__; remove to avoid conflict
        d.pop("n_in", None)
        # default n_food_types for runs predating multi-food-type support
        d.setdefault("n_food_types", 1)
        # default fitness_mode for runs predating food-score fitness support
        d.setdefault("fitness_mode", "survival")
        # default position_sensors for runs predating proprioceptive sensor support
        d.setdefault("position_sensors", False)
        # default penalty_warmup_gens for runs predating warm-up support
        d.setdefault("penalty_warmup_gens", 0)
        d.setdefault("penalty_cycle_gens", 0)
        d.setdefault("penalty_cycle_free_gens", 0)
        d.setdefault("mutation_warmup_scale", 1.0)
        # defaults for proportional penalty fracs (runs predating this feature)
        d.setdefault("dist_frac",  0.0)
        d.setdefault("act_frac",   0.0)
        d.setdefault("edge_frac",  0.0)
        d.setdefault("C0_wiring",  77.0)
        d.setdefault("C0_act",     1.0)
        d.setdefault("C0_edge",    154.0)
        return d

    def _migrate_wcfg(d: dict) -> dict:
        d = dict(d)
        # default position_sensors for world configs predating proprioceptive sensors
        d.setdefault("position_sensors", False)
        return d

    with open(Path(run_dir) / "config.json") as f:
        data = json.load(f)
    cfg   = Config(**_fix_tuples(_migrate(data["config"])))
    wcfg  = WorldConfig(**_migrate_wcfg(data["world_config"]))
    rates = MutationRates(**data["mutation_rates"])
    return cfg, wcfg, rates


# ── Genome serialisation ──────────────────────────────────────────────────────

def save_genome(path: str | Path, genome: Genome) -> None:
    """
    Save all 7 genome fields to a .npz archive.

    Field order in the archive matches _GENOME_FIELDS, which is the same
    order as the Genome pytree registration so load_genome can reconstruct
    via Genome(*children) without a dict lookup.
    """
    arrays = {field: np.array(getattr(genome, field)) for field in _GENOME_FIELDS}
    np.savez(str(path), **arrays)


def load_genome(path: str | Path) -> Genome:
    """Reconstruct a Genome from a .npz archive saved by save_genome."""
    archive  = np.load(str(path))
    children = [jnp.array(archive[field]) for field in _GENOME_FIELDS]
    return Genome(*children)


# ── History (newline-delimited JSON) ──────────────────────────────────────────

def append_history(run_dir: Path, stats: dict) -> None:
    """
    Append one stats dict as a single JSON line to history.jsonl.

    Uses append mode — never rewrites existing content, safe for partial runs.
    """
    with open(Path(run_dir) / "history.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")


def load_history(run_dir: Path) -> list[dict]:
    """
    Read all lines from history.jsonl and parse each as JSON.

    Returns an empty list if the file is missing or empty.
    """
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── Logger factory ────────────────────────────────────────────────────────────

def make_logger(
    run_dir: Path,
    checkpoint_every: int = 100,
    verbose: bool = True,
) -> Callable[[dict, Genome], None]:
    """
    Return a callback(stats, best_genome) for use with run_evolution.

    Each call:
      1. Appends stats to history.jsonl.
      2. Overwrites best_genome.npz with the current best.
      3. Saves checkpoints/gen_{N:06d}.npz every checkpoint_every generations.
      4. Prints a one-line summary if verbose=True.

    Parameters
    ----------
    run_dir          : directory created by make_run_dir
    checkpoint_every : save a named checkpoint every this many generations
    verbose          : print progress to stdout each generation
    """
    run_dir = Path(run_dir)

    def callback(stats: dict, best_genome: Genome) -> None:
        gen = stats["generation"]

        # 1. Append to history
        append_history(run_dir, stats)

        # 2. Overwrite current best
        save_genome(run_dir / "best_genome.npz", best_genome)

        # 3. Named checkpoint
        if gen % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"gen_{gen:06d}.npz"
            save_genome(ckpt_path, best_genome)

        # 4. Progress line
        if verbose:
            print(
                f"gen {gen:04d} | "
                f"fit {stats['max_fitness']:.4f} "
                f"(mean {stats['mean_fitness']:.4f}) | "
                f"steps {stats['mean_steps']:.1f} | "
                f"nodes {stats['mean_n_active']:.1f}"
            )

    return callback


# ── Full training-state snapshots (resume support) ────────────────────────────

def save_training_state(
    path: str | Path,
    pop: Genome,
    fitness: "jax.Array",
    steps: "jax.Array",
    key: "jax.Array",
    generation: int,
    raw_food: "jax.Array | None" = None,
) -> None:
    """
    Save the complete training state needed to resume an interrupted run.

    Stores:
      • All batched genome fields (prefixed with ``pop_``)
      • fitness  [pop_size] — current adjusted fitness scores
      • steps    [pop_size] — raw mean step counts (needed for collect_stats)
      • raw_food [pop_size] — mean cumulative raw food score (optional; omitted
                             when None so old snapshots remain valid)
      • rng_key  [2]        — current JAX PRNGKey
      • generation          — the generation number that will be collected NEXT
                             (i.e. the generation about to run when you resume)

    Typical filename: ``checkpoints/state_gen_{N:06d}.npz``

    Notes
    -----
    The population stored here has already been *evaluated* (fitness/steps are
    current).  On resume, ``run_evolution`` immediately calls ``collect_stats``
    then continues the generation loop from this point.
    """
    arrays: dict[str, np.ndarray] = {}

    # Batched genome fields
    for field in _GENOME_FIELDS:
        arrays[_POP_PREFIX + field] = np.array(getattr(pop, field))

    # Scalars and vectors
    arrays["fitness"]    = np.array(fitness)
    arrays["steps"]      = np.array(steps)
    arrays["rng_key"]    = np.array(key)
    arrays["generation"] = np.array(generation, dtype=np.int64)
    if raw_food is not None:
        arrays["raw_food"] = np.array(raw_food)

    np.savez(str(path), **arrays)


def load_training_state(
    path: str | Path,
) -> "tuple[Genome, jax.Array, jax.Array, jax.Array, int, jax.Array | None]":
    """
    Reconstruct full training state from a snapshot written by save_training_state.

    Returns
    -------
    pop        : batched Genome [pop_size]
    fitness    : float32 [pop_size]
    steps      : float32 [pop_size]
    key        : JAX PRNGKey [2]
    generation : int — generation number to resume from
    raw_food   : float32 [pop_size] or None (absent in snapshots from older runs)
    """
    archive  = np.load(str(path))
    children = [jnp.array(archive[_POP_PREFIX + field]) for field in _GENOME_FIELDS]
    pop      = Genome(*children)

    fitness    = jnp.array(archive["fitness"])
    steps      = jnp.array(archive["steps"])
    key        = jnp.array(archive["rng_key"])
    generation = int(archive["generation"])
    raw_food   = jnp.array(archive["raw_food"]) if "raw_food" in archive else None

    return pop, fitness, steps, key, generation, raw_food


def latest_state_checkpoint(run_dir: str | Path) -> "Path | None":
    """
    Find the most recent ``state_gen_*.npz`` file in the run's checkpoints dir.

    Returns the Path to the latest file, or None if no state snapshots exist.
    Useful for automatic resume after a crash:

        state_path = latest_state_checkpoint(run_dir)
        if state_path:
            pop, fitness, steps, key, gen = load_training_state(state_path)
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    candidates = sorted(ckpt_dir.glob("state_gen_*.npz"))
    return candidates[-1] if candidates else None
