#!/usr/bin/env python3
"""
run_experiment.py — entry point for one forge-queue job.

A job is one arm at one penalty setting, run for --n-replicates independent
seeds.  Each replicate gets its own run directory under --output-dir with its
own config.json, history.jsonl, checkpoints and summary.json, so replicates can
be compared without re-deriving what settings produced them.

Every knob the experiment varies is exposed as a flag, because run.sh maps
FORGE_PARAM_* straight onto flags — anything not here cannot be set from a job
spec, and would silently take its default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

from ctrnn_lattice_evo import Config, WorldConfig
from ctrnn_lattice_evo.mutation import MutationRates
from ctrnn_lattice_evo.evolution import run_evolution
from ctrnn_lattice_evo.analysis import summarise_run
from ctrnn_lattice_evo.logger import make_run_dir, save_config, make_logger


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # ── Arm ──────────────────────────────────────────────────────────────────
    p.add_argument("--init-mode", default="grid",
                   choices=["grid", "uniform", "sparse"],
                   help="which experimental arm this job runs")
    p.add_argument("--n-max", type=int, default=64)
    p.add_argument("--grid-w", type=int, default=8, help="lattice ROWS")
    p.add_argument("--grid-h", type=int, default=None,
                   help="lattice COLUMNS (defaults to --grid-w)")
    p.add_argument("--grid-r", type=int, default=2,
                   help="Chebyshev radius; dist_frac is collinear with "
                        "edge_frac at r=1 and only independent at r>=2")
    p.add_argument("--sparse-n-active", type=int, default=None)
    p.add_argument("--init-edge-density", type=float, default=0.15)
    p.add_argument("--node-ops-enabled", type=_bool, default=False,
                   help="sparse arm only; Config rejects it elsewhere")

    # ── Penalties ────────────────────────────────────────────────────────────
    p.add_argument("--edge-frac", type=float, default=0.0)
    p.add_argument("--dist-frac", type=float, default=0.0)
    p.add_argument("--act-frac", type=float, default=0.0)
    p.add_argument("--c0-edge", type=float, default=None,
                   help="override; leave unset to derive from the lattice")
    p.add_argument("--c0-dist", type=float, default=None)

    # ── Penalty schedule ─────────────────────────────────────────────────────
    p.add_argument("--penalty-warmup-gens", type=int, default=0)
    p.add_argument("--mutation-warmup-scale", type=float, default=1.0)
    p.add_argument("--penalty-cycle-gens", type=int, default=0)
    p.add_argument("--penalty-cycle-free-gens", type=int, default=0)

    # ── Evolution ────────────────────────────────────────────────────────────
    p.add_argument("--n-generations", type=int, default=500)
    p.add_argument("--n-replicates", type=int, default=10)
    p.add_argument("--pop-size", type=int, default=1000)
    p.add_argument("--tournament-size", type=int, default=4)
    p.add_argument("--n-evals", type=int, default=5)
    p.add_argument("--fitness-mode", default="survival",
                   choices=["survival", "food"])

    # ── Mutation ─────────────────────────────────────────────────────────────
    p.add_argument("--weight-sigma", type=float, default=0.1)
    p.add_argument("--tau-sigma", type=float, default=0.1)
    p.add_argument("--bias-sigma", type=float, default=0.1)
    p.add_argument("--type-flip-prob", type=float, default=0.05)
    p.add_argument("--add-edge-prob", type=float, default=0.1)
    p.add_argument("--remove-edge-p-per-edge", type=float, default=0.003,
                   help="per-edge Bernoulli; sets how fast edges CAN die, "
                        "which is a different lever from edge-frac")
    p.add_argument("--add-node-prob", type=float, default=0.05)
    p.add_argument("--remove-node-prob", type=float, default=0.05)

    # ── World ────────────────────────────────────────────────────────────────
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--n-food-types", type=int, default=1)
    p.add_argument("--position-sensors", type=_bool, default=False)

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs",
                   help="MUST be distinct per job or the sweep collides")
    p.add_argument("--run-id", default=None)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--state-checkpoint-every", type=int, default=100)
    p.add_argument("--verbose", type=_bool, default=True)

    return p.parse_args(argv)


def build_configs(a: argparse.Namespace):
    cfg = Config(
        N_max=a.n_max,
        grid_W=a.grid_w,
        grid_H=a.grid_h,
        grid_r=a.grid_r,
        init_mode=a.init_mode,
        sparse_n_active=a.sparse_n_active,
        init_edge_density=a.init_edge_density,
        node_ops_enabled=a.node_ops_enabled,
        edge_frac=a.edge_frac,
        dist_frac=a.dist_frac,
        act_frac=a.act_frac,
        C0_edge=a.c0_edge,
        C0_dist=a.c0_dist,
        penalty_warmup_gens=a.penalty_warmup_gens,
        mutation_warmup_scale=a.mutation_warmup_scale,
        penalty_cycle_gens=a.penalty_cycle_gens,
        penalty_cycle_free_gens=a.penalty_cycle_free_gens,
        population_size=a.pop_size,
        tournament_size=a.tournament_size,
        fitness_mode=a.fitness_mode,
        n_food_types=a.n_food_types,
        position_sensors=a.position_sensors,
    )
    wcfg = WorldConfig(
        episode_steps=a.episode_steps,
        n_food_types=a.n_food_types,
        position_sensors=a.position_sensors,
    )
    rates = MutationRates(
        weight_sigma=a.weight_sigma,
        tau_sigma=a.tau_sigma,
        bias_sigma=a.bias_sigma,
        type_flip_prob=a.type_flip_prob,
        add_edge_prob=a.add_edge_prob,
        remove_edge_p_per_edge=a.remove_edge_p_per_edge,
        add_node_prob=a.add_node_prob,
        remove_node_prob=a.remove_node_prob,
    )
    return cfg, wcfg, rates


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg, wcfg, rates = build_configs(a)

    print(f"[run_experiment] jax devices: {jax.devices()}", flush=True)
    print(f"[run_experiment] arm={cfg.init_mode} "
          f"lattice={cfg.grid_W}x{cfg.grid_H} r={cfg.grid_r} "
          f"C0_edge={cfg.C0_edge:.0f} C0_dist={cfg.C0_dist:.0f}", flush=True)
    print(f"[run_experiment] edge_frac={cfg.edge_frac} "
          f"dist_frac={cfg.dist_frac} act_frac={cfg.act_frac} "
          f"remove_edge_p={rates.remove_edge_p_per_edge}", flush=True)

    # A CPU-only JAX build will run, just unusably slowly — worth seeing in the
    # log rather than discovering from the wall-clock time.
    if not any(d.platform == "gpu" for d in jax.devices()):
        print("[run_experiment] WARNING: no GPU device visible", flush=True)

    base = Path(a.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    for rep in range(a.n_replicates):
        seed = a.seed + rep
        run_id = f"{a.run_id or cfg.init_mode}_rep{rep:02d}"
        run_dir = make_run_dir(base, run_id=run_id)
        save_config(run_dir, cfg, wcfg, rates)

        print(f"\n[run_experiment] replicate {rep + 1}/{a.n_replicates} "
              f"seed={seed} -> {run_dir}", flush=True)

        callback = make_logger(run_dir,
                               checkpoint_every=a.checkpoint_every,
                               verbose=a.verbose)

        best, final_fitness, history = run_evolution(
            jax.random.PRNGKey(seed),
            a.n_generations,
            cfg, wcfg, rates,
            n_evals=a.n_evals,
            callback=callback,
            state_checkpoint_dir=run_dir / "checkpoints",
            state_checkpoint_every=a.state_checkpoint_every,
        )

        # summarise_run re-derives the population metrics with networkx, which
        # is slow; it runs once per replicate, never per generation.
        summary = summarise_run(history, _final_pop_placeholder(best), best, cfg)
        summary["seed"] = seed
        summary["init_mode"] = cfg.init_mode
        summary["edge_frac"] = cfg.edge_frac
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        last = history[-1]
        print(f"[run_experiment] rep {rep} done — "
              f"fit={last['max_fitness']:.4f} "
              f"edges={last['mean_n_edges']:.0f} "
              f"local={last['mean_local_fraction']:.3f} "
              f"nodes={last['mean_n_active']:.1f}", flush=True)

    return 0


def _final_pop_placeholder(best):
    """summarise_run wants a batched population; give it the best genome as a
    population of one.

    The alternative is returning the whole final population from
    run_evolution, which would mean holding [pop_size, N, N] in memory through
    the networkx loop for metrics we only report as means.  If per-individual
    final structure ever matters, change run_evolution to return the
    population and pass it here instead.
    """
    import jax.numpy as jnp
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), best)


if __name__ == "__main__":
    sys.exit(main())