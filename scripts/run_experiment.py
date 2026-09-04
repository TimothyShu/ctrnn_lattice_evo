#!/usr/bin/env python3
"""
run_experiment.py — entry point for one forge-queue job.

A job is one arm at one penalty setting, run for --n-replicates independent
seeds.  Each replicate gets its own subdirectory with config.json,
history.jsonl, checkpoints and best_genome.npz; the job as a whole also emits
the three artifacts the forge-queue dashboard reads:

    series.csv     per-generation traces, aggregated across replicates.
                   contract/state.py parses this with csv.reader and the
                   template plots column 0 as x — so it must be CSV, not
                   JSON-lines, and every cell must parseFloat.
    metrics.json   final scalars, shown as a key/value table and used by
                   project_table for cross-job comparison.
    summary.png    picked up by the gallery (any image under the run dir is).

Output goes to $FORGE_RUN_DIR when set.  run_spec.sh creates that directory on
NVMe, exports it, and archive.sh moves it to /mnt/archive/<project>/ afterwards
— anything written elsewhere is never archived and never appears in the
dashboard.

Every knob the experiment varies is exposed as a flag, because run.sh maps
FORGE_PARAM_* straight onto flags: anything not here cannot be set from a job
spec and would silently take its default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np

from ctrnn_lattice_evo import Config, WorldConfig
from ctrnn_lattice_evo.mutation import MutationRates
from ctrnn_lattice_evo.evolution import run_evolution
from ctrnn_lattice_evo.logger import make_run_dir, save_config, make_logger, load_history

# Per-generation traces to export.  First entry is the x axis.
SERIES_KEYS = [
    "max_fitness",
    "mean_fitness",
    "mean_n_edges",
    "mean_local_fraction",
    "mean_n_active",
    "mean_steps",
]


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # ── Arm ──────────────────────────────────────────────────────────────────
    p.add_argument("--init-mode", default="grid",
                   choices=["grid", "uniform", "sparse"])
    p.add_argument("--n-max", type=int, default=64)
    p.add_argument("--grid-w", type=int, default=8, help="lattice ROWS")
    p.add_argument("--grid-h", type=int, default=None, help="lattice COLUMNS")
    p.add_argument("--grid-r", type=int, default=2)
    p.add_argument("--sparse-n-active", type=int, default=None)
    p.add_argument("--init-edge-density", type=float, default=0.15)
    p.add_argument("--node-ops-enabled", type=_bool, default=False)

    # ── Penalties ────────────────────────────────────────────────────────────
    p.add_argument("--edge-frac", type=float, default=0.0)
    p.add_argument("--dist-frac", type=float, default=0.0)
    p.add_argument("--act-frac", type=float, default=0.0)
    p.add_argument("--c0-edge", type=float, default=None)
    p.add_argument("--c0-dist", type=float, default=None)
    p.add_argument("--add-kernel-lambda", type=float, default=0.0,
                   help="e-folding reach (lattice units) of the add_edges "
                        "proposal kernel; 0 = uniform addition (disabled)")

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
    p.add_argument("--edge-churn", type=float, default=0.003,
                   help="one rate drives BOTH edge operators, so add and "
                        "remove cancel in expectation at any density")
    p.add_argument("--add-node-prob", type=float, default=0.05)
    p.add_argument("--remove-node-prob", type=float, default=0.05)

    # ── World ────────────────────────────────────────────────────────────────
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--n-food-types", type=int, default=1)
    p.add_argument("--position-sensors", type=_bool, default=False)

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None,
                   help="defaults to $FORGE_RUN_DIR, else ./runs")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--state-checkpoint-every", type=int, default=100)
    p.add_argument("--verbose", type=_bool, default=True)

    return p.parse_args(argv)


def build_configs(a: argparse.Namespace):
    cfg = Config(
        N_max=a.n_max, grid_W=a.grid_w, grid_H=a.grid_h, grid_r=a.grid_r,
        init_mode=a.init_mode,
        sparse_n_active=a.sparse_n_active,
        init_edge_density=a.init_edge_density,
        node_ops_enabled=a.node_ops_enabled,
        edge_frac=a.edge_frac, dist_frac=a.dist_frac, act_frac=a.act_frac,
        add_kernel_lambda=a.add_kernel_lambda,
        C0_edge=a.c0_edge, C0_dist=a.c0_dist,
        penalty_warmup_gens=a.penalty_warmup_gens,
        mutation_warmup_scale=a.mutation_warmup_scale,
        penalty_cycle_gens=a.penalty_cycle_gens,
        penalty_cycle_free_gens=a.penalty_cycle_free_gens,
        population_size=a.pop_size, tournament_size=a.tournament_size,
        fitness_mode=a.fitness_mode,
        n_food_types=a.n_food_types, position_sensors=a.position_sensors,
    )
    wcfg = WorldConfig(
        episode_steps=a.episode_steps,
        n_food_types=a.n_food_types,
        position_sensors=a.position_sensors,
    )
    rates = MutationRates(
        weight_sigma=a.weight_sigma, tau_sigma=a.tau_sigma,
        bias_sigma=a.bias_sigma, type_flip_prob=a.type_flip_prob,
        edge_churn=a.edge_churn,
        add_node_prob=a.add_node_prob, remove_node_prob=a.remove_node_prob,
    )
    return cfg, wcfg, rates


# ── Dashboard artifacts ───────────────────────────────────────────────────────

def _stack_histories(histories: list[list[dict]]) -> dict[str, np.ndarray]:
    """[n_reps, n_gens] per metric, NaN-padded.

    Replicates can differ in length if one stopped early, so pad rather than
    truncate — a short replicate should shorten the band, not drag the mean
    toward zero.
    """
    n_gens = max(len(h) for h in histories)
    out = {}
    for key in SERIES_KEYS:
        arr = np.full((len(histories), n_gens), np.nan)
        for i, hist in enumerate(histories):
            for j, row in enumerate(hist):
                if key in row:
                    arr[i, j] = row[key]
        out[key] = arr
    return out


def write_series_csv(path: Path, stacked: dict[str, np.ndarray]) -> None:
    """Mean across replicates, one column per metric.

    contract/state.py reads this with csv.reader and the template plots
    column 0 as x, so: generation first, numeric cells only, no NaN literals
    (the template's parseFloat would yield NaN and break the trace — empty
    string is what it treats as a gap).
    """
    n_gens = next(iter(stacked.values())).shape[1]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation"] + SERIES_KEYS)
        for g in range(n_gens):
            row = [g]
            for key in SERIES_KEYS:
                v = np.nanmean(stacked[key][:, g])
                row.append("" if np.isnan(v) else f"{v:.6g}")
            w.writerow(row)


def write_metrics_json(path: Path, stacked: dict[str, np.ndarray],
                       cfg: Config, a: argparse.Namespace) -> None:
    """Final scalars for the dashboard's metrics table and project_table.

    Only int/float values are picked up as comparable metrics by
    project_table, so keep the numbers flat and put strings last.

    fit_delta_tail is the one that stops a gap being misread: a difference
    between two plateaued arms is a real capacity finding, whereas a
    difference between one converged and one still climbing just means the run
    was too short.
    """
    fit = np.nanmean(stacked["max_fitness"], axis=0)
    n_gens = len(fit)
    tail = min(100, max(1, n_gens // 2))

    def final(key: str) -> float:
        return float(np.nanmean(stacked[key], axis=0)[-1])

    metrics = {
        "n_replicates":        int(stacked["max_fitness"].shape[0]),
        "n_generations":       int(n_gens),
        "fit_final":           float(fit[-1]),
        "fit_best":            float(np.nanmax(fit)),
        "fit_delta_tail":      float(fit[-1] - fit[-tail]),
        "fit_final_sd":        float(np.nanstd(stacked["max_fitness"][:, -1])),
        "final_n_edges":       final("mean_n_edges"),
        "final_local_frac":    final("mean_local_fraction"),
        "final_n_active":      final("mean_n_active"),
        "final_mean_steps":    final("mean_steps"),
        "pruned_fraction":     1.0 - final("mean_n_edges") / float(cfg.C0_edge),
        "C0_edge":             float(cfg.C0_edge),
        "C0_dist":             float(cfg.C0_dist),
        "edge_frac":           float(cfg.edge_frac),
        "dist_frac":           float(cfg.dist_frac),
        "act_frac":            float(cfg.act_frac),
        "add_kernel_lambda":   float(cfg.add_kernel_lambda),
        "grid_r":              int(cfg.grid_r),
        "init_mode":           cfg.init_mode,
    }
    path.write_text(json.dumps(metrics, indent=2))


def write_plot(path: Path, stacked: dict[str, np.ndarray], title: str) -> None:
    """Four-panel PNG; the gallery picks up any image under the run dir.

    Optional — matplotlib is not a hard dependency, and a job should not fail
    at the very end for want of a plot after hours of compute.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[run_experiment] matplotlib not available, skipping plot")
        return

    panels = [
        ("max_fitness",         "max fitness"),
        ("mean_n_edges",        "mean edge count"),
        ("mean_local_fraction", "mean local fraction"),
        ("mean_n_active",       "mean active nodes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    gens = np.arange(next(iter(stacked.values())).shape[1])

    for ax, (key, label) in zip(axes.ravel(), panels):
        arr = stacked[key]
        mean = np.nanmean(arr, axis=0)
        ax.plot(gens, mean)
        if arr.shape[0] > 1:
            sd = np.nanstd(arr, axis=0)
            ax.fill_between(gens, mean - sd, mean + sd, alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[1, 0].set_xlabel("generation")
    axes[1, 1].set_xlabel("generation")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    a = parse_args(argv)
    cfg, wcfg, rates = build_configs(a)

    # run_spec.sh creates this on NVMe and archive.sh moves it afterwards.
    # Anything written outside it never reaches /mnt/archive and never appears
    # in the dashboard.
    out_root = Path(a.output_dir or os.environ.get("FORGE_RUN_DIR") or "runs")
    out_root.mkdir(parents=True, exist_ok=True)

    job_name = os.environ.get("FORGE_JOB_NAME", cfg.init_mode)

    print(f"[run_experiment] jax devices: {jax.devices()}", flush=True)
    print(f"[run_experiment] output -> {out_root}", flush=True)
    print(f"[run_experiment] arm={cfg.init_mode} "
          f"lattice={cfg.grid_W}x{cfg.grid_H} r={cfg.grid_r} "
          f"C0_edge={cfg.C0_edge:.0f} C0_dist={cfg.C0_dist:.0f}", flush=True)
    print(f"[run_experiment] edge_frac={cfg.edge_frac} "
          f"dist_frac={cfg.dist_frac} act_frac={cfg.act_frac} "
          f"edge_churn={rates.edge_churn}", flush=True)

    if not any(d.platform == "gpu" for d in jax.devices()):
        print("[run_experiment] WARNING: no GPU device visible", flush=True)

    histories: list[list[dict]] = []

    for rep in range(a.n_replicates):
        seed = a.seed + rep
        run_dir = make_run_dir(out_root, run_id=f"{cfg.init_mode}_rep{rep:02d}")
        save_config(run_dir, cfg, wcfg, rates)

        print(f"\n[run_experiment] replicate {rep + 1}/{a.n_replicates} "
              f"seed={seed} -> {run_dir.name}", flush=True)

        callback = make_logger(run_dir,
                               checkpoint_every=a.checkpoint_every,
                               verbose=a.verbose,
                               cfg=cfg)

        _best, _fitness, history = run_evolution(
            jax.random.PRNGKey(seed),
            a.n_generations,
            cfg, wcfg, rates,
            n_evals=a.n_evals,
            callback=callback,
            state_checkpoint_dir=run_dir / "checkpoints",
            state_checkpoint_every=a.state_checkpoint_every,
        )
        histories.append(history)

        last = history[-1]
        print(f"[run_experiment] rep {rep} done — "
              f"fit={last['max_fitness']:.4f} "
              f"edges={last['mean_n_edges']:.0f} "
              f"local={last['mean_local_fraction']:.3f} "
              f"nodes={last['mean_n_active']:.1f}", flush=True)

    # ── Job-level artifacts ──────────────────────────────────────────────────
    stacked = _stack_histories(histories)
    write_series_csv(out_root / "series.csv", stacked)
    write_metrics_json(out_root / "metrics.json", stacked, cfg, a)
    write_plot(out_root / "summary.png", stacked,
               f"{job_name} — {cfg.init_mode}, edge_frac={cfg.edge_frac}")

    m = json.loads((out_root / "metrics.json").read_text())
    print(f"\n[run_experiment] {a.n_replicates} reps — "
          f"fit={m['fit_final']:.4f}±{m['fit_final_sd']:.4f} "
          f"(Δtail {m['fit_delta_tail']:+.4f}) "
          f"edges={m['final_n_edges']:.0f} "
          f"pruned={m['pruned_fraction']:.1%} "
          f"local={m['final_local_frac']:.3f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())