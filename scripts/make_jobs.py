#!/usr/bin/env python3
"""
make_jobs.py — emit forge-queue job specs for the lattice experiment.

Writes one YAML per (arm, edge_frac) cell into --out, default
~/queue/pending/.  Nothing here talks to the queue; it just drops files where
the watcher picks them up.

    python scripts/make_jobs.py --stage smoke     1 job,  minutes
    python scripts/make_jobs.py --stage gate      2 jobs, ~10h
    python scripts/make_jobs.py --stage pilot     4 jobs, ~10h
    python scripts/make_jobs.py --stage sweep    18 jobs, ~30h serial

Stages, in the order they should be run:

  smoke  One tiny job.  Confirms the FORGE_PARAM_* -> flag mapping actually
         works.  If the transform differs from what run.sh assumes, the run
         silently uses defaults, which is indistinguishable from a null
         result — so check the echoed command line before anything else.

  gate   grid vs sparse at edge_frac=0, full length.  If the lattice cannot
         match the sparse arm with NO cost pressure, the topology is the
         bottleneck and the pruning experiment is unrunnable at this size.
         Raise grid_r or N_max and repeat rather than queueing the sweep.

  pilot  {grid, uniform} x {0, 0.2}.  Checks the final edge count lands
         anywhere near the anticipated band before committing 30 hours.

  sweep  All three arms across the frac range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# edge_frac under the clamped proportional penalty is the fraction of fitness
# surrendered at reference cost, so it lives in [0, 1].  ctrnn_evo's 0.001-scale
# lambda values do not carry over.
SWEEP_FRACS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]

STAGES = {
    "smoke": [("grid", 0.0)],
    "gate":  [("grid", 0.0), ("sparse", 0.0)],
    "pilot": [("grid", 0.0), ("grid", 0.2),
              ("uniform", 0.0), ("uniform", 0.2)],
    "sweep": [(arm, f) for arm in ("grid", "uniform", "sparse")
              for f in SWEEP_FRACS],
}

SPEC = """\
project: ctrnn_lattice_evo
entrypoint: run.sh
venv: ~/jax-env
gpu: auto
notify: {notify}
notes: "{notes}"
params:
  init-mode: {arm}
  n-max: {n_max}
  grid-w: {grid_w}
  grid-r: {grid_r}
  edge-frac: {frac}
  dist-frac: 0.0
  act-frac: 0.0
  add-kernel-lambda: {add_kernel_lambda}
  penalty-warmup-gens: {warmup}
  edge-churn: {churn}
  n-generations: {gens}
  n-replicates: {reps}
  pop-size: {pop}
  n-evals: {n_evals}
  episode-steps: {episode_steps}
  seed: {seed}
{extra}"""


def make_spec(arm: str, frac: float, idx: int, a) -> tuple[str, str]:
    name = f"{arm}_ef{frac:g}".replace(".", "p")

    # Node operators are the sparse arm's whole identity — "start small and
    # grow".  Config rejects them on the fixed-lattice arms.
    extra = "  node-ops-enabled: true\n" if arm == "sparse" else ""

    return name, SPEC.format(
        notify="true",
        notes=f"{a.stage}: {arm} arm, edge_frac={frac}",
        arm=arm,
        n_max=a.n_max,
        grid_w=a.grid_w,
        grid_r=a.grid_r,
        frac=frac,
        add_kernel_lambda=a.add_kernel_lambda,
        warmup=a.penalty_warmup_gens,
        churn=a.edge_churn,
        gens=a.n_generations,
        reps=a.n_replicates,
        pop=a.pop_size,
        n_evals=a.n_evals,
        episode_steps=a.episode_steps,
        seed=a.seed + 1000 * idx,   # disjoint seed blocks per cell
        extra=extra,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=list(STAGES), default="pilot")
    p.add_argument("--out", type=Path, default=Path.home() / "queue" / "pending")
    p.add_argument("--n-max", type=int, default=64)
    p.add_argument("--grid-w", type=int, default=8)
    p.add_argument("--grid-r", type=int, default=2)
    p.add_argument("--n-generations", type=int, default=500)
    p.add_argument("--n-replicates", type=int, default=10)
    p.add_argument("--pop-size", type=int, default=1000)
    p.add_argument("--n-evals", type=int, default=5)
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--penalty-warmup-gens", type=int, default=100)
    p.add_argument("--edge-churn", type=float, default=0.003)
    p.add_argument("--add-kernel-lambda", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    # A smoke job must finish in minutes, so shrink it regardless of flags.
    if a.stage == "smoke":
        a.n_generations = 3
        a.n_replicates = 1
        a.pop_size = 20
        a.n_evals = 1
        a.episode_steps = 200
        a.penalty_warmup_gens = 0

    cells = STAGES[a.stage]
    specs = [make_spec(arm, frac, i, a) for i, (arm, frac) in enumerate(cells)]

    if a.dry_run:
        for name, text in specs:
            print(f"--- {name}.yaml ---\n{text}")
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, text in specs:
            path = a.out / f"{a.stage}_{name}.yaml"
            path.write_text(text)
            print(f"wrote {path}")

    n = len(specs)
    print(f"\n{a.stage}: {n} job(s).  One GPU means these run strictly "
          f"serially.")
    if a.stage != "smoke":
        print(f"At {a.n_replicates} reps x {a.n_generations} gens each, "
              f"budget roughly {2 * n}-{3 * n}h total.")


if __name__ == "__main__":
    main()