#!/usr/bin/env bash
# forge-queue entrypoint for ctrnn_lattice_evo.
#
# Contract: forge-queue exports each `params:` entry from the job spec as
# FORGE_PARAM_<KEY>, activates the venv, sets CUDA_VISIBLE_DEVICES, and calls
# this script.  Everything below maps those env vars onto CLI flags, so adding
# a parameter to a job spec needs no change here.
#
# The exact name transform run_spec.sh applies is not documented in the
# forge-queue README, and YAML keys like `n-replicates` are not legal POSIX
# variable names, so it must be rewriting them somehow.  This loop handles both
# plausible forms:
#
#   FORGE_PARAM_N_REPLICATES=10   (uppercased, - -> _)  -> --n-replicates 10
#   FORGE_PARAM_n-replicates=10   (verbatim)            -> --n-replicates 10
#
# Reading the environment with `env -0` rather than bash variable expansion is
# deliberate: a verbatim name containing a hyphen is not a valid bash
# identifier and would be invisible to ${!FORGE_PARAM_*} expansion, but is
# still present in the process environment.
#
# Booleans are passed as values, not bare flags, because run_experiment.py
# parses them with a _bool type — so `false` in a spec actually disables the
# option instead of being silently dropped.
#
# The echoed command line below is the point of the smoke test: if the
# transform is something neither branch handles, the flags simply will not
# appear and the run proceeds on defaults, which looks exactly like a null
# result.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

set -euo pipefail
cd "$(dirname "$0")"

ARGS=()
while IFS= read -r -d '' entry; do
  name="${entry%%=*}"
  value="${entry#*=}"
  [[ "$name" == FORGE_PARAM_* ]] || continue

  flag="${name#FORGE_PARAM_}"
  flag="$(printf '%s' "$flag" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"

  [[ -z "$flag" ]] && continue
  ARGS+=("--$flag" "$value")
done < <(env -0)

echo "[run.sh] pwd=$(pwd)"
echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run.sh] python=$(command -v python)"
echo "[run.sh] $((${#ARGS[@]} / 2)) params from job spec"
echo "[run.sh] exec: python scripts/run_experiment.py ${ARGS[*]}"

exec python scripts/run_experiment.py "${ARGS[@]}"