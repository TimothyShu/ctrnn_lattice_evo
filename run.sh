#!/usr/bin/env bash
# forge-queue entrypoint for ctrnn_lattice_evo.
#
# Contract: forge-queue exports each `params:` entry from the job spec as
# FORGE_PARAM_<KEY>, activates the venv, sets CUDA_VISIBLE_DEVICES, and calls
# this script.  Everything below maps those env vars onto CLI flags, so adding
# a parameter to a job spec needs no change here.
#
# ── Parameter name transform ────────────────────────────────────────────────
# The exact transform run_spec.sh applies is not documented in the forge-queue
# README, and YAML keys like `n-replicates` are not legal POSIX variable names,
# so it must be rewriting them somehow.  Both plausible forms are handled:
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
# The echoed exec line below is the point of the smoke test: if the transform
# is something neither branch handles, the flags simply will not appear and the
# run proceeds on defaults, which looks exactly like a null result.
#
# ── Interpreter ─────────────────────────────────────────────────────────────
# Never exec bare `python`.  Ubuntu ships python3 with no unversioned alias,
# so `python` exists only while a venv is active — and whether run_spec.sh
# activates the venv or merely sets PATH is not documented.  Resolving
# explicitly means a missing interpreter fails with a clear message here rather
# than as `exec: python: not found` from the last line of the script, hours
# after the job was submitted.

set -euo pipefail
cd "$(dirname "$0")"

# XLA otherwise preallocates most of the card and probes downward on failure,
# filling the log with OOM lines that are not errors.  Disabling preallocation
# lets it grow on demand instead.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# ── Map FORGE_PARAM_* onto CLI flags ────────────────────────────────────────

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

# ── Resolve the interpreter ─────────────────────────────────────────────────
# FORGE_VENV is a guess at what run_spec.sh might export; if it uses a
# different name the branch simply never fires and the PATH lookup below
# applies, which is the behaviour without this block at all.  Confirm with:
#   grep -nE "VENV|activate|PATH=" ~/forge-queue/backend/lib/run_spec.sh

PY="${FORGE_VENV:-}"
if [[ -n "$PY" && -x "$PY/bin/python" ]]; then
  PY="$PY/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "[run.sh] no python on PATH and no venv active" >&2
  exit 1
fi

# ── Report and exec ─────────────────────────────────────────────────────────

echo "[run.sh] pwd=$(pwd)"
echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run.sh] python=$PY"
echo "[run.sh] $((${#ARGS[@]} / 2)) params from job spec"
echo "[run.sh] exec: $PY scripts/run_experiment.py ${ARGS[*]}"

exec "$PY" scripts/run_experiment.py "${ARGS[@]}"