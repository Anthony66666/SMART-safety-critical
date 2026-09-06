#!/usr/bin/env bash
# Run the server's evaluation on this laptop, against the 4090.
#
# A thin wrapper rather than a second script. run_val14.sh already takes every
# path from the environment; duplicating it here is how table.py came to know
# about two reactivities while run_val14.sh knew about three, and the runs that
# fell through the gap did so silently. So this sets the environment and hands
# over.
#
# What is genuinely different here is the size of the machine: 32 cores against
# the server's 192, and a 24 GB card against 46 GB. nuPlan is CPU-bound -- the
# map queries, not the models -- so the core count is what sets the wall time,
# and test14-hard's 272 scenarios against val14's 1118 is what makes it fit.
#
#   bash scripts/run_local.sh baseline
#   REACTIVITY=smart bash scripts/run_local.sh occluded
#   LIMIT=4 bash scripts/run_local.sh baseline          # smoke, sequential
#   PLANNER=pdm_closed REACTIVITY=smart bash scripts/run_local.sh baseline
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)

export CONDA=${CONDA:-$HOME/anaconda3}
export ENV_NAME=${ENV_NAME:-flow_planner}
export BENCH=${BENCH:-$HERE}
export DEVKIT=${DEVKIT:-$HOME/nuplan-devkit}
# Results land under $WORK/exp, which is what score.py and table.py read.
export WORK=${WORK:-$HOME/occlusion-bench-local}
export SPLIT=${SPLIT:-test14-hard}

# The test split, unzipped from nuplan-v1.1_test.zip. VAL_SPLIT is the variable
# run_val14.sh uses for whichever split is being run -- the name is historical.
export VAL_SPLIT=${VAL_SPLIT:-/mnt/e/nuplan-test/splits/test}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps}

# 32 cores, and ray wants to leave some for the driver and the OS. Going wider
# than the machine makes every worker slower without finishing sooner.
export THREADS=${THREADS:-24}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# The card is half the server's. The smart budget is computed from free memory
# at launch, so it adapts on its own -- but the per-worker estimate it divides
# by was measured before the rollout was truncated to what replanning consumes,
# and is now pessimistic. Left overridable rather than guessed at: run the
# smoke test, read the peak, and set it.
export SMART_WORKER_MIB=${SMART_WORKER_MIB:-3000}

mkdir -p "$WORK"
[ -e "$WORK/nuplan-devkit" ] || ln -s "$DEVKIT" "$WORK/nuplan-devkit" 2>/dev/null || true

echo "local run: split=$SPLIT reactivity=${REACTIVITY:-nonreactive} mode=${1:-baseline}"
echo "  data   : $VAL_SPLIT"
echo "  maps   : $NUPLAN_MAPS_ROOT"
echo "  results: $WORK/exp"
echo

exec bash "$HERE/scripts/server/run_val14.sh" "$@"
