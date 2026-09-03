#!/usr/bin/env bash
# Run the whole planner suite on val14: every planner, reactive and not,
# with and without occlusion, then print the table.
#
# Four runs per planner, and the cell that matters is the gap between
# baseline and occluded within each reactivity -- not the absolute scores,
# which carry a systematic offset from the published numbers (see docs/PLAN.md).
#
# Runs are serialised on purpose. The bottleneck is CPU, not GPU: one run
# already saturates a good share of 192 cores, so running several at once
# makes all of them slower without finishing sooner, and it was how a control
# experiment got launched onto an occupied card and died as CUDA OOM.
#
#   bash scripts/server/sweep_val14.sh                    # everything
#   PLANNERS="idm pdm_closed" bash scripts/server/sweep_val14.sh
#   REACTIVITIES=nonreactive bash scripts/server/sweep_val14.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$HOME/occlusion-bench}
LOGS=${LOGS:-$HOME/sweep_logs}
mkdir -p "$LOGS"

# Rule-based first: they are quick, need no GPU, and a failure there means the
# harness is wrong rather than the planner.
PLANNERS=${PLANNERS:-"idm pdm_closed pdm_hybrid pdm_open urban_driver gc_pgp plancnn diffusion flow"}
REACTIVITIES=${REACTIVITIES:-"nonreactive reactive"}
MODES=${MODES:-"baseline occluded"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "planners     : $PLANNERS"
echo "reactivities : $REACTIVITIES"
echo "modes        : $MODES"
echo "logs         : $LOGS"
echo

total=0; failed=0
for planner in $PLANNERS; do
  for react in $REACTIVITIES; do
    for mode in $MODES; do
      tag="${planner}_${react}_${mode}"
      log="$LOGS/$tag.log"
      # Skip anything already scored, so an interrupted sweep resumes instead
      # of repeating hours of work.
      if grep -q "Finished running simulation" "$log" 2>/dev/null; then
        echo "skip  $tag  (already done)"; continue
      fi
      total=$((total + 1))
      printf 'run   %-40s ' "$tag"
      start=$SECONDS
      # SEED only matters for the two sampling planners; the rest ignore it.
      PLANNER=$planner REACTIVITY=$react SEED=${SEED:-0} CFG_WEIGHT=${CFG_WEIGHT:-1.8} \
        VAL_SPLIT=${VAL_SPLIT:-$WORK/val14_local} SPLIT=${SPLIT:-val14} \
        bash "$HERE/scripts/server/run_val14.sh" "$mode" > "$log" 2>&1
      if grep -q "Finished running simulation" "$log"; then
        printf 'ok   %5d min\n' $(( (SECONDS - start) / 60 ))
      else
        failed=$((failed + 1))
        printf 'FAILED -- %s\n' "$(grep -m1 -oE '[A-Za-z.]*(Error|Exception)[^\"]{0,60}' "$log" | head -1)"
      fi
    done
  done
done

echo
echo "$total runs attempted, $failed failed"
echo
"${PY:-$HOME/miniforge3/envs/flow_planner/bin/python}" "$HERE/scripts/server/table.py" "$WORK/exp"
