#!/usr/bin/env bash
# Run the whole planner suite on val14 and print the table: every planner,
# every traffic model, with and without occlusion.
#
# The cell that matters is the gap between baseline and occluded within one
# reactivity -- not the absolute scores, which carry a systematic offset from
# the published numbers (see docs/PLAN.md).
#
# Three traffic models, and they are three different experiments rather than
# three settings of one. nonreactive replays the log, reactive drives the
# background cars with IDM, and smart drives them with a learned model. The
# first two are nuPlan's own challenges; the third is what this benchmark adds,
# and it is the one where occlusion and traffic behaviour can interact.
#
# Runs are serialised on purpose. The bottleneck is CPU, not GPU: one run
# already saturates a good share of 192 cores, so running several at once makes
# all of them slower without finishing sooner, and it was how a control
# experiment got launched onto an occupied card and died as CUDA OOM.
#
#   bash scripts/server/sweep_val14.sh                    # everything available
#   PLANNERS="idm pdm_closed" bash scripts/server/sweep_val14.sh
#   REACTIVITIES=smart bash scripts/server/sweep_val14.sh
#   CHECK_ONLY=1 bash scripts/server/sweep_val14.sh       # preflight, run nothing
set -uo pipefail

HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$HOME/occlusion-bench}
LOGS=${LOGS:-$HOME/sweep_logs}
mkdir -p "$LOGS"

# Rule-based first: they are quick, need no GPU, and a failure there means the
# harness is wrong rather than the planner.
PLANNERS=${PLANNERS:-"idm pdm_closed pdm_hybrid pdm_open urban_driver gc_pgp plancnn diffusion flow dtpp carl"}
REACTIVITIES=${REACTIVITIES:-"nonreactive reactive smart"}
MODES=${MODES:-"baseline occluded"}
# Card 3 by default: the emptiest on this box, and card 0 is usually taken.
# run_val14.sh still refuses to start below 8 GB free, so a bad choice fails
# loudly rather than dying as CUDA OOM inside the ray workers -- which once
# cost 1089 of 1118 simulations and still produced a score, 78.89 over the 29
# survivors, that read exactly like a result.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

# "Finished running simulation" only means the runner's loop ended. It is
# printed just as happily when every simulation inside it threw, which is how a
# sweep once reported ok for a run where 911 of 1118 scenarios died of CUDA OOM
# and scored the 207 survivors as though that were the result. A run counts as
# done only if it finished *and* nearly everything in it succeeded.
MAX_FAILURE_RATE=${MAX_FAILURE_RATE:-2}   # percent

run_succeeded() {
  local log=$1
  grep -q "Finished running simulation" "$log" 2>/dev/null || return 1
  local ok bad
  ok=$(grep -oE 'Number of successful simulations: [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  bad=$(grep -oE 'Number of failed simulations: [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  [ -z "$ok" ] && return 1
  bad=${bad:-0}
  [ $(( (bad * 100) / (ok + bad) )) -le "$MAX_FAILURE_RATE" ]
}

failure_note() {
  local log=$1 ok bad
  ok=$(grep -oE 'Number of successful simulations: [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  bad=$(grep -oE 'Number of failed simulations: [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  [ -n "$bad" ] && [ "${bad:-0}" -gt 0 ] && echo "${bad} of $(( ok + bad )) scenarios failed"
}

common_env() {
  # SEED only matters for the two sampling planners; the rest ignore it.
  echo "PLANNER=$1 REACTIVITY=$2 SEED=${SEED:-0} CFG_WEIGHT=${CFG_WEIGHT:-1.8}" \
       "VAL_SPLIT=${VAL_SPLIT:-$WORK/val14_local} SPLIT=${SPLIT:-val14}"
}

# Preflight. Every combination is resolved and its files checked before
# anything runs, so a missing checkpoint surfaces now rather than after the
# hours of queue ahead of it. Planners that are not installed on this machine
# are reported once and skipped, instead of failing eleven times over.
# Counters rather than array lengths: under `set -u` bash 4 treats an empty
# associative array as unbound, and ${#a[@]:-0} is not a legal substitution,
# so the obvious guard is itself the bug.
echo "preflight"
n_runnable=0
n_blocked=0
declare -A why_not
for planner in $PLANNERS; do
  for react in $REACTIVITIES; do
    for mode in $MODES; do
      tag="${planner}_${react}_${mode}"
      if run_succeeded "$LOGS/$tag.log"; then
        continue        # already scored; reported in the run loop below
      fi
      reason=$(env $(common_env "$planner" "$react") DRY_RUN=1 \
                 bash "$HERE/scripts/server/run_val14.sh" "$mode" 2>&1 >/dev/null)
      if [ -z "$reason" ]; then
        n_runnable=$((n_runnable + 1))
      else
        n_blocked=$((n_blocked + 1))
        why_not["$tag"]=$(echo "$reason" | tr '\n' ';' | sed 's/;$//')
      fi
    done
  done
done

if [ "$n_blocked" -gt 0 ]; then
  echo "  not runnable here:"
  for tag in $(printf '%s\n' "${!why_not[@]}" | sort); do
    printf '    %-38s %s\n' "$tag" "${why_not[$tag]}"
  done
fi
echo "  $n_runnable runnable, $n_blocked blocked"
echo

if [ -n "${CHECK_ONLY:-}" ]; then
  exit 0
fi

total=0; failed=0
for planner in $PLANNERS; do
  for react in $REACTIVITIES; do
    for mode in $MODES; do
      tag="${planner}_${react}_${mode}"
      log="$LOGS/$tag.log"
      # Skip anything already scored, so an interrupted sweep resumes instead
      # of repeating hours of work.
      if run_succeeded "$log"; then
        echo "skip  $tag  (already done)"; continue
      fi
      # Blocked combinations were reported in the preflight; do not spend a
      # startup on them just to fail.
      [ -n "${why_not[$tag]:-}" ] && continue
      total=$((total + 1))
      printf 'run   %-40s ' "$tag"
      start=$SECONDS
      env $(common_env "$planner" "$react") \
        bash "$HERE/scripts/server/run_val14.sh" "$mode" > "$log" 2>&1
      if run_succeeded "$log"; then
        printf 'ok   %5d min\n' $(( (SECONDS - start) / 60 ))
      elif note=$(failure_note "$log"); [ -n "$note" ]; then
        failed=$((failed + 1))
        printf 'FAILED -- %s\n' "$note"
      else
        failed=$((failed + 1))
        # Report the failure that stopped the run, not the first line in the
        # log that contains the word Error. Ray announces that its telemetry
        # exporter could not start on almost every run -- harmless, it says so
        # itself -- and taking the first match reported that instead of the
        # CUDA OOM further down, which sent a diagnosis off in the wrong
        # direction for a while.
        reason=$(grep -A 30 'Simulation failed: with the following trace' "$log" 2>/dev/null \
                   | grep -oE '[A-Za-z_.]*(Error|Exception): [^"]{0,70}' | head -1)
        [ -z "$reason" ] && reason=$(grep -oE '[A-Za-z_.]*(Error|Exception): [^"]{0,70}' "$log" \
                                       | grep -v 'metrics agent' | head -1)
        printf 'FAILED -- %s\n' "${reason:-see $log}"
      fi
    done
  done
done

echo
echo "$total runs attempted, $failed failed"
echo
"${PY:-$HOME/miniforge3/envs/flow_planner/bin/python}" "$HERE/scripts/server/table.py" "$WORK/exp"
