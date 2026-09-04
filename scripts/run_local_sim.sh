#!/usr/bin/env bash
# One nuPlan closed-loop run on the local mini split, through the official
# runner rather than the hand-written loop in run_benchmark.py.
#
# This exists to answer "does this configuration actually work end to end",
# not to produce comparable scores. A handful of mini scenarios is far too few
# for that, and the scores it prints should not be quoted -- see CLAUDE.md.
# What it does check is that the observation builds, the simulation steps, and
# the official metrics come out the other side.
#
#   OBS=smart_agents_observation bash scripts/run_local_sim.sh
#   PLANNER=idm_planner LIMIT=2 OBS=occluded_smart_agents_observation \
#       bash scripts/run_local_sim.sh
set -euo pipefail

PY=${PY:-$HOME/anaconda3/envs/flow_planner/bin/python}
DEVKIT=${DEVKIT:-$HOME/nuplan-devkit}
BENCH=${BENCH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA=${DATA:-/mnt/e/nuplan-mini/nuplan-v1.1_mini/data/cache/mini}
MAPS=${MAPS:-/mnt/e/nuplan-mini/nuplan-maps-v1.0/maps}

PLANNER=${PLANNER:-idm_planner}
OBS=${OBS:-}
LIMIT=${LIMIT:-2}
CHALLENGE=${CHALLENGE:-closed_loop_nonreactive_agents}
OUT=${OUT:-$BENCH/exp_local}

# SMART's compiled extensions are CPU-only in this environment, and the model
# is the bottleneck at roughly seven minutes per scenario. Ray's overhead is
# pure cost at this scale, so run in-process.
OBS_ARG=""
[ -n "$OBS" ] && OBS_ARG="observation=$OBS"

echo "planner=$PLANNER  observation=${OBS:-<default>}  scenarios=$LIMIT"

NUPLAN_DATA_ROOT="$DATA" NUPLAN_MAPS_ROOT="$MAPS" NUPLAN_EXP_ROOT="$OUT" \
PYTHONPATH="$BENCH:${PYTHONPATH:-}" \
"$PY" "$DEVKIT/nuplan/planning/script/run_simulation.py" \
    +simulation="$CHALLENGE" \
    planner="$PLANNER" \
    $OBS_ARG \
    ~callback.simulation_log_callback \
    scenario_builder=nuplan_mini \
    scenario_builder.data_root="$DATA" \
    scenario_builder.map_root="$MAPS" \
    scenario_filter=all_scenarios \
    scenario_filter.scenario_types="[traversing_intersection]" \
    scenario_filter.num_scenarios_per_type=null \
    scenario_filter.limit_total_scenarios="$LIMIT" \
    scenario_filter.timestamp_threshold_s=20 \
    experiment_uid="local/${PLANNER}/${OBS:-baseline}/$(date +%Y-%m-%d-%H-%M-%S)" \
    worker=sequential \
    number_of_gpus_allocated_per_simulation=0 \
    enable_simulation_progress_bar=true \
    verbose=false \
    hydra.searchpath="[file://$BENCH/configs/nuplan, \
pkg://nuplan.planning.script.config.common, \
pkg://nuplan.planning.script.experiments]"
