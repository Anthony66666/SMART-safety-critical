#!/usr/bin/env bash
# Full val14, twice: as nuPlan ships it, and through the occlusion wrapper.
#
# The baseline run exists to be checked against the published number before
# anything is claimed about occlusion. Flow Planner reports 90.43 on Val14
# non-reactive; if this run does not land near that, the harness is wrong and
# the occluded number means nothing. Run the baseline first and look at it.
#
#   LIMIT=8 bash scripts/server/run_val14.sh baseline   # smoke test first
#   bash scripts/server/run_val14.sh baseline
#   bash scripts/server/run_val14.sh occluded
#
# Run the smoke test before committing to the full split. Eight scenarios take
# minutes and catch a broken path, a missing map or a planner that will not
# load; 1118 take hours to tell you the same thing.
#
# 1118 scenarios, 15 s each, both conditions. Everything except the observation
# is nuPlan's own: two_stage_controller, the official closed-loop metrics and
# the official weighted-average aggregator that produces the headline score.
set -u

MODE=${1:-baseline}
CONDA=${CONDA:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-flow_planner}
WORK=${WORK:-$HOME/occlusion-bench}
BENCH=${BENCH:-$HOME/SMART-safety-critical}   # this repository, for the wrapper

PY="$CONDA/envs/$ENV_NAME/bin/python"
DEVKIT="$WORK/nuplan-devkit"

export NUPLAN_DEVKIT_ROOT="$DEVKIT"
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/hqlab/dataset_nas3/nuplan/raw}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/hqlab/dataset_nas3/nuplan/raw/maps}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$WORK/exp}
export PYTHONPATH="$BENCH:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# Three of the four cards were busy when this was written. Check before
# running and set this to whatever is actually free.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

# The simulation is CPU-bound -- measured at roughly 70% CPU against 25% GPU --
# so the GPU fraction is what caps parallelism, not the model. 0.1 allows ten
# concurrent simulations per card; raise the threads, not the fraction, if the
# box has cores to spare.
GPU_FRACTION=${GPU_FRACTION:-0.1}
THREADS=${THREADS:-48}

# LIMIT caps the scenario count, for a smoke test. Unset means the whole split.
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="scenario_filter.limit_total_scenarios=$LIMIT"
    echo "SMOKE TEST: $LIMIT scenarios only"
fi

if [ "$MODE" = "occluded" ]; then
    OBSERVATION="observation=occluded_box_observation"
    TAG=occluded
else
    OBSERVATION=""
    TAG=baseline
fi

mkdir -p "$NUPLAN_EXP_ROOT"

$PY "$DEVKIT/nuplan/planning/script/run_simulation.py" \
    +simulation=closed_loop_nonreactive_agents \
    planner=flow_planner \
    planner.flow_planner.config_path="$WORK/checkpoints/model_config_resolved.yaml" \
    planner.flow_planner.ckpt_path="$WORK/checkpoints/model.pth" \
    planner.flow_planner.enable_ema=false \
    $OBSERVATION \
    scenario_builder=nuplan \
    scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val" \
    scenario_filter=val14 \
    $LIMIT_ARG \
    experiment_uid="flow_planner/val14/$TAG/$(date +%Y-%m-%d-%H-%M-%S)" \
    worker=ray_distributed \
    worker.threads_per_node=$THREADS \
    distributed_mode=SINGLE_NODE \
    number_of_gpus_allocated_per_simulation=$GPU_FRACTION \
    enable_simulation_progress_bar=true \
    verbose=false \
    hydra.searchpath="[file://$BENCH/configs/nuplan, \
pkg://flow_planner.nuplan_simulation.scenario_filter, \
pkg://flow_planner.nuplan_simulation, \
pkg://nuplan.planning.script.config.common, \
pkg://nuplan.planning.script.experiments]"
