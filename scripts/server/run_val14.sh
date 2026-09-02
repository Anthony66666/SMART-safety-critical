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

# These are probed rather than taken from the environment. A shell that has
# already exported NUPLAN_DATA_ROOT -- pointing at the versioned directory
# rather than its parent, say -- silently doubles the path, and the failure
# arrives as a devkit ValueError about a load path that does not exist.
find_dir() {
    for candidate in "$@"; do
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

VAL_SPLIT=$(find_dir \
    "${VAL_SPLIT:-}" \
    /hqlab/dataset_nas3/nuplan/raw/nuplan-v1.1/splits/val \
    "${NUPLAN_DATA_ROOT:-}/splits/val" \
    "${NUPLAN_DATA_ROOT:-}/nuplan-v1.1/splits/val") || {
    echo "cannot find the val split; set VAL_SPLIT to the directory of .db files" >&2
    exit 1
}

MAPS=$(find_dir \
    "${NUPLAN_MAPS_ROOT:-}" \
    /hqlab/dataset_nas3/nuplan/raw/maps) || {
    echo "cannot find the maps; set NUPLAN_MAPS_ROOT" >&2
    exit 1
}

export NUPLAN_DEVKIT_ROOT="$DEVKIT"
export NUPLAN_MAPS_ROOT="$MAPS"
# Forced, not defaulted: results have to land where score.py looks for them,
# and an inherited NUPLAN_EXP_ROOT sends them somewhere else entirely.
export NUPLAN_EXP_ROOT="$WORK/exp"
export PYTHONPATH="$BENCH:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

echo "val split : $VAL_SPLIT  ($(ls "$VAL_SPLIT"/*.db 2>/dev/null | wc -l) db files)"
echo "maps      : $MAPS"
echo "results   : $NUPLAN_EXP_ROOT"

# Three of the four cards were busy when this was written. Check before
# running and set this to whatever is actually free.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

# This is CPU-bound, and by a wide margin. Profiling one step: 296 ms goes into
# observation_adapter -- nuPlan's GPKG map queries and the agent-history
# assembly, all CPU -- against a 54 MB model doing four ODE steps, which is
# noise on an L40S. Measured overall at roughly 70% CPU against 25% GPU.
#
# So the GPU fraction is a purely artificial cap: at 0.1 it allows ten
# simulations per card while the box has 192 cores sitting idle. A 300-step
# scenario takes about 2.5 minutes, so 1118 of them is ~47 hours serial; ten
# ways makes that 4.7 hours, thirty ways about 1.5. Each worker holds a CUDA
# context of a few hundred MB, so thirty-odd fit in 46 GB with room to spare.
GPU_FRACTION=${GPU_FRACTION:-0.03}
THREADS=${THREADS:-64}

# LIMIT caps the scenario count, for a smoke test. Unset means the whole split.
LIMIT_ARG=""
WORKER=${WORKER:-ray_distributed}
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="scenario_filter.limit_total_scenarios=$LIMIT"
    # Ray swallows worker tracebacks: a smoke test that fails under it shows up
    # as a progress bar that never moves, which says nothing about why. Run the
    # small case in-process so the error is the error. Set WORKER to override.
    WORKER=${WORKER_OVERRIDE:-sequential}
    echo "SMOKE TEST: $LIMIT scenarios, worker=$WORKER"
fi

# threads_per_node is a ray option; passing it to the sequential worker fails.
THREADS_ARG=""
[ "$WORKER" = "ray_distributed" ] && THREADS_ARG="worker.threads_per_node=$THREADS"

# SPLIT selects the scenario filter, and with it the data and the builder.
# test14 lives in a different split entirely -- nuplan_challenge reads
# nuplan-v1.1/test/, not the val directory -- so pointing the val data at a
# test14 filter would quietly evaluate whichever of its tokens happened to
# also be in val.
SPLIT=${SPLIT:-val14}
case "$SPLIT" in
    val14)         BUILDER=nuplan ;;
    test14-hard|test14-random) BUILDER=nuplan_challenge ;;
    *) echo "unknown SPLIT '$SPLIT' (val14 | test14-hard | test14-random)" >&2; exit 1 ;;
esac

# REACTIVITY picks nuPlan's own challenge. Non-reactive replays the log;
# reactive drives the background traffic with IDM. They are scored by different
# aggregators and published as separate columns, so they are separate runs.
REACTIVITY=${REACTIVITY:-nonreactive}
case "$REACTIVITY" in
    nonreactive) CHALLENGE=closed_loop_nonreactive_agents; OCCLUDED_OBS=occluded_box_observation ;;
    reactive)    CHALLENGE=closed_loop_reactive_agents;    OCCLUDED_OBS=occluded_idm_agents_observation ;;
    *) echo "unknown REACTIVITY '$REACTIVITY' (nonreactive | reactive)" >&2; exit 1 ;;
esac

case "$MODE" in
    occluded)
        # The occluded observation has to wrap whatever the challenge uses, or
        # the reactive condition silently reverts to log replay.
        OBSERVATION="observation=$OCCLUDED_OBS"; TAG=occluded ;;
    random)
        # Control: same number of objects withheld per frame, chosen at random
        # rather than by sight line. Only meaningful against an occluded run
        # over the same scenarios.
        OBSERVATION="observation=random_withholding_observation"; TAG=random ;;
    baseline)
        OBSERVATION=""; TAG=baseline ;;
    *) echo "unknown mode '$MODE' (baseline | occluded | random)" >&2; exit 1 ;;
esac

echo "split $SPLIT   $REACTIVITY   $TAG"

# Check the card has room before spending hours finding out it does not. A run
# on an already-occupied GPU dies as CUDA OOM inside the ray workers, which
# surfaces as a pile of failed simulations and an aggregate score computed over
# whichever handful survived -- a number that looks entirely reasonable.
if command -v nvidia-smi >/dev/null; then
    VIS=${CUDA_VISIBLE_DEVICES:-0}
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${VIS%%,*}" 2>/dev/null | head -1)
    if [ -n "$FREE" ]; then
        echo "gpu ${VIS%%,*}: ${FREE} MiB free"
        # Each concurrent simulation holds a CUDA context plus the model; the
        # default fraction allows about thirty of them.
        if [ "$FREE" -lt 8000 ]; then
            echo "only ${FREE} MiB free on gpu ${VIS%%,*} -- pick an idle one, or" >&2
            echo "raise GPU_FRACTION to run fewer simulations at once." >&2
            exit 1
        fi
    fi
fi


# Each 1118-scenario run writes about 57 GB of per-frame simulation logs, and
# scoring needs none of it -- the aggregator parquet is a few megabytes. Keep
# them only when the run is one whose scenarios will be rendered or examined;
# otherwise three runs fill a disk quota and everything dies at once.
# TOKENS re-runs a named handful of scenarios instead of the whole split. The
# point is the logs: a full run with them costs about 57 GB, almost all of it
# the planner object pickled once per scenario, but a dozen scenarios cost a
# few hundred MB. So score the split without logs, then come back for whichever
# scenarios turned out to be worth looking at.
TOKEN_ARG=""
if [ -n "${TOKENS:-}" ]; then
    TOKEN_ARG="scenario_filter.scenario_tokens=[${TOKENS}]                scenario_filter.num_scenarios_per_type=null                scenario_filter.limit_total_scenarios=null"
    KEEP_LOGS=1
    # A handful of scenarios does not need ray, and running in-process means a
    # failure prints its own traceback instead of a stalled progress bar.
    WORKER=${WORKER_OVERRIDE:-sequential}
    THREADS_ARG=""
    echo "re-running $(echo "$TOKENS" | tr ',' '\n' | wc -l) named scenarios with logs"
fi

KEEP_LOGS=${KEEP_LOGS:-0}
if [ "$KEEP_LOGS" = "1" ]; then
    LOG_ARG=""
    echo "keeping per-frame simulation logs (~57 GB for a full split)"
else
    # Remove just this callback from the composed config. `callback=[]` looks
    # right and is not: callback is a defaults-list group, so hydra reads that
    # as a value override and stops with "Key 'callback' is not in struct".
    # Deleting the one key leaves cfg.callback an empty dict, which both
    # build_simulation_callbacks and the simulation builder handle.
    LOG_ARG="~callback.simulation_log_callback"
    echo "not writing per-frame simulation logs; set KEEP_LOGS=1 if you need them"
fi

mkdir -p "$NUPLAN_EXP_ROOT"

# enable_ema is prefixed with + because it is not a key in Flow Planner's own
# flow_planner.yaml, and hydra refuses to override what is not there. It has to
# be false: the published checkpoint is already exported EMA weights, a flat
# state_dict, so the unwrap branch would look for an ema_state_dict that does
# not exist.

$PY "$DEVKIT/nuplan/planning/script/run_simulation.py" \
    +simulation=$CHALLENGE \
    planner=flow_planner \
    planner.flow_planner.config_path="$WORK/checkpoints/model_config_resolved.yaml" \
    planner.flow_planner.ckpt_path="$WORK/checkpoints/model.pth" \
    +planner.flow_planner.enable_ema=false \
    $OBSERVATION \
    $LOG_ARG \
    $TOKEN_ARG \
    scenario_builder=$BUILDER \
    scenario_builder.data_root="$VAL_SPLIT" \
    scenario_builder.map_root="$MAPS" \
    scenario_filter=$SPLIT \
    $LIMIT_ARG \
    experiment_uid="flow_planner/$SPLIT/$REACTIVITY/$TAG/$(date +%Y-%m-%d-%H-%M-%S)" \
    worker=$WORKER \
    distributed_mode=SINGLE_NODE \
    $THREADS_ARG \
    number_of_gpus_allocated_per_simulation=$GPU_FRACTION \
    enable_simulation_progress_bar=true \
    verbose=false \
    hydra.searchpath="[file://$BENCH/configs/nuplan, \
pkg://flow_planner.nuplan_simulation.scenario_filter, \
pkg://flow_planner.nuplan_simulation, \
pkg://nuplan.planning.script.config.common, \
pkg://nuplan.planning.script.experiments]"
