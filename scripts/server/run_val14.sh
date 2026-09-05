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
# Whether the caller chose this matters below, so record it before defaulting.
# This box cannot reach huggingface.co. Anything that tries -- timm fetching
# ImageNet weights, a tokenizer, a config -- should go to the mirror instead of
# retrying until it fails.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

GPU_FRACTION_EXPLICIT=${GPU_FRACTION:+yes}
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

# REACTIVITY picks what drives the background traffic. Non-reactive replays the
# log and reactive uses IDM; both are nuPlan's own challenges, scored by
# different aggregators and published as separate columns, so they are separate
# runs. `smart` is ours: a learned traffic model in place of IDM, scored by the
# reactive aggregator because that is the challenge it is standing in for.
#
# BASE_OBS is what the challenge would use on its own. It is empty for the two
# official ones because the challenge config already selects them, and set for
# smart because nothing else would -- leaving it empty there silently reverts
# to log replay and produces a plausible set of numbers for the wrong thing.
REACTIVITY=${REACTIVITY:-nonreactive}
case "$REACTIVITY" in
    nonreactive) CHALLENGE=closed_loop_nonreactive_agents
                 BASE_OBS=""; OCCLUDED_OBS=occluded_box_observation ;;
    reactive)    CHALLENGE=closed_loop_reactive_agents
                 BASE_OBS=""; OCCLUDED_OBS=occluded_idm_agents_observation ;;
    smart)       CHALLENGE=closed_loop_reactive_agents
                 BASE_OBS=smart_agents_observation
                 OCCLUDED_OBS=occluded_smart_agents_observation
                 # Every ray worker loads its own copy of the traffic model, so
                 # here the GPU budget is set by the observation rather than by
                 # the planner. One worker measured 1250 MiB, and the 0.03
                 # default lets ray pack 33 of them onto a card that also has
                 # other people's jobs on it -- which is exactly the CUDA OOM
                 # this hit, arriving as a torch error inside match_token_map
                 # rather than as anything about scheduling. How much
                 # headroom is enough depends on the planner as well, so the
                 # number is settled after the planner is known, below.
                 SMART_GPU_BUDGET=yes ;;
    *) echo "unknown REACTIVITY '$REACTIVITY' (nonreactive | reactive | smart)" >&2; exit 1 ;;
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
        if [ "$REACTIVITY" = smart ]; then
            echo "random control is not wired for smart traffic yet" >&2; exit 1
        fi
        OBSERVATION="observation=random_withholding_observation"; TAG=random ;;
    baseline)
        OBSERVATION=${BASE_OBS:+observation=$BASE_OBS}; TAG=baseline ;;
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

# Classifier-free guidance. The published model_config sets cfg_weight: 1.0,
# which the guidance formula u = (1-w)*u_uncond + w*u_cond turns into plain
# u_cond -- guidance off, as the config's own comment says. The paper reports
# 90.43 at 1.8, so the shipped default evaluates a weakened planner.
#
# It has to be set on the model config, not on FlowPlanner: flow_ode.generate
# takes no cfg_weight parameter, so the planner's argument falls into **kwargs
# and never reaches VelocityModel. Verified by measurement -- same seed, only
# model.cfg_weight changed, trajectories 2.26 m apart.
CFG_WEIGHT=${CFG_WEIGHT:-1.8}
RESOLVED="$WORK/checkpoints/model_config_cfg${CFG_WEIGHT}.yaml"
$PY - "$WORK" "$CFG_WEIGHT" "$RESOLVED" <<'CFGEOF'
import sys
from omegaconf import OmegaConf
work, weight, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
config = OmegaConf.load(f'{work}/checkpoints/model_config_resolved.yaml')
config.model.cfg_weight = weight
OmegaConf.save(config, out)
CFGEOF
echo "cfg_weight $CFG_WEIGHT  ->  $(basename "$RESOLVED")"

# PLANNER picks what is being evaluated. The benchmark needs contrast across
# paradigms more than it needs every planner in the table:
#   idm         rule-based, consults only its lead vehicle -- structurally
#               immune to occlusion, and measured at exactly zero gap
#   pdm_closed  rule-based but reasons over the whole scene; the nuPlan
#               challenge winner. Deterministic, so it needs no seeding, and
#               purely rule-based, so it needs no checkpoint
#   flow        learned
# If pdm_closed also loses points, the finding stops being about one learned
# planner and becomes about planners that use the wider scene.
PLANNER=${PLANNER:-flow}
EXTRA_SEARCHPATH=""

case "$PLANNER" in
  idm)
    PLANNER_ARG="planner=idm_planner"
    echo "planner: IDM (devkit)"
    ;;
  pdm_closed)
    PLANNER_ARG="planner=pdm_closed_planner"
    # Rule-based and CPU-only, so reserving GPU for it just caps concurrency
    # on a machine whose bottleneck is cores.
    GPU_FRACTION=${GPU_FRACTION:-0}
    # tuplan_garage keeps its planner configs in its own package.
    EXTRA_SEARCHPATH="pkg://tuplan_garage.planning.script.config.common, \
        pkg://tuplan_garage.planning.script.config.simulation, "
    echo "planner: PDM-Closed (tuplan_garage, rule-based, no checkpoint)"
    ;;
  pdm_hybrid)
    PLANNER_ARG="planner=pdm_hybrid_planner \
        planner.pdm_hybrid_planner.checkpoint_path=$WORK/checkpoints/tuplan/pdm_offset_checkpoint.ckpt"
    EXTRA_SEARCHPATH="pkg://tuplan_garage.planning.script.config.common, \
        pkg://tuplan_garage.planning.script.config.simulation, "
    echo "planner: PDM-Hybrid (tuplan_garage)"
    ;;
  pdm_open)
    PLANNER_ARG="planner=pdm_open_planner \
        planner.pdm_open_planner.checkpoint_path=$WORK/checkpoints/tuplan/pdm_open_checkpoint.ckpt"
    EXTRA_SEARCHPATH="pkg://tuplan_garage.planning.script.config.common, \
        pkg://tuplan_garage.planning.script.config.simulation, "
    echo "planner: PDM-Open (tuplan_garage)"
    ;;
  urban_driver|gc_pgp|plancnn)
    # These three run through the devkit's generic ml_planner, differing only
    # in which model config and checkpoint they are handed. raster_model and
    # urban_driver_open_loop_model are the devkit's own; gc_pgp_model comes
    # from tuplan_garage.
    case "$PLANNER" in
      urban_driver) MODEL=urban_driver_open_loop_model; CKPT=urbandriver_checkpoint.ckpt; EXTRA="" ;;
      gc_pgp)       MODEL=gc_pgp_model;                 CKPT=gc_pgp_checkpoint.ckpt
                    EXTRA="model.aggregator.pre_train=false" ;;
      # raster_model defaults to pretrained=true, which sends timm to
      # HuggingFace for ImageNet ResNet50 weights. This box cannot reach
      # huggingface.co, so every plancnn run died in the model constructor --
      # and the download is pointless anyway, since the plancnn checkpoint
      # loaded a moment later overwrites all of it.
      plancnn)      MODEL=raster_model;                 CKPT=plancnn_checkpoint.ckpt
                    EXTRA="model.pretrained=false" ;;
    esac
    PLANNER_ARG="planner=ml_planner \
        planner.ml_planner.model_config=\${model} \
        planner.ml_planner.checkpoint_path=$WORK/checkpoints/tuplan/$CKPT \
        model=$MODEL $EXTRA"
    EXTRA_SEARCHPATH="pkg://tuplan_garage.planning.script.config.common, \
        pkg://tuplan_garage.planning.script.config.simulation, "
    echo "planner: $PLANNER (ml_planner + $MODEL)"
    ;;
  dtpp)
    # DTPP's repository is a flat set of top-level modules rather than a
    # package, so its checkout has to be on PYTHONPATH for `planner.Planner`
    # to resolve and for torch.load to unpickle the model's classes. Its
    # checkpoint ships inside the repository.
    DTPP_ROOT=${DTPP_ROOT:-$WORK/DTPP}
    export PYTHONPATH="$DTPP_ROOT:$PYTHONPATH"
    PLANNER_ARG="planner=dtpp_planner \
        planner.dtpp_planner.model_path=$DTPP_ROOT/base_model.pth"
    echo "planner: DTPP (tree policy planning)"
    ;;
  carl)
    # The only RL entry in the table, and the only one that needs a different
    # simulation config: it emits control actions rather than a trajectory, so
    # it runs under one_stage_controller instead of two_stage_controller. That
    # makes its absolute score incomparable with the other planners -- but the
    # gap is not, since both of its conditions use the same controller.
    CARL_ROOT=${CARL_ROOT:-$WORK/CaRL/nuPlan}
    CHALLENGE="${CHALLENGE}_action"
    PLANNER_ARG="planner=ppo_planner \
        planner.ppo_planner.checkpoint_path=$CARL_ROOT/checkpoints/${CARL_CKPT:-nuplan_51892_1B}/model_best.pth"
    EXTRA_SEARCHPATH="pkg://carl_nuplan.planning.script.config.common, \
        pkg://carl_nuplan.planning.script.config.simulation, \
        pkg://carl_nuplan.planning.script.experiments, "
    echo "planner: CaRL / PPO (one_stage_controller -- absolute score not comparable)"
    ;;
  diffusion)
    # Same lab as Flow Planner and its direct predecessor, so the two are not
    # independent samples -- but it is the only strong learned planner whose
    # weights are still obtainable. PLUTO's and PlanTF's are 404 on the
    # authors' OneDrive.
    PLANNER_ARG="planner=seeded_diffusion_planner \
        planner.seeded_diffusion_planner.seed=${SEED:-0} \
        planner.seeded_diffusion_planner.planner.config.args_file=$WORK/checkpoints/diffusion/args.json \
        planner.seeded_diffusion_planner.planner.ckpt_path=$WORK/checkpoints/diffusion/model.pth"
    EXTRA_SEARCHPATH="pkg://diffusion_planner.config, "
    echo "planner: Diffusion Planner, seed=${SEED:-0}"
    ;;
  flow)
    :
    ;;
  *)
    echo "unknown PLANNER '$PLANNER'" >&2
    echo "  rule-based : idm | pdm_closed" >&2
    echo "  hybrid     : pdm_hybrid | dtpp" >&2
    echo "  RL         : carl" >&2
    echo "  learned    : pdm_open | urban_driver | gc_pgp | plancnn | diffusion | flow" >&2
    exit 1 ;;
esac

# SEED pins Flow Planner's sampling noise to the simulation step. Without it
# two identical runs disagree on 621 of 1118 scenarios and 0.78 points, which
# is the same size as the effect being measured. Unset SEED to reproduce the
# original unpinned behaviour. Only Flow Planner samples; the rule-based
# planners are deterministic already.
if [ "$PLANNER" != "flow" ]; then   # non-flow planners set PLANNER_ARG above
    :
elif [ -n "${SEED:-}" ]; then
    PLANNER_ARG="planner=seeded_flow_planner \
        planner.seeded_flow_planner.seed=$SEED \
        planner.seeded_flow_planner.planner.config_path=$RESOLVED \
        planner.seeded_flow_planner.planner.ckpt_path=$WORK/checkpoints/model.pth"
    echo "seeded planner, seed=$SEED"
else
    PLANNER_ARG="planner=flow_planner \
        planner.flow_planner.config_path=$RESOLVED \
        planner.flow_planner.ckpt_path=$WORK/checkpoints/model.pth \
        +planner.flow_planner.enable_ema=false"
    echo "UNSEEDED planner -- results will not reproduce; set SEED=0 to pin them"
fi

mkdir -p "$NUPLAN_EXP_ROOT"

# enable_ema is prefixed with + because it is not a key in Flow Planner's own
# flow_planner.yaml, and hydra refuses to override what is not there. It has to
# be false: the published checkpoint is already exported EMA weights, a flat
# state_dict, so the unwrap branch would look for an ema_state_dict that does
# not exist.

# With smart traffic the card holds one copy of the traffic model per ray
# worker, and for most planners a copy of the planner's model too. 0.1 was
# enough for the rule-based planners and for diffusion -- which still peaked at
# 45 of 48 GB -- but urban_driver and gc_pgp went over. Splitting the two cases
# keeps the cheap planners fast instead of slowing everything to the worst
# case. Ray divides by the number of visible cards, so running on two halves
# the pressure on each without changing throughput.
if [ -n "${SMART_GPU_BUDGET:-}" ] && [ -z "$GPU_FRACTION_EXPLICIT" ]; then
    # A fixed fraction cannot work here: it encodes an assumption about how
    # much of the card is ours, and on a shared box that assumption is wrong
    # within the hour. Measured, one worker running smart traffic wants about
    # 7.5 GB -- not the 1.25 GB the sequential smoke test suggested, because
    # that run had one worker and no planner model beside it. The budget is
    # therefore computed from the memory actually free at launch.
    #
    # This is what the earlier failures were: 0.2 asked for five workers on a
    # 46 GB card that a neighbouring job had already taken 29 GB of, so the
    # allocation died 48 seconds in, on a 2 MiB request.
    SMART_WORKER_MIB=${SMART_WORKER_MIB:-8000}
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                 -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$free_mib" ]; then
        # Leave one worker's worth of slack for the neighbours to grow into.
        slots=$(( (free_mib - SMART_WORKER_MIB) / SMART_WORKER_MIB ))
        [ "$slots" -lt 1 ] && slots=1
        [ "$slots" -gt 8 ] && slots=8
        GPU_FRACTION=$(awk -v n="$slots" 'BEGIN { printf "%.3f", 1.0 / n }')
        echo "smart traffic: ${free_mib} MiB free, ${slots} concurrent (GPU_FRACTION=$GPU_FRACTION)"
    else
        GPU_FRACTION=0.34
    fi
fi

# DRY_RUN resolves everything and checks the files the run would need, then
# exits without simulating. The point is to find a missing checkpoint in a
# second rather than after the queue ahead of it has finished -- and to keep
# that knowledge here, where the paths are actually built, instead of copied
# into the sweep script where it would drift.
if [ -n "${DRY_RUN:-}" ]; then
    missing=0
    # Every argument of the form key=/absolute/path names a file or directory
    # the run will open.
    for token in $PLANNER_ARG $OBSERVATION; do
        case "$token" in
            *=/*) path=${token#*=}
                  if [ ! -e "$path" ]; then
                      echo "  missing: $path" >&2; missing=1
                  fi ;;
        esac
    done
    if [ "$REACTIVITY" = smart ]; then
        ckpt=${SMART_CHECKPOINT:-$BENCH/checkpoints/bosch_nuplan_smart.ckpt}
        [ -e "$ckpt" ] || { echo "  missing: $ckpt" >&2; missing=1; }
        $PY -c 'import torch_geometric, torch_scatter, torch_cluster' 2>/dev/null \
            || { echo "  missing: torch_geometric/scatter/cluster" >&2; missing=1; }
    fi
    [ -d "$VAL_SPLIT" ] || { echo "  missing: $VAL_SPLIT" >&2; missing=1; }
    exit $missing
fi

$PY "$DEVKIT/nuplan/planning/script/run_simulation.py" \
    +simulation=$CHALLENGE \
    $PLANNER_ARG \
    $OBSERVATION \
    $LOG_ARG \
    $TOKEN_ARG \
    scenario_builder=$BUILDER \
    scenario_builder.data_root="$VAL_SPLIT" \
    scenario_builder.map_root="$MAPS" \
    scenario_filter=$SPLIT \
    $LIMIT_ARG \
    experiment_uid="$PLANNER/$SPLIT/$REACTIVITY/$TAG/$(date +%Y-%m-%d-%H-%M-%S)" \
    worker=$WORKER \
    distributed_mode=SINGLE_NODE \
    $THREADS_ARG \
    number_of_gpus_allocated_per_simulation=$GPU_FRACTION \
    enable_simulation_progress_bar=true \
    verbose=false \
    hydra.searchpath="[file://$BENCH/configs/nuplan, \
    $EXTRA_SEARCHPATH \
pkg://flow_planner.nuplan_simulation.scenario_filter, \
pkg://flow_planner.nuplan_simulation, \
pkg://nuplan.planning.script.config.common, \
pkg://nuplan.planning.script.experiments]"
