#!/usr/bin/env bash
# Flow Planner's own launch_sim_nuplan.sh, filled in for the L40S box.
#
# Kept as close to the authors' script as it can be run, so that a number from
# it is an independent check on our harness rather than a rerun of it. Two
# changes are unavoidable and both are marked ADDED below; everything else,
# including number_of_gpus_allocated_per_simulation=0.15 and the simulation
# logs, is left as they wrote it.
export CUDA_VISIBLE_DEVICES=1        # a free card on this box; theirs assumes 0-7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
export NUPLAN_DEVKIT_ROOT=/lab/haoq_lab/12432702/occlusion-bench/nuplan-devkit
export NUPLAN_DATA_ROOT=/hqlab/dataset_nas3/nuplan/raw
export NUPLAN_MAPS_ROOT=/hqlab/dataset_nas3/nuplan/raw/maps
export NUPLAN_EXP_ROOT=/lab/haoq_lab/12432702/occlusion-bench/exp_official

SPLIT=val14
CHALLENGE=closed_loop_nonreactive_agents
###################################

BRANCH_NAME=flow_planner_release
# Their comment says "path of .hydra/config in ckpt folder". The config
# published with the checkpoint is that file, but it still interpolates into
# training config branches that were not published, so it will not load as-is.
# This is the same file with those branches merged back in.
CONFIG_FILE=/lab/haoq_lab/12432702/occlusion-bench/checkpoints/model_config_resolved.yaml
CKPT_FILE=/lab/haoq_lab/12432702/occlusion-bench/checkpoints/model.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

# Two additions to the authors' command, both unavoidable:
#
# +planner.flow_planner.enable_ema=false
#   The published checkpoint is a flat state_dict of exported EMA weights, so
#   the default enable_ema=True looks for an 'ema_state_dict' key the file does
#   not contain and the run dies before the first scenario. Their script as
#   written cannot run against their released checkpoint.
#
# scenario_builder.data_root=$VAL_DATA
#   scenario_builder=nuplan reads $NUPLAN_DATA_ROOT/nuplan-v1.1/trainval, but
#   this NAS keeps the splits one level down under splits/, so no choice of
#   NUPLAN_DATA_ROOT resolves it. Points at the staged val copy instead, which
#   was checked to hold all 1118 val14 tokens and is far faster than the NAS.
#   The group override must come first: setting the group afterwards would
#   reset the value.
VAL_DATA=/lab/haoq_lab/12432702/occlusion-bench/val14_local

mkdir -p "$NUPLAN_EXP_ROOT"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    +planner.flow_planner.enable_ema=false \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_builder.data_root=$VAL_DATA \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=64 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
