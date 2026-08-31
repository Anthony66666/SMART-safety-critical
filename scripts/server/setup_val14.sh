#!/usr/bin/env bash
# Build the environment for a full val14 run on the L40S box.
#
# Mirrors the Flow Planner setup instructions with one deviation that is not
# optional: nuplan-devkit pins hydra-core==1.1.0rc1, which needs
# omegaconf==2.1.0.rc1, and that release no longer exists on PyPI. Installing
# the devkit requirements as written fails outright. Flow Planner requires
# hydra 1.3.2 anyway and installs it over the top, so the pin is dropped and
# the newer version stands -- which is the state their own instructions
# produce, since they install their requirements after the devkit's.
#
# Run once:  bash scripts/server/setup_val14.sh
set -u

CONDA=${CONDA:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-flow_planner}
WORK=${WORK:-$HOME/occlusion-bench}
PIP="$CONDA/envs/$ENV_NAME/bin/pip"
PY="$CONDA/envs/$ENV_NAME/bin/python"

mkdir -p "$WORK"

echo "=== 1. clone what is missing ==="
[ -d "$WORK/nuplan-devkit" ] || git clone -q --depth 1 https://github.com/motional/nuplan-devkit.git "$WORK/nuplan-devkit"
[ -d "$WORK/Flow-Planner" ] || git clone -q --depth 1 https://github.com/DiffusionAD/Flow-Planner.git "$WORK/Flow-Planner"

echo "=== 2. conda env ==="
"$CONDA/bin/conda" create -n "$ENV_NAME" python=3.9 -y -q

echo "=== 3. nuplan-devkit, without the impossible hydra pin ==="
$PIP install -q -e "$WORK/nuplan-devkit"
grep -vE "^hydra-core|^omegaconf" "$WORK/nuplan-devkit/requirements.txt" > /tmp/devkit_req.txt
$PIP install -q --timeout 60 --retries 6 -r /tmp/devkit_req.txt

echo "=== 4. flow planner (brings torch 2.3 and hydra 1.3.2) ==="
$PIP install -q -e "$WORK/Flow-Planner"
$PIP install -q --timeout 60 --retries 6 -r "$WORK/Flow-Planner/requirements.txt"

# run_simulation.py imports it; the devkit's own 1.3.8 pin cannot coexist with
# torch 2.3, and only seed_everything is actually used.
$PIP install -q --timeout 60 --retries 6 "pytorch-lightning>=2.0,<2.5"

echo "=== 5. checkpoint ==="
# huggingface.co is unreachable from this box; hf-mirror.com serves the same
# paths and does answer. Both are tried, and a failure stops the script rather
# than leaving an empty directory for a later step to trip over.
mkdir -p "$WORK/checkpoints"
HOSTS="${HF_HOSTS:-https://hf-mirror.com https://huggingface.co}"
for f in model_config.yaml model.pth; do
    [ -s "$WORK/checkpoints/$f" ] && { echo "  have $f"; continue; }
    for host in $HOSTS; do
        echo "  fetching $f from $host"
        if curl -fL --connect-timeout 20 --retry 3 -o "$WORK/checkpoints/$f" \
                "$host/ttwhy/flow-planner/resolve/main/$f"; then
            break
        fi
        rm -f "$WORK/checkpoints/$f"
    done
    if [ ! -s "$WORK/checkpoints/$f" ]; then
        echo "FAILED to download $f from any of: $HOSTS" >&2
        echo "Copy it in by hand, then re-run this script:" >&2
        echo "  scp checkpoints/flow_planner/$f <this-host>:$WORK/checkpoints/" >&2
        exit 1
    fi
done
ls -la "$WORK/checkpoints"

echo "=== 5b. fill in the config branches the checkpoint does not ship ==="
# The published config was cut out of a training config tree and still
# interpolates into parts of it that did not come along. Values are the
# repository's own defaults (flow_planner/script/data/dataset/nuplan_data.yaml);
# train.epoch is only read by the LR scheduler and never at inference.
WORK="$WORK" $PY - <<'RESOLVE'
import os
from omegaconf import OmegaConf

work = os.environ['WORK']
config = OmegaConf.load(f'{work}/checkpoints/model_config.yaml')
defaults = OmegaConf.create({
    'data': {'dataset': {'train': {
        'future_downsampling_method': 'uniform',
        'predicted_neighbor_num': '${model.neighbor_pred_num}'}}},
    'train': {'epoch': 1},
})
OmegaConf.save(OmegaConf.merge(defaults, config),
               f'{work}/checkpoints/model_config_resolved.yaml')
print('wrote model_config_resolved.yaml')
RESOLVE

echo "=== 6. verify ==="
$PY - <<'CHECK'
import torch, hydra, omegaconf, shapely, pytorch_lightning
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('hydra', hydra.__version__, 'omegaconf', omegaconf.__version__)
from nuplan.planning.simulation.simulation import Simulation
from flow_planner.planner import FlowPlanner
print('devkit + flow planner import cleanly')
CHECK
