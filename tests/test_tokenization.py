"""Tests for projecting a continuous pose back onto the token vocabulary.

An externally controlled ego leaves the token grid, but the rollout still
needs a token embedding for it at the next step. Projection must use the same
metric as preprocessing (mean corner distance between bounding-box contours),
or the ego's embedding drifts away from what the model was trained on.
"""
import pickle

import torch
from torch_geometric.loader import DataLoader

from smart.datasets.scalable_dataset import MultiDataset
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act

from smart.safety.tokenization import nearest_token

TOKEN_PATH = "smart/tokens/cluster_frame_5_2048.pkl"
VEH_WIDTH, VEH_LENGTH = 2.0, 4.8


def _veh_tokens():
    with open(TOKEN_PATH, "rb") as f:
        return torch.from_numpy(pickle.load(f)['token']['veh']).float()


def _pose_of(contour):
    """Recover (pos, heading) from a 4-corner contour, as the rollout does."""
    pos = contour.mean(dim=0)
    diff = contour[0] - contour[3]
    return pos, torch.arctan2(diff[1], diff[0])


def test_recovers_the_token_that_generated_the_pose():
    """Projecting a token's own resulting pose must return that token."""
    tokens = _veh_tokens()
    prev_pos = torch.zeros(1, 2)
    prev_heading = torch.zeros(1)

    for k in (0, 17, 1234, 2047):
        pos, heading = _pose_of(tokens[k])
        idx = nearest_token(prev_pos, prev_heading,
                            pos[None, :], heading[None],
                            tokens, VEH_WIDTH, VEH_LENGTH)
        assert idx.item() == k


def test_agrees_with_the_preprocessing_tokeniser():
    """The dataset's token_idx was produced by preprocess.py::match_token.
    Reproducing it from the logged poses proves the two use the same metric.

    Agreement is not asserted at exactly 1.0: preprocessing matches agents that
    enter mid-scenario against a separate `token_last` vocabulary, which this
    projection does not model. That path was not isolated here; it accounts for
    a handful of poses at most.
    """
    cfg = load_config_act("configs/validation/validation_scalable.yaml")
    dc = cfg.Dataset
    ds = MultiDataset(root=dc.root, split='val', raw_dir=dc.val_raw_dir,
                      processed_dir=dc.val_processed_dir,
                      transform=WaymoTargetBuilder(cfg.Model.num_historical_steps,
                                                   cfg.Model.decoder.num_future_steps))
    tokens = _veh_tokens()

    agree = total = 0
    for n, batch in enumerate(DataLoader(ds, batch_size=1, shuffle=False)):
        a = batch['agent']
        veh = a['type'] == 0
        tok_pos, tok_head = a['token_pos'][veh], a['token_heading'][veh]
        tok_idx, valid = a['token_idx'][veh], a['valid_mask'][veh]
        pos, head = a['position'][veh][..., :2], a['heading'][veh]

        # token m results from the pose at timestep (m+1)*shift, applied from
        # the pose token m-1 produced.
        for m in range(1, tok_idx.shape[1]):
            t = (m + 1) * 5
            if t >= pos.shape[1]:
                break
            usable = valid[:, t] & valid[:, (m - 1) * 5]
            if not usable.any():
                continue
            idx = nearest_token(tok_pos[usable, m - 1], tok_head[usable, m - 1],
                                pos[usable, t], head[usable, t], tokens)
            agree += (idx == tok_idx[usable, m]).sum().item()
            total += idx.numel()
        if n >= 4:
            break

    assert total > 1000
    assert agree / total >= 0.99
