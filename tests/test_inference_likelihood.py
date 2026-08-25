"""Integration tests: exact likelihood accounting through a real rollout.

These run the actual SMART rollout on one demo scenario with the trained
checkpoint, so they exercise the sampling loop in agent_decoder.inference().
"""
import pytest
import torch
from torch_geometric.loader import DataLoader

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

CONFIG = "configs/validation/validation_scalable.yaml"
CKPT = "checkpoints/epoch=31.ckpt"


def _load(beam_size=None):
    config = load_config_act(CONFIG)
    if beam_size is not None:
        config.Model.decoder.beam_size = beam_size
    dc = config.Dataset
    ds = MultiDataset(
        root=dc.root, split='val', raw_dir=dc.val_raw_dir,
        processed_dir=dc.val_processed_dir,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                     config.Model.decoder.num_future_steps))
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False)))

    model = SMART(config.Model)
    model.load_params_from_file(filename=CKPT, logger=Logging().log(level='DEBUG'))
    model.eval()

    data = model.match_token_map(batch)
    data = model.sample_pt_pred(data)
    data['agent']['av_index'] += data['agent']['ptr'][:-1]
    return model, data


@pytest.fixture(scope="module")
def rollout_full_support():
    model, data = _load(beam_size=2048)
    with torch.no_grad():
        return model.inference(data)


def test_inference_reports_log_p(rollout_full_support):
    assert 'log_p' in rollout_full_support


def test_inference_reports_log_q(rollout_full_support):
    assert 'log_q' in rollout_full_support


@pytest.fixture(scope="module")
def rollout_truncated():
    model, data = _load(beam_size=5)
    with torch.no_grad():
        return model.inference(data)


def test_rollout_likelihood_is_non_trivial(rollout_full_support):
    """Guard against vacuous assertions: simulated agents must accumulate
    real probability mass, otherwise every comparison below is meaningless."""
    log_p = rollout_full_support['log_p']
    assert (log_p < 0).any()


def test_full_support_gives_identical_log_p_and_log_q(rollout_full_support):
    """beam_size == token_size means no truncation, so q collapses onto p."""
    assert torch.allclose(rollout_full_support['log_p'],
                          rollout_full_support['log_q'], atol=1e-4)


def test_truncation_lifts_log_q_above_log_p(rollout_truncated):
    """Top-5 truncation concentrates mass, so sampled tokens are likelier under q."""
    log_p = rollout_truncated['log_p']
    log_q = rollout_truncated['log_q']
    simulated = log_p < 0
    assert simulated.any()
    assert (log_q[simulated] > log_p[simulated]).all()


def test_softmax_of_logits_is_the_cross_entropy_likelihood():
    """pred_prob is only "exact p" if softmax(next_token_prob) is the model
    distribution. Cross-check against the training loss: without label
    smoothing, cross_entropy is exactly the mean negative log-likelihood."""
    import torch.nn.functional as F

    model, data = _load()
    with torch.no_grad():
        pred = model(data)

    keep = pred['next_token_eval_mask']
    logits = pred['next_token_prob'][keep]
    gt = pred['next_token_idx_gt'][keep]

    manual_nll = -torch.log_softmax(logits, dim=-1).gather(
        -1, gt[:, None]).squeeze(-1).mean()
    ce = F.cross_entropy(logits, gt)

    assert manual_nll.item() == pytest.approx(ce.item(), rel=1e-5)
