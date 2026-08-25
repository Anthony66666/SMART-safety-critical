"""Tests for the realism scoring pipeline.

Scenario preparation is stochastic in SMART (match_token_map perturbs map
tokens, sample_pt_pred randomly masks map points). Left alone that makes a
judge's score for a fixed scenario irreproducible, so the pipeline pins it.
"""
import pytest
import torch
from torch_geometric.loader import DataLoader

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.safety.scoring import (RealismReport, SelfJudgeError,
                                  prepare_scenario, score_tokens)
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

CONFIG = "configs/validation/validation_scalable.yaml"
CKPT = "checkpoints/epoch=31.ckpt"


def _model_and_batches(n=2):
    config = load_config_act(CONFIG)
    dc = config.Dataset
    ds = MultiDataset(root=dc.root, split='val', raw_dir=dc.val_raw_dir,
                      processed_dir=dc.val_processed_dir,
                      transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                                   config.Model.decoder.num_future_steps))
    batches = []
    for i, b in enumerate(DataLoader(ds, batch_size=1, shuffle=False)):
        batches.append(b)
        if len(batches) == n:
            break
    model = SMART(config.Model)
    model.load_params_from_file(filename=CKPT, logger=Logging().log(level='DEBUG'))
    model.eval()
    return model, batches


def test_preparation_is_reproducible():
    """Two preparations of the same scenario must agree, or the judge's score
    for a fixed scenario drifts between runs."""
    model, batches = _model_and_batches(n=1)
    _, again = _model_and_batches(n=1)

    a = prepare_scenario(model, batches[0])
    b = prepare_scenario(model, again[0])

    assert torch.equal(a['pt_token']['token_idx'], b['pt_token']['token_idx'])


def test_score_matches_the_generators_own_report():
    model, batches = _model_and_batches(n=1)
    data = prepare_scenario(model, batches[0])
    torch.manual_seed(0)
    with torch.no_grad():
        rollout = model.inference(data)

    scored = score_tokens(model, data, rollout['next_token_idx'])

    assert torch.equal(scored, rollout['log_p'])


def test_self_judging_is_refused_by_default():
    with pytest.raises(SelfJudgeError, match="same checkpoint"):
        RealismReport.create(generator_ckpt=CKPT, judge_ckpt=CKPT)


def test_self_judging_is_allowed_when_asked_for_explicitly():
    report = RealismReport.create(generator_ckpt=CKPT, judge_ckpt=CKPT,
                                  allow_self_judge=True)
    assert report.self_judged is True


def test_a_distinct_judge_is_not_flagged():
    report = RealismReport.create(generator_ckpt=CKPT, judge_ckpt="other.ckpt")
    assert report.self_judged is False
