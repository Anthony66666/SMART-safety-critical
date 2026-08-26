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


# --- anchors: deliberately wrong token sequences, to check the ruler ---

def test_borrowed_tokens_match_the_target_agent_count():
    """Another scenario has a different number of agents; the donor sequence
    is tiled or truncated so it can be scored against this scenario."""
    from smart.safety.scoring import borrow_tokens
    donor = torch.arange(3 * 16).reshape(3, 16)

    assert borrow_tokens(donor, num_agents=7).shape == (7, 16)
    assert borrow_tokens(donor, num_agents=2).shape == (2, 16)


def test_borrowed_tokens_reuse_the_donor_rows():
    from smart.safety.scoring import borrow_tokens
    donor = torch.arange(3 * 16).reshape(3, 16)

    borrowed = borrow_tokens(donor, num_agents=5)

    assert torch.equal(borrowed[0], donor[0])
    assert torch.equal(borrowed[3], donor[0])   # wraps around


def test_permuting_agents_keeps_every_sequence_but_moves_it():
    """The sharpest anchor: identical token marginals, wrong pairing with map
    and history. A judge that only learned token frequencies cannot tell this
    apart from the real thing."""
    from smart.safety.scoring import permute_agents
    tokens = torch.arange(6 * 16).reshape(6, 16)

    permuted = permute_agents(tokens, seed=0)

    assert permuted.shape == tokens.shape
    assert not torch.equal(permuted, tokens)
    assert torch.equal(permuted.sort(dim=0).values, tokens.sort(dim=0).values)


def test_permuting_agents_is_reproducible():
    from smart.safety.scoring import permute_agents
    tokens = torch.arange(6 * 16).reshape(6, 16)

    assert torch.equal(permute_agents(tokens, seed=3), permute_agents(tokens, seed=3))


# --- realism statistics ------------------------------------------------------

def test_bits_per_dimension_rescales_nats():
    """BPD is the standard unit, and nothing more than a change of base: it
    carries exactly the information the nats-per-agent-step figure does."""
    from smart.safety.scoring import bits_per_dim
    import math

    assert bits_per_dim(-math.log(2.0)) == pytest.approx(1.0)
    assert bits_per_dim(-2.0 * math.log(2.0)) == pytest.approx(2.0)


def test_bits_per_dimension_preserves_ordering():
    from smart.safety.scoring import bits_per_dim

    assert bits_per_dim(-2.385) < bits_per_dim(-3.772)


def test_typicality_is_zero_at_the_reference_entropy():
    """A sequence whose average surprisal equals the reference sits exactly on
    the typical set."""
    from smart.safety.scoring import typicality

    assert typicality(log_p_per_dim=-3.772, reference_entropy=3.772) == pytest.approx(0.0)


def test_typicality_grows_in_both_directions():
    """Too likely is as atypical as too unlikely -- which is why typicality
    cannot double as a realism score when the model is over-dispersed."""
    from smart.safety.scoring import typicality

    assert typicality(-2.0, 3.0) == pytest.approx(1.0)
    assert typicality(-4.0, 3.0) == pytest.approx(1.0)


# --- single-agent perturbation ----------------------------------------------

def test_zero_offset_leaves_positions_untouched():
    from smart.safety.scoring import lateral_offset
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[5.0, 5.0], [6.0, 5.0]]])
    head = torch.zeros(2, 2)

    assert torch.equal(lateral_offset(pos, head, agent_index=0, distance=0.0), pos)


def test_offset_moves_the_agent_perpendicular_to_its_heading():
    """Heading +x means the lateral direction is +y."""
    from smart.safety.scoring import lateral_offset
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    head = torch.zeros(1, 2)

    moved = lateral_offset(pos, head, agent_index=0, distance=3.0)

    assert torch.allclose(moved[0, :, 1], torch.full((2,), 3.0))
    assert torch.allclose(moved[0, :, 0], pos[0, :, 0])


def test_offset_leaves_other_agents_alone():
    from smart.safety.scoring import lateral_offset
    pos = torch.tensor([[[0.0, 0.0]], [[5.0, 5.0]]])
    head = torch.zeros(2, 1)

    moved = lateral_offset(pos, head, agent_index=0, distance=3.0)

    assert torch.equal(moved[1], pos[1])


def test_offset_preserves_relative_motion():
    """The point of the probe: the agent's own motion -- and therefore its
    token sequence -- is unchanged. Only its place in the world moves."""
    from smart.safety.scoring import lateral_offset
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    head = torch.zeros(1, 3)

    moved = lateral_offset(pos, head, agent_index=0, distance=4.0)

    assert torch.allclose(moved[0, 1:] - moved[0, :-1], pos[0, 1:] - pos[0, :-1])
