"""Teacher-forced scoring: evaluating a token sequence the model did not sample.

A judge must score scenarios produced by someone else -- another model, a
baseline method, or a log. That requires running the rollout with the tokens
pinned instead of sampled, while still reading off the model's probability
for each pinned token.

Both models must score against the SAME prepared scenario. Preparation
(match_token_map, sample_pt_pred) is stochastic and unseeded, so preparing
twice yields different map context and log-likelihoods that differ by ~1e-1 --
noise large enough to swamp the realism differences this project measures.
"""
import pytest
import torch

from tests.test_inference_likelihood import _load


@pytest.fixture(scope="module")
def sampled_then_forced():
    model, data = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        sampled = model.inference(data)
        forced = model.inference(data, forced_tokens=sampled['next_token_idx'])
    return sampled, forced


def test_forcing_a_models_own_tokens_reproduces_its_likelihood(sampled_then_forced):
    """The strongest check available: scoring is correct exactly when
    re-scoring a model's own sample returns the number it already reported."""
    sampled, forced = sampled_then_forced
    assert torch.equal(forced['log_p'], sampled['log_p'])


def test_forced_scoring_replays_the_pinned_tokens(sampled_then_forced):
    sampled, forced = sampled_then_forced
    assert torch.equal(forced['next_token_idx'], sampled['next_token_idx'])


def test_forced_scoring_reproduces_the_trajectory(sampled_then_forced):
    sampled, forced = sampled_then_forced
    assert torch.equal(forced['pred_traj'], sampled['pred_traj'])


def test_forced_scoring_has_a_degenerate_sampling_distribution(sampled_then_forced):
    """Nothing was sampled, so q is a point mass and log q must be 0.
    Guards against forced output being fed into importance weights as if it
    had been drawn from the model."""
    _, forced = sampled_then_forced
    assert torch.equal(forced['log_q'], torch.zeros_like(forced['log_q']))


def test_scoring_is_reproducible_on_one_prepared_scenario():
    """Two scorings of the same sequence against the same prepared scenario
    must agree bit for bit, or no realism comparison downstream is stable."""
    model, data = _load()
    torch.manual_seed(0)
    with torch.no_grad():
        tokens = model.inference(data)['next_token_idx']
        first = model.inference(data, forced_tokens=tokens)['log_p']
        second = model.inference(data, forced_tokens=tokens)['log_p']
    assert torch.equal(first, second)
