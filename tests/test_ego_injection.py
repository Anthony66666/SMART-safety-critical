"""Integration tests: driving the ego with an external policy.

SMART normally rolls the AV from its own token distribution. Safety-critical
evaluation replaces it with a policy under test, which changes two things that
must both hold: the ego follows the policy, and the ego stops contributing to
the scenario likelihood (it is no longer being generated).
"""
import pytest
import torch

from smart.safety.ego import ReplayPlanner
from tests.test_inference_likelihood import _load


@pytest.fixture(scope="module")
def replay_rollout():
    model, data = _load()
    av = int(data['agent']['av_index'])
    hist = model.num_historical_steps

    logged = torch.cat([
        data['agent']['position'][av, hist:, :2],
        data['agent']['heading'][av, hist:, None],
    ], dim=-1)

    planner = ReplayPlanner(logged, shift=5)
    with torch.no_grad():
        pred = model.inference(data, ego_planner=planner)
    return pred, av, logged


def test_replay_reproduces_the_logged_ego_trajectory(replay_rollout):
    """If injection works, a replayed ego matches the log it came from."""
    pred, av, logged = replay_rollout
    assert torch.allclose(pred['pred_traj'][av], logged[:, :2], atol=1e-3)


def test_ego_is_excluded_from_the_likelihood(replay_rollout):
    """The ego is not sampled from the model, so counting it would corrupt
    every realism score and every importance weight downstream."""
    pred, av, _ = replay_rollout
    assert pred['log_p'][av].item() == 0.0
    assert pred['log_q'][av].item() == 0.0


def test_other_agents_still_accumulate_likelihood(replay_rollout):
    """Guard against excluding everyone and calling it a pass."""
    pred, av, _ = replay_rollout
    others = torch.ones_like(pred['log_p'], dtype=torch.bool)
    others[av] = False
    assert (pred['log_p'][others] < 0).any()


@pytest.fixture(scope="module")
def idm_rollout():
    from smart.safety.ego import IDMPlanner
    model, data = _load()
    av = int(data['agent']['av_index'])
    with torch.no_grad():
        pred = model.inference(data, ego_planner=IDMPlanner(desired_speed=12.0))
    return pred, av


def test_idm_ego_produces_a_finite_trajectory(idm_rollout):
    pred, av = idm_rollout
    assert torch.isfinite(pred['pred_traj'][av]).all()


def test_idm_ego_actually_moves(idm_rollout):
    """A stationary ego would make every downstream collision metric vacuous."""
    pred, av = idm_rollout
    traj = pred['pred_traj'][av]
    assert (traj[-1] - traj[0]).norm() > 1.0


def test_idm_ego_is_also_excluded_from_the_likelihood(idm_rollout):
    pred, av = idm_rollout
    assert pred['log_p'][av].item() == 0.0
