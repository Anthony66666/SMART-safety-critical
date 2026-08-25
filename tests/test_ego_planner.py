"""Tests for the ego policy under test.

SMART rolls every agent -- including the AV -- from its own token
distribution. Safety-critical evaluation needs the ego driven by an external
policy instead, so these planners produce the ego's motion and the rollout
engine injects it.
"""
import pytest
import torch

from smart.safety.ego import PlanningContext, ReplayPlanner


def _ctx(step, ego_state=None):
    return PlanningContext(
        step=step,
        ego_state=ego_state if ego_state is not None else torch.zeros(3),
        neighbor_states=torch.zeros(0, 3),
    )


def test_replay_planner_returns_the_logged_segment():
    """Replay is the control condition: it must reproduce the log exactly."""
    log = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    planner = ReplayPlanner(log, shift=5)

    assert torch.equal(planner.plan(_ctx(step=0)), log[0:5])


def test_replay_planner_advances_with_the_step():
    log = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    planner = ReplayPlanner(log, shift=5)

    assert torch.equal(planner.plan(_ctx(step=1)), log[5:10])
