"""Tests for the rule-based ego policy.

IDM longitudinal control with lane-holding lateral behaviour. This is the
first policy under test: fully reproducible, no external dependencies, and
weak enough that adversarial scenarios should be able to break it.
"""
import math

import pytest
import torch

from smart.safety.ego import IDMPlanner, PlanningContext
from smart.safety.ego.idm_planner import idm_acceleration

FREE = math.inf


def test_accelerates_when_below_desired_speed_on_a_free_road():
    a = idm_acceleration(speed=5.0, gap=FREE, lead_speed=0.0, desired_speed=15.0)
    assert a > 0


def test_holds_speed_when_already_at_desired_speed_on_a_free_road():
    a = idm_acceleration(speed=15.0, gap=FREE, lead_speed=0.0, desired_speed=15.0)
    assert a == pytest.approx(0.0, abs=1e-6)


def test_brakes_hard_for_a_close_slow_lead():
    a = idm_acceleration(speed=15.0, gap=2.0, lead_speed=0.0, desired_speed=15.0)
    assert a < -2.0


def _ctx(speed, neighbors=None):
    return PlanningContext(
        step=0,
        ego_state=torch.tensor([0.0, 0.0, 0.0]),
        neighbor_states=neighbors if neighbors is not None else torch.zeros(0, 3),
        ego_speed=speed,
    )


def test_planner_moves_the_ego_forward_along_its_heading():
    poses = IDMPlanner(desired_speed=15.0, shift=5).plan(_ctx(speed=10.0))

    assert poses.shape == (5, 3)
    assert (poses[:, 0] > 0).all()          # advanced along +x
    assert torch.allclose(poses[:, 1], torch.zeros(5), atol=1e-6)   # no drift
    assert (poses[1:, 0] > poses[:-1, 0]).all()                     # monotonic


def test_planner_slows_for_a_stopped_vehicle_directly_ahead():
    blocked = IDMPlanner(desired_speed=15.0, shift=5).plan(
        _ctx(speed=10.0, neighbors=torch.tensor([[5.0, 0.0, 0.0]])))
    free = IDMPlanner(desired_speed=15.0, shift=5).plan(_ctx(speed=10.0))

    assert blocked[-1, 0] < free[-1, 0]
