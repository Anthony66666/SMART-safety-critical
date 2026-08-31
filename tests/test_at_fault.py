"""Tests for nuPlan's ego collision taxonomy."""
import math

import torch

from smart.metrics.at_fault import (
    ACTIVE_FRONT,
    ACTIVE_LATERAL,
    ACTIVE_REAR,
    STOPPED_EGO,
    STOPPED_TRACK,
    at_fault,
    ego_collisions,
)


def scene(ego_track, other_track, steps=None):
    """Two agents; each track is a list of (x, y, heading) per step."""
    steps = steps or len(ego_track)
    position = torch.zeros(2, steps, 2)
    heading = torch.zeros(2, steps)
    for i, track in enumerate((ego_track, other_track)):
        for t, (x, y, h) in enumerate(track):
            position[i, t] = torch.tensor([x, y])
            heading[i, t] = h
    shape = torch.tensor([[5.0, 2.0, 1.5], [5.0, 2.0, 1.5]])
    valid = torch.ones(2, steps, dtype=torch.bool)
    return position, heading, shape, valid


def still(x, y, h=0.0, steps=3):
    return [(x, y, h)] * steps


def approaching(x0, y, dx, h=0.0, steps=3):
    return [(x0 + dx * t, y, h) for t in range(steps)]


def test_driving_into_a_stopped_car_is_at_fault():
    counts, _ = ego_collisions(*scene(approaching(0, 0, 2.0), still(6.0, 0)), ego=0)
    assert counts[STOPPED_TRACK] == 1
    assert at_fault(counts) == 1


def test_a_stopped_ego_being_hit_is_not_at_fault():
    counts, _ = ego_collisions(*scene(still(0, 0), approaching(-8.0, 0, 2.0)), ego=0)
    assert counts[STOPPED_EGO] == 1
    assert at_fault(counts) == 0


def test_being_rear_ended_while_moving_is_not_at_fault():
    """Both moving, the other car behind the ego and closing."""
    ego = [(0.0, 0, 0.0), (1.0, 0, 0.0), (2.0, 0, 0.0)]
    other = [(-8.0, 0, 0.0), (-4.0, 0, 0.0), (-1.0, 0, 0.0)]
    counts, _ = ego_collisions(*scene(ego, other), ego=0)
    assert counts[ACTIVE_REAR] == 1
    assert at_fault(counts) == 0


def test_front_contact_with_a_moving_car_is_at_fault():
    """Ego drives forward into a car crossing in front of it."""
    ego = [(0.0, 0, 0.0), (2.0, 0, 0.0), (4.0, 0, 0.0)]
    other = [(7.0, 6.0, -math.pi / 2), (7.0, 3.0, -math.pi / 2), (7.0, 0.2, -math.pi / 2)]
    counts, _ = ego_collisions(*scene(ego, other), ego=0)
    assert counts[ACTIVE_FRONT] == 1
    assert at_fault(counts) == 1


def test_a_track_is_counted_once_however_long_contact_lasts():
    """nuPlan counts new collisions, not timesteps in contact."""
    ego = approaching(0, 0, 1.0, steps=12)
    other = still(4.0, 0, steps=12)
    counts, events = ego_collisions(*scene(ego, other, steps=12), ego=0)
    assert sum(counts) == 1
    assert len(events) == 1


def test_no_contact_is_no_collision():
    counts, events = ego_collisions(*scene(still(0, 0), still(50.0, 50.0)), ego=0)
    assert sum(counts) == 0 and events == []


def test_neighbouring_lanes_are_not_a_collision():
    counts, _ = ego_collisions(*scene(still(0, 0), still(0.0, 3.5)), ego=0)
    assert sum(counts) == 0


def test_invalid_steps_are_skipped():
    position, heading, shape, valid = scene(still(0, 0), still(0.0, 0.5))
    valid[1, :] = False
    counts, _ = ego_collisions(position, heading, shape, valid, ego=0)
    assert sum(counts) == 0
