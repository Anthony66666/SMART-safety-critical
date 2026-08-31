"""Tests for oriented box overlap."""
import math

import pytest
import torch

from smart.metrics.collision import boxes_overlap, collision_rate, overlap_matrix


def box(x, y, heading=0.0, width=2.0, length=5.0):
    return torch.tensor([x, y, heading, width, length])


def test_a_box_overlaps_itself():
    assert bool(boxes_overlap(box(0, 0), box(0, 0)))


def test_boxes_far_apart_do_not_overlap():
    assert not bool(boxes_overlap(box(0, 0), box(100, 0)))


def test_cars_in_neighbouring_lanes_do_not_overlap():
    """The failure the circle test made. Centres 3.5 m apart, boxes 2 m wide."""
    assert not bool(boxes_overlap(box(0, 0), box(0, 3.5)))


def test_cars_nose_to_tail_do_not_overlap():
    """5 m long, centres 5.5 m apart along the heading: half a metre of gap."""
    assert not bool(boxes_overlap(box(0, 0), box(5.5, 0)))


def test_cars_nose_to_tail_touching_do_overlap():
    assert bool(boxes_overlap(box(0, 0), box(4.9, 0)))


def test_rotation_matters():
    """Separated end to end, but overlapping once one car turns across."""
    assert not bool(boxes_overlap(box(0, 0), box(0, 3.0)))
    assert bool(boxes_overlap(box(0, 0), box(0, 3.0, heading=math.pi / 2)))


def test_diagonal_overlap_a_circle_test_would_miss():
    """Two boxes crossing at 45 degrees, centres apart but corners through."""
    assert bool(boxes_overlap(box(0, 0), box(3.0, 1.0, heading=math.pi / 4)))


def test_overlap_matrix_is_symmetric_with_empty_diagonal():
    boxes = torch.stack([box(0, 0), box(4.0, 0), box(50, 50)])
    matrix = overlap_matrix(boxes)
    assert bool((matrix == matrix.T).all())
    assert not bool(matrix.diagonal().any())
    assert bool(matrix[0, 1]) and not bool(matrix[0, 2])


def test_overlap_matrix_handles_degenerate_sizes():
    assert overlap_matrix(torch.zeros(0, 5)).shape == (0, 0)
    assert not overlap_matrix(torch.stack([box(0, 0)])).any()


def test_collision_rate_counts_agent_timesteps():
    # two agents, two steps; they overlap at the second step only
    trajectory = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                               [[50.0, 0.0], [3.0, 0.0]]])
    heading = torch.zeros(2, 2)
    shape = torch.tensor([[5.0, 2.0, 1.5], [5.0, 2.0, 1.5]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    assert collision_rate(trajectory, heading, shape, valid) == pytest.approx(0.5)


def test_collision_rate_ignores_invalid_agents():
    trajectory = torch.zeros(2, 1, 2)
    heading = torch.zeros(2, 1)
    shape = torch.tensor([[5.0, 2.0, 1.5], [5.0, 2.0, 1.5]])
    valid = torch.tensor([[True], [False]])
    assert collision_rate(trajectory, heading, shape, valid) == 0.0
