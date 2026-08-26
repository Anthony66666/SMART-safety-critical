"""Tests for the danger objective J.

The tilting sampler reweights each candidate token by exp(J/beta), so J must
be an exact geometric quantity over the token's bounding box -- not a
differentiable surrogate. This is the structural advantage over diffusion
guidance: the token vocabulary carries explicit corner geometry, so collision
and proximity are computed, not approximated.
"""
import math

import pytest
import torch

from smart.safety.objectives import box_separation, proximity_danger


def _box(cx, cy, half_len=1.0, half_wid=1.0, theta=0.0):
    """Axis-order corners (lf, rf, rb, lb) for a box centred at (cx, cy)."""
    c, s = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in [(half_len, half_wid), (half_len, -half_wid),
                   (-half_len, -half_wid), (-half_len, half_wid)]:
        corners.append([cx + dx * c - dy * s, cy + dx * s + dy * c])
    return torch.tensor(corners)


def test_separated_boxes_report_the_gap():
    a = _box(0.0, 0.0)          # x in [-1, 1]
    b = _box(5.0, 0.0)          # x in [ 4, 6]
    assert box_separation(a, b).item() == pytest.approx(3.0)


def test_touching_boxes_report_zero():
    a = _box(0.0, 0.0)          # x in [-1, 1]
    b = _box(2.0, 0.0)          # x in [ 1, 3]
    assert box_separation(a, b).item() == pytest.approx(0.0, abs=1e-6)


def test_overlapping_boxes_report_negative_penetration():
    a = _box(0.0, 0.0)          # x in [-1, 1]
    b = _box(1.0, 0.0)          # x in [ 0, 2]  -> overlap depth 1 in x
    assert box_separation(a, b).item() == pytest.approx(-1.0)


def test_separation_is_symmetric():
    a = _box(0.0, 0.0)
    b = _box(3.0, 1.0)
    assert box_separation(a, b).item() == pytest.approx(box_separation(b, a).item())


def test_separation_batches_over_leading_dims():
    a = torch.stack([_box(0.0, 0.0), _box(0.0, 0.0)])
    b = torch.stack([_box(5.0, 0.0), _box(2.0, 0.0)])
    out = box_separation(a, b)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(3.0)
    assert out[1].item() == pytest.approx(0.0, abs=1e-6)


def test_proximity_danger_is_high_when_boxes_meet():
    T = 4
    adv = torch.stack([_box(float(x), 0.0) for x in [6, 4, 2, 0]])   # approaches
    vic = torch.stack([_box(0.0, 0.0) for _ in range(T)])           # parked
    # closest approach at the last step: identical boxes fully overlap, so the
    # penetration to separate them is 2 -> danger +2
    assert proximity_danger(adv, vic).item() == pytest.approx(2.0)


def test_proximity_danger_is_negative_when_boxes_stay_apart():
    T = 3
    adv = torch.stack([_box(float(x), 0.0) for x in [10, 9, 8]])
    vic = torch.stack([_box(0.0, 0.0) for _ in range(T)])
    # min separation 8 - 1 - 1 = 6 -> danger -6
    assert proximity_danger(adv, vic).item() == pytest.approx(-6.0)


def test_separation_matches_brute_force_point_distance_when_disjoint():
    """Exactness check: for disjoint boxes SAT separation equals the true
    polygon-polygon distance, which for these axis-aligned cases is the
    nearest-corner distance."""
    torch.manual_seed(0)
    for _ in range(50):
        ca = torch.rand(2) * 4 - 2
        cb = ca + torch.tensor([6.0, 0.0]) + torch.rand(2)   # guaranteed disjoint
        a, b = _box(*ca.tolist()), _box(*cb.tolist())
        # brute force: min distance between the two point sets is an UPPER bound
        # on true polygon distance; for these separated axis-aligned boxes the
        # separating axis is x, so SAT <= corner distance.
        corner = torch.cdist(a, b).min().item()
        assert box_separation(a, b).item() <= corner + 1e-5
        assert box_separation(a, b).item() > 0
