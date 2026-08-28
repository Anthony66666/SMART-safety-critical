"""Tests for 2D line-of-sight visibility.

The acceptance criteria for this module are that it agrees with an independent
implementation of ray-polygon intersection on random scenes, and that it is
deterministic. Both are asserted here, alongside hand-constructed geometry whose
answer is known by inspection.
"""
import math

import torch

from smart.occlusion.visibility import (
    agent_visibility,
    boxes_to_corners,
    line_of_sight,
    segments_intersect,
    visible_mask,
)


def _cal_polygon_contour(x, y, theta, width, length):
    """Verbatim copy of smart.modules.agent_decoder.cal_polygon_contour.

    Duplicated rather than imported: that module pulls in torch_cluster and the
    whole model stack, which this geometry test has no business depending on.
    The point of the copy is to pin the corner convention -- if the original ever
    changes, this test still encodes what visibility.py was built against.
    """
    left_front = (x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta),
                  y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta))
    right_front = (x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta),
                   y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta))
    right_back = (x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta),
                  y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta))
    left_back = (x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta),
                 y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta))
    return [left_front, right_front, right_back, left_back]


def _segments_cross_parametric(a0, a1, b0, b1):
    """Independent segment intersection via a parametric solve.

    Deliberately different algebra from the orientation test in visibility.py:
    solve a0 + t(a1-a0) = b0 + u(b1-b0) by Cramer's rule and require both
    parameters strictly inside (0, 1).
    """
    r = (a1[0] - a0[0], a1[1] - a0[1])
    s = (b1[0] - b0[0], b1[1] - b0[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return False  # parallel or collinear
    diff = (b0[0] - a0[0], b0[1] - a0[1])
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * r[1] - diff[1] * r[0]) / denom
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def _visibility_reference(origin, boxes):
    """Independent, scalar, loop-based visibility. O(N^2) and obviously correct."""
    corners = [_cal_polygon_contour(*[float(v) for v in row]) for row in boxes]
    fractions = []
    for i, box_corners in enumerate(corners):
        centroid = (float(boxes[i][0]), float(boxes[i][1]))
        samples = list(box_corners) + [centroid]
        seen = 0
        for point in samples:
            blocked = False
            for j, occ in enumerate(corners):
                if j == i:
                    continue
                for k in range(4):
                    if _segments_cross_parametric(origin, point, occ[k], occ[(k + 1) % 4]):
                        blocked = True
                        break
                if blocked:
                    break
            seen += not blocked
        fractions.append(seen / len(samples))
    return fractions


def test_corner_convention_matches_agent_decoder():
    torch.manual_seed(0)
    for _ in range(50):
        x, y = torch.randn(2).tolist()
        theta = float(torch.rand(1) * 2 * math.pi - math.pi)
        width, length = float(torch.rand(1) * 2 + 1), float(torch.rand(1) * 4 + 2)

        expected = torch.tensor(_cal_polygon_contour(x, y, theta, width, length))
        got = boxes_to_corners(*[torch.tensor(v) for v in (x, y, theta, width, length)])
        assert torch.allclose(got, expected, atol=1e-6)


def test_segments_intersect_basic_cases():
    def seg(ax, ay, bx, by, cx, cy, dx, dy):
        t = lambda *v: torch.tensor(v, dtype=torch.float64)
        return bool(segments_intersect(t(ax, ay), t(bx, by), t(cx, cy), t(dx, dy)))

    assert seg(0, 0, 2, 2, 0, 2, 2, 0)        # proper crossing
    assert not seg(0, 0, 1, 1, 2, 2, 3, 3)    # collinear, disjoint
    assert not seg(0, 0, 1, 1, 1, 1, 2, 0)    # touching at an endpoint
    assert not seg(0, 0, 1, 0, 0, 1, 1, 1)    # parallel


def test_no_occluders_means_everything_visible():
    boxes = torch.tensor([[10.0, 0.0, 0.0, 2.0, 4.0],
                          [20.0, 5.0, 1.0, 2.0, 4.0]])
    origin = torch.zeros(2)
    # Occluder set empty: nothing can block, so every sample point is visible.
    no_occluders = torch.zeros(boxes.shape[0], dtype=torch.bool)
    assert torch.equal(agent_visibility(origin, boxes, no_occluders),
                       torch.ones(2))


def test_agent_never_occludes_itself():
    boxes = torch.tensor([[10.0, 0.0, 0.0, 2.0, 4.0]])
    fraction = agent_visibility(torch.zeros(2), boxes)
    # A lone box's far corners are behind its own near face; if self-occlusion
    # leaked in, this would drop below 1.
    assert float(fraction[0]) == 1.0


def test_box_directly_behind_a_wide_occluder_is_hidden():
    # Observer at the origin, a wide van at x=10, a car squarely behind it.
    boxes = torch.tensor([[10.0, 0.0, 0.0, 8.0, 2.0],    # van, broad side on
                          [20.0, 0.0, 0.0, 2.0, 4.0]])   # car directly behind
    fractions = agent_visibility(torch.zeros(2), boxes)
    assert float(fractions[0]) == 1.0
    assert float(fractions[1]) == 0.0
    assert torch.equal(visible_mask(torch.zeros(2), boxes),
                       torch.tensor([True, False]))


def test_box_beside_the_occluder_stays_visible():
    boxes = torch.tensor([[10.0, 0.0, 0.0, 8.0, 2.0],
                          [20.0, 30.0, 0.0, 2.0, 4.0]])  # far off to the side
    fractions = agent_visibility(torch.zeros(2), boxes)
    assert float(fractions[1]) == 1.0


def test_partial_occlusion_is_strictly_between_zero_and_one():
    # Occluder covers part of the target's silhouette but not all of it.
    boxes = torch.tensor([[10.0, 1.6, 0.0, 3.0, 2.0],
                          [20.0, 0.0, 0.0, 6.0, 4.0]])
    fraction = float(agent_visibility(torch.zeros(2), boxes)[1])
    assert 0.0 < fraction < 1.0


def test_occluder_mask_excludes_selected_agents_from_blocking():
    boxes = torch.tensor([[10.0, 0.0, 0.0, 8.0, 2.0],
                          [20.0, 0.0, 0.0, 2.0, 4.0]])
    # Same geometry as the "hidden" case, but the van no longer occludes.
    mask = torch.tensor([False, True])
    fractions = agent_visibility(torch.zeros(2), boxes, occluder_mask=mask)
    assert float(fractions[1]) == 1.0


def test_matches_independent_implementation_on_random_scenes():
    torch.manual_seed(20260828)
    for _ in range(30):
        num_agents = int(torch.randint(2, 9, (1,)))
        boxes = torch.stack([
            torch.rand(num_agents) * 60 - 30,             # x
            torch.rand(num_agents) * 60 - 30,             # y
            torch.rand(num_agents) * 2 * math.pi - math.pi,
            torch.rand(num_agents) * 1.5 + 1.5,           # width
            torch.rand(num_agents) * 3.0 + 3.0,           # length
        ], dim=-1).double()
        origin = (torch.rand(2).double() * 10 - 5)

        got = agent_visibility(origin, boxes).tolist()
        expected = _visibility_reference((float(origin[0]), float(origin[1])), boxes)
        assert got == expected


def test_deterministic():
    torch.manual_seed(1)
    boxes = torch.rand(12, 5) * torch.tensor([40.0, 40.0, 6.0, 2.0, 4.0])
    origin = torch.zeros(2)
    first = agent_visibility(origin, boxes)
    for _ in range(5):
        assert torch.equal(agent_visibility(origin, boxes), first)


def test_line_of_sight_handles_empty_occluder_set():
    points = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    clear = line_of_sight(torch.zeros(2), points, torch.zeros(0, 4, 2))
    assert bool(clear.all())
