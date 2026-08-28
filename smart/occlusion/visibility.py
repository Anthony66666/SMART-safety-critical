"""Line-of-sight visibility of agents under 2D occlusion.

Closed-loop planning benchmarks hand the planner ground-truth boxes for every
agent in the scene. Real perception does not work that way: a parked van, a
truck, or a bus hides whatever is behind it, and sight-line obstruction is a
high-frequency cause in real crash reports. This module computes which agents
an observer can actually see.

The model is deliberately the cheapest one that captures the dominant real
cause: 2D line of sight from a sensor origin to an agent's bounding box, with
other agents' boxes as the only occluders. It has **no free parameters** -- no
sensor range, no field of view, no detection probability -- so a visibility
result is reproducible and cannot be tuned to favour a conclusion. Sensor range
and FOV, if ever wanted, belong in a separate layer on top of this one.

Corner convention matches `smart.modules.agent_decoder.cal_polygon_contour`
(left-front, right-front, right-back, left-back). That function takes Python
floats, so it cannot run over a whole scene at once; `boxes_to_corners` here is
the vectorised equivalent and `tests/test_visibility.py` asserts the two agree.
"""
import torch


def boxes_to_corners(x: torch.Tensor,
                     y: torch.Tensor,
                     heading: torch.Tensor,
                     width: torch.Tensor,
                     length: torch.Tensor) -> torch.Tensor:
    """Corners of oriented bounding boxes.

    Args:
        x, y, heading, width, length: (...,) broadcastable tensors. `heading` is
            in radians, `length` runs along the heading, `width` across it.

    Returns:
        (..., 4, 2) corners ordered left-front, right-front, right-back,
        left-back -- the same order as `cal_polygon_contour`.
    """
    cos, sin = torch.cos(heading), torch.sin(heading)
    half_l, half_w = 0.5 * length, 0.5 * width

    # Offsets along (heading, left-of-heading) for each corner, in that order.
    longitudinal = torch.stack([half_l, half_l, -half_l, -half_l], dim=-1)
    lateral = torch.stack([half_w, -half_w, -half_w, half_w], dim=-1)

    cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)
    corner_x = x.unsqueeze(-1) + longitudinal * cos - lateral * sin
    corner_y = y.unsqueeze(-1) + longitudinal * sin + lateral * cos
    return torch.stack([corner_x, corner_y], dim=-1)


def _cross(origin: torch.Tensor,
           a: torch.Tensor,
           b: torch.Tensor) -> torch.Tensor:
    """z-component of (a - origin) x (b - origin); sign gives the turn direction."""
    return ((a[..., 0] - origin[..., 0]) * (b[..., 1] - origin[..., 1]) -
            (a[..., 1] - origin[..., 1]) * (b[..., 0] - origin[..., 0]))


def segments_intersect(a0: torch.Tensor,
                       a1: torch.Tensor,
                       b0: torch.Tensor,
                       b1: torch.Tensor) -> torch.Tensor:
    """Proper intersection test for segments a0-a1 and b0-b1.

    This is the *strict* test: segments that merely touch at an endpoint, or lie
    collinear, do not count as intersecting. That choice is deliberate and
    conservative for occlusion -- a sight line that exactly grazes a corner is
    reported as unblocked, so the module never claims more occlusion than the
    geometry forces.

    Args:
        a0, a1, b0, b1: (..., 2) broadcastable segment endpoints.

    Returns:
        (...,) bool tensor.
    """
    d1 = _cross(b0, b1, a0)
    d2 = _cross(b0, b1, a1)
    d3 = _cross(a0, a1, b0)
    d4 = _cross(a0, a1, b1)
    return ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))


def _box_edges(corners: torch.Tensor) -> torch.Tensor:
    """(N, 4, 2) corners -> (N, 4, 2, 2) edges, each as its two endpoints."""
    return torch.stack([corners, corners.roll(shifts=-1, dims=-2)], dim=-2)


def line_of_sight(origin: torch.Tensor,
                  points: torch.Tensor,
                  occluder_corners: torch.Tensor,
                  ignore: torch.Tensor = None) -> torch.Tensor:
    """Whether each point is reachable from `origin` by an unblocked segment.

    Args:
        origin: (2,) sensor position.
        points: (N, K, 2) query points.
        occluder_corners: (M, 4, 2) occluding boxes.
        ignore: optional (N, M) bool; True marks an occluder that must not block
            the corresponding row of `points` (used to stop a box from occluding
            itself).

    Returns:
        (N, K) bool tensor, True where the sight line is clear.
    """
    num_agents, num_points = points.shape[0], points.shape[1]
    if occluder_corners.shape[0] == 0:
        return points.new_ones((num_agents, num_points), dtype=torch.bool)

    edges = _box_edges(occluder_corners)                       # (M, 4, 2, 2)
    edge_start = edges[..., 0, :].reshape(1, 1, -1, 2)         # (1, 1, M*4, 2)
    edge_end = edges[..., 1, :].reshape(1, 1, -1, 2)

    origin_b = origin.reshape(1, 1, 1, 2)
    points_b = points.unsqueeze(2)                             # (N, K, 1, 2)

    blocked = segments_intersect(origin_b, points_b, edge_start, edge_end)

    if ignore is not None:
        # (N, M) -> (N, 1, M*4): every edge inherits its box's ignore flag.
        keep = ~ignore.repeat_interleave(4, dim=-1).unsqueeze(1)
        blocked = blocked & keep

    return ~blocked.any(dim=-1)


def agent_visibility(sensor_origin: torch.Tensor,
                     boxes: torch.Tensor,
                     occluder_mask: torch.Tensor = None) -> torch.Tensor:
    """Fraction of each agent's silhouette that is visible from `sensor_origin`.

    Each agent is sampled at its four corners plus its centroid, and the returned
    fraction is how many of those five points have a clear sight line. Reporting
    a fraction rather than a flag keeps the module parameter-free: callers that
    want a boolean can take `fraction > 0` (any part visible), which is the
    convention `visible_mask` uses.

    An agent never occludes itself. Agents excluded by `occluder_mask` are still
    *evaluated* for visibility -- they just do not block anyone.

    Args:
        sensor_origin: (2,) observer position.
        boxes: (N, 5) rows of [x, y, heading, width, length].
        occluder_mask: optional (N,) bool selecting which agents act as
            occluders. Defaults to all of them. Use it to drop the ego's own box,
            or to restrict occlusion to vehicle-sized agents.

    Returns:
        (N,) float tensor in [0, 1].
    """
    if boxes.shape[0] == 0:
        return boxes.new_zeros((0,))

    corners = boxes_to_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2],
                               boxes[:, 3], boxes[:, 4])        # (N, 4, 2)
    centroid = boxes[:, None, :2]
    sample_points = torch.cat([corners, centroid], dim=1)       # (N, 5, 2)

    num_agents = boxes.shape[0]
    if occluder_mask is None:
        occluder_index = torch.arange(num_agents, device=boxes.device)
    else:
        occluder_index = occluder_mask.nonzero(as_tuple=True)[0]

    occluders = corners[occluder_index]
    # An agent must not occlude itself: mark, for each row, the occluder column
    # that is that same agent.
    ignore = (torch.arange(num_agents, device=boxes.device)[:, None] ==
              occluder_index[None, :])

    clear = line_of_sight(sensor_origin, sample_points, occluders, ignore=ignore)
    return clear.to(boxes.dtype).mean(dim=-1)


def visible_mask(sensor_origin: torch.Tensor,
                 boxes: torch.Tensor,
                 occluder_mask: torch.Tensor = None) -> torch.Tensor:
    """Agents with any part of their silhouette visible from `sensor_origin`.

    Thin wrapper over `agent_visibility` with the partial-visibility convention:
    an agent glimpsed through a gap counts as seen. Args are as in
    `agent_visibility`; returns an (N,) bool tensor.
    """
    return agent_visibility(sensor_origin, boxes, occluder_mask) > 0
