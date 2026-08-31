"""Whether two oriented boxes actually overlap.

The zero-shot rollout was scored with a circle of half the box length, which
calls two 5 m cars in neighbouring lanes a collision because their centres are
under 5 m apart. On a six-lane arterial that is most of the scene, and it put
the log's own collision rate at 19%, which is not a fact about nuPlan.

This is the real test: the separating axis theorem on two rectangles. Two
convex shapes miss each other exactly when some axis separates their
projections, and for rectangles only the four edge normals need checking. The
result is exact, has no tolerance to tune, and is the same test a nuPlan metric
would apply.

Boxes are (x, y, heading, width, length) throughout, matching
smart.occlusion.visibility.
"""
import torch

from smart.occlusion.visibility import boxes_to_corners


def _axes(corners: torch.Tensor) -> torch.Tensor:
    """Unit normals of a rectangle's two distinct edge directions.

    Opposite edges of a rectangle are parallel, so four edges give two axes.

    Args:
        corners: (..., 4, 2)

    Returns:
        (..., 2, 2) unit normals.
    """
    edges = torch.stack([corners[..., 1, :] - corners[..., 0, :],
                         corners[..., 3, :] - corners[..., 0, :]], dim=-2)
    normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _separated(corners_a: torch.Tensor, corners_b: torch.Tensor,
               axes: torch.Tensor) -> torch.Tensor:
    """True where some axis separates the two boxes' projections."""
    # (..., axis, corner)
    projected_a = torch.einsum('...ck,...ak->...ac', corners_a, axes)
    projected_b = torch.einsum('...ck,...ak->...ac', corners_b, axes)
    gap = ((projected_a.min(dim=-1).values > projected_b.max(dim=-1).values)
           | (projected_b.min(dim=-1).values > projected_a.max(dim=-1).values))
    return gap.any(dim=-1)


def boxes_overlap(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Elementwise overlap of two broadcastable sets of oriented boxes.

    Args:
        boxes_a, boxes_b: (..., 5) as (x, y, heading, width, length).

    Returns:
        (...,) bool. Boxes that merely touch count as overlapping, since a
        shared edge means the vehicles are in contact.
    """
    corners_a = boxes_to_corners(boxes_a[..., 0], boxes_a[..., 1], boxes_a[..., 2],
                                 boxes_a[..., 3], boxes_a[..., 4])
    corners_b = boxes_to_corners(boxes_b[..., 0], boxes_b[..., 1], boxes_b[..., 2],
                                 boxes_b[..., 3], boxes_b[..., 4])
    axes = torch.cat([_axes(corners_a), _axes(corners_b)], dim=-2)
    return ~_separated(corners_a, corners_b, axes)


def overlap_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Which boxes in one set overlap which others.

    Args:
        boxes: (N, 5).

    Returns:
        (N, N) bool, symmetric, diagonal false.

    A circumradius test rejects most pairs before the full check, which matters
    because Las Vegas scenarios carry a few hundred agents and the pairwise
    corner projections are the expensive part.
    """
    count = len(boxes)
    overlap = torch.zeros(count, count, dtype=torch.bool, device=boxes.device)
    if count < 2:
        return overlap

    radius = 0.5 * torch.hypot(boxes[:, 3], boxes[:, 4])
    centres = boxes[:, :2]
    near = (torch.cdist(centres, centres) <= radius[:, None] + radius[None, :])
    near.fill_diagonal_(False)
    pairs = torch.triu(near).nonzero(as_tuple=False)
    if not len(pairs):
        return overlap

    hit = boxes_overlap(boxes[pairs[:, 0]], boxes[pairs[:, 1]])
    struck = pairs[hit]
    overlap[struck[:, 0], struck[:, 1]] = True
    overlap[struck[:, 1], struck[:, 0]] = True
    return overlap


def collision_rate(trajectory: torch.Tensor, heading: torch.Tensor,
                   shape: torch.Tensor, valid: torch.Tensor) -> float:
    """Fraction of agent-timesteps in which an agent overlaps another.

    Args:
        trajectory: (N, T, 2) positions.
        heading: (N, T).
        shape: (N, 3) as (length, width, height), constant over time.
        valid: (N, T) bool.

    Returns:
        Collisions divided by live agent-timesteps.
    """
    hits = 0
    total = 0
    for t in range(trajectory.shape[1]):
        live = valid[:, t].nonzero(as_tuple=True)[0]
        if len(live) < 2:
            continue
        boxes = torch.stack([trajectory[live, t, 0], trajectory[live, t, 1],
                             heading[live, t], shape[live, 1], shape[live, 0]], dim=-1)
        hits += int(overlap_matrix(boxes).any(dim=1).sum())
        total += len(live)
    return hits / max(total, 1)
