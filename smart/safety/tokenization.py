"""Projecting continuous poses back onto the discrete token vocabulary.

When the ego is driven by an external planner it leaves the token grid, but
the rollout still needs a token embedding for it at the next step. The
projection here mirrors the metric used when the dataset was tokenised
(smart/datasets/preprocess.py :: match_token): candidate tokens are rendered
into the world frame from the previous pose, and the one whose bounding-box
corners are closest on average to the target box wins.

Preprocessing uses fixed nominal box sizes rather than each agent's true
extent; VEH_WIDTH / VEH_LENGTH reproduce that choice for shift > 2.
"""
import torch

VEH_WIDTH, VEH_LENGTH = 2.0, 4.8


def polygon_contour(pos: torch.Tensor,
                    heading: torch.Tensor,
                    width: float,
                    length: float) -> torch.Tensor:
    """(N, 4, 2) box corners as (left_front, right_front, right_back, left_back)."""
    cos, sin = heading.cos(), heading.sin()
    dx_l, dy_l = 0.5 * length * cos, 0.5 * length * sin
    dx_w, dy_w = 0.5 * width * sin, 0.5 * width * cos
    x, y = pos[:, 0], pos[:, 1]
    corners = torch.stack([
        torch.stack([x + dx_l - dx_w, y + dy_l + dy_w], dim=-1),
        torch.stack([x + dx_l + dx_w, y + dy_l - dy_w], dim=-1),
        torch.stack([x - dx_l + dx_w, y - dy_l - dy_w], dim=-1),
        torch.stack([x - dx_l - dx_w, y - dy_l + dy_w], dim=-1),
    ], dim=1)
    return corners


def nearest_token(prev_pos: torch.Tensor,
                  prev_heading: torch.Tensor,
                  target_pos: torch.Tensor,
                  target_heading: torch.Tensor,
                  token_contours: torch.Tensor,
                  width: float = VEH_WIDTH,
                  length: float = VEH_LENGTH) -> torch.Tensor:
    """Index of the token whose outcome best matches a desired pose.

    Args:
        prev_pos: (N, 2) pose the token is applied from.
        prev_heading: (N,) heading the token is applied from.
        target_pos: (N, 2) desired resulting position.
        target_heading: (N,) desired resulting heading.
        token_contours: (K, 4, 2) token endpoint contours, agent-local.
        width, length: nominal box size used for matching.

    Returns:
        (N,) token indices.
    """
    cos, sin = prev_heading.cos(), prev_heading.sin()
    rot = torch.stack([torch.stack([cos, sin], dim=-1),
                       torch.stack([-sin, cos], dim=-1)], dim=-2)

    world = torch.einsum('kcd,nde->nkce', token_contours, rot)
    world = world + prev_pos[:, None, None, :]

    target = polygon_contour(target_pos, target_heading, width, length)
    dist = (target[:, None] - world).pow(2).sum(-1).sqrt().mean(-1)
    return dist.argmin(dim=-1)
