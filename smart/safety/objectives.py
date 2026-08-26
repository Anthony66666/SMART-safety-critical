"""Danger objectives for tilted sampling.

The tilting sampler draws candidate tokens with weight p * exp(J / beta), so J
has to be evaluated exactly over each candidate's bounding box. The token
vocabulary carries explicit corner geometry, so danger is a computed quantity
here, never a differentiable surrogate -- this is the structural reason the
sampler never has to leave the token manifold the way gradient guidance does.
"""
import torch


def _axes(box: torch.Tensor) -> torch.Tensor:
    """Unit outward edge normals of a convex polygon, shape (..., 4, 2)."""
    edges = box.roll(-1, dims=-2) - box
    normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def box_separation(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Signed separation between two oriented boxes (Separating Axis Theorem).

    Positive is the gap between disjoint boxes; zero is touching; negative is
    penetration depth. Exact for convex polygons.

    Args:
        box_a, box_b: (..., 4, 2) corner coordinates, broadcastable.

    Returns:
        (...) signed separation.
    """
    axes = torch.cat([_axes(box_a), _axes(box_b)], dim=-2)          # (..., 8, 2)
    proj_a = torch.einsum('...cd,...ad->...ac', box_a, axes)         # (..., 8, 4)
    proj_b = torch.einsum('...cd,...ad->...ac', box_b, axes)
    # Gap along each axis: how far the projections are from overlapping.
    gap = torch.maximum(proj_b.amin(-1) - proj_a.amax(-1),
                        proj_a.amin(-1) - proj_b.amax(-1))           # (..., 8)
    # SAT: the axis of maximum separation decides the whole relationship.
    return gap.amax(-1)


def proximity_danger(adversary: torch.Tensor,
                     victim: torch.Tensor) -> torch.Tensor:
    """Danger of an adversary trajectory against a victim trajectory.

    High when the two come close or overlap, low when they stay apart. Defined
    as the negative minimum separation over the block, so a grazing contact
    scores near zero and a collision scores positive.

    Args:
        adversary: (..., T, 4, 2) adversary box per timestep.
        victim: (..., T, 4, 2) victim box per timestep.

    Returns:
        (...) danger scalar.
    """
    sep = box_separation(adversary, victim)                         # (..., T)
    return -sep.amin(dim=-1)
