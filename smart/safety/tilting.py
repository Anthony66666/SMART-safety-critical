"""Danger-tilted sampling.

The optimal danger-vs-realism sampler under a KL budget is the exponential
tilt q proportional to p * exp(J / beta): a single knob beta trades realism
for danger, with the model distribution p recovered as beta -> infinity.

Because token danger J is computed exactly over the candidate's bounding box
(smart/safety/objectives.py), the tilt reweights probabilities on the token
manifold rather than pushing samples off it -- there is no gradient, no
surrogate, and no off-manifold drift the way there is for diffusion guidance.
"""
import torch


def tilt_weights(weights: torch.Tensor,
                 danger: torch.Tensor,
                 beta: float) -> torch.Tensor:
    """Reweight sampling weights by exp(danger / beta).

    Args:
        weights: (..., K) non-negative sampling weights (the model's p, or an
            already-truncated q). Zero-mass candidates stay zero.
        danger: (..., K) danger J per candidate.
        beta: temperature of the tilt; large recovers `weights`, small
            concentrates on maximum danger.

    Returns:
        (..., K) tilted weights, unnormalised.
    """
    # Reference the max over candidates that can actually be drawn (positive
    # weight). A top-k restriction may have zeroed the most dangerous token; if
    # its danger were still the reference, every survivor would underflow to
    # zero in float32 and multinomial would have nothing to sample.
    drawable = danger.masked_fill(weights <= 0, float('-inf'))
    ref = drawable.amax(dim=-1, keepdim=True)
    # Survivors have danger <= ref so the exponent is <= 0 and cannot overflow.
    # Zeroed candidates keep danger > ref, which would give 0 * inf = nan, so
    # they are forced back to zero explicitly.
    tilted = weights * torch.exp((danger - ref) / beta)
    return torch.where(weights > 0, tilted, torch.zeros_like(weights))


def restrict_to_topk(weights: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all but the k largest weights along the last dim.

    Full-support sampling reaches low-probability tail tokens that produce
    physically jerky motion even before any tilt. Restricting the tilt to the
    model's top-k candidates keeps it on plausible motion while still leaving
    room to pick the dangerous option among them.

    Args:
        weights: (..., K) non-negative weights.
        k: number of candidates to keep.

    Returns:
        (..., K) weights with all but the top-k zeroed.
    """
    if k >= weights.shape[-1]:
        return weights
    kth = weights.topk(k, dim=-1).values[..., -1:]
    return torch.where(weights >= kth, weights, torch.zeros_like(weights))
