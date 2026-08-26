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
    # Subtract the row max for numerical stability; it cancels on normalisation.
    scaled = (danger - danger.amax(dim=-1, keepdim=True)) / beta
    return weights * torch.exp(scaled)
