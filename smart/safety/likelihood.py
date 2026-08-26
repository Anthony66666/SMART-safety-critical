"""Exact sequence log-likelihood accounting for token rollouts.

SMART samples each agent's next token from the top-k of a full 2048-way
softmax. That makes two distributions relevant, and they are not the same:

  p  the model distribution -- the full softmax
  q  the sampling distribution -- p restricted to the top-k and renormalised

Realism is measured under p. Importance weights are w = p / q. Tracking only
one of them silently corrupts both, so this class accumulates both.
"""
from typing import Optional

import torch


class SequenceLikelihood:
    """Accumulates per-agent sequence log-likelihood over rollout steps."""

    def __init__(self, num_agents: int, device: Optional[torch.device] = None) -> None:
        self.num_agents = num_agents
        self._log_p = torch.zeros(num_agents, device=device)
        self._log_q = torch.zeros(num_agents, device=device)

    def update(self,
               p_chosen: torch.Tensor,
               q_chosen: torch.Tensor,
               valid: torch.Tensor) -> None:
        """Fold in one rollout step.

        q is passed in rather than derived from p: top-k truncation happens to
        make q a renormalisation of p, but temperature, tilting and guidance do
        not, and inferring q would silently corrupt every importance weight
        the moment the sampler stops being plain truncation.

        Args:
            p_chosen: (num_agents,) model probability of the sampled token.
            q_chosen: (num_agents,) probability the sampler actually used.
            valid: (num_agents,) bool; invalid agents contribute nothing.
        """
        zero = torch.zeros_like(p_chosen)
        step_log_p = torch.log(p_chosen)
        step_log_q = torch.log(q_chosen)
        self._log_p = self._log_p + torch.where(valid, step_log_p, zero)
        self._log_q = self._log_q + torch.where(valid, step_log_q, zero)

    @property
    def log_p(self) -> torch.Tensor:
        """(num_agents,) log-likelihood under the model distribution."""
        return self._log_p

    @property
    def log_q(self) -> torch.Tensor:
        """(num_agents,) log-likelihood under the sampling distribution."""
        return self._log_q
