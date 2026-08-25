"""Interface for the ego policy under test.

The rollout advances in blocks of `shift` timesteps (0.5 s at 10 Hz), so a
planner is asked for `shift` future poses at a time rather than one.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class PlanningContext:
    """What the ego observes at one rollout step.

    Attributes:
        step: rollout step index, counted from the start of the future.
        ego_state: (3,) current ego pose as (x, y, heading).
        neighbor_states: (num_neighbors, 3) other agents' poses.
        ego_speed: current ego speed in m/s.
    """
    step: int
    ego_state: torch.Tensor
    neighbor_states: torch.Tensor
    ego_speed: float = 0.0


class EgoPlanner(ABC):
    """Produces the ego's motion for the next rollout step."""

    @abstractmethod
    def plan(self, ctx: PlanningContext) -> torch.Tensor:
        """Return a (shift, 3) tensor of (x, y, heading) poses."""
