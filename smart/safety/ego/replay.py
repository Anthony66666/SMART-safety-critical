"""Log-replay ego policy.

The control condition for every experiment: the ego does exactly what it did
in the recorded scenario. If an injected replay rollout does not reproduce the
log, the injection machinery itself is broken.
"""
import torch

from smart.safety.ego.base import EgoPlanner, PlanningContext


class ReplayPlanner(EgoPlanner):
    """Replays a logged ego trajectory."""

    def __init__(self, logged_poses: torch.Tensor, shift: int = 5) -> None:
        """
        Args:
            logged_poses: (num_future_steps, 3) logged (x, y, heading).
            shift: timesteps consumed per rollout step.
        """
        self.logged_poses = logged_poses
        self.shift = shift

    def plan(self, ctx: PlanningContext) -> torch.Tensor:
        start = ctx.step * self.shift
        return self.logged_poses[start:start + self.shift]
