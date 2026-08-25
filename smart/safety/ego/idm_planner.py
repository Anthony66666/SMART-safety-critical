"""Rule-based ego policy: IDM longitudinal control, lane-holding lateral.

The first policy under test. It is deliberately simple and fully
reproducible -- no learned weights, no external dependencies -- so that any
failure it exhibits is attributable to the scenario rather than to opaque
policy behaviour.
"""
import math

import torch

from smart.safety.ego.base import EgoPlanner, PlanningContext


def idm_acceleration(speed: float,
                     gap: float,
                     lead_speed: float,
                     desired_speed: float,
                     min_gap: float = 2.0,
                     time_headway: float = 1.5,
                     max_accel: float = 1.5,
                     comfort_decel: float = 2.0,
                     delta: float = 4.0) -> float:
    """Intelligent Driver Model acceleration.

    Args:
        speed: current ego speed (m/s).
        gap: bumper-to-bumper distance to the lead vehicle (m); inf if none.
        lead_speed: lead vehicle speed (m/s).
        desired_speed: free-road target speed (m/s).

    Returns:
        Acceleration in m/s^2.
    """
    free_road = 1.0 - (speed / desired_speed) ** delta

    if math.isinf(gap):
        return max_accel * free_road

    approach_rate = speed - lead_speed
    desired_gap = (min_gap
                   + speed * time_headway
                   + speed * approach_rate / (2.0 * math.sqrt(max_accel * comfort_decel)))
    interaction = (desired_gap / max(gap, 1e-3)) ** 2
    return max_accel * (free_road - interaction)


class IDMPlanner(EgoPlanner):
    """Drives the ego forward along its current heading under IDM control."""

    def __init__(self,
                 desired_speed: float = 15.0,
                 shift: int = 5,
                 dt: float = 0.1,
                 **idm_params) -> None:
        self.desired_speed = desired_speed
        self.shift = shift
        self.dt = dt
        self.idm_params = idm_params

    def plan(self, ctx: PlanningContext) -> torch.Tensor:
        x, y, heading = ctx.ego_state[0], ctx.ego_state[1], ctx.ego_state[2]
        speed = float(ctx.ego_speed)

        gap, lead_speed = self._lead_vehicle(ctx, heading)
        accel = idm_acceleration(speed, gap, lead_speed, self.desired_speed,
                                 **self.idm_params)

        poses = []
        pos = torch.stack([x, y])
        direction = torch.stack([heading.cos(), heading.sin()])
        for _ in range(self.shift):
            speed = max(0.0, speed + accel * self.dt)
            pos = pos + direction * speed * self.dt
            poses.append(torch.cat([pos, heading.reshape(1)]))
        return torch.stack(poses)

    def _lead_vehicle(self, ctx: PlanningContext, heading: torch.Tensor):
        """Nearest neighbour ahead within a narrow corridor along the heading."""
        if ctx.neighbor_states.numel() == 0:
            return math.inf, 0.0

        rel = ctx.neighbor_states[:, :2] - ctx.ego_state[:2]
        direction = torch.stack([heading.cos(), heading.sin()])
        along = rel @ direction
        lateral = (rel - along[:, None] * direction).norm(dim=-1)

        ahead = (along > 1e-3) & (lateral < 2.0)
        if not ahead.any():
            return math.inf, 0.0
        return float(along[ahead].min()), 0.0
