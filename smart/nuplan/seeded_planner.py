"""Make a stochastic planner reproducible, so a paired comparison means something.

Flow Planner draws its initial noise from the global generator on every call --
`x_init = torch.randn(...)` in forward_inference -- so the same scenario run
twice gives different trajectories. Measured on one scenario in a single
process: 0.288 m apart after 299 steps. Across a full val14 run the effect is
much larger, because ray reuses workers and the order they receive scenarios
in decides the generator state each one starts from: two identical
configurations differ on 621 of 1118 scenarios, by 0.78 points overall, with
seven scenarios falling from full marks to zero on noise alone.

That is fatal for this benchmark, whose entire claim is a difference between
two conditions of about that size. So the noise is pinned: before each planner
call the generator is seeded from the step index, which makes the trajectory a
deterministic function of the observations and nothing else.

Seeding by step rather than per scenario is deliberate. Both conditions then
draw the *same* noise at step k, so a difference between them can only come
from what the planner was shown -- common random numbers, the standard
variance-reduction pairing. It also means the noise sequence repeats across
scenarios, which is harmless: the scenarios are independent and the noise is
isotropic.

This is an intervention on the planner and has to be declared with any result.
It does not change what the planner computes, only which of its equally likely
samples it computes.
"""
from typing import Type

import torch

from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class SeededPlanner(AbstractPlanner):
    """Wraps a planner and fixes its random draws to the simulation step.

    Args:
        planner: the planner to wrap. A deterministic one is unaffected.
        seed: base added to the step index. Change it to draw a different but
            equally reproducible sample, which is how to measure how much of a
            result is sampling luck.
    """

    def __init__(self, planner, seed: int = 0):
        # nuPlan resolves `_target_` to a class and reads `requires_scenario`
        # off it before instantiating, so this has to be the target rather than
        # a factory function -- which means building the inner planner here.
        # Under `_recursive_: false` it arrives as config; passed directly from
        # Python it arrives already built.
        if not isinstance(planner, AbstractPlanner):
            from hydra.utils import instantiate
            planner = instantiate(planner)
        self._planner = planner
        self._seed = seed

    def name(self) -> str:
        """Inherited, see superclass."""
        return f'seeded_{self._planner.name()}'

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return self._planner.observation_type()

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.manual_seed(self._seed)
        self._planner.initialize(initialization)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Seeds both CPU and CUDA generators, which is where the sampler draws.
        torch.manual_seed(self._seed + current_input.iteration.index)
        return self._planner.compute_planner_trajectory(current_input)

    def generate_planner_report(self, clear_stats: bool = True):
        """Inherited, see superclass."""
        return self._planner.generate_planner_report(clear_stats)
