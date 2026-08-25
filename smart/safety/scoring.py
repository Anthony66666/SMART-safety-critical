"""Scoring scenario realism under a judge model.

The number this module produces is the log-likelihood of a scenario under a
model that did not generate it. Two things make that number fragile, and both
are handled here rather than left to callers:

1. Scenario preparation is stochastic. match_token_map perturbs map tokens
   (SMART.noise, a training augmentation that is on by default) and
   sample_pt_pred randomly masks a third of the map points. Preparing the
   same scenario twice shifts its log-likelihood by ~1e-1 -- larger than the
   realism differences this project sets out to measure.

2. A judge that is the generator makes the score circular. That is sometimes
   a legitimate thing to do while building the pipeline, but it must never be
   mistaken for evidence, so it has to be requested explicitly and is carried
   in the report.
"""
from dataclasses import dataclass, field
from typing import Optional

import math

import torch


class SelfJudgeError(RuntimeError):
    """Raised when a scenario would be scored by the model that generated it."""


def prepare_scenario(model, batch, seed: int = 0):
    """Tokenise a scenario deterministically, ready for generation or scoring.

    Both the generator and the judge must be handed the SAME prepared object:
    preparing separately reintroduces the map-context noise this pins down.

    Args:
        model: a SMART instance used for its tokenisation tables.
        batch: one batch from the dataloader. Mutated in place, as upstream.
        seed: fixes sample_pt_pred's map-point masking.

    Returns:
        The prepared batch.
    """
    torch.manual_seed(seed)
    noise = model.noise
    model.noise = False          # map-token perturbation is a training augmentation
    try:
        data = model.match_token_map(batch)
        data = model.sample_pt_pred(data)
    finally:
        model.noise = noise
    data['agent']['av_index'] += data['agent']['ptr'][:-1]
    return data


def score_tokens(judge, data, token_idx: torch.Tensor) -> torch.Tensor:
    """Per-agent log-likelihood of a token sequence under `judge`.

    Args:
        judge: the scoring model.
        data: a scenario from `prepare_scenario`.
        token_idx: (num_agents, num_steps) sequence to score.

    Returns:
        (num_agents,) log-likelihood under the judge.
    """
    with torch.no_grad():
        return judge.inference(data, forced_tokens=token_idx)['log_p']


@dataclass
class RealismReport:
    """Realism scores plus the provenance needed to interpret them."""

    generator_ckpt: str
    judge_ckpt: str
    self_judged: bool
    scores: list = field(default_factory=list)

    @classmethod
    def create(cls, generator_ckpt: str, judge_ckpt: str,
               allow_self_judge: bool = False) -> 'RealismReport':
        """Start a report, refusing a circular setup unless asked for.

        Raises:
            SelfJudgeError: if generator and judge are the same checkpoint and
                `allow_self_judge` was not set.
        """
        self_judged = generator_ckpt == judge_ckpt
        if self_judged and not allow_self_judge:
            raise SelfJudgeError(
                'generator and judge are the same checkpoint, so realism '
                'scores would be circular; pass allow_self_judge=True to do '
                'this deliberately while building the pipeline')
        return cls(generator_ckpt=generator_ckpt, judge_ckpt=judge_ckpt,
                   self_judged=self_judged)

    def caveat(self) -> Optional[str]:
        """A warning to print alongside the numbers, if one is warranted."""
        if self.self_judged:
            return ('CIRCULAR: scored by the generating model. Valid for '
                    'checking the pipeline, not as evidence of realism.')
        return None


# --- anchors -----------------------------------------------------------------
#
# A likelihood is only a useful ruler if it separates plausible scenarios from
# implausible ones. These build deliberately wrong token sequences to measure
# that separation. Without them, a small gap between logged and generated
# scenarios is uninterpretable: it could mean the generator is good, or it
# could mean the ruler cannot tell anything apart.


def borrow_tokens(donor: torch.Tensor, num_agents: int) -> torch.Tensor:
    """A different scenario's token sequence, resized to this scenario.

    The donor's agents have nothing to do with this scenario's map or history,
    so this is the "obviously wrong" end of the scale. Rows wrap around when
    the donor has fewer agents.

    Args:
        donor: (num_donor_agents, num_steps) sequence from another scenario.
        num_agents: agent count to match.

    Returns:
        (num_agents, num_steps) tokens.
    """
    rows = torch.arange(num_agents, device=donor.device) % donor.shape[0]
    return donor[rows].contiguous()


def permute_agents(tokens: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """This scenario's own sequences, reassigned to the wrong agents.

    Sharper than borrowing from elsewhere: the token marginals are untouched,
    so only a judge that actually conditions on map and history can tell this
    apart from the real scenario. A judge that merely learned which tokens are
    common cannot.

    Args:
        tokens: (num_agents, num_steps).
        seed: fixes the permutation.

    Returns:
        (num_agents, num_steps) with rows permuted, never the identity for
        more than one agent.
    """
    generator = torch.Generator().manual_seed(seed)
    num_agents = tokens.shape[0]
    order = torch.randperm(num_agents, generator=generator)
    if num_agents > 1 and bool((order == torch.arange(num_agents)).all()):
        order = order.roll(1)
    return tokens[order.to(tokens.device)].contiguous()


# --- realism statistics ------------------------------------------------------
#
# Measured on 60 WOMD validation scenarios (self-judged, full support), the
# choice of statistic decides whether the realism axis works at all:
#
#   statistic   logged vs generated   generated vs permuted
#   mean              t=+21.3               t=+0.65   blind
#   median            t=+15.1               t=-3.56   reversed
#   p10               t=+22.5               t=+5.38
#   min               t=+12.7               t=+10.3
#
# Scrambled scenarios are mostly fine agents plus a few impossible ones, while
# samples from the model are uniformly typical. The mean averages a positive
# tail effect against a negative median effect and cancels itself out. The
# lower tail is what separates them -- and "is any agent doing something
# impossible" is the question safety-critical realism actually asks.


def bits_per_dim(log_p_per_dim: float) -> float:
    """Convert a per-dimension log-likelihood in nats to bits per dimension.

    The standard reporting unit, and a pure change of base: it reorders
    nothing and separates nothing that nats did not already separate. Useful
    for comparison against published numbers, not as a sharper instrument.
    """
    return -log_p_per_dim / math.log(2.0)


def typicality(log_p_per_dim: float, reference_entropy: float) -> float:
    """Distance from the typical set: |-log p per dim - reference|.

    High-dimensional samples from a model concentrate near its entropy rather
    than near its mode, so a low likelihood does not by itself mean a sample
    is unrealistic. This measures deviation in either direction.

    Note the reference choice decides what the number means, and neither
    choice yields a usable realism score here. Referenced to the model's own
    entropy it ranks real traffic as the least typical of all (the model is
    over-dispersed: H=3.77 nats against a cross-entropy of 2.39). Referenced
    to the data cross-entropy it orders things correctly but no longer
    separates generated scenarios from scrambled ones (t=-1.61).
    """
    return abs(-log_p_per_dim - reference_entropy)
