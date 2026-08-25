"""Construction of the independent judge model.

Realism is scored by a model that did not generate the scenario. That is the
whole point of the judge, and it fails silently if the judge turns out to be
the generator wearing a different seed -- nothing crashes, the numbers just
stop meaning what the paper claims. Hence the explicit architecture check.
"""
from typing import Dict, List, Mapping

from smart.safety.splits import SplitSpec

# Fields that must differ for the judge to be a genuinely distinct density
# model. A different seed alone is not enough: a reseeded twin shares every
# inductive bias with the generator.
_ARCHITECTURE_FIELDS = ('hidden_dim', 'num_heads', 'head_dim',
                        'num_agent_layers', 'num_map_layers')


def assert_distinct_architecture(judge: Mapping, generator: Mapping) -> None:
    """Raise if the judge is not architecturally distinguishable.

    Args:
        judge: judge architecture fields.
        generator: generator architecture fields.

    Raises:
        ValueError: if every architecture field matches.
    """
    shared = [f for f in _ARCHITECTURE_FIELDS if judge.get(f) == generator.get(f)]
    if len(shared) == len(_ARCHITECTURE_FIELDS):
        raise ValueError(
            'judge and generator are architecturally indistinguishable '
            f'(identical {", ".join(_ARCHITECTURE_FIELDS)}); an independent '
            'likelihood model must differ in width or depth, not only in seed')


def judge_architecture_from_config(config) -> Dict:
    """Architecture fields of a judge config."""
    return {
        'hidden_dim': config.Model.hidden_dim,
        'num_heads': config.Model.num_heads,
        'head_dim': config.Model.head_dim,
        'num_agent_layers': config.Model.decoder.num_agent_layers,
        'num_map_layers': config.Model.decoder.num_map_layers,
    }


def generator_architecture_from_checkpoint(path: str) -> Dict:
    """Architecture the generator was actually trained with.

    Read from the checkpoint rather than a config file, which may have been
    edited since the run.
    """
    import torch

    hparams = torch.load(path, map_location='cpu')['hyper_parameters']
    model = hparams['model_config']
    return {
        'hidden_dim': model['hidden_dim'],
        'num_heads': model['num_heads'],
        'head_dim': model['head_dim'],
        'num_agent_layers': model['decoder']['num_agent_layers'],
        'num_map_layers': model['decoder']['num_map_layers'],
    }


def split_spec_from_config(config) -> SplitSpec:
    """Build the dataset partition described by a judge config."""
    split_cfg = config.Split
    fractions: Dict[str, float] = {k: float(v) for k, v in dict(split_cfg.fractions).items()}
    return SplitSpec(fractions=fractions, salt=str(split_cfg.salt))


def scenario_ids_for(config, split: str, directories: List[str]) -> set:
    """Ids belonging to `split`, drawn from `directories`."""
    import os

    from smart.safety.splits import assign_split

    spec = split_spec_from_config(config)
    keep = set()
    for directory in directories:
        for name in os.listdir(os.path.expanduser(os.path.normpath(directory))):
            scenario_id = os.path.splitext(name)[0]
            if assign_split(scenario_id, spec) == split:
                keep.add(scenario_id)
    return keep
