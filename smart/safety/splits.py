"""Deterministic dataset partitioning.

Realism scores are only meaningful if the model reporting them never saw the
scenarios it is judging, and never shared training data with the generator it
is judging. Splits are therefore assigned by hashing the scenario id: the
partition is reproducible on any machine, needs no manifest file, and is
disjoint by construction.

The salt makes the partition regenerable under a different arrangement
without renaming anything.
"""
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List

_HASH_SCALE = float(1 << 64)


@dataclass(frozen=True)
class SplitSpec:
    """A named partition of the dataset.

    Attributes:
        fractions: split name -> share of the data; must sum to 1.
        salt: changes the partition without changing the split names.
    """
    fractions: Dict[str, float]
    salt: str

    def __post_init__(self) -> None:
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f'fractions must sum to 1, got {total}')


def _unit_hash(scenario_id: str, salt: str) -> float:
    """Map an id to [0, 1) uniformly and reproducibly across processes."""
    digest = hashlib.md5(f'{salt}:{scenario_id}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / _HASH_SCALE


def assign_split(scenario_id: str, spec: SplitSpec) -> str:
    """Return the split `scenario_id` belongs to."""
    position = _unit_hash(scenario_id, spec.salt)
    upper = 0.0
    for name, fraction in spec.fractions.items():
        upper += fraction
        if position < upper:
            return name
    return list(spec.fractions)[-1]


def select_split(scenario_ids: Iterable[str], spec: SplitSpec, split: str) -> List[str]:
    """Filter ids down to a single split."""
    if split not in spec.fractions:
        raise ValueError(f'unknown split {split!r}; have {sorted(spec.fractions)}')
    return [i for i in scenario_ids if assign_split(i, spec) == split]
