"""Assemble every run under an experiment root into one table.

The benchmark's claim lives in the gap between the two observation conditions
for one planner, so the table is organised around that: a row per planner and
reactivity, the two scores side by side, and their difference.

Absolute scores carry a systematic offset from the published numbers and are
not directly comparable to them -- 87.9 against a published 90.43 for Flow
Planner, though 92.8 against 93 for PDM-Closed. The gap is the measurement;
the columns either side are context.

    python scripts/server/table.py ~/occlusion-bench/exp
    python scripts/server/table.py ~/occlusion-bench/exp --metric no_ego_at_fault_collisions
    python scripts/server/table.py ~/occlusion-bench/exp --csv results.csv
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

# experiment_uid is "<planner>/<split>/<reactivity>/<mode>/<timestamp>"
PATTERN = re.compile(r'/([^/]+)/(val14|test14-hard|test14-random)/'
                     r'(nonreactive|reactive)/(baseline|occluded|random)/')


def collect(root):
    """Newest run for each (planner, split, reactivity, mode)."""
    runs = {}
    for path in sorted(glob.glob(
            os.path.join(root, '**', 'aggregator_metric', '*.parquet'), recursive=True)):
        match = PATTERN.search(path)
        if not match:
            continue
        runs[match.groups()] = path        # sorted, so the last wins
    return runs


def score(path, metric):
    frame = pd.read_parquet(path)
    row = frame[frame.scenario == 'final_score']
    if not len(row) or metric not in row.columns:
        return None, 0
    value = row[metric].iloc[0]
    return (float(value) * 100 if pd.notna(value) else None,
            int(row.num_scenarios.iloc[0]))


def failures(path):
    reports = glob.glob(os.path.join(path.split('/aggregator_metric')[0],
                                     'runner_report.parquet'))
    if not reports:
        return None
    frame = pd.read_parquet(reports[0])
    if 'succeeded' not in frame.columns:
        return None
    return int((~frame.succeeded.astype(bool)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?',
                        default=os.path.expanduser('~/occlusion-bench/exp'))
    parser.add_argument('--metric', default='score',
                        help='final_score column; e.g. no_ego_at_fault_collisions')
    parser.add_argument('--csv', help='also write the table here')
    args = parser.parse_args()

    runs = collect(args.root)
    if not runs:
        raise SystemExit(f'no runs found under {args.root}')

    cells = defaultdict(dict)
    for (planner, split, react, mode), path in runs.items():
        value, count = score(path, args.metric)
        cells[(planner, split, react)][mode] = (value, count, failures(path))

    print(f'metric: {args.metric}\n')
    header = (f"{'planner':<14} {'split':<12} {'react':<12} "
              f"{'baseline':>9} {'occluded':>9} {'gap':>8} {'n':>6}  notes")
    print(header)
    print('-' * len(header))

    rows = []
    for key in sorted(cells):
        planner, split, react = key
        base = cells[key].get('baseline', (None, 0, None))
        occ = cells[key].get('occluded', (None, 0, None))
        gap = (occ[0] - base[0]) if (base[0] is not None and occ[0] is not None) else None
        notes = []
        for name, cell in (('baseline', base), ('occluded', occ)):
            if cell[0] is None:
                notes.append(f'{name} missing')
            elif cell[2]:
                notes.append(f'{name} {cell[2]} FAILED')
        # A run whose scenario count differs from its partner is not a
        # like-for-like comparison, whatever the gap says.
        if base[1] and occ[1] and base[1] != occ[1]:
            notes.append(f'counts differ {base[1]} vs {occ[1]}')
        print(f'{planner:<14} {split:<12} {react:<12} '
              f'{base[0] if base[0] is not None else float("nan"):>9.2f} '
              f'{occ[0] if occ[0] is not None else float("nan"):>9.2f} '
              f'{gap if gap is not None else float("nan"):>+8.2f} '
              f'{max(base[1], occ[1]):>6d}  {"; ".join(notes)}')
        rows.append(dict(planner=planner, split=split, reactivity=react,
                         baseline=base[0], occluded=occ[0], gap=gap,
                         scenarios=max(base[1], occ[1]), notes='; '.join(notes)))

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
