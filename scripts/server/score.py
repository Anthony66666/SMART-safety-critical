"""Read the official nuPlan score, and the sub-metrics the headline hides.

The published number is the weighted-average aggregator's `final_score`, times
100. But that single figure is the wrong thing to look at for this benchmark.
nuPlan's closed-loop score gates a weighted average behind multiplicative
penalties, so the occlusion effect -- which should land on collisions and
time-to-collision -- gets diluted by speed limit and comfort terms that
occlusion has no reason to touch.

So this prints every sub-metric, and when two runs are present it prints the
gap per sub-metric. The gap is the result; the totals are context.

    python scripts/server/score.py ~/occlusion-bench/exp
    python scripts/server/score.py ~/occlusion-bench/exp --by-type
"""
import argparse
import glob
import os

import pandas as pd

# Ordered so the ones occlusion should actually move come first.
SUB_METRICS = [
    'no_ego_at_fault_collisions',
    'time_to_collision_within_bound',
    'drivable_area_compliance',
    'driving_direction_compliance',
    'ego_is_making_progress',
    'ego_progress_along_expert_route',
    'speed_limit_compliance',
    'ego_is_comfortable',
]


def failures(run_dir):
    """Simulations that did not complete, as (failed, total).

    A scenario that crashes is never scored, so it leaves the average untouched
    and vanishes from the output -- a run where much of the split failed can
    post a fine number. The runner report is the only place that records it.
    """
    reports = glob.glob(os.path.join(run_dir, 'runner_report.parquet'))
    if not reports:
        return None
    frame = pd.read_parquet(reports[0])
    if 'succeeded' not in frame.columns:
        return None
    return int((~frame.succeeded.astype(bool)).sum()), len(frame)


def load_runs(root):
    """Every completed run under `root`, oldest first."""
    runs = []
    for path in sorted(glob.glob(
            os.path.join(root, '**', 'aggregator_metric', '*.parquet'), recursive=True)):
        frame = pd.read_parquet(path)
        if not (frame.scenario == 'final_score').any():
            continue
        runs.append({
            'tag': path.split('/exp/')[-1].split('/aggregator_metric')[0],
            'frame': frame,
            'failed': failures(path.split('/aggregator_metric')[0]),
        })
    return runs


def summary(frame):
    """The final_score row as a plain dict of metric -> value."""
    row = frame[frame.scenario == 'final_score'].iloc[0]
    out = {'score': float(row.score), 'scenarios': int(row.num_scenarios)}
    for name in SUB_METRICS:
        if name in row.index and pd.notna(row[name]):
            out[name] = float(row[name])
    return out


def by_type(frame):
    """Score per nuPlan scenario type, weighted by scenario count."""
    rows = frame[(frame.scenario != 'final_score') & frame.scenario_type.notna()]
    if not len(rows):
        return {}
    out = {}
    for kind, group in rows.groupby('scenario_type'):
        weight = max(group.num_scenarios.sum(), 1)
        out[kind] = {'score': float((group.score * group.num_scenarios).sum() / weight),
                     'n': int(group.num_scenarios.sum())}
    return out


def paired(before, after):
    """Compare two runs scenario by scenario rather than in aggregate.

    Two averages differing by a point says nothing on its own: the sampler is
    stochastic and the runs could differ by that much for no reason at all.
    The same scenario under both conditions is a paired observation, and the
    spread across pairs is what says whether a shift in the mean is real.
    """
    def rows(frame):
        keep = frame[(frame.scenario != 'final_score') & frame.log_name.notna()]
        return keep.set_index('scenario').score

    a, b = rows(before), rows(after)
    shared = a.index.intersection(b.index)
    if len(shared) < 2:
        print('\nno shared scenarios to pair')
        return
    delta = (b[shared] - a[shared]) * 100

    worse = int((delta < -1e-9).sum())
    better = int((delta > 1e-9).sum())
    same = len(delta) - worse - better
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    stderr = sd / len(delta) ** 0.5

    print(f'\npaired over {len(delta)} scenarios')
    print(f'  mean change      {mean:+.2f} points')
    print(f'  std deviation    {sd:.2f}')
    print(f'  standard error   {stderr:.2f}   -> mean is {abs(mean) / stderr:.1f} '
          f'standard errors from zero')
    print(f'  worse / same / better   {worse} / {same} / {better}')
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f'  p{int(q * 100):02d} {float(delta.quantile(q)):+8.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?',
                        default=os.path.expanduser('~/occlusion-bench/exp'))
    parser.add_argument('--by-type', action='store_true',
                        help='also break the two newest runs down by scenario type')
    parser.add_argument('--paired', action='store_true',
                        help='compare the two newest runs scenario by scenario')
    args = parser.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f'no aggregator output under {args.root}')

    for run in runs:
        stats = summary(run['frame'])
        note = ''
        if run['failed'] and run['failed'][0]:
            note = f"   {run['failed'][0]}/{run['failed'][1]} FAILED"
        print(f"{stats['score'] * 100:6.2f}   {stats['scenarios']:5d} scenarios"
              f"{note}   {run['tag']}")

    if len(runs) < 2:
        print('\nonly one run here; the gap needs a baseline and an occluded run')
        return

    before, after = summary(runs[-2]['frame']), summary(runs[-1]['frame'])
    print(f"\n{'metric':34s} {'earlier':>9} {'later':>9} {'delta':>9}")
    print('-' * 64)
    for name in ['score'] + SUB_METRICS:
        if name not in before or name not in after:
            continue
        gap = after[name] - before[name]
        mark = '  <--' if abs(gap) > 5e-4 else ''
        print(f'{name:34s} {before[name] * 100:9.2f} {after[name] * 100:9.2f} '
              f'{gap * 100:+9.2f}{mark}')

    if args.paired:
        paired(runs[-2]['frame'], runs[-1]['frame'])

    if args.by_type:
        earlier, later = by_type(runs[-2]['frame']), by_type(runs[-1]['frame'])
        shared = sorted(set(earlier) & set(later),
                        key=lambda t: later[t]['score'] - earlier[t]['score'])
        if shared:
            print(f"\n{'scenario type':44s} {'n':>4} {'earlier':>8} "
                  f"{'later':>8} {'delta':>8}")
            print('-' * 78)
            for kind in shared:
                gap = later[kind]['score'] - earlier[kind]['score']
                print(f"{kind[:44]:44s} {int(later[kind]['n']):4d} "
                      f"{earlier[kind]['score'] * 100:8.2f} "
                      f"{later[kind]['score'] * 100:8.2f} {gap * 100:+8.2f}")


if __name__ == '__main__':
    main()
