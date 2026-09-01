"""Print the official final score from a completed run, and compare two.

The number nuPlan publishes is the weighted-average aggregator's `final_score`
row, times 100. Reading it directly avoids re-deriving anything.

    python scripts/server/score.py ~/occlusion-bench/exp
"""
import glob
import os
import sys


def failures(run_dir):
    """How many simulations in this run did not complete.

    A scenario that crashes is not scored, so it leaves the final average alone
    and vanishes -- a run where half the split failed can post a fine number.
    The runner report is the only place that says so.
    """
    import pandas as pd

    reports = glob.glob(os.path.join(run_dir, 'runner_report.parquet'))
    if not reports:
        return None
    frame = pd.read_parquet(reports[0])
    if 'succeeded' not in frame.columns:
        return None
    return int((~frame.succeeded.astype(bool)).sum()), len(frame)


def scores(root):
    """Every run under `root`, newest last, as (tag, final score, counts)."""
    found = []
    for path in sorted(glob.glob(os.path.join(root, '**', 'aggregator_metric', '*.parquet'),
                                 recursive=True)):
        import pandas as pd
        frame = pd.read_parquet(path)
        row = frame[frame.scenario == 'final_score']
        if len(row):
            run_dir = path.split('/aggregator_metric')[0]
            found.append((path.split('/exp/')[-1].split('/aggregator_metric')[0],
                          float(row.score.iloc[0]), len(frame) - 1,
                          failures(run_dir)))
    return found


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/occlusion-bench/exp')
    found = scores(root)
    if not found:
        raise SystemExit(f'no aggregator output under {root}')
    for tag, score, count, failed in found:
        note = ''
        if failed is not None and failed[0]:
            note = f'   {failed[0]}/{failed[1]} FAILED'
        print(f'{score * 100:6.2f}   {count:5d} scenarios{note}   {tag}')
    if len(found) >= 2:
        print(f'\nlatest two differ by {(found[-1][1] - found[-2][1]) * 100:+.2f} points')


if __name__ == '__main__':
    main()
