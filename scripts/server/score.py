"""Print the official final score from a completed run, and compare two.

The number nuPlan publishes is the weighted-average aggregator's `final_score`
row, times 100. Reading it directly avoids re-deriving anything.

    python scripts/server/score.py ~/occlusion-bench/exp
"""
import glob
import os
import sys


def scores(root):
    """Every run under `root`, newest last, as (tag, final score)."""
    found = []
    for path in sorted(glob.glob(os.path.join(root, '**', 'aggregator_metric', '*.parquet'),
                                 recursive=True)):
        import pandas as pd
        frame = pd.read_parquet(path)
        row = frame[frame.scenario == 'final_score']
        if len(row):
            found.append((path.split('/exp/')[-1].split('/aggregator_metric')[0],
                          float(row.score.iloc[0]), len(frame) - 1))
    return found


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/occlusion-bench/exp')
    found = scores(root)
    if not found:
        raise SystemExit(f'no aggregator output under {root}')
    for tag, score, count in found:
        print(f'{score * 100:6.2f}   {count:5d} scenarios   {tag}')
    if len(found) >= 2:
        print(f'\nlatest two differ by {(found[-1][1] - found[-2][1]) * 100:+.2f} points')


if __name__ == '__main__':
    main()
