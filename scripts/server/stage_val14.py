"""Copy only the log files val14 actually needs onto local disk.

The val split is 1381 logs and about 150 GB, but val14 is 1118 scenarios, and a
scenario lives in exactly one log. Reading them over NAS is what makes the
simulation slow, so the fix is to stage the subset locally -- not the whole
split.

Scenario tokens are lidar_pc primary keys, so membership is an index lookup
rather than a table scan, which keeps the survey itself cheap over NAS. Logs
are probed in parallel because the cost is per-file latency, not CPU.

The survey reports coverage. If some val14 tokens are not found in the split,
the run would silently evaluate fewer scenarios than the published benchmark
and the comparison to 90.43 would not be like for like -- so that number is
printed whether or not anything is copied.

    python scripts/server/stage_val14.py --dest ~/occlusion-bench/val14_local
    python scripts/server/stage_val14.py --dest ~/occlusion-bench/val14_local --copy
"""
import argparse
import glob
import os
import re
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

BATCH = 400  # well under SQLite's variable limit


def read_tokens(path):
    """The scenario tokens listed in a nuPlan scenario_filter yaml.

    Tokens appear both quoted and bare in these files -- val14 quotes all of
    its, test14-hard quotes only the 182 that would otherwise parse as YAML
    numbers and leaves 90 bare. Matching only the quoted form silently found
    two thirds of them and staged a subset that looked like a complete run.
    """
    with open(path) as handle:
        text = handle.read()
    found = re.findall(r'^\s*-\s*"?([0-9a-f]{16})"?\s*$', text, re.M)
    return [bytes.fromhex(t) for t in found]


def tokens_in_log(db_path, tokens):
    """Which of `tokens` this log contains, by indexed lookup."""
    found = set()
    try:
        connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        try:
            for start in range(0, len(tokens), BATCH):
                chunk = tokens[start:start + BATCH]
                marks = ','.join('?' * len(chunk))
                rows = connection.execute(
                    f'select token from lidar_pc where token in ({marks})', chunk)
                found.update(row[0] for row in rows)
        finally:
            connection.close()
    except sqlite3.Error:
        return db_path, set()
    return db_path, found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-split',
                        default='/hqlab/dataset_nas3/nuplan/raw/nuplan-v1.1/splits/val')
    parser.add_argument('--filter', default=os.path.expanduser(
        '~/occlusion-bench/Flow-Planner/flow_planner/nuplan_simulation/'
        'scenario_filter/val14.yaml'))
    parser.add_argument('--dest', required=True)
    parser.add_argument('--copy', action='store_true',
                        help='actually copy; without it this only surveys')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    tokens = read_tokens(args.filter)
    if not tokens:
        raise SystemExit(f'no scenario tokens found in {args.filter}')
    logs = sorted(glob.glob(os.path.join(args.val_split, '*.db')))
    if not logs:
        raise SystemExit(f'no .db files under {args.val_split}')
    print(f'{len(tokens)} val14 tokens, {len(logs)} logs in the split\n'
          f'surveying with {args.workers} workers...', flush=True)

    needed = {}
    seen = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (path, found) in enumerate(
                pool.map(lambda p: tokens_in_log(p, tokens), logs), 1):
            if found:
                needed[path] = found
                seen |= found
            if done % 200 == 0:
                print(f'  {done}/{len(logs)} logs, {len(seen)}/{len(tokens)} tokens '
                      f'located in {len(needed)} logs', flush=True)

    total = sum(os.path.getsize(p) for p in needed)
    print(f'\n{len(needed)} logs hold {len(seen)}/{len(tokens)} tokens, '
          f'{total / 2**30:.1f} GB')
    missing = len(tokens) - len(seen)
    if missing:
        print(f'WARNING: {missing} val14 tokens are not in this split. A run over '
              f'it is not the published benchmark and cannot be compared to 90.43.')

    if not args.copy:
        print('\nsurvey only; pass --copy to stage them')
        return

    os.makedirs(args.dest, exist_ok=True)

    def stage(path):
        """Copy one log, skipping it if a complete copy is already there.

        Copies go through a .partial name so an interrupted run leaves nothing
        that looks finished -- the size check would otherwise wave through a
        truncated file, and a truncated db fails deep inside the simulation
        rather than here.
        """
        target = os.path.join(args.dest, os.path.basename(path))
        if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(path):
            return False
        partial = target + '.partial'
        shutil.copy2(path, partial)
        os.replace(partial, target)
        return True

    # Copying is bound by NAS latency, not by this machine, so the same
    # concurrency that made the survey quick applies here.
    copied = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, did in enumerate(pool.map(stage, sorted(needed)), 1):
            copied += bool(did)
            if index % 25 == 0:
                print(f'  {index}/{len(needed)}', flush=True)
    print(f'\nstaged {copied} new files into {args.dest}')
    print(f'now run with: VAL_SPLIT={args.dest} bash scripts/server/run_val14.sh baseline')


if __name__ == '__main__':
    main()
