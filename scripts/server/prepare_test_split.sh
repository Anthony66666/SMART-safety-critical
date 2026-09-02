#!/usr/bin/env bash
# Join, verify and extract the nuPlan test split.
#
# Run under tmux so an SSH drop does not take it with it. Each step is
# skippable on a rerun, so an interruption costs only the step it was in.
set -euo pipefail

RAW=${RAW:-/hqlab/dataset_nas3/nuplan/raw}
ZIP="$RAW/nuplan-v1.1_test.zip"
URL="https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip"
HERE=$(cd "$(dirname "$0")/../.." && pwd)

echo "=== 1. join the parts in the right order ==="
# parallel_fetch skips any part that is already complete, so with the download
# finished this only re-joins -- but it must be the fixed version: joining with
# a glob orders part.10 before part.2 and produces a scrambled file of exactly
# the right size, which every later check would pass.
bash "$HERE/scripts/server/parallel_fetch.sh" "$URL" "$ZIP" 16

echo "=== 2. central directory ==="
# Reads only the directory at the end of the archive, so it is quick and it
# fails loudly if the join went wrong.
unzip -l "$ZIP" | tail -3

echo "=== 3. extract ==="
# unzip checks each entry's CRC as it writes it, so this is the real integrity
# test; there is no point paying for a separate -t pass over 89 GB first.
cd "$RAW"
unzip -q -o "$ZIP"

echo "=== 4. what landed ==="
# The test archive unpacks to data/cache/test, the same convention the mini
# download uses -- not the nuplan-v1.1/splits layout the existing val split
# sits in, which this NAS's download script had rearranged. Report whichever
# directories actually appeared rather than assuming either.
for d in "$RAW"/data/cache/*/ "$RAW"/nuplan-v1.1/splits/*/; do
    [ -d "$d" ] || continue
    printf '  %-46s %s db files\n' "$d" "$(ls "$d"/*.db 2>/dev/null | wc -l)"
done
echo "done"
