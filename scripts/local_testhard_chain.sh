#!/usr/bin/env bash
# Wait for the test split to arrive, unpack it, smoke it, then run test14-hard.
#
# Each step checks the one before it rather than assuming: the transfer is
# verified by size, the unpack by finding .db files, and the full run is only
# started if a four-scenario smoke test actually succeeded. A chain that starts
# the eight-hour job on a bad unpack is worse than one that stops.
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
ZIP=${ZIP:-/mnt/e/nuplan-test/nuplan-v1.1_test.zip}
DEST=${DEST:-/mnt/e/nuplan-test}
LOG=${LOG:-$DEST/chain.log}
PLANNERS=${PLANNERS:-"pdm_closed gc_pgp urban_driver diffusion flow"}

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$DEST"
say "waiting for the transfer to finish"
SIZE=95919476643
while :; do
    have=$( [ -f "$ZIP" ] && stat -c%s "$ZIP" || echo 0 )
    [ "$have" -ge "$SIZE" ] && break
    sleep 300
done
say "transfer complete: $(stat -c%s "$ZIP") bytes"

# Unpack once. An interrupted unzip leaves a partial tree that looks fine, so
# the marker is written only after unzip returns cleanly.
if [ ! -f "$DEST/.unpacked" ]; then
    say "unpacking (this reads 89 GB)"
    if unzip -q -o "$ZIP" -d "$DEST" >> "$LOG" 2>&1; then
        touch "$DEST/.unpacked"
        say "unpacked"
    else
        say "unzip failed -- stopping"; exit 1
    fi
fi

# Find the directory holding the logs rather than assuming the layout.
SPLIT_DIR=$(find "$DEST" -type d -name test -print -quit 2>/dev/null)
[ -n "$SPLIT_DIR" ] || SPLIT_DIR=$(dirname "$(find "$DEST" -name '*.db' -print -quit 2>/dev/null)" 2>/dev/null)
count=$(ls "$SPLIT_DIR"/*.db 2>/dev/null | wc -l)
say "split dir: $SPLIT_DIR ($count db files)"
[ "$count" -gt 0 ] || { say "no .db files found -- stopping"; exit 1; }
export VAL_SPLIT="$SPLIT_DIR"

# Preflight every planner first. It resolves the arguments and checks the files
# each run would open, in seconds -- the alternative is what happened the first
# time through, where a checkpoint in the wrong place took ninety seconds of
# ray startup to report and would have taken hours if it had been the third
# planner in the queue rather than the first.
say "preflight"
blocked=0
for planner in $PLANNERS; do
    for mode in baseline occluded; do
        reason=$(PLANNER=$planner REACTIVITY=smart DRY_RUN=1 \
                   bash "$HERE/scripts/run_local.sh" "$mode" 2>&1 >/dev/null)
        [ -z "$reason" ] || { say "  ${planner}_smart_${mode}: $reason"; blocked=1; }
    done
done
[ "$blocked" -eq 0 ] || { say "preflight found missing files -- stopping"; exit 1; }
say "preflight ok"

# Smoke next, in the execution mode the real run uses. A smoke test that
# quietly switches to a sequential worker proves the model loads and nothing
# about running several at once, which is how the server lost a day.
say "smoke test (4 scenarios, ray)"
if LIMIT=4 WORKER_OVERRIDE=ray_distributed THREADS=8 REACTIVITY=smart \
     bash "$HERE/scripts/run_local.sh" baseline >> "$LOG" 2>&1; then
    say "smoke ok"
else
    say "smoke FAILED -- stopping, see $LOG"; exit 1
fi

for planner in $PLANNERS; do
    for mode in baseline occluded; do
        tag="${planner}_smart_${mode}"
        rl="$DEST/${tag}.log"
        if grep -q "Number of successful simulations" "$rl" 2>/dev/null; then
            say "skip $tag (done)"; continue
        fi
        say "run $tag"
        start=$SECONDS
        PLANNER=$planner REACTIVITY=smart bash "$HERE/scripts/run_local.sh" "$mode" \
            > "$rl" 2>&1
        ok=$(grep -oE 'Number of successful simulations: [0-9]+' "$rl" | tail -1 | grep -oE '[0-9]+$')
        bad=$(grep -oE 'Number of failed simulations: [0-9]+' "$rl" | tail -1 | grep -oE '[0-9]+$')
        say "  $tag: ${ok:-0} ok, ${bad:-?} failed, $(( (SECONDS - start) / 60 )) min"
    done
done
say "done"
