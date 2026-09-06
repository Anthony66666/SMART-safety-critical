#!/usr/bin/env bash
# Pull one large file off the server, resuming from wherever it stopped.
#
# scp restarts from zero when the connection drops, and this link drops often
# enough that an 89 GB transfer would rarely finish. Reading with dd from a byte
# offset and appending locally means an interruption costs only the seconds
# since the last block.
#
# Parallel connections do not help here and are not offered: four streams
# measured 11.4 MB/s against 11.8 for one, so the link rather than the
# per-connection rate is the limit. Fetching the same file from its origin is
# worse still, at 0.4 MB/s through the proxy.
#
#   bash scripts/fetch_from_server.sh <remote-path> <local-path> [rate-KB/s]
set -uo pipefail

REMOTE=${1:?usage: fetch_from_server.sh <remote-path> <local-path> [rate-KB/s]}
LOCAL=${2:?usage: fetch_from_server.sh <remote-path> <local-path> [rate-KB/s]}
# The server reads this file over NFS at about 20 MB/s and the sweeps are
# reading their scenarios from the same mount, so taking all of it would slow
# them down. Empty means no limit.
RATE=${3:-}

SSH="ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

TOTAL=$($SSH l40s "stat -c%s '$REMOTE'" 2>/dev/null | tr -d '\r')
[ -n "$TOTAL" ] || { echo "cannot stat $REMOTE on the server" >&2; exit 1; }
mkdir -p "$(dirname "$LOCAL")"

while :; do
    have=$( [ -f "$LOCAL" ] && stat -c%s "$LOCAL" || echo 0 )
    if [ "$have" -ge "$TOTAL" ]; then
        echo "complete: $LOCAL ($have bytes)"
        break
    fi
    awk -v h="$have" -v t="$TOTAL" 'BEGIN {
        printf "%.1f / %.1f GB (%.1f%%), resuming\n", h/1073741824, t/1073741824, 100*h/t }'
    # iflag=skip_bytes so the offset does not have to land on a block boundary.
    if [ -n "$RATE" ] && command -v pv >/dev/null 2>&1; then
        $SSH l40s "dd if='$REMOTE' bs=1M iflag=skip_bytes skip=$have 2>/dev/null" \
            | pv -q -L "${RATE}k" >> "$LOCAL"
    else
        $SSH l40s "dd if='$REMOTE' bs=1M iflag=skip_bytes skip=$have 2>/dev/null" >> "$LOCAL"
    fi
    sleep 5
done

echo "local size:  $(stat -c%s "$LOCAL")"
echo "server size: $TOTAL"
