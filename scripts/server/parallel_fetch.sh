#!/usr/bin/env bash
# Fetch one large file over many connections, resumably.
#
# A single wget to S3 was averaging about 1 MB/s from this box, which is a day
# for the 89 GB test split. The object advertises Accept-Ranges, so the limit is
# per-connection rather than the link, and several ranges at once fix it.
# aria2 would be the obvious tool but the conda-forge build fails its TLS
# handshake against S3; curl on the same box is fine, so this uses that.
#
# Parts are kept separate until every one is complete and the right size, and
# only then concatenated -- an interrupted run resumes each part from wherever
# it stopped rather than leaving one plausible-looking truncated file.
#
#   bash scripts/server/parallel_fetch.sh <url> <output> [connections]
set -euo pipefail

URL=${1:?usage: parallel_fetch.sh <url> <output> [connections]}
OUT=${2:?usage: parallel_fetch.sh <url> <output> [connections]}
N=${3:-16}

TOTAL=$(curl -fsI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
[ -n "$TOTAL" ] || { echo "could not read Content-Length from $URL" >&2; exit 1; }
printf 'total %.1f GB over %d connections\n' "$(echo "$TOTAL/1073741824" | bc -l)" "$N"

if [ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$TOTAL" ]; then
    echo "already complete: $OUT"; exit 0
fi

PARTS="$OUT.parts"
mkdir -p "$PARTS"
CHUNK=$(( (TOTAL + N - 1) / N ))

for i in $(seq 0 $((N - 1))); do
    START=$(( i * CHUNK ))
    END=$(( START + CHUNK - 1 ))
    [ "$END" -ge "$TOTAL" ] && END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    PART="$PARTS/part.$i"

    (
        HAVE=0
        [ -f "$PART" ] && HAVE=$(stat -c%s "$PART")
        if [ "$HAVE" -ge "$WANT" ]; then
            exit 0                      # this slice is already done
        fi
        # Resume this slice from where it stopped, not from its start.
        curl -fsS --retry 20 --retry-delay 5 --retry-all-errors \
             -r "$(( START + HAVE ))-$END" "$URL" >> "$PART"
    ) &
done

echo "waiting for $N connections; watch progress with:  du -sh $PARTS"
FAILED=0
wait -n 2>/dev/null || true
wait || FAILED=1

for i in $(seq 0 $((N - 1))); do
    START=$(( i * CHUNK )); END=$(( START + CHUNK - 1 ))
    [ "$END" -ge "$TOTAL" ] && END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    HAVE=$(stat -c%s "$PARTS/part.$i" 2>/dev/null || echo 0)
    if [ "$HAVE" -ne "$WANT" ]; then
        echo "part $i is $HAVE of $WANT bytes -- rerun this script to resume" >&2
        exit 1
    fi
done

echo "all parts complete, joining..."
# Explicit numeric order. `cat part.*` sorts lexicographically, which puts
# part.10 before part.2 and scrambles the file -- while leaving its total size
# exactly right, so the check below would wave it through. The 8-part test that
# validated this script never showed it, because with parts 0-7 the two
# orderings coincide.
for i in $(seq 0 $((N - 1))); do cat "$PARTS/part.$i"; done > "$OUT"
GOT=$(stat -c%s "$OUT")
if [ "$GOT" -ne "$TOTAL" ]; then
    echo "joined file is $GOT bytes, expected $TOTAL -- not deleting parts" >&2
    exit 1
fi
rm -rf "$PARTS"
printf 'done: %s (%.1f GB)\n' "$OUT" "$(echo "$GOT/1073741824" | bc -l)"
