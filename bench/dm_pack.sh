#!/bin/bash
# Pack a decision-model VM's saved results (/var/lib/azslm/results) into a small bundle that
# `deploy.py fetch` can copy in a few minutes: Run Command returns 4 KiB per call, and the full
# prediction files are MBs. Keeps every small file as is, digests long step logs, scores every
# prediction file with dm_score.py (full metrics: NLL, Brier, temperature), and compacts the
# per-item predictions with dm_compact.py. Needs dm_compact.py and dm_score.py next to this script.
#   deploy.py push VM <dir with dm_pack.sh dm_compact.py dm_score.py>
#   deploy.py run VM -c 'bash /opt/azslm/bench/dm_pack.sh'
#   deploy.py fetch VM /var/lib/azslm/bundle.tar.xz <local>/bundle.tar.xz
set -uo pipefail
B=$(dirname "$(readlink -f "$0")")
R=${R:-/var/lib/azslm/results}
OUT=/var/lib/azslm/bundle
rm -rf $OUT /var/lib/azslm/bundle.tar.xz && mkdir -p $OUT
cd "$R" || exit 1
shopt -s nullglob
preds=( *-p-*.jsonl *.preds.jsonl )
python3 $B/dm_compact.py pack $OUT/compact-preds.jsonl "${preds[@]}" || echo "compact FAILED"
python3 $B/dm_score.py "${preds[@]}" --md > $OUT/score-all.md 2> $OUT/score-all.err || echo "score FAILED"
python3 $B/dm_score.py "${preds[@]}" --json $OUT/score-all.jsonl > /dev/null 2>&1
[ -s $OUT/score-all.err ] || rm -f $OUT/score-all.err
for f in *; do
  [ -f "$f" ] || continue
  case "$f" in *-p-*.jsonl|*.preds.jsonl) continue ;; esac
  if [[ "$f" == *.log && "$f" != chain.log && "$f" != *-top.log && $(stat -c %s "$f") -gt 8192 ]]; then
    { head -n 3 "$f" | cut -c1-300; echo "[...] distinct notable lines:"
      grep -E "loaded |Error|ERROR|Traceback|Killed|WARNING|out of memory" "$f" | grep -v '^SUMMARY' | cut -c1-300 | awk '!seen[$0]++' | head -n 30
      echo "[...] last lines:"; tail -n 3 "$f" | cut -c1-300; } > "$OUT/$f.digest.txt"
  else
    cp "$f" $OUT/
  fi
done
tar -C $OUT -cf - . | xz -9e > /var/lib/azslm/bundle.tar.xz
ls -la $OUT | awk 'NR>3 {print $5, $9}' | tail -n 60
stat -c "bundle %s bytes" /var/lib/azslm/bundle.tar.xz
