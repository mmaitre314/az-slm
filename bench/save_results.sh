#!/bin/bash
# Copy small result files to the OS disk (/var/lib/azslm/results), which survives deallocation,
# and mark the VM as holding unfetched results: the idle watchdog then deallocates it instead of
# deleting it. Fetch with `scripts/deploy.py harvest <vm> <local-dir> --release`.
#
#   save_results.sh FILE_OR_GLOB...   (relative paths are taken from /mnt/data/results)
# Call it after every step of a job chain, so a lost job or VM never loses finished steps.
set -uo pipefail
DEST=/var/lib/azslm/results
mkdir -p "$DEST"
cd /mnt/data/results 2>/dev/null || cd /
for f in "$@"; do
  for g in $f; do
    [ -f "$g" ] || continue
    size=$(stat -c %s "$g")
    if [ "$size" -gt 2000000 ]; then
      # keep large logs bounded: the harvest goes through 3 KiB Run Command chunks
      tail -c 200000 "$g" > "$DEST/$(basename "$g").tail"
    else
      cp "$g" "$DEST/"
    fi
  done
done
touch /var/lib/azslm/keep
echo "saved: $(ls "$DEST" | wc -l) files, $(du -sh "$DEST" | cut -f1) in $DEST"
