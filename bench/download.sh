#!/bin/bash
# Download model files from Hugging Face to /mnt/data/models/<repo>. Env:
#   REPO     owner/repo
#   INCLUDE  space-separated glob patterns (default: everything)
set -euo pipefail
: "${REPO:?set REPO}"
dest=/mnt/data/models/$REPO
mkdir -p "$dest"
args=()
for p in ${INCLUDE:-}; do args+=(--include "$p"); done
export HF_XET_HIGH_PERFORMANCE=1
start=$(date +%s)
/opt/azslm/venv/bin/hf download "$REPO" "${args[@]}" --local-dir "$dest"
secs=$(( $(date +%s) - start ))
bytes=$(du -sb "$dest" | cut -f1)
echo "downloaded $REPO: $(( bytes / 1000000000 )) GB in ${secs}s ($(( bytes / 1000000 / (secs > 0 ? secs : 1) )) MB/s)"
ls -la "$dest"
