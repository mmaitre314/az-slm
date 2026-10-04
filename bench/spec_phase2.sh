#!/bin/bash
# E12 follow-up chain: wait for the single-sequence job, then tie check, then 4/16-sequence runs.
set -uo pipefail
cd "$(dirname "$0")"
while systemctl is-active --quiet azslm-job-single; do sleep 10; done
bash spec_tiecheck.sh
CONFIGS=${BATCH_CONFIGS:-"base mtp3 d08q4-4"} NPS="4 16" OUT=/mnt/data/results/spec-batch.jsonl bash spec_batch.sh
