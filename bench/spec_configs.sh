#!/bin/bash
# Speculative-decoding configurations for E12, sourced by spec_single.sh / spec_batch.sh.
# spec_args NAME prints the llama-server flags for configuration NAME.
M=/mnt/data/models
MTP_Q4_0=$M/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
D08_Q4=$M/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf
D08_Q8=$M/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
D2_Q4=$M/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_0.gguf
spec_args() {
  case "$1" in
    base)      echo "" ;;
    mtp1)      echo "--spec-type draft-mtp --spec-draft-n-max 1" ;;
    mtp2)      echo "--spec-type draft-mtp --spec-draft-n-max 2" ;;
    mtp3)      echo "--spec-type draft-mtp --spec-draft-n-max 3" ;;
    mtp4)      echo "--spec-type draft-mtp --spec-draft-n-max 4" ;;
    mtp6)      echo "--spec-type draft-mtp --spec-draft-n-max 6" ;;
    mtpu3)     echo "--spec-type draft-mtp --spec-draft-model $MTP_Q4_0 --spec-draft-n-max 3" ;;
    d08q4-3)   echo "--spec-type draft-simple --spec-draft-model $D08_Q4 --spec-draft-n-max 3" ;;
    d08q4-4)   echo "--spec-type draft-simple --spec-draft-model $D08_Q4 --spec-draft-n-max 4" ;;
    d08q4-8)   echo "--spec-type draft-simple --spec-draft-model $D08_Q4 --spec-draft-n-max 8" ;;
    d08q8-4)   echo "--spec-type draft-simple --spec-draft-model $D08_Q8 --spec-draft-n-max 4" ;;
    d2q4-4)    echo "--spec-type draft-simple --spec-draft-model $D2_Q4 --spec-draft-n-max 4" ;;
    ngsimple)  echo "--spec-type ngram-simple" ;;
    ngsimple-s) echo "--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 6" ;;
    ngmod)     echo "--spec-type ngram-mod" ;;
    ngmod-s)   echo "--spec-type ngram-mod --spec-ngram-mod-n-match 4 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 8" ;;
    ngmapk)    echo "--spec-type ngram-map-k" ;;
    ngmapk-s)  echo "--spec-type ngram-map-k --spec-ngram-map-k-size-n 3 --spec-ngram-map-k-size-m 6" ;;
    ngk4v)     echo "--spec-type ngram-map-k4v" ;;
    ngk4v-s)   echo "--spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 3 --spec-ngram-map-k4v-size-m 6" ;;
    ngcache)   echo "--spec-type ngram-cache" ;;
    mtp3+ngmod) echo "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3" ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}
