#!/bin/bash
# Build the speculative-decoding targets in the native (AMX) build and in the AMX-free control build.
# llama-server is needed for --spec-type (draft-mtp, ngram-*, draft-simple) and parallel slots.
set -uo pipefail
cd /opt/llama.cpp
for b in build build-native-noamx; do
  echo "== $b $(date -u +%T)"
  cmake --build $b -j"$(nproc)" --target llama-server llama-speculative llama-speculative-simple llama-completion 2>&1 | tail -n 3
  ls $b/bin | grep -E '^llama-(server|speculative|speculative-simple|completion)$' | tr '\n' ' '; echo
done
echo "done $(date -u +%T)"
