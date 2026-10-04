#!/bin/bash
# Build a second llama.cpp from the same commit with AMX compiled out (all AVX-512 features kept),
# for a clean AMX on/off comparison. Output: /opt/llama.cpp/build-noamx/bin.
set -euxo pipefail
cd /opt/llama.cpp
cmake -B build-noamx -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX_VNNI=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
  -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF
cmake --build build-noamx -j"$(nproc)" --target llama-bench llama-batched-bench llama-perplexity llama-completion llama-batched llama-parallel
for b in build build-noamx; do
  echo "$b: $(objdump -d $b/bin/libggml-cpu.so | grep -cE 'tdpb|tileloadd') AMX instructions in libggml-cpu.so"
done
