#!/bin/bash
# Control build for AMX A/B: same as the native build (-march=native) except the AMX instruction sets
# are disabled, so only AMX differs. (build-noamx from build_noamx.sh uses an explicit flag list,
# which also changes other code paths.) Output: /opt/llama.cpp/build-native-noamx/bin.
set -euxo pipefail
cd /opt/llama.cpp
flags="-march=native -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-amx-fp16 -mno-amx-complex"
cmake -B build-native-noamx -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX512=OFF \
  -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags"
cmake --build build-native-noamx -j"$(nproc)" --target llama-bench llama-batched-bench llama-perplexity llama-completion llama-parallel
for b in build build-noamx build-native-noamx; do
  [ -f "$b/bin/libggml-cpu.so" ] || continue
  echo "$b: $(objdump -d $b/bin/libggml-cpu.so | grep -cE 'tdpb|tileloadd') AMX instructions," \
    "$(objdump -d $b/bin/libggml-cpu.so | grep -cE 'vpdpbusd') VNNI instructions"
done
