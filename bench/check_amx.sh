#!/bin/bash
# Verify Intel AMX end to end on the VM: CPU flags, kernel permission (arch_prctl), and the
# llama.cpp build. Run via: scripts/deploy.py run <vm> bench/check_amx.sh
echo "== cpu"
lscpu | grep -E '^(Model name|Socket|Core|Thread|NUMA node\(s\)|CPU\(s\)|L3)' | sed 's/  */ /g'
echo "amx flags (cpuinfo): $(grep -o 'amx[a-z0-9_]*' /proc/cpuinfo | sort -u | tr '\n' ' ')"
command -v cpuid >/dev/null && echo "amx (cpuid leaf 7): $(cpuid -1 -l 7 -s 1 2>/dev/null | grep -i 'amx' | sed 's/  */ /g' | tr '\n' ';')"
echo "kernel: $(uname -r)"

echo "== kernel AMX permission (arch_prctl)"
python3 - <<'EOF'
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
SYS_arch_prctl, ARCH_GET_XCOMP_SUPP, ARCH_REQ_XCOMP_PERM, XTILEDATA = 158, 0x1021, 0x1023, 18
mask = ctypes.c_uint64()
rc = libc.syscall(SYS_arch_prctl, ARCH_GET_XCOMP_SUPP, ctypes.byref(mask))
print(f"XCOMP_SUPP rc={rc} mask=0x{mask.value:x} xtiledata_supported={bool(mask.value >> XTILEDATA & 1)}")
rc = libc.syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XTILEDATA)
print(f"REQ_XCOMP_PERM(XTILEDATA) rc={rc} errno={ctypes.get_errno()} -> {'AMX usable' if rc == 0 else 'AMX NOT usable'}")
EOF

echo "== memory / disks"
free -g | head -2
df -h /mnt/data | tail -1

echo "== setup"
echo "setup-done: $(cat /var/lib/azslm/setup-done 2>/dev/null || echo running)"
echo "llama.cpp: $(cat /var/lib/azslm/llama.cpp.version 2>/dev/null)"
if [ -f /opt/llama.cpp/build/CMakeCache.txt ]; then
  grep -E '^GGML_(NATIVE|AMX[A-Z_]*|AVX512[A-Z_]*):' /opt/llama.cpp/build/CMakeCache.txt | tr '\n' ' '; echo
  ls /opt/llama.cpp/build/bin/ | grep -E '^llama-(bench|batched-bench|perplexity|cli|server)$' | tr '\n' ' '; echo
fi
