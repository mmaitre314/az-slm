# E02: Is AMX usable inside an Azure VM?

| | |
| --- | --- |
| Status | done (2026-10-04) |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular), Ubuntu 24.04, kernel 6.17.0-1022-azure |
| Stack | `bench/check_amx.sh`; llama.cpp `11fe0215` (2026-10-04) |

## Question

Does the hypervisor expose AMX to the guest, and does the guest kernel let processes use it? A
CPU flag alone isn't enough: Linux requires each process to request the AMX tile state (`arch_prctl`).

## Hypotheses

- H1: The E16ds_v7 guest sees `amx_tile`, `amx_bf16`, `amx_int8`, plus AMX-FP16 (Granite Rapids).
- H2: Kernel 6.x grants the XTILEDATA permission, so AMX instructions actually run.
- H3: A native llama.cpp build compiles AMX kernels and uses them for supported quant types.

## Method

`bench/check_amx.sh` via Run Command: `lscpu`, `/proc/cpuinfo` flags, `cpuid` leaf 7 subleaf 1,
`arch_prctl(ARCH_GET_XCOMP_SUPP)` and `arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA)` through Python
ctypes. Counted AMX instructions in llama.cpp's `libggml-cpu.so` (`objdump`), and read the
model-load log of llama-bench.

## Measurements

| Check | Result |
| --- | --- |
| CPU | Intel Xeon 6 6973P-C, 8 cores / 16 threads, 1 socket slice, 480 MiB L3 visible |
| `/proc/cpuinfo` | `amx_bf16 amx_int8 amx_tile` |
| CPUID leaf 7.1 | `AMX-FP16: FP16 tile operations = true` (not listed in cpuinfo) |
| `ARCH_GET_XCOMP_SUPP` | mask `0x600e7`, XTILEDATA (bit 18) supported |
| `ARCH_REQ_XCOMP_PERM(XTILEDATA)` | rc 0: AMX usable |
| llama.cpp native build | 203 AMX instructions (`tdpb*`, `tileloadd`) in libggml-cpu.so |
| llama.cpp build with AMX compiled out | 0 AMX instructions |
| Model load, Q4_0 | `AMX model buffer size = 14660.88 MiB` (450 tensors repacked for AMX) |

## Analysis

All three hypotheses hold. AMX is fully available to applications on E16ds_v7 with Ubuntu 24.04.
The AMX-FP16 extension is present but hidden from `/proc/cpuinfo`, so checks must use CPUID or
oneDNN's ISA report.

## Next steps

E03 (does it help?) and E04 (is the output correct?).
