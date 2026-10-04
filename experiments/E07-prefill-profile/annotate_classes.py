import collections, re, subprocess, sys
data, lib, pat = sys.argv[1:4]
txt = subprocess.run(["nice", "-n", "19", "perf", "script", "-i", data, "-F", "ip,sym,symoff", "-G", "--no-demangle"],
                     capture_output=True, text=True).stdout
hist = collections.Counter(); total = 0; sym = None; symoffs = collections.Counter()
for line in txt.splitlines():
    p = line.split()
    if len(p) < 2: continue
    total += 1
    ip, rest = p[0], p[1]
    m = re.match(r"(.+)\+0x([0-9a-f]+)$", rest)
    if not m: continue
    s, off = m.group(1), int(m.group(2), 16)
    if re.search(pat, s):
        sym = s; symoffs[off] += 1
n = sum(symoffs.values())
print(f"total samples {total}, in symbol {n} ({100*n/total:.1f}%)  sym={sym[:90]}")
nm = subprocess.run(["nm", "-S", lib], capture_output=True, text=True).stdout
start = size = None
for l in nm.splitlines():
    f = l.split()
    if len(f) == 4 and f[3] == sym:
        start, size = int(f[0], 16), int(f[1], 16)
print(f"symbol start 0x{start:x} size {size} bytes")
dis = subprocess.run(["nice", "-n", "19", "objdump", "-d", "--no-show-raw-insn", f"--start-address={start}", f"--stop-address={start+size}", lib],
                     capture_output=True, text=True).stdout
ins = {}
for l in dis.splitlines():
    m = re.match(r"\s*([0-9a-f]+):\s+(\S+)(.*)", l)
    if m: ins[int(m.group(1), 16) - start] = (m.group(2), m.group(3).strip())
offs = sorted(ins)
def cls(mn, ops):
    if mn.startswith("tdp"): return "AMX tile multiply (tdpb*)"
    if mn.startswith("tile"): return "AMX tile load/store/zero"
    if mn.startswith("v") and "zmm" in ops: return "AVX-512 (zmm) " + ("VNNI vpdpbusd" if mn.startswith("vpdp") else "other")
    if mn.startswith("v") and "ymm" in ops: return "AVX2 (ymm)"
    if mn.startswith("v"): return "AVX (xmm/other v*)"
    if mn.startswith("j") or mn in ("call", "ret", "cmp", "test"): return "branch/compare"
    return "scalar/other"
by = collections.Counter()
mn_hist = collections.Counter()
for off, c in symoffs.items():
    # attribute to the instruction at or before this offset (the IP is the next instruction to retire)
    i = max([o for o in offs if o <= off], default=None)
    mn, ops = ins.get(i, ("?", ""))
    by[cls(mn, ops)] += c
    mn_hist[mn] += c
# static instruction mix
static = collections.Counter(cls(*ins[o]) for o in offs)
print("class                              samples%  static-instr")
for k, c in by.most_common():
    print(f"{k:36s} {100*c/n:6.1f}   {static.get(k,0)}")
print("top mnemonics by samples%:", ", ".join(f"{m} {100*c/n:.1f}" for m, c in mn_hist.most_common(8)))
