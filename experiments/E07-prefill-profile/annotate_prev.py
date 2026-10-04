import collections, re, subprocess, sys
data, lib, pat = sys.argv[1:4]
txt = subprocess.run(["nice", "-n", "19", "perf", "script", "-i", data, "-F", "ip,sym,symoff", "-G", "--no-demangle"],
                     capture_output=True, text=True).stdout
symoffs = collections.Counter(); sym = None
for line in txt.splitlines():
    p = line.split()
    if len(p) < 2: continue
    m = re.match(r"(.+)\+0x([0-9a-f]+)$", p[1])
    if m and re.search(pat, m.group(1)):
        sym = m.group(1); symoffs[int(m.group(2), 16)] += 1
n = sum(symoffs.values())
nm = subprocess.run(["nm", "-S", lib], capture_output=True, text=True).stdout
for l in nm.splitlines():
    f = l.split()
    if len(f) == 4 and f[3] == sym: start, size = int(f[0], 16), int(f[1], 16)
dis = subprocess.run(["nice", "-n", "19", "objdump", "-d", "--no-show-raw-insn", f"--start-address={start}", f"--stop-address={start+size}", lib],
                     capture_output=True, text=True).stdout
ins = []
for l in dis.splitlines():
    m = re.match(r"\s*([0-9a-f]+):\s+(\S+)(.*)", l)
    if m: ins.append((int(m.group(1), 16) - start, m.group(2), m.group(3).strip()))
pos = {o: i for i, (o, _, _) in enumerate(ins)}
prev = collections.Counter()
for off, c in symoffs.items():
    i = pos.get(off)
    if i is None: continue
    prev[(ins[i-1][1] if i else "?")] += c
print("previous-instruction mnemonic of sampled IP, % of symbol samples:", ", ".join(f"{m} {100*c/n:.1f}" for m, c in prev.most_common(6)))
for off, c in symoffs.most_common(4):
    i = pos[off]; print(f"  +0x{off:x} {100*c/n:.1f}%  prev: {ins[i-1][1]} {ins[i-1][2][:40]} | at: {ins[i][1]} {ins[i][2][:30]}")
print("hot offsets (samples % of symbol):")
for off, c in symoffs.most_common(6):
    i = pos.get(off)
    print(f"  +0x{off:x} {100*c/n:5.1f}%  {ins[i][1] if i is not None else '?'}")

# all tile instructions (static)
print("tile instructions in symbol:", collections.Counter(mn for _, mn, _ in ins if mn.startswith(('tile', 'tdp'))))
