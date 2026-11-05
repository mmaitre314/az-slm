#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model's safetensors so vLLM can serve it without LoRA
support (E32). Our own code; runs inside the vLLM CPU container (torch + safetensors).

  dm_lora_merge.py --base /mnt/data/models/Qwen/Qwen3.5-4B --adapter <dir with adapter_config.json and
                   adapter_model.safetensors> --out /mnt/data/models/merged/<name>

W' = W + (lora_alpha / r) * B @ A for every adapted module (alpha / sqrt(r) with use_rslora), computed in
fp32 and cast back to the base tensor's dtype. Base and adapter keys are matched on the part after
"layers.", so a text-only adapter merges into a multimodal checkpoint (model.language_model.layers.*).
Every LoRA pair must find exactly one base tensor, or the script fails. Non-weight files (config,
tokenizer, chat template) are copied unchanged.
"""
import argparse
import json
import math
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def lora_pairs(adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    pairs = {}
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            for part in (".lora_A.weight", ".lora_B.weight"):
                if k.endswith(part):
                    mod = k[: -len(part)]
                    if "layers." not in mod:
                        raise SystemExit(f"unexpected adapter key {k}")
                    suffix = "layers." + mod.split("layers.", 1)[1] + ".weight"
                    pairs.setdefault(suffix, {})["A" if "lora_A" in part else "B"] = f.get_tensor(k)
    for s, d in pairs.items():
        if set(d) != {"A", "B"}:
            raise SystemExit(f"incomplete LoRA pair for {s}")
    return pairs, scale, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs, scale, cfg = lora_pairs(args.adapter)
    print(f"adapter: {len(pairs)} LoRA pairs, r={cfg['r']}, alpha={cfg['lora_alpha']}, scale={scale}")
    os.makedirs(args.out, exist_ok=True)
    shards = sorted(f for f in os.listdir(args.base) if f.endswith(".safetensors"))
    used = set()
    for shard in shards:
        tensors = {}
        with safe_open(os.path.join(args.base, shard), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                hit = [s for s in pairs if k.endswith(s) and (k == s or k[: -len(s)].endswith("."))]
                if len(hit) > 1:
                    raise SystemExit(f"ambiguous match for {k}: {hit}")
                if hit:
                    s = hit[0]
                    a, b = pairs[s]["A"].float(), pairs[s]["B"].float()
                    delta = (b @ a) * scale
                    if delta.shape != t.shape:
                        raise SystemExit(f"shape mismatch {k}: base {tuple(t.shape)} vs delta {tuple(delta.shape)}")
                    t = (t.float() + delta).to(t.dtype)
                    used.add(s)
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(args.out, shard), metadata=meta or {"format": "pt"})
        print(f"{shard}: {len(tensors)} tensors")
    missing = set(pairs) - used
    if missing:
        raise SystemExit(f"{len(missing)} LoRA pairs matched no base tensor, e.g. {sorted(missing)[:3]}")
    for f in os.listdir(args.base):
        src = os.path.join(args.base, f)
        if not f.endswith(".safetensors") and os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, f))
    print(f"merged {len(used)} modules into {args.out}")


if __name__ == "__main__":
    main()
