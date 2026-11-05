#!/usr/bin/env python3
"""Training-free decision readout from a causal LM with vLLM (E28/E29). Runs inside the vLLM CPU
container. One prefill per question, no decoding: the prompt lists the options with letter labels
and the model's next-token probabilities over those letters are the decision distribution (the
"listwise readout" idea of LLM2Jev).

  dm_readout.py MODEL --tag TAG --data a.jsonl[,b.jsonl] --out preds.jsonl --summary summary.jsonl
                [--limit N] [--max-num-seqs 64] [--max-model-len 4096] [--extra JSON] [--repeat 1]
                [--format list|semif] [--temperature T]

Prompt formats:
  list   (default) one user message listing options as "A. ...", read at "A"/" A" ... (52 labels).
  semif  the SemIf protocol (MIT, TheoLeeCJ/SemIf) used by trained letter-readout decision models
         such as FlyMy decision-4b and JevK5: a fixed system prompt and one JSON user message
         {evidence, criterion, options[{letter, description "<key>: <text>"}]}, yes/no as "true"/"false",
         read at the bare uppercase letters (26 labels). --temperature divides the letter logits
         (FlyMy decision-4b-v1.2: 1.45; JevK5 v0.3: 1.22) before the softmax.
Input items: dm_data.py's canonical format (state, question, qtype, options, gold). Items with
more options than the format's single-token labels are written with an "error" field.
Per item it writes {model, dataset, id, gold, probs, label_mass, n_prompt_tokens, run}; label_mass
is the probability the model put on any valid label (format compliance). Per dataset it writes a
summary row with wall time, prompt tokens and tokens/s, so throughput comes from the same run.
"""
import argparse
import json
import math
import os
import time

LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
UPPER = LABELS[:26]
SEMIF_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                "Respond with only its uppercase letter, with no explanation or reasoning.")


def prompt_of(it):
    opts = "\n".join(f"{LABELS[i]}. {o}" for i, o in enumerate(it["options"]))
    state = it["state"] if len(it["state"]) <= 8000 else it["state"][:8000] + " [...]"  # stay inside max-model-len
    return (f"Read the input and answer the question by choosing exactly one option.\n\n"
            f"Input:\n{state}\n\nQuestion: {it['question']}\n\nOptions:\n{opts}\n\n"
            f"Reply with the letter of the best option only.")


def capped(state):
    return state if len(state) <= 8000 else state[:8000] + " [...]"


def semif_messages(it):
    """SemIf rendering of a canonical item (option strings may already read "key: description")."""
    if it.get("qtype") == "noul":
        rows = [("true", "The proposition is true."), ("false", "The proposition is false.")]
    elif it.get("qtype") == "score":
        rows = [(str(i), o) for i, o in enumerate(it["options"])]
    else:
        rows = [tuple(o.split(": ", 1)) if ": " in o else (o, "") for o in it["options"]]
    payload = {"evidence": capped(it["state"]), "criterion": it["question"],
               "options": [{"letter": UPPER[i], "description": k + ": " + (d or k)} for i, (k, d) in enumerate(rows)]}
    return [{"role": "system", "content": SEMIF_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True, help="comma-separated item files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra vllm.LLM kwargs")
    ap.add_argument("--repeat", type=int, default=1, help="timed passes per dataset (throughput noise)")
    ap.add_argument("--format", choices=("list", "semif"), default="list")
    ap.add_argument("--temperature", type=float, default=1.0, help="divides the label logits before the softmax")
    args = ap.parse_args()
    labels = LABELS if args.format == "list" else UPPER

    def messages(it):
        if args.format == "semif":
            return semif_messages(it)
        return [{"role": "user", "content": prompt_of(it)}]

    import vllm
    from vllm import LLM, SamplingParams

    run = os.environ.get("RUN", "")
    t0 = time.time()
    kwargs = dict(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  max_logprobs=128, limit_mm_per_prompt={"image": 0, "video": 0},
                  enable_prefix_caching=False)  # repeated passes must redo the prefill, or throughput is inflated
    kwargs.update(json.loads(args.extra))
    llm = LLM(**kwargs)
    load_s = round(time.time() - t0, 1)
    tok = llm.get_tokenizer()
    # label -> token ids of "X" and " X" when each is a single token (semif: the bare letter only)
    label_ids = []
    for L in labels:
        ids = set()
        for s in ((L, " " + L) if args.format == "list" else (L,)):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        label_ids.append(sorted(ids))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=128)
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    print(f"loaded {args.model} in {load_s}s; vllm {vllm.__version__}", flush=True)

    # untimed warm-up (graph/kernel setup would otherwise land in the first dataset's throughput)
    first = [json.loads(l) for l in open(args.data.split(",")[0]) if l.strip()][:16]
    wm = [messages(it) for it in first if len(it["options"]) <= len(labels)]
    if wm:
        llm.chat(wm, sp, use_tqdm=False, **kw)

    for path in args.data.split(","):
        items = [json.loads(l) for l in open(path) if l.strip()]
        if args.limit:
            items = items[: args.limit]
        ok = [it for it in items if len(it["options"]) <= len(labels)]
        bad = [it for it in items if len(it["options"]) > len(labels)]
        msgs = [messages(it) for it in ok]
        ds = items[0]["dataset"] if items else os.path.basename(path)
        for rep in range(args.repeat):
            t = time.time()
            outs = llm.chat(msgs, sp, use_tqdm=False, **kw) if msgs else []
            wall = time.time() - t
            ptoks = sum(len(o.prompt_token_ids) for o in outs)
            summ = {"run": run, "model": args.tag, "path": args.model, "dataset": ds, "rep": rep, "n": len(ok),
                    "n_skipped": len(bad), "wall_s": round(wall, 2), "prompt_tokens": ptoks,
                    "tok_per_s": round(ptoks / wall, 1) if wall else None,
                    "items_per_s": round(len(ok) / wall, 2) if wall else None, "load_s": load_s,
                    "max_num_seqs": args.max_num_seqs, "vllm": vllm.__version__, "extra": json.loads(args.extra),
                    "format": args.format, "temperature": args.temperature}
            with open(args.summary, "a") as f:
                f.write(json.dumps(summ) + "\n")
            print("SUMMARY " + json.dumps(summ), flush=True)
        with open(args.out, "a") as f:
            for it, o in zip(ok, outs):
                lp = o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}
                scores = []
                for k in range(len(it["options"])):
                    p = sum(math.exp(lp[i].logprob) for i in label_ids[k] if i in lp)
                    scores.append(p)
                mass = sum(scores)
                if args.temperature != 1.0 and mass > 0:  # softmax(logit / T) over the labels
                    scores = [math.exp(math.log(s) / args.temperature) if s > 0 else 0.0 for s in scores]
                tot = sum(scores)
                probs = [s / tot for s in scores] if tot > 0 else [1.0 / len(scores)] * len(scores)
                f.write(json.dumps({"run": run, "model": args.tag, "dataset": it["dataset"], "id": it["id"],
                                    "gold": it["gold"], "probs": [round(p, 6) for p in probs],
                                    "label_mass": round(mass, 4), "n_prompt_tokens": len(o.prompt_token_ids)}) + "\n")
            for it in bad:
                f.write(json.dumps({"run": run, "model": args.tag, "dataset": it["dataset"], "id": it["id"],
                                    "gold": it["gold"], "n_options": len(it["options"]),
                                    "error": f"{len(it['options'])} options > {len(labels)} single-token labels"}) + "\n")


if __name__ == "__main__":
    main()
