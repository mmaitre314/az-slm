#!/usr/bin/env python3
"""E16 task-quality benchmarks (GSM8K, MMLU) for one model in the vLLM CPU container.

Runs inside the container (entrypoint python3) with offline `vllm.LLM`, greedy decoding, thinking
disabled through the chat template, and ONE batched `llm.chat(...)` call per benchmark. The model is
loaded once for all benchmarks of the invocation.

* GSM8K: the first N (default 200) test questions. The prompt asks for step-by-step reasoning that
  ends with `#### <number>`; max 512 new tokens. Score: the number after the last `####` (else the
  last number in the text), commas stripped, compared numerically with the reference.
* MMLU: N (default 400) test questions sampled with a fixed seed (random.Random(SEED).sample over
  the whole test split, indices sorted) so every subject is covered. The prompt asks for the letter
  only; max 8 new tokens. Score: first standalone A-D letter in the output.

Datasets are the JSONL files made by bench/e16_convert.py (/mnt/data/datasets/{gsm8k,mmlu}_test.jsonl).

Output (appended; every row carries `run`, taken from $RUN):
  --out-tasks    one line per question: benchmark, id, tag, answer (extracted), reference, correct,
                 text (first 300 chars), n_out_tokens, finish_reason
  --out-summary  one line per (tag, benchmark): accuracy, n_correct, n, Wilson 95% CI, wall_seconds,
                 output_tokens, prompt_tokens, n_no_answer, n_truncated, vllm version, ...

usage: quality_tasks.py MODEL_PATH --tag TAG [--limit N] [--benchmarks gsm8k,mmlu] [--enforce-eager]
"""
import argparse
import json
import math
import os
import random
import re
import time

MMLU_SEED = 16
GSM8K_N = 200
MMLU_N = 400
GSM8K_MAX_TOKENS = 512
MMLU_MAX_TOKENS = 8

GSM8K_PROMPT = (
    "Solve the following math problem step by step, keeping the reasoning concise. Finish with a final line "
    "of the form '#### <number>', where <number> is only the final numeric answer (no units, no commas).\n\n"
    "Problem: {question}"
)
MMLU_PROMPT = (
    "The following is a multiple choice question about {subject}. "
    "Answer with only the letter (A, B, C or D) of the correct answer.\n\n"
    "{question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    "Answer:"
)

NUM_RE = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?")
LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])")


def canon_number(s):
    """'1,234.50' -> '1234.5', '18.00' -> '18'; None if it is not a number."""
    s = s.replace(",", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isfinite(v) and v == int(v):
        return str(int(v))
    return repr(round(v, 6))


def extract_gsm8k(text):
    """Returns (answer, method): the first number after the last '####', else the last number."""
    if "####" in text:
        nums = NUM_RE.findall(text.rsplit("####", 1)[1])
        if nums:
            return canon_number(nums[0]), "marker"
    nums = NUM_RE.findall(text)
    return (canon_number(nums[-1]), "last_number") if nums else (None, "none")


def extract_mmlu(text):
    m = LETTER_RE.search(text)
    return m.group(1) if m else None


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_gsm8k(data_dir, limit):
    rows = read_jsonl(f"{data_dir}/gsm8k_test.jsonl")[: limit or GSM8K_N]
    items = []
    for i, r in enumerate(rows):
        ref = canon_number(r["answer"].rsplit("####", 1)[1])
        items.append({"id": i, "prompt": GSM8K_PROMPT.format(question=r["question"]), "reference": ref})
    return items, GSM8K_MAX_TOKENS


def build_mmlu(data_dir, limit):
    rows = read_jsonl(f"{data_dir}/mmlu_test.jsonl")
    idx = sorted(random.Random(MMLU_SEED).sample(range(len(rows)), MMLU_N))[: limit or MMLU_N]
    items = []
    for i in idx:
        r = rows[i]
        a, b, c, d = r["choices"]
        prompt = MMLU_PROMPT.format(subject=r["subject"].replace("_", " "), question=r["question"].strip(),
                                    a=a, b=b, c=c, d=d)
        items.append({"id": i, "prompt": prompt, "reference": "ABCD"[int(r["answer"])], "subject": r["subject"]})
    return items, MMLU_MAX_TOKENS


def make_progress():
    """tqdm bar that updates every 60 s, so the log shows progress without megabytes of redraws."""
    try:
        from tqdm import tqdm
    except ImportError:
        return False

    def factory(*a, **k):
        k.update(mininterval=60, maxinterval=60, miniters=1)
        return tqdm(*a, **k)
    return factory


def spec_counters(llm):
    """Speculative-decoding counters from vLLM's in-process metrics ({} if none or unsupported)."""
    try:
        metrics = llm.get_metrics()
    except Exception as e:  # older vLLM or stats disabled
        return {"error": str(e)[:200]}
    out = {}
    for m in metrics:
        if "spec_decode" not in m.name:
            continue
        if hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
        elif hasattr(m, "values"):
            prev = out.get(m.name, [0] * len(m.values))
            out[m.name] = [x + y for x, y in zip(prev, m.values)]
    return out


def spec_delta(before, after):
    d = {}
    for k, v in after.items():
        b = before.get(k)
        if isinstance(v, list):
            d[k] = [x - y for x, y in zip(v, b or [0] * len(v))]
        elif isinstance(v, (int, float)):
            d[k] = v - (b or 0)
    drafts = d.get("vllm:spec_decode_num_drafts")
    drafted = d.get("vllm:spec_decode_num_draft_tokens")
    accepted = d.get("vllm:spec_decode_num_accepted_tokens")
    if drafted:
        d["acceptance_rate"] = round(accepted / drafted, 4)
    if drafts:
        d["mean_acceptance_length"] = round(1 + accepted / drafts, 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True, help="short model tag, e.g. bf16, w8a8, w4a16")
    ap.add_argument("--benchmarks", default="gsm8k,mmlu")
    ap.add_argument("--limit", type=int, default=0, help="questions per benchmark (0 = full set)")
    ap.add_argument("--data-dir", default="/mnt/data/datasets")
    ap.add_argument("--out-tasks", default="/mnt/data/results/quality-tasks.jsonl")
    ap.add_argument("--out-summary", default="/mnt/data/results/quality-summary.jsonl")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--enforce-eager", action="store_true", help="skip torch.compile (vllm_correct.py default)")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra LLM() kwargs")
    ap.add_argument("--show", type=int, default=0, help="print this many example outputs per benchmark")
    args = ap.parse_args()

    import vllm  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    run = os.environ.get("RUN", "")
    t0 = time.time()
    extra = json.loads(args.extra)
    if "speculative_config" in extra:
        extra.setdefault("disable_log_stats", False)  # get_metrics() needs stats for the acceptance counters
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0}, enforce_eager=args.enforce_eager, **extra)
    load_s = round(time.time() - t0, 1)
    print(f"loaded {args.model} in {load_s}s", flush=True)
    kw = {"chat_template_kwargs": {"enable_thinking": False}}

    for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        items, max_tokens = {"gsm8k": build_gsm8k, "mmlu": build_mmlu}[bench](args.data_dir, args.limit)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        msgs = [[{"role": "user", "content": it["prompt"]}] for it in items]
        print(f"== {bench}: {len(items)} questions, max_tokens={max_tokens} {time.strftime('%T')}", flush=True)
        spec0 = spec_counters(llm) if "speculative_config" in extra else {}
        t = time.time()
        results = llm.chat(msgs, sp, use_tqdm=make_progress(), **kw)
        wall = round(time.time() - t, 1)
        spec = spec_delta(spec0, spec_counters(llm)) if "speculative_config" in extra else None

        rows, n_correct, n_none, n_trunc, out_tok, in_tok = [], 0, 0, 0, 0, 0
        for it, res in zip(items, results):
            o = res.outputs[0]
            if bench == "gsm8k":
                ans, method = extract_gsm8k(o.text)
            else:
                ans, method = extract_mmlu(o.text), "letter"
            ok = ans is not None and ans == it["reference"]
            n_correct += ok
            n_none += ans is None
            n_trunc += o.finish_reason == "length"
            out_tok += len(o.token_ids)
            in_tok += len(res.prompt_token_ids)
            rows.append({"run": run, "benchmark": bench, "id": it["id"], "tag": args.tag, "answer": ans,
                         "reference": it["reference"], "correct": bool(ok), "method": method,
                         "n_out_tokens": len(o.token_ids), "finish_reason": o.finish_reason,
                         "text": o.text[:300]})
        n = len(rows)
        summary = {"run": run, "tag": args.tag, "model": args.model, "benchmark": bench,
                   "accuracy": round(n_correct / n, 4), "n_correct": n_correct, "n": n,
                   "ci95_wilson": wilson(n_correct, n), "wall_seconds": wall, "output_tokens": out_tok,
                   "prompt_tokens": in_tok, "n_no_answer": n_none, "n_truncated": n_trunc,
                   "load_seconds": load_s, "vllm": vllm.__version__, "dtype": args.dtype,
                   "max_num_seqs": args.max_num_seqs, "enforce_eager": args.enforce_eager,
                   "limit": args.limit, "mmlu_seed": MMLU_SEED, "extra": extra, "spec": spec}
        with open(args.out_tasks, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out_summary, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
        for r in rows[: args.show]:
            print("EXAMPLE " + json.dumps({k: r[k] for k in ("id", "answer", "reference", "correct", "method",
                                                              "finish_reason", "text")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":  # required: vLLM spawns the engine process
    main()
