"""Shared helpers for the OpenVINO GenAI experiments (E10): pipeline loading, prompt building,
peak-memory reporting. Imported by openvino_correct.py and openvino_bench.py (same directory)."""
import os
import resource
import time

PROMPTS = [
    "List the three largest cities in France by population.",
    "Translate to German: The quick brown fox jumps over the lazy dog.",
    "What is 17 * 23? Answer with the number only.",
]


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)


def read_proc_status(key):
    """Current value of a /proc/self/status field in GB (VmRSS, VmHWM)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return round(int(line.split()[1]) / 1e6, 1)
    return None


def make_scheduler(max_num_seqs=32, cache_gb=16, prefix_caching=False, max_batched=None, dynamic_split_fuse=True):
    import openvino_genai as g
    sc = g.SchedulerConfig()
    sc.max_num_seqs = max_num_seqs
    sc.cache_size = cache_gb
    sc.enable_prefix_caching = prefix_caching
    sc.dynamic_split_fuse = dynamic_split_fuse
    if max_batched:
        sc.max_num_batched_tokens = max_batched
    return sc


def load_pipe(model_dir, kind="auto", sched=None, props=None):
    """Return (pipe, kind, load_seconds). kind: llm | vlm | cb | auto (llm, then cb)."""
    import openvino_genai as g
    props = dict(props or {})
    kinds = ["llm", "cb"] if kind == "auto" else [kind]
    last = None
    for k in kinds:
        t0 = time.time()
        try:
            if k == "llm":
                cfg = dict(props)
                if sched:
                    cfg["scheduler_config"] = sched
                pipe = g.LLMPipeline(model_dir, "CPU", cfg)
            elif k == "vlm":
                pipe = g.VLMPipeline(model_dir, "CPU", scheduler_config=sched, **props) if sched \
                    else g.VLMPipeline(model_dir, "CPU", **props)
            else:
                pipe = g.ContinuousBatchingPipeline(model_dir, sched, "CPU", props)
            return pipe, k, time.time() - t0
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"load with {k} failed after {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:600]}", flush=True)
    raise RuntimeError(f"no pipeline could load {model_dir}: {last}")


def chat_prompt(tokenizer, user_text, thinking=False):
    """Apply the model's chat template (thinking disabled unless requested). Returns a string."""
    ctx = {} if thinking else {"enable_thinking": False}
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], True, extra_context=ctx)


def gen_texts(res):
    """Texts out of whatever generate() returned (DecodedResults, list of results, ...)."""
    if hasattr(res, "texts"):
        return list(res.texts)
    if isinstance(res, (list, tuple)):
        out = []
        for r in res:
            if hasattr(r, "texts"):
                out.append(r.texts[0])
            elif hasattr(r, "m_generation_ids"):
                out.append(r.m_generation_ids[0])
            else:
                out.append(str(r))
        return out
    return [str(res)]


def env_info():
    import openvino as ov
    import openvino_genai as g
    return {"openvino": ov.__version__, "openvino_genai": getattr(g, "__version__", "?"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA")}
