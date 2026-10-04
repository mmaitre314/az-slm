#!/bin/bash
# List model files and sizes in Hugging Face repos (run on the VM; the sandbox cannot reach HF).
# Env: REPOS="owner/repo ..." ; MATCH: regex filter on file names (default: weights only).
REPOS=${REPOS:-"bartowski/Qwen3.8-27B-GGUF"} MATCH=${MATCH:-'\.(gguf|safetensors)$'} \
/opt/azslm/venv/bin/python - <<'EOF'
import collections, os, re
from huggingface_hub import HfApi
api = HfApi()
for repo in os.environ["REPOS"].split():
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        print(f"== {repo}: ERROR {type(e).__name__}: {e}")
        continue
    print(f"== {repo} @ {info.sha[:10]}")
    groups = collections.OrderedDict()
    for s in info.siblings:
        if re.search(os.environ["MATCH"], s.rfilename):
            # Collapse split files (-00001-of-00003) into one line per model variant.
            key = re.sub(r"-\d{5}-of-\d{5}", "", s.rfilename)
            n, size = groups.get(key, (0, 0))
            groups[key] = (n + 1, size + (s.size or 0))
    for key, (n, size) in groups.items():
        print(f"{size / 1e9:7.2f} GB {'(' + str(n) + ' files) ' if n > 1 else ''}{key}")
EOF
