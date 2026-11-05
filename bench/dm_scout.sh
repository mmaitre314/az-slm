#!/bin/bash
# Decision-model scout (E28): fetch model cards, READMEs, papers and leaderboards that the agent
# sandbox can't reach, strip them to text, and save a compact bundle for `deploy.py harvest`.
# Runs on a VM (which has open Internet). Output: /mnt/data/results/scout/*.txt and hf-*.json.
# Env: MAXB (bytes kept per document, default 12000).
set -uo pipefail
OUT=/mnt/data/results/scout
MAXB=${MAXB:-12000}
mkdir -p "$OUT"
cd "$OUT" || exit 1
UA="Mozilla/5.0 (X11; Linux x86_64) decision-model-scout"

# HTML -> text with the standard library only (scripts, styles and tags dropped).
cat > /tmp/html2txt.py <<'EOF'
import html, re, sys
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.out = []; self.skip = 0
    def handle_starttag(self, t, a):
        if t in ("script", "style", "noscript", "svg"): self.skip += 1
        if t in ("p", "br", "li", "tr", "h1", "h2", "h3", "h4", "pre", "div", "table"): self.out.append("\n")
        if t in ("td", "th"): self.out.append(" | ")
    def handle_endtag(self, t):
        if t in ("script", "style", "noscript", "svg") and self.skip: self.skip -= 1
    def handle_data(self, d):
        if not self.skip: self.out.append(d)
raw = sys.stdin.read()
p = P(); p.feed(raw)
txt = html.unescape("".join(p.out))
txt = re.sub(r"[ \t]+", " ", txt); txt = re.sub(r"\n\s*\n+", "\n\n", txt)
sys.stdout.write(txt.strip() + "\n")
EOF

get() {  # name url [html|raw]
  local name=$1 url=$2 kind=${3:-html}
  if curl -sSL --max-time 40 -A "$UA" "$url" -o "/tmp/scout.$name" 2>/dev/null && [ -s "/tmp/scout.$name" ]; then
    { echo "### $name"; echo "### $url"; if [ "$kind" = html ]; then python3 /tmp/html2txt.py < "/tmp/scout.$name"; else cat "/tmp/scout.$name"; fi; } \
      | head -c "$MAXB" > "$name.txt"
    echo "ok   $name $(wc -c < "$name.txt")B"
  else
    echo "FAIL $name $url"
  fi
}

# 1. Discovery through the Hugging Face API (models and datasets mentioning these names).
for q in jev openjev laya verdict decider decision-model decision-4b decision-2b systemone kev intern-decision; do
  curl -sS --max-time 40 "https://huggingface.co/api/models?search=$q&limit=60&sort=downloads&full=false" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(m.get("id"),m.get("downloads"),m.get("likes"),m.get("pipeline_tag"),m.get("library_name"),m.get("createdAt","")[:10],sep="\t") for m in d]' \
    > "hf-models-$q.tsv" 2>/dev/null; echo "hf models '$q': $(wc -l < "hf-models-$q.tsv")"
done
curl -sS --max-time 40 "https://huggingface.co/api/datasets?search=jev&limit=40" \
  | python3 -c 'import json,sys; [print(m.get("id"),m.get("downloads"),m.get("likes"),sep="\t") for m in json.load(sys.stdin)]' > hf-datasets-jev.tsv 2>/dev/null

# 2. Model cards (raw README, first 5 KB) of the candidates named in search results so far, plus the
# 30 most-downloaded models the searches found.
FOUND=$(cat hf-models-*.tsv 2>/dev/null | sort -t$'\t' -k2,2nr | cut -f1 | awk '!s[$0]++' | head -n 30)
for repo in shenjunhao/OpenJev-4B alanhuangya/OpenJev-8B Praveenrajus/jevify-qwen3.5-4b Manav2op/verdict-small \
            jaredpalmer/kev-4b flymy-ai/decision-4b-v1.2 tozp/laya-onnx $FOUND; do
  [ -f "card_$(echo "$repo" | tr '/' '_').txt" ] && continue
  MAXB=5000
  n=$(echo "$repo" | tr '/' '_')
  get "card_$n" "https://huggingface.co/$repo/raw/main/README.md" raw
  curl -sS --max-time 30 "https://huggingface.co/api/models/$repo" | python3 -c '
import json,sys; d=json.load(sys.stdin)
print(json.dumps({k:d.get(k) for k in ("id","sha","pipeline_tag","library_name","tags","cardData","config")})[:4000])
print("files:", [s.get("rfilename") for s in d.get("siblings",[])][:60])' > "api_$n.txt" 2>/dev/null
done
MAXB=12000
get "card_ds_jev-bench" "https://huggingface.co/datasets/Praveenrajus/jev-bench/raw/main/README.md" raw
curl -sS --max-time 30 "https://huggingface.co/api/datasets/Praveenrajus/jev-bench" | python3 -c '
import json,sys; d=json.load(sys.stdin); print(d.get("sha")); print([s.get("rfilename") for s in d.get("siblings",[])][:80])' > api_ds_jev-bench.txt 2>/dev/null

# 3. GitHub READMEs (raw.githubusercontent.com is reachable from the VM).
for repo in Yinsongxu/LLM2Jev cyberspace-cs/llm2jev Zefan-Cai/Open-Jev razorback16/OpenJev SiliconLabAI/OpenJev \
            monsterlag/verdict Manavarya09/verdict Heman10x-NGU/Verdict-open-jev strands-labs/strands-decider; do
  n=$(echo "$repo" | tr '/' '_')
  get "gh_$n" "https://raw.githubusercontent.com/$repo/HEAD/README.md" raw
done
get "gh_strands_jevbench" "https://raw.githubusercontent.com/strands-labs/strands-decider/HEAD/evaluation/jevbench.md" raw

# 4. Papers, leaderboards and the comparison article.
get "arxiv_llm2jev" "https://arxiv.org/abs/2610.02076"
get "arxiv_inner_jev" "https://arxiv.org/abs/2610.17205"
get "arxiv_llm2jev_html" "https://arxiv.org/html/2610.02076v2"
get "bh_jevbench" "https://benchmarkheaven.com/jev-models"
get "bh_jevbench_v162" "https://benchmarkheaven.com/jev-models/v1.6.2"
get "openjev_bench" "https://zefan-cai.github.io/open-jev/benchmarks/"
get "tds_seven" "https://towardsdatascience.com/jev-your-engines-seven-open-decision-models-vs-a-generative-llm/"
get "digitalapplied_clef" "https://www.digitalapplied.com/blog/open-decision-models-compared-clef-decider-jev"
get "layer3_guide" "https://www.layer3labs.io/guides/system-one-decision-models"
get "marktechpost_jev" "https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/"
get "openrouter_jev" "https://openrouter.ai/typesafe/jev-1.13"
ls -la "$OUT" | tail -n +2 | wc -l
