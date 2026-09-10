#!/usr/bin/env bash
# Bootstrap a RunPod (Linux, CUDA) instance for the cross-modal refusal research.
# Driven over SSH from the local machine. Assumes a RunPod PyTorch template
# (torch + CUDA preinstalled). Idempotent-ish; safe to re-run.
set -euo pipefail

WORK=${WORK:-/workspace}
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

INSTRUCT=${INSTRUCT:-Qwen/Qwen2-VL-7B-Instruct}
BASE=${BASE:-Qwen/Qwen2-VL-7B}

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== python deps ==="
python -m pip install -q -U pip
# transformers v5 + PEFT/TRL for Soup; datasets/vision utils; plotting.
python -m pip install -q -U \
  "transformers>=5.16" accelerate peft trl datasets \
  pillow einops qwen-vl-utils matplotlib scipy pyyaml safetensors
# Soup (the user's fine-tuning CLI) with training + vision extras.
python -m pip install -q -U "soup-cli[train,vision]" || echo "WARN: soup-cli install failed; retry manually"
# bitsandbytes works on a proper CUDA build here (unlike the local cu132 box).
python -m pip install -q -U bitsandbytes || echo "WARN: bitsandbytes optional"

echo "=== versions ==="
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
for m in ("peft","trl","datasets","soup"):
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", e)
PY

echo "=== download models (bf16, ungated) ==="
hf download "$INSTRUCT"
hf download "$BASE" || echo "WARN: base $BASE not fetched (check name/availability)"

echo "=== smoke: load instruct + resolve taps ==="
cd "$WORK/transFORme-r" 2>/dev/null || cd "$WORK"
python - <<PY
import os
os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
from research.scripts import loaders
from research.src import extraction
m,p = loaders.load_vlm("$INSTRUCT")
sites = extraction.resolve_taps(m,"qwen2_vl",{"vision_out":True,"resampler_latents":True,"text_residual":True,"cross_attn":True})
tr=[s for s in sites if s.startswith("text_residual")]; vo=[s for s in sites if s.startswith("vision_out")]
print(f"load OK: {len(tr)} text_residual + {len(vo)} vision_out taps")
PY
echo "=== setup done ==="
