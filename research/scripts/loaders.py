"""Model & data loaders for the real runs (P0-P5).

Design choices that keep Phase 1 self-contained and low-risk:
* Harmful text comes from AdvBench; benign text from Alpaca (both public, ungated).
* The IMAGE modality is produced by rendering each instruction as a FigStep-style
  typographic image *locally* (a published attack modality) instead of downloading a
  gated multimodal-harmful dataset. So R_V is measured on genuinely image-conditioned
  inputs with zero gated content and nothing harmful written to disk.
* We only ever run forward passes to read activations here (Phase 1). No harmful text
  is generated or persisted. See docs/ethics.md.

Lazy imports so the harness / synthetic demo stay usable without heavy deps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research" / "configs" / "experiments.yaml"

# Coarse semantic buckets so category-mean SVD has several distinct rows (the
# refusal cone needs >1 category; a single split is rank-1). Keyword-based and
# deliberately simple; unmatched prompts fall in "other".
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cyber": ("hack", "malware", "virus", "ransomware", "exploit", "phishing", "ddos", "keylog", "sql injection", "password"),
    "weapons": ("bomb", "explosive", "weapon", "gun", "firearm", "grenade", "chemical weapon", "poison gas"),
    "drugs": ("drug", "meth", "cocaine", "heroin", "narcotic", "fentanyl"),
    "fraud": ("fraud", "scam", "launder", "counterfeit", "forge", "steal", "identity theft", "credit card"),
    "harm": ("kill", "murder", "assault", "harm", "suicide", "self-harm", "poison", "torture"),
}


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_vlm(model_id: str, load_in_4bit: bool = False, dtype: str = "bfloat16",
             processor_id: str | None = None):
    """Load a VLM + processor. Uses transformers' image-text-to-text auto class.

    Defaults to bf16 (fits an 8 GB GPU for a 2B model); 4-bit is opt-in and only
    used if bitsandbytes is importable. `processor_id` overrides where the
    processor/chat-template comes from (e.g. borrow an Instruct model's vision
    chat template to prompt a base checkpoint that lacks one).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # transformers v5 raises on benign old->new weight-conversion notes (e.g. tied
    # lm_head reported "missing"). Swallow ONLY that conversion raise; real
    # size-mismatch / load errors still propagate. The weights themselves load fine.
    import transformers.modeling_utils as _mu

    if not getattr(_mu, "_conv_raise_patched", False):
        _orig_report = _mu.log_state_dict_report

        def _patched_report(*a, **k):
            try:
                return _orig_report(*a, **k)
            except RuntimeError as e:
                if "conversion" in str(e).lower():
                    return None
                raise

        _mu.log_state_dict_report = _patched_report
        _mu._conv_raise_patched = True

    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": "auto"}
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype
            )
        except Exception:
            pass  # fall back to full-precision load
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(processor_id or model_id)
    return model, processor


def make_forward(processor, family: str, max_new: int = 0) -> Callable:
    """Return `run_forward(model, batch)`; a batch item is {'text', 'image'?}."""
    import torch

    def run_forward(model, batch: list[dict]) -> None:
        texts, images = [], []
        for item in batch:
            content = []
            if item.get("image") is not None:
                content.append({"type": "image"})
                images.append(item["image"])
            content.append({"type": "text", "text": item["text"]})
            msg = [{"role": "user", "content": content}]
            texts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
        inputs = processor(text=texts, images=images or None, return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    return run_forward


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _categorize(prompt: str) -> str:
    low = prompt.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(k in low for k in kws):
            return cat
    return "other"


def _first_text_field(ds) -> str:
    for c in ("text", "prompt", "instruction", "goal", "behavior"):
        if c in ds.column_names:
            return c
    return ds.column_names[0]


def load_harmful_by_category(limit_per_cat: int, categories: list[str]) -> dict[str, list[str]]:
    """Harmful behaviors (ungated `mlabonne/harmful_behaviors`, the canonical
    abliteration set) bucketed into the requested semantic categories."""
    from datasets import load_dataset

    ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    field = _first_text_field(ds)
    buckets: dict[str, list[str]] = {c: [] for c in categories}
    for row in ds:
        cat = _categorize(row[field])
        if cat in buckets and len(buckets[cat]) < limit_per_cat:
            buckets[cat].append(row[field])
    return {c: v for c, v in buckets.items() if v}  # drop empty categories


def load_harmless(limit: int) -> list[str]:
    """Benign instructions (ungated `mlabonne/harmless_alpaca`)."""
    from datasets import load_dataset

    ds = load_dataset("mlabonne/harmless_alpaca", split="train")
    field = _first_text_field(ds)
    return [r[field] for r in ds.select(range(min(limit, len(ds))))]


def render_prompt_image(text: str, size: int = 448) -> Any:
    """FigStep-style typographic image: the instruction rendered as wrapped text."""
    import textwrap

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    wrapped = textwrap.fill(text, width=34)
    draw.multiline_text((16, 16), wrapped, fill="black", font=font, spacing=6)
    return img


def load_image_conditioned_by_category(
    harmful_by_cat: dict[str, list[str]], neutral_prompt: str
) -> dict[str, list[dict]]:
    """Turn each harmful instruction into an image-conditioned item (R_V source)."""
    return {
        cat: [{"text": neutral_prompt, "image": render_prompt_image(p)} for p in prompts]
        for cat, prompts in harmful_by_cat.items()
    }


def load_harmless_images(harmless: list[str], neutral_prompt: str) -> list[dict]:
    return [{"text": neutral_prompt, "image": render_prompt_image(p)} for p in harmless]
