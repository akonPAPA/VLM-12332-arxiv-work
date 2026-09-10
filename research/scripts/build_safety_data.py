"""Build controlled safety-alignment SFT data for C2 (unified schema for our
own trainer, research/scripts/train_lora.py).

Rows are {"prompt", "response", optional "image"}:
  * safety_text.jsonl — condition B (TRULY text-only): harmful -> refusal,
    benign -> real answer. No image.
  * safety_img.jsonl  — image-conditioned: FigStep image of a harmful instruction
    -> refusal. Condition C trains on BOTH files (text + image safety).

Only refusals / benign answers are authored; harmful prompts are the standard
public set; no harmful responses written (docs/ethics.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.scripts import loaders

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data"
IMG_DIR = DATA / "images"
REFUSALS = [
    "I can't help with that request.",
    "I'm sorry, but I can't assist with that.",
    "I can't provide that. It could cause harm.",
    "That's something I won't help with.",
    "I'm not able to help with that request.",
]
NEUTRAL_IMG_PROMPT = "Follow the instruction shown in the image."


def _benign_pairs(n: int) -> list[dict]:
    from datasets import load_dataset

    for ds_id in ("mlabonne/harmless_alpaca", "tatsu-lab/alpaca"):
        try:
            ds = load_dataset(ds_id, split="train")
        except Exception:
            continue
        cols = ds.column_names
        instr = "instruction" if "instruction" in cols else ("text" if "text" in cols else cols[0])
        out = "output" if "output" in cols else ("response" if "response" in cols else None)
        rows = []
        for r in ds:
            if r.get(instr) and (out is None or r.get(out)):
                rows.append({"prompt": r[instr], "response": r[out] if out else "Sure, here is a helpful answer."})
            if len(rows) >= n:
                break
        if rows:
            return rows
    raise RuntimeError("no benign dataset available")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-harmful", type=int, default=150)
    ap.add_argument("--n-benign", type=int, default=150)
    ap.add_argument("--n-vision", type=int, default=80)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cats = ["cyber", "weapons", "drugs", "fraud", "harm"]

    harmful = loaders.load_harmful_by_category(max(args.n_harmful, args.n_vision), cats)
    pool = [p for v in harmful.values() for p in v]

    text_rows = [{"prompt": p, "response": REFUSALS[i % len(REFUSALS)]} for i, p in enumerate(pool[: args.n_harmful])]
    text_rows += _benign_pairs(args.n_benign)
    (DATA / "safety_text.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in text_rows), encoding="utf-8")
    print(f"safety_text.jsonl: {len(text_rows)} rows (text-only)")

    img_rows = []
    for i, p in enumerate(pool[: args.n_vision]):
        fn = f"harm_{i:04d}.png"
        loaders.render_prompt_image(p).save(IMG_DIR / fn)
        img_rows.append({"image": fn, "prompt": NEUTRAL_IMG_PROMPT, "response": REFUSALS[i % len(REFUSALS)]})
    (DATA / "safety_img.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in img_rows), encoding="utf-8")
    print(f"safety_img.jsonl: {len(img_rows)} image rows (+images in {IMG_DIR})")


if __name__ == "__main__":
    main()
