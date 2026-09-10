"""Minimal LoRA SFT for the C2 causal probe (transformers + peft, no Soup).

Reuses the same Qwen2-VL input construction as the extraction/eval code, so image
handling is identical to C1/C3. LoRA is restricted to the LANGUAGE-model attention
(regex target) so the vision tower stays frozen — any visual refusal cone that
appears at eval is then genuine cross-modal generalization, not trained visual
weights. Saves a MERGED full checkpoint that load_vlm can read directly.

Usage:
  python -m research.scripts.train_lora --base Qwen/Qwen2-VL-7B \
     --data research/data/safety_text.jsonl --out research/checkpoints/textsafety
  # condition C trains on both text and image rows:
  python -m research.scripts.train_lora --base Qwen/Qwen2-VL-7B \
     --data research/data/safety_text.jsonl research/data/safety_img.jsonl \
     --out research/checkpoints/imgsafety
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image

from research.scripts import loaders

TARGET_RE = r".*language_model\..*\.(q_proj|v_proj)$"  # LM attention only; vision frozen


def _rows(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _example(proc, row: dict, image_dir: Path):
    content, imgs = [], None
    if row.get("image"):
        content.append({"type": "image"})
        imgs = [Image.open(image_dir / row["image"]).convert("RGB")]
    content.append({"type": "text", "text": row["prompt"]})
    msgs = [{"role": "user", "content": content}]
    prompt_text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + row["response"] + (proc.tokenizer.eos_token or "")
    full = proc(text=[full_text], images=imgs, return_tensors="pt")
    plen = proc(text=[prompt_text], images=imgs, return_tensors="pt").input_ids.shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processor", default=None, help="borrow processor/chat-template (e.g. the Instruct model)")
    ap.add_argument("--image-dir", default="research/data/images")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model

    model, proc = loaders.load_vlm(args.base, processor_id=args.processor)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=TARGET_RE, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.train()

    rows = _rows(args.data)
    image_dir = Path(args.image_dir)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    dev = model.device

    step = 0
    for ep in range(args.epochs):
        random.shuffle(rows)
        opt.zero_grad()
        run = 0.0
        for i, row in enumerate(rows):
            try:
                batch = _example(proc, row, image_dir)
            except Exception as e:
                print(f"skip row {i}: {e}"); continue
            batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            run += float(out.loss)
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"epoch {ep} step {step} loss {run/ (args.grad_accum*5):.4f}", flush=True); run = 0.0
        opt.step(); opt.zero_grad()
    print("training done; merging + saving", flush=True)

    merged = model.merge_and_unload()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    proc.save_pretrained(args.out)
    print(f"saved merged checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
