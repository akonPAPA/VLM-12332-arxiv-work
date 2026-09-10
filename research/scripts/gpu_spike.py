"""Load a large model and run continuous benign generation to drive GPU
telemetry high (for a utilization screenshot). Not part of the research; purely a
load generator. Ctrl-C or --seconds to stop.

Usage: python -m research.scripts.gpu_spike --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ --seconds 300
"""

from __future__ import annotations

import argparse
import time

import torch

from research.scripts import loaders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    model, proc = loaders.load_vlm(args.model)
    model.eval()
    prompt = "Explain, in as much technical detail as possible, how a transformer neural network works."
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text] * args.batch, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    free, total = torch.cuda.mem_get_info()
    print(f"loaded {args.model} | VRAM used {(total-free)/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    t0 = time.time(); it = 0
    while time.time() - t0 < args.seconds:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=args.tokens, do_sample=True, temperature=0.8)
        it += 1
        used = (total - torch.cuda.mem_get_info()[0]) / 1e9
        rate = args.batch * args.tokens * it / (time.time() - t0)
        print(f"iter {it}  ~{rate:.0f} tok/s  VRAM {used:.1f} GB  elapsed {time.time()-t0:.0f}s", flush=True)
    print("spike done", flush=True)


if __name__ == "__main__":
    main()
