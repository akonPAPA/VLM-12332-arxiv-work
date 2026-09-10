"""C2 causal comparison: does a refusal cone FORM after self-injected safety,
and does a VISUAL cone (img_sep) appear under TEXT-ONLY safety data?

Reads research/results/phase1_{base,textsafety,imgsafety}.json (whichever exist)
and writes research/paper/figures/fig_c2_causal.png + a printed summary.

Metrics per decoder layer:
  text_sep / img_sep  = scale-free refusal-signal strength (||mu_h-mu_l|| / norm)
  RV_minus_RT_residual = fraction of the visual cone outside the text cone (P1)
The headline read: img_sep in condition 'textsafety' vs 'base' — if it rises, the
visual refusal cone generalizes from text-only safety data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESD = ROOT / "research" / "results"
FIG = ROOT / "research" / "paper" / "figures" / "fig_c2_causal.png"
CONDS = [("base", "#999999"), ("textsafety", "#4C72B0"), ("imgsafety", "#C44E52")]


def _load(tag: str):
    p = RESD / f"phase1_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _series(data, key):
    layers, vals = [], []
    for name, e in data["sites"].items():
        if name.startswith("text_residual.") and key in e:
            layers.append(int(name.split(".")[1])); vals.append(e[key])
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    return [layers[i] for i in order], [vals[i] for i in order]


def main() -> None:
    loaded = [(t, c, _load(t)) for t, c in CONDS]
    loaded = [(t, c, d) for t, c, d in loaded if d is not None]
    if not loaded:
        print("no phase1_{base,textsafety,imgsafety}.json found — run the C2 extractions first")
        return

    fig, (axT, axV) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for tag, color, d in loaded:
        lt, vt = _series(d, "text_sep")
        lv, vv = _series(d, "img_sep")
        if vt:
            axT.plot(lt, vt, "-o", ms=3, color=color, label=tag)
        if vv:
            axV.plot(lv, vv, "-s", ms=3, color=color, label=tag)
    axT.set_title("Text refusal signal (text_sep)"); axV.set_title("Visual refusal signal (img_sep)")
    for ax in (axT, axV):
        ax.set_xlabel("decoder layer"); ax.set_ylabel("||mu_h - mu_l|| / norm"); ax.legend(fontsize=8)
    fig.suptitle("C2 causal probe: refusal-cone formation under self-injected safety")
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)
    print(f"wrote {FIG}")

    # printed summary: mean signal over late layers (>=20)
    print(f"\n{'condition':12s} {'text_sep(late)':>14s} {'img_sep(late)':>14s}")
    for tag, _c, d in loaded:
        lt, vt = _series(d, "text_sep"); lv, vv = _series(d, "img_sep")
        mt = sum(v for l, v in zip(lt, vt) if l >= 20) / max(1, sum(l >= 20 for l in lt))
        mv = sum(v for l, v in zip(lv, vv) if l >= 20) / max(1, sum(l >= 20 for l in lv))
        print(f"{tag:12s} {mt:14.3f} {mv:14.3f}")


if __name__ == "__main__":
    main()
