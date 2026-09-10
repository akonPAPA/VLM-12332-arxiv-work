"""Overlay C1 R_V\\R_T energy vs normalized depth for the three models (scaling)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "research" / "results"
FIG = ROOT / "research" / "paper" / "figures" / "fig_c1_scaling.png"
MODELS = [
    ("phase1_cross_modal_map.json", "Qwen2-VL-2B", "#4C72B0", "o"),
    ("phase1_7b_instruct.json", "Qwen2-VL-7B", "#55A868", "s"),
    ("phase1_qwen3vl_30b.json", "Qwen3-VL-30B (MoE)", "#C44E52", "^"),
]


def series(fn):
    d = json.loads((RES / fn).read_text(encoding="utf-8"))
    pts = []
    for name, e in d["sites"].items():
        if name.startswith("text_residual.") and e.get("has_text"):
            pts.append((int(name.split(".")[1]), e["RV_minus_RT_residual"]))
    pts.sort()
    n = pts[-1][0]
    xs = [i / n for i, _ in pts]
    ys = [v for _, v in pts]
    return xs, ys


def main():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for fn, label, c, mk in MODELS:
        xs, ys = series(fn)
        ax.plot(xs, ys, marker=mk, ms=3, lw=1.5, color=c, label=label)
    ax.axhline(0.5, ls=":", c="#999", lw=1)
    ax.set_xlabel("normalized decoder depth (layer / final layer)")
    ax.set_ylabel(r"$R_V \setminus R_T$ energy")
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.set_title("Cross-modal refusal separation vs depth, across scale\n"
                 "(higher = more of the visual refusal cone text-space methods miss)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
