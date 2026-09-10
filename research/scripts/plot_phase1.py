"""Plot the real Phase-1 cross-modal map: overlap and R_V\\R_T energy vs depth.

Reads research/results/phase1_cross_modal_map.json and writes a figure to
research/paper/figures/fig_p1_real_depth.png. Empirical, from a real VLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "research" / "results" / "phase1_cross_modal_map.json"
FIG = ROOT / "research" / "paper" / "figures" / "fig_p1_real_depth.png"


def main() -> None:
    data = json.loads(RES.read_text(encoding="utf-8"))
    sites = data["sites"]
    layers, overlap, residual, first_angle = [], [], [], []
    for name, e in sites.items():
        if name.startswith("text_residual.") and e.get("has_text"):
            layers.append(int(name.split(".")[1]))
            overlap.append(e["overlap"])
            residual.append(e["RV_minus_RT_residual"])
            first_angle.append(e["principal_angles_deg"][0])
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    layers = [layers[i] for i in order]
    overlap = [overlap[i] for i in order]
    residual = [residual[i] for i in order]
    first_angle = [first_angle[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(layers, residual, "-o", ms=3, color="#4C72B0", label=r"$R_V\setminus R_T$ energy")
    ax1.plot(layers, overlap, "-s", ms=3, color="#55A868", label="subspace overlap")
    ax1.axhline(0.5, ls=":", c="#999", lw=1)
    ax1.set_xlabel("decoder layer"); ax1.set_ylabel("fraction of energy"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(layers, first_angle, "-^", ms=3, color="#C44E52", alpha=0.7, label="1st principal angle (deg)")
    ax2.set_ylabel("1st principal angle (deg)", color="#C44E52"); ax2.set_ylim(0, 90)
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    n = data.get("n_per_class"); k = data.get("subspace_k"); cats = len(data.get("categories", []))
    ax1.set_title(f"Cross-modal refusal geometry vs depth — {data['model'].split('/')[-1]}\n"
                  f"(n={n}/class, {cats} harm categories, k={k})")
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)
    print(f"wrote {FIG}")
    print(f"layers={len(layers)}  residual range [{min(residual):.2f},{max(residual):.2f}]  "
          f"overlap range [{min(overlap):.2f},{max(overlap):.2f}]")


if __name__ == "__main__":
    main()
