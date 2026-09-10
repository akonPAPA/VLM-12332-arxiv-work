"""End-to-end SYNTHETIC validation of the analysis pipeline.

This does NOT produce empirical findings. It plants a known text refusal cone
`R_T` and visual refusal cone `R_V` that share one axis (partial overlap), then
runs the *real* harness functions to confirm the pipeline (a) recovers the planted
geometry, and (b) makes predictions P1-P4 measurable through the code. Every
number here is a property of the synthetic construction, included so a reviewer
can run the whole chain with zero harmful content and see the figures render.

Run: python -m research.scripts.demo_synthetic
Outputs: research/results/synthetic/metrics.json  and  research/paper/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from research.src.geometry import (
    ablate_activations,
    category_refusal_subspace,
    orthonormalize,
    orthogonal_residual_energy,
    principal_angles,
    subspace_overlap,
)

SEED = 0
D = 256
N = 400
TAU = 1.0  # refusal threshold on the readout
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "research" / "paper" / "figures"
RES = ROOT / "research" / "results" / "synthetic"


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


def build_axes():
    """Four orthonormal axes: shared refusal, text-only, visual-only, capability."""
    g = torch.Generator().manual_seed(SEED)
    q = orthonormalize(torch.randn(D, 4, generator=g))
    return g, q[:, 0], q[:, 1], q[:, 2], q[:, 3]


def sample_categories(g, shared, specific, n_cat=5):
    """Harm categories whose MEANS span {shared, specific} -> a genuine 2D cone.

    Each category mean = a_c*shared + b_c*specific with distinct (a_c, b_c), so the
    SVD of the category-mean contrasts recovers span{shared, specific}.
    """
    cats = []
    for i in range(n_cat):
        a = 2.0 + 0.6 * i
        b = 3.0 - 0.4 * i
        mean = a * shared + b * specific
        cats.append(mean + torch.randn(N, D, generator=g) * 0.5)
    return cats


def asr_with_weight(acts: torch.Tensor, w: torch.Tensor) -> float:
    """ASR on a harmful set = fraction whose refusal readout falls below threshold."""
    return float((acts @ w <= TAU).float().mean())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    g, shared, text_ax, vis_ax, cap_ax = build_axes()

    harmless = torch.randn(N, D, generator=g) * 0.5

    # --- C1 / P1: recover both cones; measure shared vs orthogonal structure ---
    RT_hat = category_refusal_subspace(sample_categories(g, shared, text_ax), harmless, k=2)
    RV_hat = category_refusal_subspace(sample_categories(g, shared, vis_ax), harmless, k=2)
    angles = principal_angles(RT_hat, RV_hat)          # one ~0 (shared), one ~90 (specific)
    overlap = subspace_overlap(RT_hat, RV_hat)         # ~0.5 by construction
    residual = orthogonal_residual_energy(RV_hat, RT_hat)  # ~0.5: the R_V \ R_T that text edits miss
    both = orthonormalize(torch.cat([RT_hat, RV_hat], dim=1))  # R_T ∪ R_V (rank 3)

    # --- C3 / P2: DEFENDED model reads refusal mostly from the VISUAL axis ------
    # (text-space hardening left the visual channel carrying refusal). Text-only
    # abliteration removes {shared,text} but not vis -> refusal persists; only
    # cross-modal abliteration (both) collapses it.
    w_ref = _unit(0.6 * shared + 0.3 * text_ax + 1.8 * vis_ax)
    w_cap = cap_ax.clone()

    xe = 3.0 * shared + 2.5 * text_ax + 3.0 * vis_ax + torch.randn(N, D, generator=g) * 0.3

    asr_base = asr_with_weight(xe, w_ref)
    asr_text_only = asr_with_weight(ablate_activations(xe, RT_hat), w_ref)
    asr_cross = asr_with_weight(ablate_activations(xe, both), w_ref)

    def cap_retained(ablated: torch.Tensor) -> float:
        base = (xe @ w_cap).abs().mean().clamp_min(1e-6)
        return float(1.0 - (ablated @ w_cap - xe @ w_cap).abs().mean() / base)

    cap_text = cap_retained(ablate_activations(xe, RT_hat))
    cap_cross = cap_retained(ablate_activations(xe, both))

    # --- C5 / P4: safety basin — restoring refusal from a cross-abliterated ckpt.
    # Start = both-cone abliterated readout. Text-only re-alignment can only bring
    # back the {shared,text} part (visual stays broken -> residual ASR). Both-cone
    # merge restores everything.
    start = ablate_activations(w_ref, both)
    text_realign_target = ablate_activations(w_ref, RV_hat)  # visual still removed
    both_target = w_ref
    alphas = torch.linspace(0, 1, 11)
    basin_text = [asr_with_weight(xe, torch.lerp(start, text_realign_target, a)) for a in alphas]
    basin_both = [asr_with_weight(xe, torch.lerp(start, both_target, a)) for a in alphas]

    metrics = {
        "note": "SYNTHETIC pipeline validation - not empirical results.",
        "P1_principal_angles_deg": [round(float(a) * 180 / torch.pi, 2) for a in angles],
        "P1_subspace_overlap": round(overlap, 3),
        "P1_RV_minus_RT_residual_energy": round(residual, 3),
        "P2_ASR_base": round(asr_base, 3),
        "P2_ASR_text_only_abliteration": round(asr_text_only, 3),
        "P2_ASR_cross_modal_abliteration": round(asr_cross, 3),
        "P3_capability_retained_text_only": round(cap_text, 3),
        "P3_capability_retained_cross_modal": round(cap_cross, 3),
        "P4_residual_ASR_text_realign_at_alpha1": round(basin_text[-1], 3),
        "P4_residual_ASR_both_merge_at_alpha1": round(basin_both[-1], 3),
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _plot_angles(angles, residual)
    _plot_attack(asr_base, asr_text_only, asr_cross, cap_text, cap_cross)
    _plot_basin(alphas, basin_text, basin_both)

    print(json.dumps(metrics, indent=2))
    print(f"\nfigures -> {FIG}\nmetrics -> {RES / 'metrics.json'}")


def _plot_angles(angles: torch.Tensor, residual: float) -> None:
    deg = [float(a) * 180 / torch.pi for a in angles]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(range(1, len(deg) + 1), deg, color="#4C72B0")
    ax.axhline(90, ls="--", c="#888", lw=1)
    ax.set_xticks(range(1, len(deg) + 1))
    ax.set_xlabel("principal component"); ax.set_ylabel("angle (deg)")
    ax.set_title(f"R_T vs R_V principal angles\n(R_V \\ R_T energy = {residual:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "fig_p1_principal_angles.png", dpi=150); plt.close(fig)


def _plot_attack(a0, at, ac, ct, cc) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = range(3)
    ax.bar([i - 0.2 for i in x], [a0, at, ac], width=0.4, label="ASR (harmful)", color="#C44E52")
    ax.bar([i + 0.2 for i in x], [1.0, ct, cc], width=0.4, label="capability retained", color="#55A868")
    ax.set_xticks(list(x)); ax.set_xticklabels(["base", "text-only\nablation", "cross-modal\nablation"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate"); ax.legend(fontsize=8)
    ax.set_title("Attack success vs capability retention")
    fig.tight_layout(); fig.savefig(FIG / "fig_p3_attack_retention.png", dpi=150); plt.close(fig)


def _plot_basin(alphas, bt, bb) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    a = [float(x) for x in alphas]
    ax.plot(a, bt, "-o", ms=3, label="text-only re-alignment", color="#C44E52")
    ax.plot(a, bb, "-s", ms=3, label="both-cone merge", color="#4C72B0")
    ax.set_xlabel("merge α  (0 = abliterated → 1 = re-aligned)"); ax.set_ylabel("ASR (lower = safer)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Safety basin: only both-cone merge fully restores refusal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_p4_safety_basin.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
