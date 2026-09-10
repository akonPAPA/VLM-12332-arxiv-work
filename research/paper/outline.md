# Paper outline

**Title:** Where Does a VLM Refuse? Modality-Entangled Safety Subspaces,
Cross-Modal Abliteration, and Retained Offensive-Cyber Capability

**Target:** arXiv (cs.CR / cs.LG) → SaTML / TrustNLP / a NeurIPS-ICLR safety workshop.

## Abstract (claim skeleton)
Refusal in gated-cross-attention VLMs is a union of modality-conditioned cones; a
visual cone `R_V` carries a component text-space defenses miss; exploiting it beats
text-only abliteration against those defenses while retaining offensive-cyber
capability; a both-cone geometry-aware merge defends it at lower utility cost.

## 1. Introduction
- Abliteration & its (text-space) defenses; the VLM blind spot.
- Contributions C1–C5. Falsifiable predictions P1–P4.

## 2. Background
- Refusal direction/cone (2406.11717, 2502.17420).
- Abliteration: activation projection & weight orthogonalization.
- Model merging / task arithmetic / safety basin (2502.16770, 2604.12384).
- VLM architecture: ViT → resampler → gated cross-attn → decoder.

## 3. Related work & delta
- Use `docs/related_work.md` matrix. Explicit one-paragraph delta.

## 4. Method
- 4.1 Activation taps across the stack (`src/extraction.py`).
- 4.2 Direction & cone estimation; principal angles (`src/geometry.py`).
- 4.3 Ablation operators: activation projection & weight orthogonalization.
- 4.4 Self-injected alignment (LoRA) for causal control (C2).

## 5. C1 — Cross-modal refusal map
- `R_T` vs `R_V` per site; principal-angle spectrum; `R_V \ R_T` energy (P1).
- Figure: angle heatmap across the stack.

## 6. C2 — Controlled causal probe
- Inject known safety alignment on a base VLM; show where refusal *forms*.
- Ablating the injected data removes the cone → causal claim.

## 7. C3 — Cross-modal abliteration attack
- Text-only vs cross-modal abliteration vs a reproduced text-space defense (P2).
- Table: ASR (undefended / defended) × (text-only / cross-modal).

## 8. C4 — Offensive-cyber capability retention
- Capability retained after each abliteration variant on the curated cyber subset (P3).
- Table: refusal↓ vs capability→ ; the retention gap.

## 9. C5 — Geometry-aware defense & safety basin
- Basin sweep over merge alphas; anisotropy; both-cone-preserving merge (P4).
- Pareto plot: refusal restored vs utility retained, vs text-only re-alignment.

## 10. Ethics / broader impact
- From `docs/ethics.md`.

## 11. Limitations & future work
- 2B/4-bit scale; small cyber subset; from-scratch transFORme-r probe as future
  fully-controlled testbed.

## Reproducibility
- Seeds, configs (`configs/experiments.yaml`), scripts; no harmful content or
  abliterated weights released.
