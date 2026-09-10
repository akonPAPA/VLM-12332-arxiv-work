# Related work & the explicit delta

Verified Sept 2026. Grouped by what the paper must position against. The **Delta**
column states what our work adds; the thesis is *modality-entangled safety
subspaces* (`R_T` text cone, `R_V` visual cone, `R_V \ R_T` missed by text edits).

## Refusal geometry (text)

| Work | arXiv | Claim | Delta for us |
|---|---|---|---|
| Refusal is a single direction | 2406.11717 | Refusal mediated by one residual-stream direction | We test whether a *second, visual* direction exists |
| Concept cones / representational independence | 2502.17420 | Refusal is a multi-dim cone, several independent directions | We ask if one of those independent directions is image-borne |
| Linear instability (CLS) | 2606.22686 | Refusal vector removable by arithmetic, ~95% ASR | We show text-only removal is *incomplete* for VLMs |
| Fast multi-dim subspaces (RFM-AGOP) | 2607.02396 | Efficient multi-directional refusal subspace extraction | We reuse as an extractor; extend to visual taps |
| Refusal geometry reflects training | 2608.25390 | Diverse refusal prefixes raise stable rank, harden ablation | Informs our defense (C5) |

## Weight-space abliteration & merging

| Work | arXiv | Claim | Delta |
|---|---|---|---|
| Gabliteration | 2512.18901 | Multi-directional weight edit, less distortion | We port the weight-edit dual to the cross-attn write matrices |
| Surgical / spectral cleaning | 2601.08489 | Disentangle safety from capability via spectral edit | Contrast with our cross-modal capability-retention result |
| Cross-architecture comparison | 2512.13655 | Compares abliteration methods across arch | We add the modality axis |
| LED-Merging | 2502.16770 | Merging induces safety-utility conflict | Basis for the P4 safety-basin sweep |
| Safety drift via coupled constraints | 2604.12384 | Merging quietly destroys alignment | Motivates geometry-aware merge (C5) |

## Defenses (all text-space — the blind spot we exploit)

| Work | arXiv | Mechanism |
|---|---|---|
| AMRA (refusal aliases) | 2608.18093 | Weight edit hides the extractable refusal direction |
| DeepRefusal | 2509.15202 | Probabilistic ablation during FT rebuilds refusal, ~95% ASR drop |
| Extended-refusal FT | 2505.19056 | Justify-then-refuse data; cheapest to reproduce as our P2/P3 target |

## Multimodal safety

| Work | arXiv | Claim | Delta |
|---|---|---|---|
| Textual refusal directions for MM safety | 2606.31876 | Transfer text refusal dir to VLM steering | We *separate* `R_V` from `R_T` rather than transfer |
| Safety geometry collapse in MLLMs | 2605.18104 | Observational drift + correction on off-the-shelf VLMs | We add *causal* control via self-injected alignment |
| Concept-specific refusal vectors (RepIt) | 2509.13281 | Per-concept refusal steering | We map concepts onto modality cones |
| Visual-modality jailbreaks | 2605.00583 | Image channel bypasses safety | We give the geometric explanation + defense |

## Offensive-security link

| Work | arXiv | Claim | Delta |
|---|---|---|---|
| Ablating safety for security apps | 2605.17413 | Abliteration for security use cases | We measure *retention gap* text vs cross-modal |
| Refusal ≠ capability in code LLMs | 2606.05396 | Refusal and capability separable | We test the separation in the multimodal + cyber setting |
| Safety fails cybersecurity at scale | 2607.02714 | Alignment underperforms on cyber | Frames the C4 benchmark |

## One-line positioning

> Prior work removes or transfers a **text** refusal direction; VLM safety work is
> mostly observational. We show refusal in a gated-cross-attention VLM is a **union
> of modality-conditioned cones**, that a visual cone `R_V` carries a component
> text-space defenses miss, that exploiting it beats text-only abliteration against
> those defenses while retaining offensive-cyber capability, and that a
> geometry-aware (both-cone) merge defends it at lower utility cost.
