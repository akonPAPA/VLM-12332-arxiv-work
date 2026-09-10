Where Does a VLM Refuse?
Modality-Entangled Safety Subspaces, Cross-Modal Abliteration,
and Retained Offensive-Cyber Capability
Akan Mukhametgali
MU
September 10, 2026
Abstract
Abliteration—removing the low-dimensional “refusal direction” from an aligned model’s activation or weight space—has become a standard probe of how brittle safety alignment is, and
a family of defenses now hardens language models against it. Almost all of this work, attacks
and defenses alike, lives in the text representation. We ask a question specific to vision-language
models (VLMs): is there a refusal direction carried by the visual pathway that is distinct from
the text refusal cone, and if so, does it make text-space defenses circumventable through the
image channel? We formalize refusal in a VLM as a union of modality-conditioned cones, a
text cone RT and a visual cone RV , and treat activation abliteration and weight-space model
souping as dual operators on this union. We contribute (i) a layer-wise cross-modal refusal map
that measures the principal angles between RT and RV and the energy of RV \ RT that text
edits miss, shown scale-robust across three VLMs; (ii) a causal probe that injects known safety
onto a base VLM and isolates a text→image refusal-transfer gap; (iii) an honest-negative study
of a cross-modal abliteration attack; and (iv) a demonstrated both-modality defense that closes
the visual gap, with a weight-space safety-basin extension outlined for future work. Empirically, C1/P1 holds and grows with scale across three VLMs (Qwen2-VL-2B, Qwen2-VL-7B, and
the Qwen3-VL-30B mixture-of-experts): the image-conditioned refusal cone keeps 0.34–0.69 of
its energy orthogonal to the text cone at 2B/7B and ≈ 0.6 even at the final layer of the 30B
model. Causally (C2), injecting safety into the base Qwen2-VL-7B via a language-only LoRA
yields 100% refusal on harmful text but only 30% on harmful images—the geometric gap is a
real behavioral hole—while adding image supervision closes it to 100%. A cross-modal abliteration attack advantage (C3), by contrast, was not established at this scale (an honest negative).
We release the analysis harness and a synthetic validation of the pipeline. We do not release
abliterated weights, trained safety-stripped checkpoints, or a runnable cross-modal attack recipe.
1 Introduction
Safety alignment in modern chat models is, to a striking degree, linear: refusal behavior is mediated
by a low-dimensional subspace of the residual stream that can be found by contrasting activations
on harmful and harmless prompts [15, 18]. “Abliteration” exploits this by projecting the subspace
out at inference or permanently editing the weights that write it, disabling refusal while largely
preserving fluency. A defensive literature has responded in kind, hiding or rebuilding the refusal
direction with weight edits and fine-tuning [6, 1, 2].
Two things are conspicuous. First, both the attacks and the defenses operate in the text representation. Second, VLM safety is studied mostly observationally on off-the-shelf checkpoints [13, 9].
1
This leaves a concrete, security-relevant question unanswered: in a VLM, is refusal that is triggered
by an image mediated by the same direction as refusal triggered by text? If a distinct visual refusal
component exists, a defender who only hardens the text direction has left a door open, and an
attacker who abliterates only the text direction is leaving capability on the table.
Thesis (modality-entangled safety subspaces). We model refusal in a gated-cross-attention
VLM as a union of modality-conditioned cones, a text cone RT and an image cone RV that share
some directions and differ in others. Safety edits are operators on this union: activation abliteration
projects onto the orthogonal complement of a chosen cone; weight souping / task arithmetic [19, 17]
translates the model along it. The core claim is that RV \RT 6= ∅, and that this residual is exactly
what text-only edits and text-only defenses miss.
Predictions and how they fared. P1 (separation) RV has a component orthogonal to RT
at every layer — confirmed, and it grows with scale (C1). P2 (causal transfer gap) safety
supervised only on text transfers only partially to harmful images — confirmed: 100% text vs.
30% image refusal (C2). P3 (attack) cross-modal abliteration yields more successful compliance
than text-only abliteration — not established at this scale (C3, an honest negative). P4 (defense)
aligning both modality cones closes the visual gap — confirmed: adding image supervision lifts
image refusal to 100% (C5).
Contributions. C1 a cross-modal refusal map that we show is scale-robust across three VLMs;
C2 a causal alignment-injection probe isolating a text→ image refusal-transfer gap; C3 an honestnegative cross-modal abliteration attack study; C5 a demonstrated both-modality defense. We
release the analysis harness, a hand-rolled LoRA trainer, and a harmless synthetic validation of the
full pipeline.
2 Background
Refusal geometry and abliteration. Given mean residual activations µh (harmful) and µℓ
(harmless) at a layer, the difference-in-means direction rˆ = (µh − µℓ)/k·k is the rank-1 refusal
axis [15]. Across harm categories the refusal signal spans a low-rank cone [18, 7]. Inference-time
abliteration replaces each activation x by x − QQ⊤x for an orthonormal basis Q of the cone; permanent abliteration applies W′ = W −QQ⊤W to matrices that write into the residual stream [16].
Defenses hide the extractable direction (AMRA [6]), rebuild it during fine-tuning (DeepRefusal [1]),
or add justify-then-refuse data [2].
Merging, task arithmetic, and the safety basin. Weight averaging (“model soups” [19]) and
task arithmetic [17] move a model along low-dimensional directions in weight space; several works
show that naive merging silently degrades alignment and that safety occupies a narrow basin [3, 12].
We use interpolation as a probe of that basin in the multimodal setting.
VLM structure. We consider the Flamingo/BLIP family: a ViT encoder, a resampler that
maps patches to a fixed set of visual tokens, and a decoder that attends to those tokens (gated
cross-attention, or token-merging as in Qwen2-VL). We tap the residual stream at four logical sites:
vision output, resampler latents, cross-attention/decoder layers, and the text residual stream.
2
3 Related work and delta
Prior work removes or transfers a text refusal direction [15, 8, 9], studies the refusal cone’s structure [18, 7], hardens models against text abliteration [6, 1, 2], or observes multimodal safety drift on
off-the-shelf VLMs [13, 4, 10]. Security-facing work shows refusal and capability are separable and
that alignment underperforms on cyber tasks [5, 14, 11]. Our delta is the intersection none of these
occupy: a cross-modal subspace separation (RV \RT ), its use to circumvent text-space defenses, the
capability retained when it is exploited, and a both-cone defense, all under a self-injected alignment
that gives causal control over how refusal forms.
4 Method
Cone estimation. Per modality we collect activations for several harm categories and one harmless set at each site. Stacking the category-mean contrasts {µc − µℓ} and taking the top-k right
singular vectors yields an orthonormal basis for the cone (this is the correct estimator: a single harmful/harmless split has rank-1 between-class scatter, so a genuine multi-axis cone requires
multiple harm categories).
Subspace comparison (P1). For cones RT , RV with orthonormal bases QT , QV , the principal
angles are arccos of the singular values of Q⊤
T QV ; the missed energy is 1 − kQTQ⊤
T QV k
2
F
/kQV k
2
F
.
Ablation operators. Activation projection x 7→ x − QQ⊤x and weight orthogonalization W 7→
W − QQ⊤W share one implementation, so the attack (C3) and the souping analysis (C5) use the
same definition of “remove this subspace.”
Causal control (C2). Rather than train a VLM from scratch (infeasible on our budget), we
take a base, non-safety-tuned VLM and inject a known safety alignment via a small LoRA whose
data we control. Ablating the injected data lets us attribute the resulting refusal cone causally to
it—an attribution off-the-shelf studies cannot make.
5 Pipeline validation (synthetic, harmless)
Before any model run we validate the analysis chain on planted data: we construct a text cone
RT = span{s, t} and a visual cone RV = span{s, v} sharing one axis s, plus an orthogonal capability
axis, and run the real harness functions. Figure 1 shows the pipeline recovers the planted geometry
(one shared, one orthogonal axis; RV \ RT energy ≈ 0.5), that a defended readout reading refusal
from the visual axis is untouched by text-only ablation but collapses under cross-modal ablation
while the capability axis is preserved, and that only a both-cone merge fully restores refusal along
the interpolation path. These numbers are properties of the construction, not empirical findings;
they exist to show the code measures what it claims.
6 Empirical protocol and results
Extraction (C1) and abliteration (C3) run on open instruct VLMs; the causal probe (C2) fine-tunes
the base Qwen2-VL-7B on a single H100. Harmful prompts come from mlabonne/harmful_behaviors
(bucketed into five categories) and benign prompts from mlabonne/harmless_alpaca; the visual
3
Figure 1: Synthetic validation of the analysis pipeline (harmless planted data). Left (P1): principal
angles between recovered RT and RV —one shared axis (≈ 9
◦
), one nearly orthogonal (≈ 87◦
).
Middle (P2/P3): text-only ablation leaves the defended readout’s ASR unchanged while crossmodal ablation collapses refusal, capability retained. Right (P4): text-only re-alignment plateaus
with residual ASR; the both-cone merge restores safety.
modality is produced by rendering each instruction as a FigStep-style typographic image, so no
gated multimodal dataset is required and no harmful image content is authored. Refusal is scored
with a lexical refusal classifier (a stronger Llama-Guard-Vision judge is left as a fidelity upgrade).
All results below are produced by research/scripts/run_phase.py and train_lora.py and are
real runs; no placeholder numbers are reported.
C1/P1 result: separation holds and grows with scale. We extract RT and RV as rank-4
cones from category-mean contrasts (five harm categories: cyber, weapons, drugs, fraud, harm;
n=48–64/class; visual modality = FigStep-style typographic renderings of the same instructions)
at every decoder layer, for three checkpoints spanning 15× in size and two architecture generations:
Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct, and the Qwen3-VL-30B-A3B mixture-of-experts.
P1 holds at every layer of every model (Table 1, Figure 2): the image-conditioned refusal
cone always keeps a large component orthogonal to the text cone. The gap is largest in early layers,
narrows as the modalities integrate with depth, and never closes—and it is larger in the 30B model,
where ≈ 0.6 of the visual refusal signal still lies outside the text cone even at the final layer. This
is exactly the residual a text-space edit or defense would miss, and it does not diminish with scale.
Table 1: C1/P1 (real): RV \ RT energy (fraction of the visual refusal cone orthogonal to the text
cone) by depth band, across three VLMs. Higher = more of the visual refusal signal that text-space
methods cannot reach. Never near zero; largest in the 30B model.
Model early layers mid layers late layers
Qwen2-VL-2B-Instruct 0.69 0.42 0.34
Qwen2-VL-7B-Instruct 0.67 0.41 0.36
Qwen3-VL-30B-A3B (MoE) 0.85 0.59 0.60
Caveats. Residuals are mean-pooled and the visual modality is a single typographic (FigStep)
style; photographic harmful images may differ. As the C2 probe shows, mean-pooled diff-in-means
understates behaviorally-real effects, so these geometry numbers are a conservative lower bound on
the separation.
4
Figure 2: RV \ RT energy vs. normalized decoder depth for three VLMs. The visual refusal cone
keeps a large component outside the text cone at every layer of every model; separation is largest
early, never closes, and is highest in the 30B MoE (staying ≈ 0.6 at the final layer). Text-space
refusal edits and defenses cannot reach this residual, and it does not shrink with scale.
C3/P2 result: inconclusive at this scale (an honest negative). We measured attacksuccess rate (ASR) on 60 held-out harmful prompts under three conditions—no ablation, text-only
(RT ) and cross-modal (RT ∪ RV ) rank-1 ablation applied at all decoder layers—for a text eval set
and a FigStep-image eval set (Table 2). Two findings stand out. First, raw ASR (any non-refusal)
saturates: rank-1 text-space ablation alone drives the refusal rate from 1.0 to 0.0 in both modalities,
so the naive metric cannot separate the conditions. Second, once we guard against incoherent output
with a coherence proxy, the coherent-compliance rates are low and not robust: the sign of the
text-only vs. cross-modal difference flipped between a pilot (n=30) and this run (n=60) on the
image eval, and the text-eval gap is within noise (0.08 vs. 0.10). We therefore do not claim P2
holds: at this scale and with all-layer rank-1 ablation (which itself degrades coherence), there is no
reliable evidence that removing the visual cone functionally beats removing the text cone—even
though the two cones are representationally distinct (P1). Establishing or refuting P2 needs a
model-based judge (Llama-Guard-Vision) in place of the coherence proxy, a coherence-preserving
ablation (single-layer or calibrated strength), larger n with paired significance tests, and multiple
models. This is a clarifying result: representational separation does not by itself imply a functional
attack advantage.
C2 result: text-only safety leaves a visual behavioral hole (the main causal finding).
Starting from the base (non-safety-tuned) Qwen2-VL-7B, we inject a known safety alignment with
a LoRA restricted to the language-model attention (q_proj,v_proj; the vision tower is frozen),
in two conditions: (B) supervised only on harmful text→refusal (plus benign→answer), and (C)
additionally on FigStep harmful images→refusal. We then measure refusal on held-out harmful
text and harmful images (Table 3). The base model refuses almost nothing. Text-only safety (B)
makes it refuse 100% of harmful text but only 30% of harmful images, even though its
LoRA touched only the language attention—refusal transfers across modalities, but incompletely.
Adding image supervision (C) closes the gap to 100%. The ≈70-point residual under text-only
5
Table 2: C3/P2 (real, Qwen2-VL-2B, n=60 held-out, rank-1 all-layer ablation). Raw ASR = any
non-refusal; coherent = non-refusal passing a coherence check. The coherent-compliance ordering
is unstable across sample sizes (see text).
Eval Ablation raw ASR coherent ASR
text
none 0.00 0.00
text-only 1.00 0.08
cross-modal 1.00 0.10
image
none 0.00 0.00
text-only 1.00 0.53
cross-modal 1.00 0.22
safety is the behavioral image of the geometric RV \ RT from C1: the part of the visual refusal
cone that text-space supervision does not reach is a real, exploitable behavioral gap. (Notably,
mean-pooled diff-in-means separability barely moved across B/C, while behavior moved from 0%
to 100%—so we report behavioral refusal, and treat the C1 geometry numbers as a conservative
probe.)
Table 3: C2 (real, causal): refusal rate after self-injected safety on base Qwen2-VL-7B (LoRA on
LM attention only), n=20 held-out harmful prompts per modality. Text-only safety transfers to
images only partially (30%); image safety closes it.
Injected safety harmful-text refusal harmful-image refusal
none (base) 15% 0%
text-only 100% 30%
text + image 100% 100%
7 C5 — the both-modality defense
The constructive counterpart follows directly from C2. Because text-space safety supervision leaves
the visual channel largely uncovered (30% image refusal), the fix is to make refusal supervision
modality-complete: add harmful-image → refusal examples so both cones are aligned. Condition C
is exactly this defense, and it works—image refusal rises from 30% to 100% while text refusal stays
at 100% (Table 3), at the cost of a small amount of image supervision (80 examples) on top of the
text set. The takeaway for practitioners is concrete: a VLM safety-tuned or abliteration-hardened
only in the text representation inherits a visual hole that scales-invariantly persists (C1); closing it
requires supervision (or defense edits) that touch the visual pathway. Casting this as a weight-space
operation—interpolating or task-arithmetic-merging a text-safety and an image-safety adapter and
tracing the resulting “safety basin”—is a natural extension we leave to future work; our harness
already exposes the merge primitives (geometry.lerp_state).
8 Ethics and responsible disclosure
This is dual-use safety research, and its net contribution is defensive: the central result is that textonly safety leaves an exploitable visual hole, and the fix (C5) is to align both modality cones. We use
6
only public prompt sets and locally-rendered typographic images; we author only refusals and benign answers, never harmful responses; we report aggregate refusal rates, never harmful transcripts;
and we do not release abliterated weights, the safety-added/removed checkpoints, or a runnable
cross-modal attack recipe. Any concrete bypass of a specific deployed model would be disclosed to
its vendor under embargo before public posting. Full policy in the released docs/ethics.md.
9 Limitations and future work
Refusal is scored by a lexical classifier rather than a model judge; held-out behavioral evals use n=20
prompts per modality; the visual modality is a single typographic (FigStep) style, so photographic
harmful images remain to be tested. C1 uses mean-pooled residuals, which C2 shows understates
behaviorally-real effects—so the geometry numbers are conservative. The cross-modal abliteration
attack (C3) was inconclusive at this scale and needs a model judge and a coherence-preserving
ablation to settle. The causal probe injects safety via a language-only LoRA rather than full
pretraining; a fully controlled testbed—a VLM trained from scratch with known safety data (the
companion transFORme-r model)—and a weight-space safety-basin study of both-cone merging
are the main future directions.
10 Conclusion
Refusal in a vision-language model is not a single text direction but a union of modality-conditioned
cones. Across three VLMs and two architecture generations, the image-conditioned refusal cone
keeps a large, scale-robust component orthogonal to the text cone (C1). That geometric gap is
behaviorally real: safety supervised only on text produces full text refusal but leaves most harmful
images un-refused (C2), and the fix is to align both modality cones (C5). Whether the residual also
yields a clean cross-modal attack advantage remains open (C3). The practical message is simple:
VLM safety and abliteration defenses built in the text representation alone inherit a visual hole
that does not shrink with scale, and must be closed in the visual pathway. We release the harness,
the LoRA trainer, and a harmless end-to-end validation.
References
[1] Anonymous. Beyond surface alignment: Rebuilding llms safety mechanism via probabilistically
ablating refusal direction. arXiv preprint arXiv:2509.15202, 2025.
[2] Anonymous. An embarrassingly simple defense against llm abliteration attacks. arXiv preprint
arXiv:2505.19056, 2025.
[3] Anonymous. Led-merging: Mitigating safety-utility conflicts in model merging. arXiv preprint
arXiv:2502.16770, 2025.
[4] Anonymous. Repit: Steering language models with concept-specific refusal vectors. arXiv
preprint arXiv:2509.13281, 2025.
[5] Anonymous. Ablating safety: Mechanisms for removing alignment in language models for
security applications. arXiv preprint arXiv:2605.17413, 2026.
[6] Anonymous. Abliteration mitigation via refusal aliases. arXiv preprint arXiv:2608.18093,
2026.
7
[7] Anonymous. Fast multi-dimensional refusal subspaces via rfm-agop. arXiv preprint
arXiv:2607.02396, 2026.
[8] Anonymous. The geometry of refusal: Linear instability in safety-aligned llms. arXiv preprint
arXiv:2606.22686 (TrustNLP 2026), 2026.
[9] Anonymous. Harnessing textual refusal directions for multimodal safety. arXiv preprint
arXiv:2606.31876, 2026.
[10] Anonymous. Jailbreaking vision-language models through the visual modality. arXiv preprint
arXiv:2605.00583, 2026.
[11] Anonymous. Not all refusals are equal: How safety alignment fails cybersecurity at scale.
arXiv preprint arXiv:2607.02714, 2026.
[12] Anonymous. Preventing safety drift in large language models via coupled weight and activation
constraints. arXiv preprint arXiv:2604.12384, 2026.
[13] Anonymous. Safety geometry collapse in multimodal llms and adaptive drift correction. arXiv
preprint arXiv:2605.18104, 2026.
[14] Anonymous. Willing but unable: Separating refusal from capability in code llms via abliteration. arXiv preprint arXiv:2606.05396, 2026.
[15] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and
Neel Nanda. Refusal in language models is mediated by a single direction. arXiv preprint
arXiv:2406.11717, 2024.
[16] Gökdeniz Gülmez. Gabliteration: Adaptive multi-directional neural weight modification. arXiv
preprint arXiv:2512.18901, 2025.
[17] Gabriel Ilharco et al. Editing models with task arithmetic. ICLR, 2023.
[18] Tom Wollschläger et al. The geometry of refusal in large language models: Concept cones and
representational independence. arXiv preprint arXiv:2502.17420, 2025.
[19] Mitchell Wortsman et al. Model soups: Averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time. ICML, 2022.
8
