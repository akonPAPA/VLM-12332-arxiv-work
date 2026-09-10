# **Where Does a VLM Refuse?** 

# Modality-Entangled Safety Subspaces, Cross-Modal Abliteration, and Retained Offensive-Cyber Capability 



### **Abstract** 

Abliteration—removing the low-dimensional “refusal direction” from an aligned model’s activation or weight space—has become a standard probe of how brittle safety alignment is, and a family of defenses now hardens language models against it. Almost all of this work, attacks and defenses alike, lives in the _text_ representation. We ask a question specific to vision-language models (VLMs): is there a refusal direction carried by the _visual_ pathway that is distinct from the text refusal cone, and if so, does it make text-space defenses circumventable through the image channel? We formalize refusal in a VLM as a _union of modality-conditioned cones_ , a text cone _RT_ and a visual cone _RV_ , and treat activation abliteration and weight-space model souping as dual operators on this union. We contribute (i) a layer-wise cross-modal refusal map that measures the principal angles between _RT_ and _RV_ and the energy of _RV \ RT_ that text edits miss, shown scale-robust across three VLMs; (ii) a _causal_ probe that injects known safety onto a base VLM and isolates a text _→_ image refusal-transfer gap; (iii) an honest-negative study of a cross-modal abliteration attack; and (iv) a demonstrated both-modality defense that closes the visual gap, with a weight-space safety-basin extension outlined for future work. Empirically, C1/P1 holds and _grows with scale_ across three VLMs (Qwen2-VL-2B, Qwen2-VL-7B, and the Qwen3-VL-30B mixture-of-experts): the image-conditioned refusal cone keeps 0 _._ 34–0 _._ 69 of its energy orthogonal to the text cone at 2B/7B and _≈_ 0 _._ 6 even at the final layer of the 30B model. Causally (C2), injecting safety into the _base_ Qwen2-VL-7B via a language-only LoRA yields 100% refusal on harmful text but only 30% on harmful images—the geometric gap is a real behavioral hole—while adding image supervision closes it to 100%. A cross-modal abliteration _attack_ advantage (C3), by contrast, was not established at this scale (an honest negative). We release the analysis harness and a synthetic validation of the pipeline. _We do not release abliterated weights, trained safety-stripped checkpoints, or a runnable cross-modal attack recipe._ 

## **1 Introduction** 

Safety alignment in modern chat models is, to a striking degree, _linear_ : refusal behavior is mediated by a low-dimensional subspace of the residual stream that can be found by contrasting activations on harmful and harmless prompts [15, 18]. “Abliteration” exploits this by projecting the subspace out at inference or permanently editing the weights that write it, disabling refusal while largely preserving fluency. A defensive literature has responded in kind, hiding or rebuilding the refusal direction with weight edits and fine-tuning [6, 1, 2]. 

Two things are conspicuous. First, both the attacks and the defenses operate in the _text_ representation. Second, VLM safety is studied mostly _observationally_ on off-the-shelf checkpoints [13, 9]. 

1 

This leaves a concrete, security-relevant question unanswered: in a VLM, is refusal that is triggered _by an image_ mediated by the same direction as refusal triggered by text? If a distinct visual refusal component exists, a defender who only hardens the text direction has left a door open, and an attacker who abliterates only the text direction is leaving capability on the table. 

**Thesis (modality-entangled safety subspaces).** We model refusal in a gated-cross-attention VLM as a union of modality-conditioned cones, a text cone _RT_ and an image cone _RV_ that share some directions and differ in others. Safety edits are operators on this union: activation abliteration projects onto the orthogonal complement of a chosen cone; weight souping / task arithmetic [19, 17] translates the model along it. The core claim is that _RV \ RT_ = ∅, and that this residual is exactly what text-only edits and text-only defenses miss. 

**Predictions and how they fared. P1 (separation)** _RV_ has a component orthogonal to _RT_ at every layer — _confirmed_ , and it grows with scale (C1). **P2 (causal transfer gap)** safety supervised only on text transfers only partially to harmful images — _confirmed_ : 100% text vs. 30% image refusal (C2). **P3 (attack)** cross-modal abliteration yields more successful compliance than text-only abliteration — _not established_ at this scale (C3, an honest negative). **P4 (defense)** aligning both modality cones closes the visual gap — _confirmed_ : adding image supervision lifts image refusal to 100% (C5). 

**Contributions.** C1 a cross-modal refusal map that we show is scale-robust across three VLMs; C2 a causal alignment-injection probe isolating a text _→_ image refusal-transfer gap; C3 an honestnegative cross-modal abliteration attack study; C5 a demonstrated both-modality defense. We release the analysis harness, a hand-rolled LoRA trainer, and a harmless synthetic validation of the full pipeline. 

## **2 Background** 

**Refusal geometry and abliteration.** Given mean residual activations _µh_ (harmful) and _µℓ_ ˆ (harmless) at a layer, the difference-in-means direction _r_ = ( _µh − µℓ_ ) _/∥·∥_ is the rank-1 refusal axis [15]. Across harm categories the refusal signal spans a low-rank _cone_ [18, 7]. Inference-time abliteration replaces each activation _x_ by _x − QQ_<sup>_⊤_</sup> _x_ for an orthonormal basis _Q_ of the cone; permanent abliteration applies _W_<sup>_′_</sup> = _W − QQ_<sup>_⊤_</sup> _W_ to matrices that write into the residual stream [16]. Defenses hide the extractable direction (AMRA [6]), rebuild it during fine-tuning (DeepRefusal [1]), or add justify-then-refuse data [2]. 

**Merging, task arithmetic, and the safety basin.** Weight averaging (“model soups” [19]) and task arithmetic [17] move a model along low-dimensional directions in weight space; several works show that naive merging silently degrades alignment and that safety occupies a narrow basin [3, 12]. We use interpolation as a probe of that basin in the multimodal setting. 

**VLM structure.** We consider the Flamingo/BLIP family: a ViT encoder, a resampler that maps patches to a fixed set of visual tokens, and a decoder that attends to those tokens (gated cross-attention, or token-merging as in Qwen2-VL). We tap the residual stream at four logical sites: vision output, resampler latents, cross-attention/decoder layers, and the text residual stream. 

2 

## **3 Related work and delta** 

Prior work removes or transfers a _text_ refusal direction [15, 8, 9], studies the refusal cone’s structure [18, 7], hardens models against text abliteration [6, 1, 2], or observes multimodal safety drift on off-the-shelf VLMs [13, 4, 10]. Security-facing work shows refusal and capability are separable and that alignment underperforms on cyber tasks [5, 14, 11]. Our delta is the intersection none of these occupy: a _cross-modal subspace separation_ ( _RV \ RT_ ), its use to _circumvent text-space defenses_ , the _capability retained_ when it is exploited, and a _both-cone defense_ , all under a self-injected alignment that gives causal control over how refusal forms. 

## **4 Method** 

**Cone estimation.** Per modality we collect activations for several harm categories and one harmless set at each site. Stacking the category-mean contrasts _{µc − µℓ}_ and taking the top- _k_ right singular vectors yields an orthonormal basis for the cone (this is the correct estimator: a single harmful/harmless split has rank-1 between-class scatter, so a genuine multi-axis cone requires multiple harm categories). 

**Subspace comparison (P1).** For cones _RT , RV_ with orthonormal bases _QT , QV_ , the principal angles are arccos of the singular values of _Q_<sup>_⊤_</sup> _T_<sup>_QV_;themissedenergyis1</sup><sup>_−∥QT Q⊤_</sup> _T_<sup>_QV ∥_2</sup> _F_<sup>_/∥QV ∥_2</sup> _F_<sup>.</sup> 

**Ablation operators.** Activation projection _x �→ x − QQ_<sup>_⊤_</sup> _x_ and weight orthogonalization _W �→ W − QQ_<sup>_⊤_</sup> _W_ share one implementation, so the attack (C3) and the souping analysis (C5) use the same definition of “remove this subspace.” 

**Causal control (C2).** Rather than train a VLM from scratch (infeasible on our budget), we take a _base_ , non-safety-tuned VLM and inject a known safety alignment via a small LoRA whose data we control. Ablating the injected data lets us attribute the resulting refusal cone causally to it—an attribution off-the-shelf studies cannot make. 

## **5 Pipeline validation (synthetic, harmless)** 

Before any model run we validate the analysis chain on planted data: we construct a text cone _RT_ = span _{s, t}_ and a visual cone _RV_ = span _{s, v}_ sharing one axis _s_ , plus an orthogonal capability axis, and run the _real_ harness functions. Figure 1 shows the pipeline recovers the planted geometry (one shared, one orthogonal axis; _RV \ RT_ energy _≈_ 0 _._ 5), that a defended readout reading refusal from the visual axis is untouched by text-only ablation but collapses under cross-modal ablation while the capability axis is preserved, and that only a both-cone merge fully restores refusal along the interpolation path. These numbers are properties of the construction, not empirical findings; they exist to show the code measures what it claims. 

## **6 Empirical protocol and results** 

Extraction (C1) and abliteration (C3) run on open instruct VLMs; the causal probe (C2) fine-tunes the _base_ Qwen2-VL-7B on a single H100. Harmful prompts come from `mlabonne/harmful_behaviors` (bucketed into five categories) and benign prompts from `mlabonne/harmless_alpaca` ; the visual 

3 



<!-- Start of picture text -->
R_T vs R_V principal angles<br>(R_V\ R_T energy = 0.51)<br>80<br>~<br>~~me}ovDo 60<br>@co 40<br>20<br>0 H<br>principal component<br><!-- End of picture text -->



<!-- Start of picture text -->
a :<br>Attack success vs capability retention<br>1.07 mmm ASR (harmful)<br>mm capability retained<br>08<br>2g 0.6<br>©<br>0.4<br>02<br>0.0<br>base text-onlyablation cross-modalablation<br><!-- End of picture text -->



<!-- Start of picture text -->
i<br>Safety basin: only both-cone merge fully restores refu:<br>1.0<br>508<br>‘oa<br>L1 06 —e text-only re-alignment7<br>2g 0.4 s+ both-cone merge<br>a20.2<br>0.0<br>0.0 merge0.2 a (0 = abliterated0.4 0.6> 1 = re-aligned)0.8 1.0<br><!-- End of picture text -->



<!-- Start of picture text -->
Cross-modal refusal separation vs depth, across scale<br>1 o (higher = more of the visual refusal cone text-space methods miss)<br>—— Qwen2-VL-2B<br>—=— Qwen2-VL-7B<br>0.8 4\ —— Qwen3-VL-30B (MoE)<br>io)<br>©8 0.6 INS .<br>Cc<br>a—©BR Se<br>=<br>> 0.4 = AN<br>a<br>0.2<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>normalized decoder depth (layer / final layer)<br><!-- End of picture text -->

Table 2: C3/P2 (real, Qwen2-VL-2B, _n_ =60 held-out, rank-1 all-layer ablation). Raw ASR = any non-refusal; coherent = non-refusal passing a coherence check. The coherent-compliance ordering is unstable across sample sizes <u>(see</u> text). 

|Eval|Ablation|raw ASR|coherent ASR|
|---|---|---|---|
||none|0.00|0.00|
|text|text-only|1.00|0.08|
||cross-modal|1.00|0.10|
||none|0.00|0.00|
|image|text-only|1.00|0.53|
||cross-modal|1.00|0.22|



safety is the behavioral image of the geometric _RV \ RT_ from C1: the part of the visual refusal cone that text-space supervision does not reach is a real, exploitable behavioral gap. (Notably, mean-pooled diff-in-means separability barely moved across B/C, while behavior moved from 0% to 100%—so we report behavioral refusal, and treat the C1 geometry numbers as a conservative probe.) 

Table 3: C2 (real, causal): refusal rate after self-injected safety on _base_ Qwen2-VL-7B (LoRA on LM attention only), _n_ =20 held-out harmful prompts per modality. Text-only safety transfers to images only partially <u>(30%);</u> image safety closes it. 

|Injected safety|harmful-text refusal|harmful-image refusal|
|---|---|---|
|none (base)|15%|0%|
|text-only|100%|**30%**|
|text + image|100%|100%|



## **— 7 C5 the both-modality defense** 

The constructive counterpart follows directly from C2. Because text-space safety supervision leaves the visual channel largely uncovered (30% image refusal), the fix is to make refusal supervision _modality-complete_ : add harmful-image _→_ refusal examples so both cones are aligned. Condition C is exactly this defense, and it works—image refusal rises from 30% to 100% while text refusal stays at 100% (Table 3), at the cost of a small amount of image supervision (80 examples) on top of the text set. The takeaway for practitioners is concrete: a VLM safety-tuned or abliteration-hardened only in the text representation inherits a visual hole that scales-invariantly persists (C1); closing it requires supervision (or defense edits) that touch the visual pathway. Casting this as a weight-space operation—interpolating or task-arithmetic-merging a text-safety and an image-safety adapter and tracing the resulting “safety basin”—is a natural extension we leave to future work; our harness already exposes the merge primitives ( `geometry.lerp_state` ). 

## **8 Ethics and responsible disclosure** 

This is dual-use safety research, and its net contribution is defensive: the central result is that textonly safety leaves an exploitable visual hole, and the fix (C5) is to align both modality cones. We use 

6 

only public prompt sets and locally-rendered typographic images; we author only refusals and benign answers, never harmful responses; we report aggregate refusal rates, never harmful transcripts; and we do _not_ release abliterated weights, the safety-added/removed checkpoints, or a runnable cross-modal attack recipe. Any concrete bypass of a specific deployed model would be disclosed to its vendor under embargo before public posting. Full policy in the released `docs/ethics.md` . 

## **9 Limitations and future work** 

Refusal is scored by a lexical classifier rather than a model judge; held-out behavioral evals use _n_ =20 prompts per modality; the visual modality is a single typographic (FigStep) style, so photographic harmful images remain to be tested. C1 uses mean-pooled residuals, which C2 shows understates behaviorally-real effects—so the geometry numbers are conservative. The cross-modal abliteration _attack_ (C3) was inconclusive at this scale and needs a model judge and a coherence-preserving ablation to settle. The causal probe injects safety via a language-only LoRA rather than full pretraining; a fully controlled testbed—a VLM trained from scratch with known safety data (the companion _transFORme-r_ model)—and a weight-space safety-basin study of both-cone merging are the main future directions. 

## **10 Conclusion** 

Refusal in a vision-language model is not a single text direction but a union of modality-conditioned cones. Across three VLMs and two architecture generations, the image-conditioned refusal cone keeps a large, scale-robust component orthogonal to the text cone (C1). That geometric gap is behaviorally real: safety supervised only on text produces full text refusal but leaves most harmful _images_ un-refused (C2), and the fix is to align both modality cones (C5). Whether the residual also yields a clean cross-modal _attack_ advantage remains open (C3). The practical message is simple: VLM safety and abliteration defenses built in the text representation alone inherit a visual hole that does not shrink with scale, and must be closed in the visual pathway. We release the harness, the LoRA trainer, and a harmless end-to-end validation. 

## **References** 

- [1] Anonymous. Beyond surface alignment: Rebuilding llms safety mechanism via probabilistically ablating refusal direction. _arXiv preprint arXiv:2509.15202_ , 2025. 

- [2] Anonymous. An embarrassingly simple defense against llm abliteration attacks. _arXiv preprint arXiv:2505.19056_ , 2025. 

- [3] Anonymous. Led-merging: Mitigating safety-utility conflicts in model merging. _arXiv preprint arXiv:2502.16770_ , 2025. 

- [4] Anonymous. Repit: Steering language models with concept-specific refusal vectors. _arXiv preprint arXiv:2509.13281_ , 2025. 

- [5] Anonymous. Ablating safety: Mechanisms for removing alignment in language models for security applications. _arXiv preprint arXiv:2605.17413_ , 2026. 

- [6] Anonymous. Abliteration mitigation via refusal aliases. _arXiv preprint arXiv:2608.18093_ , 2026. 

7 

- [7] Anonymous. Fast multi-dimensional refusal subspaces via rfm-agop. _arXiv preprint arXiv:2607.02396_ , 2026. 

- [8] Anonymous. The geometry of refusal: Linear instability in safety-aligned llms. _arXiv preprint arXiv:2606.22686 (TrustNLP 2026)_ , 2026. 

- [9] Anonymous. Harnessing textual refusal directions for multimodal safety. _arXiv preprint arXiv:2606.31876_ , 2026. 

- [10] Anonymous. Jailbreaking vision-language models through the visual modality. _arXiv preprint arXiv:2605.00583_ , 2026. 

- [11] Anonymous. Not all refusals are equal: How safety alignment fails cybersecurity at scale. _arXiv preprint arXiv:2607.02714_ , 2026. 

- [12] Anonymous. Preventing safety drift in large language models via coupled weight and activation constraints. _arXiv preprint arXiv:2604.12384_ , 2026. 

- [13] Anonymous. Safety geometry collapse in multimodal llms and adaptive drift correction. _arXiv preprint arXiv:2605.18104_ , 2026. 

- [14] Anonymous. Willing but unable: Separating refusal from capability in code llms via abliteration. _arXiv preprint arXiv:2606.05396_ , 2026. 

- [15] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. _arXiv preprint arXiv:2406.11717_ , 2024. 

- [16] Gökdeniz Gülmez. Gabliteration: Adaptive multi-directional neural weight modification. _arXiv preprint arXiv:2512.18901_ , 2025. 

- [17] Gabriel Ilharco et al. Editing models with task arithmetic. _ICLR_ , 2023. 

- [18] Tom Wollschläger et al. The geometry of refusal in large language models: Concept cones and representational independence. _arXiv preprint arXiv:2502.17420_ , 2025. 

- [19] Mitchell Wortsman et al. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. _ICML_ , 2022. 

8 

