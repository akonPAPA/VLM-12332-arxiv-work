# Cross-Modal Refusal Geometry — research harness

Companion code + notes for the paper *Where Does a VLM Refuse? Modality-Entangled
Safety Subspaces, Cross-Modal Abliteration, and Retained Offensive-Cyber Capability*.

Plan of record: `~/.dscps-qwen3.8-flash-next/plans/research-glistening-wozniak.md`.

## Layout

```
research/
  configs/experiments.yaml   models, datasets, layer taps, budget knobs
  docs/related_work.md       verified literature matrix + the explicit delta
  docs/ethics.md             dual-use handling & responsible disclosure
  src/geometry.py            subspaces, principal angles, projection ablation (pure tensor; no model needed)
  src/extraction.py          hook-based activation capture across the VLM stack
  src/abliterate.py          inference-hook + permanent weight-edit abliteration
  src/eval_refusal.py        refusal / ASR scoring
  paper/outline.md           section-by-section skeleton
  requirements-research.txt  extra deps beyond the repo root
```

## Phases (see plan for exit criteria)

- **P0** harness + reproduce text-only abliteration on one 2B instruct VLM *(local)*
- **P1** extract `R_T` / `R_V` across the stack, principal angles *(local)*
- **P2** self-injected safety-LoRA on a base VLM — causal formation *(~$ cloud)*
- **P3** cross-modal vs text-only abliteration against a text-space defense *(local)*
- **P4** cyber capability-retention table *(mixed)*
- **P5** safety-basin sweep + subspace-preserving merge defense *(mixed)*

MVP paper = P0–P3 (C1–C3). Full paper = through P5.

## Ground rules

- Everything heavy must run at 2B / 4-bit locally. The ~$20 cloud budget is only for the
  LoRA (P2) and one defense reproduction.
- No harmful content, no abliterated weights, and no runnable cross-modal attack recipe are
  committed. See `docs/ethics.md`.

## Quick check (no model, no download)

```bash
python -m research.src.geometry --selftest
```
