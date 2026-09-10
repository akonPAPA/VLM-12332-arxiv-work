# Ethics & responsible disclosure

This project studies how safety (refusal) is represented in vision-language models
and how that representation can be circumvented, in order to build better defenses.
It is dual-use. The following rules are binding on the repo and the paper.

## What we do

- Use **only existing public benchmarks** (AdvBench, HarmBench, JailbreakBench,
  StrongREJECT, MM-SafetyBench, FigStep, HADES, VLGuard) and a **small curated
  subset** of a published cyber benchmark (e.g. CyberSecEval / SecLLMHolmes).
- Report **aggregate** metrics: attack-success rate, refusal rate, capability
  retention, subspace angles. Not step-by-step harmful transcripts.
- Frame the contribution around **measurement + defense** (C5 is the payoff).

## What we do NOT do

- No novel exploits, malware, or operational offensive-cyber content is produced.
- **No abliterated weights are released.** No runnable cross-modal attack recipe or
  attack images are committed or published; the method is described at the level
  needed for reproduction of the *defense*, with the attack detail withheld.
- No harmful prompts/images are committed to the repo. Datasets are downloaded at
  runtime from their official sources under their licenses; only IDs/hashes and
  scoring code live here.

## Disclosure

- Any concrete bypass of a specific deployed model's safety is disclosed to that
  vendor before public posting, with a reasonable embargo.
- The paper includes a Broader Impact / Ethics statement matching the norms of the
  cited security papers (2605.17413, 2607.02714) and venue policy (SaTML/TrustNLP).

## Handling generated content

- The judge pipeline scores outputs; harmful generations are hashed/counted, then
  discarded, never persisted in plaintext in the repo or artifacts.

## Scope guard

If an experiment would require producing genuinely operational harmful content to
make its point, it is out of scope — substitute a proxy metric or drop it.
