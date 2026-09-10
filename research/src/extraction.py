"""Activation capture across the VLM stack (C1).

Registers forward hooks at logical *sites* (vision_out, resampler_latents,
cross_attn, text_residual) and returns mean-pooled residual-stream activations,
one vector per prompt per site. Model-family-specific module resolution is the
one piece that must be filled per checkpoint — see `resolve_taps`.

torch is imported lazily so `geometry` stays importable on machines without a GPU
build; this module is only exercised once a model is loaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import torch


# --------------------------------------------------------------------------- #
# Tap resolution (per model family) — THE model-specific TODO.
# --------------------------------------------------------------------------- #
# Per-family regexes over module names. Keep this table small and auditable;
# it is the only model-specific surface and belongs in the paper appendix.
# Each entry maps a logical site -> (regex, per_layer?). `per_layer` sites yield
# one hook per matching module, named "<site>.<index>".
_FAMILY_PATTERNS: dict[str, dict[str, tuple[str, bool]]] = {
    # Qwen2-VL: visual tower `.visual.blocks.*`, patch merger `.visual.merger`,
    # decoder `.model.layers.*`. Visual tokens are merged into the sequence, so
    # there is no separate gated cross-attn module — `cross_attn` stays empty and
    # the interaction is observed in the decoder layers.
    "qwen2_vl": {
        "vision_out": (r"visual\.blocks\.\d+$", True),
        "resampler_latents": (r"visual\.merger$", False),
        "text_residual": (r"language_model\.layers\.\d+$", True),  # v5 naming
        "cross_attn": (r".*cross_attn$", True),
    },
    # InternVL2: ViT `.vision_model.encoder.layers.*`, MLP connector `.mlp1`,
    # LLM `.language_model.model.layers.*`.
    "internvl2": {
        "vision_out": (r"vision_model\.encoder\.layers\.\d+$", True),
        "resampler_latents": (r"(?:^|\.)mlp1$", False),
        "text_residual": (r"language_model\.model\.layers\.\d+$", True),
        "cross_attn": (r".*cross_attn$", True),
    },
    # Flamingo/IDEFICS-style gated cross-attention (and the transFORme-r design):
    # here `cross_attn` is the real, load-bearing site.
    "gated_xattn": {
        "vision_out": (r"(?:vision|visual).*blocks?\.\d+$", True),
        "resampler_latents": (r".*(resampler|perceiver|connector).*$", False),
        "text_residual": (r".*layers\.\d+$", True),
        "cross_attn": (r".*(gated_cross_attn|cross_attn|xattn).*\.\d+$", True),
    },
}


def resolve_taps(model: Any, family: str, taps: dict[str, bool]) -> dict[str, Any]:
    """Map logical site names -> concrete nn.Module objects to hook.

    Pattern-based so it survives minor version drift: matches `model.named_modules()`
    against a small per-family table. Per-layer sites become "<site>.<idx>". Only
    sites enabled in `taps` are returned. If an enabled site matches nothing the
    caller gets a clear error (better than silently dropping a tap).
    """
    import re

    if family not in _FAMILY_PATTERNS:
        raise ValueError(f"unknown family '{family}'; add it to _FAMILY_PATTERNS")
    patterns = _FAMILY_PATTERNS[family]
    named = list(model.named_modules())
    resolved: dict[str, Any] = {}

    for site, want in taps.items():
        if not want or site not in patterns:
            continue
        rx, per_layer = patterns[site]
        matches = [(n, m) for n, m in named if re.search(rx, n)]
        if per_layer:
            for idx, (_n, m) in enumerate(matches):
                resolved[f"{site}.{idx}"] = m
        elif matches:
            resolved[site] = matches[-1][1]  # deepest single match
        if want and not matches and site != "cross_attn":
            raise LookupError(
                f"tap '{site}' matched no module in family '{family}' "
                f"(pattern {rx!r}); inspect model.named_modules() and update the table"
            )
    return resolved


# --------------------------------------------------------------------------- #
# Hook recorder
# --------------------------------------------------------------------------- #
class HookRecorder:
    """Collects mean-pooled outputs of the given modules during forward passes."""

    def __init__(self, sites: dict[str, Any]) -> None:
        self._sites = sites
        self._buffers: dict[str, list[torch.Tensor]] = {k: [] for k in sites}
        self._handles: list[Any] = []

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach().float()
            # Pool to one (1, d) residual vector per forward pass. Handles both
            # (batch, seq, d) decoder outputs and (patches, d) vision-block outputs.
            # Assumes batch size 1 so no padding tokens pollute the mean.
            if t.ndim == 3:
                pooled = t.mean(dim=1)              # (batch, d)
            elif t.ndim == 2:
                pooled = t.mean(dim=0, keepdim=True)  # (1, d)
            else:
                pooled = t.reshape(1, -1)
            self._buffers[name].append(pooled.cpu())
        return hook

    def __enter__(self) -> "HookRecorder":
        for name, module in self._sites.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stack(self) -> dict[str, torch.Tensor]:
        """Concatenate captured batches into (n_samples, d) per site."""
        return {k: torch.cat(v, dim=0) for k, v in self._buffers.items() if v}


@contextmanager
def record(model: Any, family: str, taps: dict[str, bool]):
    sites = resolve_taps(model, family, taps)
    rec = HookRecorder(sites)
    with rec:
        yield rec


def collect_activations(
    model: Any,
    run_forward,
    prompts: Iterable[Any],
    family: str,
    taps: dict[str, bool],
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Run `run_forward(model, batch)` over prompts, capturing per-site activations.

    `run_forward` is a caller-supplied closure that handles the processor and does
    a no-grad forward (it varies by family / whether an image is attached), keeping
    this function model-agnostic. Returns {site: (n_prompts, d)}.
    """
    prompts = list(prompts)
    with record(model, family, taps) as rec:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                run_forward(model, prompts[i : i + batch_size])
    return rec.stack()
