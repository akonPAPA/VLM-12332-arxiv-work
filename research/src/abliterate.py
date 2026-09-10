"""Abliteration operators (C3): inference-time projection & permanent weight edit.

Both delegate the linear algebra to `geometry`, so the attack and the weight-space
souping analysis (P4) share one definition of "remove this refusal subspace".

The weight-edit path is fully functional at the tensor level (operates on a state
dict); the inference-hook path needs the per-site module map from `extraction`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

try:  # works both as a package module and as a direct script
    from .geometry import ablate_activations, orthogonalize_weight
except ImportError:  # pragma: no cover
    from geometry import ablate_activations, orthogonalize_weight


# --------------------------------------------------------------------------- #
# Inference-time abliteration (activation projection via forward hooks)
# --------------------------------------------------------------------------- #
@contextmanager
def abliterate_inference(sites_to_modules: dict[str, Any], basis_by_site: dict[str, torch.Tensor]):
    """Temporarily project each hooked site onto the complement of its refusal basis.

    `sites_to_modules` maps site -> nn.Module (from extraction.resolve_taps);
    `basis_by_site` maps site -> (d, k) basis. Sites present in both are ablated.
    Restores the model on exit. Use with `basis_by_site = {text sites: R_T}` for a
    text-only attack, or include the visual sites' R_V for the cross-modal attack.
    """
    handles = []

    def make_hook(basis: torch.Tensor):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cleaned = ablate_activations(t.float(), basis.to(t.device)).to(t.dtype)
            if isinstance(out, tuple):
                return (cleaned, *out[1:])
            return cleaned
        return hook

    try:
        for site, module in sites_to_modules.items():
            if site in basis_by_site:
                handles.append(module.register_forward_hook(make_hook(basis_by_site[site])))
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Permanent (weight-space) abliteration — dual used by the souping analysis
# --------------------------------------------------------------------------- #
def abliterate_weights(
    state_dict: dict[str, torch.Tensor],
    basis: torch.Tensor,
    target_key_substrings: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Orthogonalize every output-writing matrix whose key matches a substring.

    Applies W' = W - Q Qᵀ W to matrices whose first dim equals d_model (basis rows).
    Returns a NEW state dict (immutability). `target_key_substrings` selects which
    projections write to the residual stream, e.g. ("o_proj", "down_proj",
    "cross_attn.o_proj"). Non-matching / wrong-shape tensors are copied unchanged.
    """
    d_model = basis.shape[0]
    out: dict[str, torch.Tensor] = {}
    for key, w in state_dict.items():
        hit = any(s in key for s in target_key_substrings)
        if hit and w.ndim == 2 and w.shape[0] == d_model:
            out[key] = orthogonalize_weight(w.float(), basis).to(w.dtype)
        else:
            out[key] = w.clone()
    return out
