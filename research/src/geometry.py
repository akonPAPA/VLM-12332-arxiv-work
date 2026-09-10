"""Safety-subspace geometry — the math backbone of the paper.

Pure tensor operations: nothing here loads a model, so it is cheap to unit-test
and runs on a laptop. All functions are pure (no in-place mutation of inputs),
matching the repo's immutability rule.

Conventions
-----------
* An "activation matrix" `A` has shape (n_samples, d_model): one residual-stream
  vector per prompt at a fixed layer/site.
* A "direction" is a unit vector of shape (d_model,).
* A "subspace basis" `Q` has shape (d_model, k): orthonormal columns spanning a
  k-dimensional refusal subspace (a cone's linear hull).

Run `python -m research.src.geometry --selftest` for a dependency-light check.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Direction & subspace estimation
# --------------------------------------------------------------------------- #
def difference_in_means(harmful: Tensor, harmless: Tensor) -> Tensor:
    """Arditi-style refusal direction: normalized mean(harmful) - mean(harmless).

    Returns a unit vector (d_model,). Both inputs are (n, d_model).
    """
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise ValueError("expected (n, d_model) activation matrices")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("harmful/harmless d_model mismatch")
    diff = harmful.mean(dim=0) - harmless.mean(dim=0)
    return _unit(diff)


def refusal_subspace(harmful: Tensor, harmless: Tensor) -> Tensor:
    """Rank-1 refusal subspace for a single harmful/harmless split.

    Returns a (d_model, 1) orthonormal basis. For two classes the between-class
    scatter is rank-1, so the only well-defined refusal axis is the mean
    difference — this is `difference_in_means` reshaped to a basis. Use
    `category_refusal_subspace` to recover a genuine multi-dimensional cone.
    """
    return difference_in_means(harmful, harmless).unsqueeze(1)  # (d_model, 1)


def category_refusal_subspace(
    harmful_by_category: list[Tensor], harmless: Tensor, k: int = 1
) -> Tensor:
    """Top-k refusal *cone* from multiple harm categories (cf. arXiv:2502.17420).

    Each harm category contributes a (mean_category - mean_harmless) contrast row;
    SVD of the stacked contrasts yields the linear hull spanning how refusal
    varies across harm types. `harmful_by_category` is a list of (n_c, d) matrices;
    `harmless` is (m, d). Returns an orthonormal basis (d_model, k), k clamped to
    the number of categories. This is the estimator behind the R_T / R_V cones.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(harmful_by_category) < 1:
        raise ValueError("need at least one harm category")
    hl_mean = harmless.mean(dim=0)
    rows = torch.stack([cat.mean(dim=0) - hl_mean for cat in harmful_by_category])
    _, _, vh = torch.linalg.svd(rows, full_matrices=False)
    k = min(k, vh.shape[0])
    return vh[:k].T.contiguous()  # (d_model, k), orthonormal columns


def orthonormalize(vectors: Tensor) -> Tensor:
    """QR-orthonormalize columns of a (d_model, k) matrix. Pure."""
    q, _ = torch.linalg.qr(vectors)
    return q


# --------------------------------------------------------------------------- #
# Subspace comparison — the P1 measurement (R_T vs R_V)
# --------------------------------------------------------------------------- #
def principal_angles(basis_a: Tensor, basis_b: Tensor) -> Tensor:
    """Principal angles (radians, ascending) between two subspaces.

    `basis_a`, `basis_b` are (d_model, k*) matrices; columns need not be
    orthonormal (we orthonormalize defensively). Number of angles = min(k_a, k_b).
    Angle 0 => shared direction; angle pi/2 => orthogonal component.
    This is the core statistic behind prediction P1 (R_V has a component
    orthogonal to R_T).
    """
    qa = orthonormalize(basis_a)
    qb = orthonormalize(basis_b)
    svals = torch.linalg.svdvals(qa.T @ qb)
    cos = svals.clamp(-1.0, 1.0)
    return torch.arccos(cos).sort().values


def subspace_overlap(basis_a: Tensor, basis_b: Tensor) -> float:
    """Scalar overlap in [0, 1]: mean squared cosine of principal angles.

    1.0 => subspaces coincide; 0.0 => fully orthogonal. Convenient single number
    for tables; report the full angle spectrum for the paper.
    """
    ang = principal_angles(basis_a, basis_b)
    return float((torch.cos(ang) ** 2).mean())


def orthogonal_residual_energy(basis_target: Tensor, basis_other: Tensor) -> float:
    """Fraction of `basis_target`'s energy lying outside `basis_other`.

    This is the direct 'R_V \\ R_T' quantity: how much of the visual refusal
    subspace text-only edits would miss. Returns a value in [0, 1].
    """
    qt = orthonormalize(basis_target)
    qo = orthonormalize(basis_other)
    proj = qo @ (qo.T @ qt)              # project target basis into `other`
    captured = (proj ** 2).sum()
    total = (qt ** 2).sum().clamp_min(_EPS)
    return float(1.0 - captured / total)


# --------------------------------------------------------------------------- #
# Ablation operators (activation-space projection & permanent weight edit)
# --------------------------------------------------------------------------- #
def ablate_activations(acts: Tensor, basis: Tensor) -> Tensor:
    """Project activations onto the orthogonal complement of `basis`.

    x' = x - Q Qᵀ x. Works for a single direction (d,), a batch (n, d), or a
    basis with k>1 columns. Pure — returns a new tensor.
    """
    q = orthonormalize(basis if basis.ndim == 2 else basis.unsqueeze(1))
    if acts.ndim == 1:
        return acts - q @ (q.T @ acts)
    return acts - (acts @ q) @ q.T


def orthogonalize_weight(weight: Tensor, basis: Tensor) -> Tensor:
    """Permanent (weight-space) abliteration of an output-writing matrix.

    For a matrix that *writes into* the residual stream (rows = d_model outputs),
    remove the refusal subspace from its column space: W' = W - Q Qᵀ W.
    `weight` is (d_model, d_in). Pure. This is the weight-edit dual of
    `ablate_activations` and the hook point for the souping analysis (P4).
    """
    if weight.shape[0] != basis.shape[0]:
        raise ValueError("weight rows must equal d_model (basis rows)")
    q = orthonormalize(basis if basis.ndim == 2 else basis.unsqueeze(1))
    return weight - q @ (q.T @ weight)


# --------------------------------------------------------------------------- #
# Souping / basin helpers (P4)
# --------------------------------------------------------------------------- #
def lerp_state(
    state_a: dict[str, Tensor], state_b: dict[str, Tensor], alpha: float
) -> dict[str, Tensor]:
    """Linear weight interpolation (model soup point) between two state dicts.

    alpha=0 -> a, alpha=1 -> b. Only overlapping float tensors of equal shape are
    interpolated; everything else is copied from `state_a`. Pure.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    out: dict[str, Tensor] = {}
    for key, va in state_a.items():
        vb = state_b.get(key)
        if vb is not None and vb.shape == va.shape and va.is_floating_point():
            out[key] = torch.lerp(va, vb, alpha)
        else:
            out[key] = va.clone()
    return out


# --------------------------------------------------------------------------- #
def _unit(v: Tensor) -> Tensor:
    return v / v.norm().clamp_min(_EPS)


def _selftest() -> None:
    torch.manual_seed(0)
    d = 64
    # Build a known refusal direction and synthesize class activations around it.
    r = _unit(torch.randn(d))
    base = torch.randn(256, d)
    harmless = base + 0.0 * r
    harmful = base + 3.0 * r  # harmful pushed along r

    est = difference_in_means(harmful, harmless)
    align = abs(float(est @ r))
    assert align > 0.99, f"diff-in-means should recover r, got cos={align:.3f}"

    sub = refusal_subspace(harmful, harmless)
    assert sub.shape == (d, 1)
    assert abs(float((sub[:, 0]) @ r)) > 0.98

    # Multi-category cone: three harm types sharing a planted 2D refusal cone.
    r2 = _unit(torch.randn(d))
    cone = orthonormalize(torch.stack([r, r2], dim=1))  # true 2D subspace
    cats = []
    for _ in range(3):
        coeffs = torch.randn(200, 2) @ cone.T  # samples inside the cone
        cats.append(base[:200] + 3.0 * coeffs + 3.0 * cone[:, 0])
    est_cone = category_refusal_subspace(cats, harmless[:200], k=2)
    assert est_cone.shape == (d, 2)
    assert orthogonal_residual_energy(cone, est_cone) < 0.15

    # Ablation should remove the r-component.
    cleaned = ablate_activations(harmful, r)
    leak = float((cleaned @ r).abs().mean())
    assert leak < 1e-4, f"residual along r after ablation: {leak:.2e}"

    # Two subspaces sharing r plus orthogonal noise -> nonzero overlap, nonzero residual.
    a = torch.stack([r, _unit(torch.randn(d))], dim=1)
    b = torch.stack([r, _unit(torch.randn(d))], dim=1)
    ov = subspace_overlap(a, b)
    res = orthogonal_residual_energy(a, b)
    assert 0.0 < ov < 1.0 and 0.0 < res < 1.0

    # Weight orthogonalization kills the row-space component along r.
    w = torch.randn(d, 32)
    w2 = orthogonalize_weight(w, r)
    assert float((r @ w2).abs().max()) < 1e-4

    # Soup endpoints.
    sa = {"w": torch.zeros(4)}
    sb = {"w": torch.ones(4)}
    assert torch.allclose(lerp_state(sa, sb, 0.5)["w"], torch.full((4,), 0.5))

    print("geometry selftest: OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python -m research.src.geometry --selftest")
