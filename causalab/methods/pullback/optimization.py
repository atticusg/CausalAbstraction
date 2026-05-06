"""Optimization routines for pullback trajectory fitting."""

from __future__ import annotations

import functools
import logging
import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as sp_shortest_path
from torch import Tensor
from torch.optim import LBFGS

import torch.nn.functional as F
from causalab.methods.distances import (
    _DIFFERENTIABLE_METRICS,
)
from causalab.methods.steer.collect import collect_grid_distributions
from causalab.neural.activations.interchange_mode import prepare_intervenable_inputs
from causalab.neural.activations.interpolate import set_interventions_interpolation

logger = logging.getLogger(__name__)


def extract_concept_dists_batch(
    logits_BV: torch.Tensor,
    concept_indices: torch.Tensor | list[list[int]],
) -> torch.Tensor:
    """Extract (W+1)-simplex distributions: concept probs + "other" bin.

    Applies softmax over the full vocabulary, extracts concept token
    probabilities (summing variant tokens), and appends complementary
    "other" mass.

    Returns:
        (B, W+1) probability distributions.
    """
    probs_full = F.softmax(logits_BV, dim=-1)
    if isinstance(concept_indices, torch.Tensor):
        concept_probs = probs_full[:, concept_indices.to(logits_BV.device)]
    else:
        W = len(concept_indices)
        concept_probs = torch.zeros(
            probs_full.shape[0], W, device=logits_BV.device, dtype=probs_full.dtype
        )
        for c, variant_ids in enumerate(concept_indices):
            ids = torch.tensor(variant_ids, device=logits_BV.device)
            concept_probs[:, c] = probs_full[:, ids].sum(dim=-1)
    other = (1.0 - concept_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    return torch.cat([concept_probs, other], dim=-1)


def _subsample_snapshots(
    snapshots: list[Tensor],
    max_snapshots: int,
    log_spaced: bool = False,
) -> list[Tensor]:
    """Keep at most max_snapshots entries, always including first and last."""
    n = len(snapshots)
    if max_snapshots <= 0 or n <= max_snapshots:
        return snapshots
    if log_spaced:
        # Log-space over [1, n-1] then prepend index 0
        raw = np.geomspace(1, n - 1, max_snapshots - 1)
        indices = [0] + np.round(raw).astype(int).tolist()
    else:
        indices = np.round(np.linspace(0, n - 1, max_snapshots)).astype(int).tolist()
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique = [i for i in indices if not (i in seen or seen.add(i))]
    return [snapshots[i] for i in unique]


class _ForceLastTokenLogits:
    """Patch the underlying HF model so its forward defaults to
    ``logits_to_keep=1``, computing the lm_head only at the final position.

    On 2048-ctx 8B Llama this drops ~1 GB per forward (the per-position vocab
    tensor) — material savings during pullback's per-timestep autograd graph.
    The pullback loss only ever reads ``logits[:, -1, :]``, so the change is
    semantically identical. Pyvene's ``IntervenableModel`` doesn't pass model
    kwargs through, so we patch the inner module's ``forward`` directly.
    """

    def __init__(self, intervenable_model):
        # Walk to the actual HF model (pyvene wraps it).
        self._target = getattr(intervenable_model, "model", intervenable_model)
        self._orig = None

    def __enter__(self):
        orig = self._target.forward
        self._orig = orig

        @functools.wraps(orig)
        def _patched(*args, **kwargs):
            kwargs.setdefault("logits_to_keep", 1)
            return orig(*args, **kwargs)

        self._target.forward = _patched
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self._target.forward = self._orig
            self._orig = None


def _run_lbfgs(
    v_params: nn.Parameter,
    compute_loss_fn,
    steps: int,
    lr: float,
    max_iter: int,
    line_search_fn: str,
    print_every: int = 1,
    max_snapshots: int = 0,
    log_spaced_snapshots: bool = False,
    convergence_window: int = 5,
    convergence_tol: float = 0.001,
) -> tuple[list[float], list[list[float]], list[Tensor]]:
    """Run L-BFGS on v_params in-place. Returns (loss_history, per_step_history, param_snapshots)."""
    optimizer = LBFGS(
        [v_params], lr=lr, max_iter=max_iter, line_search_fn=line_search_fn
    )
    loss_history = []
    per_step_history = []
    param_snapshots: list[Tensor] = [v_params.data.clone().cpu()]

    for step in range(steps):
        step_per_t: list[float] = []

        def closure():
            optimizer.zero_grad()
            total, per_step = compute_loss_fn()
            total.backward()
            step_per_t.clear()
            step_per_t.extend(per_step)
            return total

        loss_val = optimizer.step(closure).item()
        loss_history.append(loss_val)
        per_step_history.append(list(step_per_t))
        param_snapshots.append(v_params.data.clone().cpu())

        if step % print_every == 0 or step == steps - 1:
            logger.info("    L-BFGS step %4d: loss = %.6f", step, loss_val)

        if len(loss_history) >= convergence_window:
            recent = torch.tensor(loss_history[-convergence_window:])
            diffs = (recent[1:] - recent[:-1]) / (recent[:-1].abs() + 1e-8)
            max_diff = diffs.abs().max().item()
            if max_diff < convergence_tol:
                logger.info(
                    "    Early stop at step %d: <%.4f%% change over %d iterations",
                    step,
                    convergence_tol * 100,
                    convergence_window,
                )
                break

    return (
        loss_history,
        per_step_history,
        _subsample_snapshots(param_snapshots, max_snapshots, log_spaced_snapshots),
    )


def _run_adam(
    v_params: nn.Parameter,
    compute_loss_fn,
    steps: int,
    lr: float,
    print_every: int = 5,
    max_snapshots: int = 0,
    log_spaced_snapshots: bool = False,
) -> tuple[list[float], list[list[float]], list[Tensor]]:
    """Run AdamW on v_params in-place. Returns (loss_history, per_step_history, param_snapshots)."""
    optimizer = torch.optim.AdamW([v_params], lr=lr)
    loss_history = []
    per_step_history = []
    param_snapshots: list[Tensor] = [v_params.data.clone().cpu()]

    for step in range(steps):
        optimizer.zero_grad()
        total, per_step = compute_loss_fn()
        total.backward()
        optimizer.step()

        loss_history.append(total.item())
        per_step_history.append(list(per_step))
        param_snapshots.append(v_params.data.clone().cpu())

        if step % print_every == 0 or step == steps - 1:
            logger.info("    AdamW step %4d: loss = %.6f", step, total.item())

    return (
        loss_history,
        per_step_history,
        _subsample_snapshots(param_snapshots, max_snapshots, log_spaced_snapshots),
    )


def _neb_project_embedding_gradients(
    grad: Tensor,
    path: Tensor,
    spring_k: float,
    eps: float = 1e-8,
) -> Tensor:
    """NEB force decomposition for embedding-space trajectory gradients.

    Mirror of ``_neb_project_gradients`` (distances.py) operating in
    k-dimensional embedding space instead of (W+1)-dimensional probability
    space.

    Args:
        grad: (n_interior, k) raw gradient w.r.t. interior embedding params.
        path: (n_steps, k) full embedding trajectory (with fixed endpoints).
        spring_k: Spring constant for parallel spacing force.
        eps: Numerical stability.

    Returns:
        (n_interior, k) NEB-projected gradient.
    """
    # Tangent at each interior point: central difference
    tangents = path[2:] - path[:-2]  # (n_interior, k)
    tangent_norms = tangents.norm(dim=-1, keepdim=True).clamp(min=eps)
    tau = tangents / tangent_norms

    # Perpendicular potential: remove parallel component
    grad_par = (grad * tau).sum(dim=-1, keepdim=True) * tau
    grad_perp = grad - grad_par

    # Parallel spring force: penalise unequal step sizes
    d_steps = (path[1:] - path[:-1]).norm(dim=-1)  # (n_steps-1,)
    spring_mag = spring_k * (d_steps[1:] - d_steps[:-1])  # (n_interior,)
    spring_grad = -spring_mag.unsqueeze(-1) * tau

    return grad_perp + spring_grad


def _run_neb(
    v_params: nn.Parameter,
    compute_loss_fn,
    steps: int,
    lr: float,
    spring_k: float,
    f_start_k: Tensor,
    f_end_k: Tensor,
    fix_start: bool,
    fix_end: bool,
    n_steps: int,
    interior_mask: Tensor,
    print_every: int = 5,
    max_snapshots: int = 0,
    log_spaced_snapshots: bool = False,
    convergence_window: int = 5,
    convergence_tol: float = 0.001,
) -> tuple[list[float], list[list[float]], list[Tensor]]:
    """Run NEB with AdamW on embedding trajectory params.

    Like ``_run_adam`` but inserts NEB gradient projection between backward
    and optimizer step to decouple path shape from waypoint spacing.
    """
    device = v_params.device
    k_dim = v_params.shape[-1]
    optimizer = torch.optim.AdamW([v_params], lr=lr)
    loss_history: list[float] = []
    per_step_history: list[list[float]] = []
    param_snapshots: list[Tensor] = [v_params.data.clone().cpu()]

    for step in range(steps):
        optimizer.zero_grad()
        total, per_step = compute_loss_fn()
        total.backward()

        # Reconstruct full path for tangent computation
        with torch.no_grad():
            full_path = torch.zeros(n_steps, k_dim, device=device)
            full_path[interior_mask] = v_params.data
            if fix_start:
                full_path[0] = f_start_k.to(device)
            if fix_end:
                full_path[-1] = f_end_k.to(device)

            v_params.grad.copy_(
                _neb_project_embedding_gradients(
                    v_params.grad,
                    full_path,
                    spring_k,
                )
            )

        optimizer.step()

        loss_val = total.item()
        loss_history.append(loss_val)
        per_step_history.append(list(per_step))
        param_snapshots.append(v_params.data.clone().cpu())

        if step % print_every == 0 or step == steps - 1:
            logger.info("    NEB step %4d: loss = %.6f", step, loss_val)

        if len(loss_history) >= convergence_window:
            recent = torch.tensor(loss_history[-convergence_window:])
            diffs = (recent[1:] - recent[:-1]) / (recent[:-1].abs() + 1e-8)
            max_diff = diffs.abs().max().item()
            if max_diff < convergence_tol:
                logger.info(
                    "    NEB early stop at step %d: <%.4f%% change over %d iterations",
                    step,
                    convergence_tol * 100,
                    convergence_window,
                )
                break

    return (
        loss_history,
        per_step_history,
        _subsample_snapshots(param_snapshots, max_snapshots, log_spaced_snapshots),
    )


def _run_basin_hopping(
    v_init: Tensor,
    make_loss_fn,
    bh_cfg: dict,
    device: torch.device,
    print_every: int = 1,
    max_snapshots: int = 0,
    log_spaced_snapshots: bool = False,
    convergence_window: int = 5,
    convergence_tol: float = 0.001,
) -> tuple[nn.Parameter, list[float], list[list[float]], list[Tensor]]:
    """Hybrid basin hopping: cheap grad-free search, then L-BFGS refinement of top candidates.

    Phase 1: Perturb -> forward-only loss eval -> Metropolis accept/reject (no gradients).
    Phase 2: Run full L-BFGS from the top-N candidates found in phase 1.

    Args:
        v_init: Initial parameter values (n_interior, k).
        make_loss_fn: Callable(v_params) -> compute_loss_fn.
        bh_cfg: Basin hopping config dict.
        device: Torch device.

    Returns:
        (best_v_params, loss_history, per_step_history, param_snapshots)
    """
    n_hops = bh_cfg["n_hops"]
    step_size = bh_cfg["step_size"]
    temperature = bh_cfg["temperature"]
    n_candidates = bh_cfg.get("n_candidates", 5)
    lbfgs_kw = {
        "steps": bh_cfg["lbfgs_steps"],
        "lr": bh_cfg["lbfgs_lr"],
        "max_iter": bh_cfg["lbfgs_max_iter"],
        "line_search_fn": bh_cfg["lbfgs_line_search_fn"],
    }

    init_scale = v_init.std().item() + 1e-6

    # --- Phase 1: cheap candidate search (forward-only, no gradients) ---
    logger.info("  Basin hopping phase 1: searching %d hops (grad-free)", n_hops)
    current_data = v_init.clone().to(device)
    with torch.no_grad():
        current_loss = make_loss_fn(nn.Parameter(current_data))()[0].item()
    candidates: list[tuple[Tensor, float]] = [(current_data.clone(), current_loss)]

    for hop in range(n_hops):
        perturbation = torch.randn_like(current_data) * step_size * init_scale
        candidate_data = current_data + perturbation

        with torch.no_grad():
            candidate_loss = make_loss_fn(nn.Parameter(candidate_data))()[0].item()
        candidates.append((candidate_data.clone(), candidate_loss))

        delta = candidate_loss - current_loss
        if delta < 0 or random.random() < math.exp(-delta / (temperature + 1e-12)):
            current_data = candidate_data
            current_loss = candidate_loss
            status = "accepted"
        else:
            status = "rejected"

        if (hop + 1) % print_every == 0 or hop == n_hops - 1:
            logger.info(
                "    hop %d/%d: loss = %.6f (%s)",
                hop + 1,
                n_hops,
                candidate_loss,
                status,
            )

    candidates.sort(key=lambda x: x[1])
    n_cand = min(n_candidates, len(candidates))
    logger.info(
        "  Phase 1 done: best candidate loss = %.6f, refining top %d",
        candidates[0][1],
        n_cand,
    )

    # --- Phase 2: L-BFGS refinement on top-N candidates ---
    conv_kw = {
        "convergence_window": convergence_window,
        "convergence_tol": convergence_tol,
    }
    best_params: nn.Parameter | None = None
    best_loss = float("inf")
    all_loss_history: list[float] = []
    all_per_step_history: list[list[float]] = []
    param_snapshots: list[Tensor] = [v_init.clone().cpu()]

    for i, (cand_data, cand_search_loss) in enumerate(candidates[:n_cand]):
        logger.info(
            "  Phase 2: refining candidate %d/%d (search loss = %.6f)",
            i + 1,
            n_cand,
            cand_search_loss,
        )
        v = nn.Parameter(cand_data.clone().to(device))
        compute_loss_fn = make_loss_fn(v)
        lh, psh, _snaps = _run_lbfgs(
            v, compute_loss_fn, print_every=print_every, **lbfgs_kw, **conv_kw
        )
        all_loss_history.extend(lh)
        all_per_step_history.extend(psh)
        param_snapshots.append(v.data.clone().cpu())

        if lh[-1] < best_loss:
            best_loss = lh[-1]
            best_params = v

    assert best_params is not None
    logger.info("  Basin hopping done: best loss = %.6f", best_loss)
    return (
        best_params,
        all_loss_history,
        all_per_step_history,
        _subsample_snapshots(param_snapshots, max_snapshots, log_spaced_snapshots),
    )


def _knn_graph_init_embedding(
    f_start_k: Tensor,
    f_end_k: Tensor,
    sample_embeddings_k: Tensor,
    interior_t: Tensor,
    k_neighbors: int,
) -> Tensor:
    """Build k-NN graph over k-dim embeddings, find shortest path, interpolate.

    Args:
        f_start_k: (k,) start centroid in PCA space.
        f_end_k: (k,) end centroid in PCA space.
        sample_embeddings_k: (M, k) per-sample embeddings.
        interior_t: (n_interior,) t values strictly between 0 and 1.
        k_neighbors: Number of nearest neighbors for graph construction.

    Returns:
        (n_interior, k) initial path for embedding-space optimizer (same device as inputs).
    """
    dev = f_start_k.device
    M = sample_embeddings_k.shape[0]
    N = M + 2  # M samples + virtual start + virtual end
    idx_start = M
    idx_end = M + 1

    # All nodes: (N, k)
    all_points = torch.cat(
        [
            sample_embeddings_k,
            f_start_k.unsqueeze(0),
            f_end_k.unsqueeze(0),
        ]
    )

    # Pairwise Euclidean distances (N, N) — on device
    dist_matrix = torch.cdist(all_points.unsqueeze(0), all_points.unsqueeze(0)).squeeze(
        0
    )

    # Build symmetric k-NN adjacency with Euclidean weights (vectorized)
    dists_copy = dist_matrix.clone()
    dists_copy.fill_diagonal_(float("inf"))
    _, topk_idx = dists_copy.topk(k_neighbors, largest=False, dim=1)  # (N, k_neighbors)
    rows_t = torch.arange(N, device=dev).unsqueeze(1).expand_as(topk_idx).reshape(-1)
    cols_t = topk_idx.reshape(-1)
    weights_t = dist_matrix[rows_t, cols_t]
    # Symmetrize and transfer to CPU for scipy
    all_rows = torch.cat([rows_t, cols_t]).cpu().numpy()
    all_cols = torch.cat([cols_t, rows_t]).cpu().numpy()
    all_weights = torch.cat([weights_t, weights_t]).cpu().numpy()

    graph = csr_matrix(
        (all_weights.astype(np.float32), (all_rows, all_cols)),
        shape=(N, N),
    )

    # Shortest path (scipy, CPU-only)
    dists_sp, predecessors = sp_shortest_path(
        graph,
        directed=False,
        indices=idx_start,
        return_predecessors=True,
    )

    _linear_fallback = lambda: torch.stack(
        [(1 - a) * f_start_k + a * f_end_k for a in interior_t.to(dev)]
    )

    if np.isinf(dists_sp[idx_end]):
        logger.warning(
            "knn_graph_init_embedding: no path from start to end with k=%d, "
            "falling back to linear interpolation",
            k_neighbors,
        )
        return _linear_fallback()

    # Reconstruct path from predecessors
    path_indices = []
    node = idx_end
    while node != idx_start:
        path_indices.append(node)
        node = predecessors[node]
    path_indices.append(idx_start)
    path_indices.reverse()

    waypoints = all_points[path_indices]  # (W, k) on device

    logger.info(
        "knn_graph_init_embedding: found path with %d waypoints (k=%d)",
        len(path_indices),
        k_neighbors,
    )

    # Compute Euclidean distance per segment for t allocation
    segment_dists = (waypoints[1:] - waypoints[:-1]).norm(dim=-1)  # (W-1,)
    total_dist = segment_dists.sum()
    if total_dist < 1e-12:
        return _linear_fallback()

    # Allocate interior points proportionally to segment distance
    n_interior = len(interior_t)
    raw_alloc = (segment_dists / total_dist * n_interior).cpu().numpy()
    alloc = np.floor(raw_alloc).astype(int)
    remainders = raw_alloc - alloc
    deficit = n_interior - alloc.sum()
    for idx in np.argsort(-remainders)[:deficit]:
        alloc[idx] += 1

    # Linearly interpolate within each segment (on device)
    parts = []
    for s in range(len(alloc)):
        n_pts = alloc[s]
        if n_pts == 0:
            continue
        seg_alphas = torch.linspace(0, 1, n_pts + 2, device=dev)[1:-1]
        for a in seg_alphas:
            parts.append((1 - a) * waypoints[s] + a * waypoints[s + 1])

    return torch.stack(parts)


def _embedding_path_length(
    v_interior: Tensor,
    f_start_k: Tensor | None,
    f_end_k: Tensor | None,
    fix_start: bool,
    fix_end: bool,
) -> Tensor:
    """Sum of consecutive Euclidean distances along the embedding trajectory."""
    parts = []
    if fix_start and f_start_k is not None:
        parts.append(f_start_k.unsqueeze(0))
    parts.append(v_interior)
    if fix_end and f_end_k is not None:
        parts.append(f_end_k.unsqueeze(0))
    path = torch.cat(parts, dim=0)  # (T, k)
    return (path[1:] - path[:-1]).norm(dim=-1).sum()


def _project_to_polyline(queries: Tensor, polyline: Tensor) -> Tensor:
    """For each query, find its closest point on the polyline curve.

    The polyline is the piecewise-linear curve through ``polyline[0..P-1]``.
    For each query, projects onto every segment, clamps to the segment, and
    returns the closest such projection (i.e. min distance to the curve, not
    1-NN to discrete vertices).

    Args:
        queries: (Q, k) query points.
        polyline: (P, k) polyline vertices in order; P >= 2 expected.

    Returns:
        (Q, k) closest points on the polyline.
    """
    Q = queries.shape[0]
    P = polyline.shape[0]
    if P < 2:
        return polyline[0:1].expand(Q, -1).clone()

    a = polyline[:-1]  # (P-1, k)
    ab = polyline[1:] - a  # (P-1, k)
    ab_sq = ab.pow(2).sum(dim=-1).clamp(min=1e-12)  # (P-1,)

    qa = queries.unsqueeze(1) - a.unsqueeze(0)  # (Q, P-1, k)
    t = (qa * ab.unsqueeze(0)).sum(dim=-1) / ab_sq  # (Q, P-1)
    t = t.clamp(0.0, 1.0)

    proj = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)  # (Q, P-1, k)
    dist_sq = (queries.unsqueeze(1) - proj).pow(2).sum(dim=-1)  # (Q, P-1)
    best_seg = dist_sq.argmin(dim=-1)  # (Q,)
    return proj[torch.arange(Q, device=proj.device), best_seg]  # (Q, k)


def path_recapitulation_metrics(
    v: Tensor,
    v_geo: Tensor,
    variance_threshold: float = 0.99,
) -> dict[str, float]:
    """Three scalars measuring how v recapitulates v_geo as a curve.

    Both inputs are (T, k) trajectories in the same feature space (T may differ
    between them). All three metrics are parameterization-invariant: each v[t]
    is compared to its closest point on the v_geo polyline, not to v_geo[t].

    ``r_squared`` is computed in v_geo's *intrinsic* subspace: the smallest set
    of leading SVD directions of centered v_geo that explain at least
    ``variance_threshold`` (default 99%) of v_geo's total variance. This
    matches what 3D visualizations show — shape agreement *within v_geo's
    natural plane*, discarding v's variance in directions where v_geo has no
    extent. ``mean_dist_from_geometric`` and ``arc_length_ratio`` stay in full
    input space; they're the off-plane / overall-length signals.

    Returns:
        r_squared: 1 - RSS/TSS in v_geo's intrinsic subspace, clamped [0, 1].
            High if v's projection onto v_geo's natural plane tracks v_geo.
        intrinsic_dim: how many singular directions were retained.
        mean_dist_from_geometric: mean closest-point distance from v[t] to
            the v_geo polyline, in full-space input units.
        arc_length_ratio: L(v) / L(v_geo) in full space. ~1 = comparable
            length, <1 = shorter, >1 = stuttering. NaN if v_geo is degenerate.
    """
    v = v.detach().cpu().to(torch.float32)
    v_geo = v_geo.detach().cpu().to(torch.float32)

    # Full-space distances: closest-point projection + orthogonal residual.
    proj_full = _project_to_polyline(v, v_geo)
    perp_full = (v - proj_full).norm(dim=-1)

    # Intrinsic R²: project both into v_geo's principal-direction basis.
    mean_geo = v_geo.mean(dim=0)
    v_geo_c = v_geo - mean_geo
    _, S, Vh = torch.linalg.svd(v_geo_c, full_matrices=False)
    energy = S.pow(2)
    total = energy.sum().clamp(min=1e-12)
    cumulative = torch.cumsum(energy / total, dim=0)
    # Smallest d such that cumulative[d-1] >= variance_threshold.
    d = int((cumulative < variance_threshold).sum().item()) + 1
    d = max(1, min(d, S.shape[0]))
    basis = Vh[:d].T  # (k, d)
    v_d = (v - mean_geo) @ basis  # (T, d)
    v_geo_d = v_geo_c @ basis  # (T, d)

    proj_d = _project_to_polyline(v_d, v_geo_d)
    rss_d = (v_d - proj_d).pow(2).sum()
    tss_d = (v_d - v_d.mean(dim=0)).pow(2).sum()
    r2 = (1.0 - rss_d / tss_d.clamp(min=1e-12)).clamp(min=0.0, max=1.0)

    L_v = (v[1:] - v[:-1]).norm(dim=-1).sum()
    L_geo = (v_geo[1:] - v_geo[:-1]).norm(dim=-1).sum()
    arc_ratio = float("nan") if L_geo < 1e-12 else float(L_v / L_geo)

    return {
        "r_squared": float(r2),
        "intrinsic_dim": d,
        "mean_dist_from_geometric": float(perp_full.mean()),
        "arc_length_ratio": arc_ratio,
    }


def _compute_trajectory_loss(
    v_interior: Tensor,
    intervenable_model,
    batched_base: dict[str, Tensor],
    batched_sources: list[dict[str, Tensor]],
    inv_locations: dict[str, tuple],
    feature_indices: list[list],
    var_indices: Tensor | list[list[int]],
    interior_target_AW1: Tensor,
    N_pair: int,
    device: torch.device,
    *,
    f_start_k: Tensor | None = None,
    f_end_k: Tensor | None = None,
    fix_start: bool = False,
    fix_end: bool = False,
    path_length_weight: float = 0.0,
    norm_reg_weight: float = 0.0,
    base_metric: str = "hellinger",
) -> tuple[Tensor, list[float]]:
    """Compute loss over all interior timesteps using pyvene interventions.

    Uses intervenable_model() for differentiable forward passes that go through
    FeatureInterpolateIntervention, which preserves the base activation's
    residual via inverse_featurizer(f_out, base_err).

    When path_length_weight > 0, adds a regularization term penalizing the
    total Euclidean path length in embedding space.

    Returns:
        (total_loss, per_step_losses)
    """
    n_interior = v_interior.shape[0]
    total = torch.tensor(0.0, device=device, requires_grad=True)
    per_step = []

    for t_idx in range(n_interior):
        target_point = v_interior[t_idx]  # (k_eff,) — carries gradient

        def replace_fn(
            f_base,
            f_src,
            *,
            _target=target_point,
            **_kw,
        ):
            B = f_base.shape[0]
            k_full = f_base.shape[-1]
            k_eff = _target.shape[-1]
            opt = _target.unsqueeze(0).expand(B, -1)
            if k_eff < k_full:
                # Hold dropped dims at each sample's base values.
                return torch.cat([opt, f_base[:, k_eff:]], dim=-1)
            return opt

        set_interventions_interpolation(intervenable_model, replace_fn)
        _, cf_output = intervenable_model(
            batched_base,
            batched_sources,
            unit_locations=inv_locations,
            subspaces=feature_indices,
        )
        logits_BV = cf_output.logits[:, -1, :]  # last token logits
        B = logits_BV.shape[0]

        p_BW1 = extract_concept_dists_batch(logits_BV, var_indices).to(device)
        target_expanded = interior_target_AW1[t_idx].unsqueeze(0).expand(B, -1)
        metric_fn = _DIFFERENTIABLE_METRICS[base_metric]
        loss_batch = metric_fn(p_BW1, target_expanded).pow(2)
        step_loss = loss_batch.sum() / N_pair
        total = total + step_loss
        per_step.append(step_loss.item())

    if path_length_weight > 0:
        pl = _embedding_path_length(v_interior, f_start_k, f_end_k, fix_start, fix_end)
        total = total + path_length_weight * pl

    if norm_reg_weight > 0 and f_start_k is not None and f_end_k is not None:
        # Penalize per-step deviation of ||v_path[t]|| from the linearly-interpolated
        # centroid norm. Centroids on Mh have natural scale ~||centroid_i||; without
        # this prior, optimization can find low-loss "shortcut" paths at large norm
        # that produce the right belief trajectory but live far off Mh.
        n_int = v_interior.shape[0]
        target_t = torch.linspace(0.0, 1.0, n_int + 2, device=device)[
            1:-1
        ]  # interior t values
        target_norm = (1.0 - target_t) * f_start_k.norm() + target_t * f_end_k.norm()
        path_norms = v_interior.norm(dim=-1)
        norm_reg = ((path_norms - target_norm) ** 2).mean()
        total = total + norm_reg_weight * norm_reg

    return total, per_step


def run_pair_optimization(
    selected_pair_indices: list[tuple[int, int]],
    pair_groups: dict[tuple[int, int], list[int]],
    filtered_samples: list[dict],
    pipeline,
    intervenable_model,
    interchange_target,
    k: int,
    var_indices: Tensor | list[list[int]],
    geodesic_paths: dict[tuple[int, int], Tensor],
    centroid_k: dict[int, Tensor],
    P_start_centroid_WW1: Tensor,
    P_end_centroid_WW1: Tensor,
    t_values: Tensor,
    optimizer_cfg,
    device: torch.device,
    concept_names: list[str],
    sample_embeddings_k: Tensor | None = None,
    base_metric: str = "hellinger",
    skip_optimization: bool = False,
    precomputed_comparisons: dict[tuple[int, int], dict[str, Tensor]] | None = None,
) -> dict[tuple[int, int], dict]:
    """Run optimization for all selected pairs.

    Uses pyvene's IntervenableModel for differentiable forward passes through
    FeatureInterpolateIntervention, which preserves the base activation's
    residual via inverse_featurizer(f_out, base_err).

    Comparison path distributions (geometric, linear) are passed in via
    precomputed_comparisons, collected separately with the full featurizer.

    Returns dict of (ci, cj) -> result_dict with optimized trajectories,
    loss histories, belief probs.
    """
    n_steps = len(t_values)

    interior_mask = torch.ones(n_steps, dtype=torch.bool)
    if optimizer_cfg.fix_start:
        interior_mask[0] = False
    if optimizer_cfg.fix_end:
        interior_mask[-1] = False
    interior_t = t_values[interior_mask]
    n_interior = len(interior_t)

    if isinstance(var_indices, Tensor):
        var_indices = var_indices.to(device)
    max_snapshots = getattr(optimizer_cfg, "max_path_snapshots", 0)
    log_spaced = getattr(optimizer_cfg, "log_spaced_snapshots", False)

    # Resolve effective optimization dim (k_opt slice of full k).
    k_opt = getattr(optimizer_cfg, "k_opt", None)
    k_eff = int(k_opt) if (k_opt is not None and 0 < int(k_opt) < k) else k
    if k_eff < k:
        if sample_embeddings_k is None:
            raise ValueError(
                "embedding_optim.k_opt < k_features requires sample_embeddings_k "
                "(training_features) to pad dropped dims with their mean."
            )
        tail_mean_k = sample_embeddings_k.mean(dim=0)[k_eff:].to(device)  # (k - k_eff,)
        logger.info(
            "Pullback: optimizing in k_opt=%d slice of k_features=%d "
            "(dropped dims held at per-sample base values during opt; stored "
            "trajectory padded with training-feature mean)",
            k_eff,
            k,
        )
    else:
        tail_mean_k = None

    def _lift_to_full_k(v_eff: Tensor) -> Tensor:
        """Lift (n, k_eff) to (n, k) by appending training-mean tail."""
        if k_eff == k:
            return v_eff
        n = v_eff.shape[0]
        tail = tail_mean_k.to(v_eff.device).unsqueeze(0).expand(n, -1)
        return torch.cat([v_eff, tail], dim=-1)

    logger.info("Steps: %d points, interior: %d", n_steps, n_interior)

    results = {}

    for ci, cj in selected_pair_indices:
        sample_indices = pair_groups[(ci, cj)]
        N_pair = len(sample_indices)
        logger.info(
            "Pair (%s -> %s): %d samples",
            concept_names[ci],
            concept_names[cj],
            N_pair,
        )

        # Build self-intervention examples for pyvene (source == base)
        pair_samples = [filtered_samples[n] for n in sample_indices]
        cf_examples = [
            {"input": s["input"], "counterfactual_inputs": [s["input"]]}
            if "counterfactual_inputs" not in s
            else s
            for s in pair_samples
        ]

        # Use prepare_intervenable_inputs for position mapping
        batched_base, batched_sources, inv_locations, feature_indices = (
            prepare_intervenable_inputs(pipeline, cf_examples, interchange_target)
        )

        # Geodesic paths and centroids are already (W+1) with 'other' bin
        p_target_AW1 = geodesic_paths[(ci, cj)]
        p_start_W1 = P_start_centroid_WW1[ci]
        p_end_W1 = P_end_centroid_WW1[cj]

        # Append "other" bin for model output distributions (concept-only from collect_grid_distributions)
        def _append_other(p: Tensor) -> Tensor:
            if p.dim() == 1:
                other = (1.0 - p.sum()).clamp(min=0.0).unsqueeze(0)
                return torch.cat([p, other])
            other = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            return torch.cat([p, other], dim=-1)

        f_start_k_full = centroid_k[ci].to(device)
        f_end_k_full = centroid_k[cj].to(device)
        # Sliced versions used inside optimization (init, loss, NEB endpoints,
        # path-length regularization). Centroids stored in pair_result remain full-k.
        f_start_k = f_start_k_full[:k_eff]
        f_end_k = f_end_k_full[:k_eff]

        # Override belief target if using geometric path
        embedding_init = getattr(optimizer_cfg, "init", "linear")
        geo_available = (
            precomputed_comparisons is not None
            and (ci, cj) in precomputed_comparisons
            and "v_geometric_k" in precomputed_comparisons[(ci, cj)]
        )
        if embedding_init == "knn_graph" and sample_embeddings_k is not None:
            k_init_emb = getattr(optimizer_cfg, "k_init", 5)
            sample_emb_eff = (
                sample_embeddings_k[:, :k_eff].to(device)
                if k_eff < k
                else sample_embeddings_k.to(device)
            )
            v_init = _knn_graph_init_embedding(
                f_start_k,
                f_end_k,
                sample_emb_eff,
                interior_t,
                k_init_emb,
            )
        elif embedding_init == "geometric" and geo_available:
            v_geo_full_k = precomputed_comparisons[(ci, cj)]["v_geometric_k"].to(device)
            v_init = v_geo_full_k[interior_mask][:, :k_eff]
        else:
            if embedding_init == "geometric" and not geo_available:
                logger.warning(
                    "init='geometric' requested but no precomputed v_geometric_k "
                    "for pair (%d, %d); falling back to linear",
                    ci,
                    cj,
                )
            v_init = torch.stack(
                [(1 - a) * f_start_k + a * f_end_k for a in interior_t]
            )

        # Snapshot v_init as full-k trajectory (start/end pinned the same way
        # as the optimized output) so visualization can render the actual init.
        v_init_eff_full = torch.zeros(n_steps, k_eff, device=device)
        v_init_eff_full[interior_mask] = v_init.to(device)
        if optimizer_cfg.fix_start:
            v_init_eff_full[0] = f_start_k.to(device)
        if optimizer_cfg.fix_end:
            v_init_eff_full[-1] = f_end_k.to(device)
        v_init_k = _lift_to_full_k(v_init_eff_full).cpu()

        interior_target_AW1 = p_target_AW1[interior_mask].to(device)

        loss_history: list[float] = []
        per_step_loss_history: list[float] = []
        path_history_k: list[Tensor] = []

        if skip_optimization:
            # Use linear init as placeholder, still collect comparison paths below
            v_eff_full = torch.zeros(n_steps, k_eff, device=device)
            v_eff_full[interior_mask] = v_init.to(device)
            if optimizer_cfg.fix_start:
                v_eff_full[0] = f_start_k.to(device)
            if optimizer_cfg.fix_end:
                v_eff_full[-1] = f_end_k.to(device)
            v_all_k = _lift_to_full_k(v_eff_full)
            logger.info("  Skipping optimization (skip_optimization=true)")
        else:
            pl_weight = getattr(optimizer_cfg, "path_length_weight", 0.0)
            norm_reg_weight = getattr(optimizer_cfg, "norm_reg_weight", 0.0)
            conv_window = getattr(optimizer_cfg, "convergence_window", 5)
            conv_tol = getattr(optimizer_cfg, "convergence_tol", 0.001)

            # Path parameterization: "free_points" optimizes each interior path
            # point directly; "tps" optimizes a smaller set of control values
            # of a 1D natural cubic spline that decodes to the interior path.
            trajectory_param = getattr(optimizer_cfg, "trajectory_param", "free_points")
            tps_path: Any = None
            if trajectory_param == "tps":
                from causalab.methods.pullback.tps_path import TPSPathModule

                tps_cfg = getattr(optimizer_cfg, "tps", None)
                n_ctrl = (
                    int(getattr(tps_cfg, "n_control_points", 10)) if tps_cfg else 10
                )
                tps_smooth = (
                    float(getattr(tps_cfg, "smoothness", 0.0)) if tps_cfg else 0.0
                )
                tps_path = TPSPathModule(
                    n_control=n_ctrl,
                    n_eval=v_init.shape[0],
                    k_features=k_eff,
                    smoothness=tps_smooth,
                ).to(device)
                tps_path.initialize_from_path(v_init.to(device))
                logger.info(
                    "  trajectory_param=tps: %d control points, smoothness=%.3f, "
                    "%d eval points (k_eff=%d)",
                    n_ctrl,
                    tps_smooth,
                    v_init.shape[0],
                    k_eff,
                )

            def make_loss_fn(v_params_ref: nn.Parameter):
                # When TPS is active, v_params_ref IS tps_path.values; we decode
                # the path from it through the spline before computing the loss
                # so gradients flow control_values → spline → path → model.
                def compute_loss() -> tuple[Tensor, list[float]]:
                    v_interior = tps_path() if tps_path is not None else v_params_ref
                    return _compute_trajectory_loss(
                        v_interior,
                        intervenable_model,
                        batched_base,
                        batched_sources,
                        inv_locations,
                        feature_indices,
                        var_indices,
                        interior_target_AW1,
                        N_pair,
                        device,
                        f_start_k=f_start_k,
                        f_end_k=f_end_k,
                        fix_start=optimizer_cfg.fix_start,
                        fix_end=optimizer_cfg.fix_end,
                        path_length_weight=pl_weight,
                        norm_reg_weight=norm_reg_weight,
                        base_metric=base_metric,
                    )

                return compute_loss

            if tps_path is not None and optimizer_cfg.name not in ("lbfgs", "adam"):
                raise NotImplementedError(
                    f"trajectory_param=tps is currently supported only with "
                    f"name=lbfgs or name=adam (got name={optimizer_cfg.name!r})"
                )

            # Patch the underlying HF model so its forward defaults to
            # logits_to_keep=1 (compute lm_head only at the last position) for
            # the duration of optimization. The loss only ever reads the last
            # token's logits, so this is semantically a no-op while saving
            # ~1 GB per forward at 2048-ctx.
            with _ForceLastTokenLogits(intervenable_model):
                if optimizer_cfg.name == "basin_hopping":
                    bh_cfg = {
                        "n_hops": optimizer_cfg.basin_hopping.n_hops,
                        "step_size": optimizer_cfg.basin_hopping.step_size,
                        "temperature": optimizer_cfg.basin_hopping.temperature,
                        "n_candidates": optimizer_cfg.basin_hopping.n_candidates,
                        "lbfgs_steps": optimizer_cfg.basin_hopping.lbfgs_steps,
                        "lbfgs_lr": optimizer_cfg.basin_hopping.lbfgs_lr,
                        "lbfgs_max_iter": optimizer_cfg.basin_hopping.lbfgs_max_iter,
                        "lbfgs_line_search_fn": optimizer_cfg.basin_hopping.lbfgs_line_search_fn,
                    }
                    v_params, loss_history, per_step_loss_history, raw_snapshots = (
                        _run_basin_hopping(
                            v_init,
                            make_loss_fn,
                            bh_cfg,
                            device,
                            max_snapshots=max_snapshots,
                            log_spaced_snapshots=log_spaced,
                            convergence_window=conv_window,
                            convergence_tol=conv_tol,
                        )
                    )
                elif optimizer_cfg.name == "adam":
                    if tps_path is not None:
                        v_params = tps_path.values
                    else:
                        v_params = nn.Parameter(v_init.clone().to(device))
                    loss_history, per_step_loss_history, raw_snapshots = _run_adam(
                        v_params,
                        make_loss_fn(v_params),
                        steps=optimizer_cfg.adam.steps,
                        lr=optimizer_cfg.adam.lr,
                        max_snapshots=max_snapshots,
                        log_spaced_snapshots=log_spaced,
                    )
                elif optimizer_cfg.name == "neb":
                    v_params = nn.Parameter(v_init.clone().to(device))
                    loss_history, per_step_loss_history, raw_snapshots = _run_neb(
                        v_params,
                        make_loss_fn(v_params),
                        steps=optimizer_cfg.adam.steps,
                        lr=optimizer_cfg.adam.lr,
                        spring_k=getattr(optimizer_cfg, "spring_k", 1.0),
                        f_start_k=f_start_k,
                        f_end_k=f_end_k,
                        fix_start=optimizer_cfg.fix_start,
                        fix_end=optimizer_cfg.fix_end,
                        n_steps=n_steps,
                        interior_mask=interior_mask,
                        max_snapshots=max_snapshots,
                        log_spaced_snapshots=log_spaced,
                        convergence_window=conv_window,
                        convergence_tol=conv_tol,
                    )
                else:
                    if tps_path is not None:
                        v_params = tps_path.values
                    else:
                        v_params = nn.Parameter(v_init.clone().to(device))
                    loss_history, per_step_loss_history, raw_snapshots = _run_lbfgs(
                        v_params,
                        make_loss_fn(v_params),
                        steps=optimizer_cfg.lbfgs.steps,
                        lr=optimizer_cfg.lbfgs.lr,
                        max_iter=optimizer_cfg.lbfgs.max_iter,
                        line_search_fn=optimizer_cfg.lbfgs.line_search_fn,
                        max_snapshots=max_snapshots,
                        log_spaced_snapshots=log_spaced,
                        convergence_window=conv_window,
                        convergence_tol=conv_tol,
                    )

            # Decode raw snapshots (control values when TPS, path values when
            # free_points) into K-step path tensors for path_history_k.
            if tps_path is not None:
                with torch.no_grad():
                    raw_snapshots = [
                        tps_path.evaluate_at_values(s.to(device)).cpu()
                        for s in raw_snapshots
                    ]

            # Build snapshots in k_eff, lift to full k via training-mean tail.
            f_start_cpu = f_start_k.cpu()
            f_end_cpu = f_end_k.cpu()
            for snap in raw_snapshots:
                v_snap_eff = torch.zeros(n_steps, k_eff)
                v_snap_eff[interior_mask] = snap
                if optimizer_cfg.fix_start:
                    v_snap_eff[0] = f_start_cpu
                if optimizer_cfg.fix_end:
                    v_snap_eff[-1] = f_end_cpu
                path_history_k.append(_lift_to_full_k(v_snap_eff))

            # Assemble full k-dim trajectory. For TPS, decode the final control
            # values into the K-step interior path before lifting.
            v_eff_full = torch.zeros(n_steps, k_eff, device=device)
            if tps_path is not None:
                with torch.no_grad():
                    v_eff_full[interior_mask] = tps_path()
            else:
                v_eff_full[interior_mask] = v_params.data
            if optimizer_cfg.fix_start:
                v_eff_full[0] = f_start_k.to(device)
            if optimizer_cfg.fix_end:
                v_eff_full[-1] = f_end_k.to(device)
            v_all_k = _lift_to_full_k(v_eff_full)

        # Collect optimized path distributions (per-sample for error bars).
        # When k_eff < k, pass the k_eff-dim trajectory so collect_grid_distributions'
        # replace_fn preserves each sample's base values in dropped dims — matching
        # the actual injection used during optimization.
        with torch.no_grad():
            opt_probs_raw = collect_grid_distributions(
                pipeline=pipeline,
                grid_points=v_eff_full if k_eff < k else v_all_k,
                interchange_target=interchange_target,
                filtered_samples=pair_samples,
                var_indices=var_indices,
                n_base_samples=len(pair_samples),
                average=False,
                full_vocab_softmax=True,
            )
            opt_probs_AW1 = _append_other(opt_probs_raw.mean(dim=1))

        pair_result = {
            "base_class": ci,
            "cf_class": cj,
            "n_samples": N_pair,
            "v_optimized_k": v_all_k.cpu(),
            "v_init_k": v_init_k,
            "opt_probs_raw": opt_probs_raw,
            "f_start_k": f_start_k_full.cpu(),
            "f_end_k": f_end_k_full.cpu(),
            "opt_probs_AW1": opt_probs_AW1,
            "loss_history": loss_history,
            "per_step_loss": torch.tensor(per_step_loss_history),
            "p_target_AW1": p_target_AW1,
            "p_start": p_start_W1,
            "p_end": p_end_W1,
            "path_history_k": path_history_k,
        }

        # Merge pre-computed comparison distributions
        if precomputed_comparisons and (ci, cj) in precomputed_comparisons:
            pair_result.update(precomputed_comparisons[(ci, cj)])

        results[(ci, cj)] = pair_result

    return results
