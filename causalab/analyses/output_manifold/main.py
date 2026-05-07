"""Output manifold analysis: fit a TPS in Hellinger belief space."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
import numpy as np

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    resolve_task,
    _task_config_for_metadata,
    generate_datasets,
    get_output_token_ids,
)
from causalab.io.artifacts import save_tensor_results
from causalab.io.sklearn_pca import save_pca, load_pca
from causalab.io.pipelines import load_pipeline
from causalab.io.plots.plot_utils import resolve_task_colormap

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "output_manifold"


def _build_centroid_alignment(task):
    """Centroid sort permutation + pre-remapped graph edges for output_manifold viz.

    plot_3d extracts coord components from train_dataset (e.g. node_coordinates_0,
    node_coordinates_1) and lex-sorts the resulting centroids by alphabetical
    param-key order. The spline's centroids are in intervention_values (class
    index) order, so we return a permutation that maps class-index order →
    plot_3d's lex-sort order. Identity for grid_5x5; non-trivial for cylinder_9x9.

    For graph_walk, also return adjacency edges already remapped from node IDs
    to lex-sort centroid indices — passing them directly to plot_3d sidesteps
    the multi-key edge_node_coords dance path_steering avoids by hashing on
    1D class indices. Returns ``None`` edges for non-graph tasks.
    """
    iv = task.intervention_variable
    embed_fn = (task.causal_model.embeddings or {}).get(iv)

    def _embed(v):
        if embed_fn is not None:
            return list(embed_fn(v))
        return [float(x) for x in v] if isinstance(v, (tuple, list)) else [float(v)]

    coords_per_class = np.array(
        [_embed(v) for v in task.intervention_values],
        dtype=float,
    )
    d = coords_per_class.shape[1]
    if d == 1:
        sort_perm = np.argsort(coords_per_class[:, 0])
    else:
        # np.lexsort: last key is primary; reverse so col 0 is primary
        # (matches torch.unique lex sort under alphabetical key order ..._0, _1).
        sort_perm = np.lexsort([coords_per_class[:, j] for j in reversed(range(d))])

    edges = None
    graph = getattr(task.causal_model, "_graph", None)
    if graph is not None:
        node_to_centroid = np.argsort(sort_perm)
        edges = [
            (int(node_to_centroid[i]), int(node_to_centroid[j]))
            for i, ns in graph.adjacency.items()
            for j in ns
            if j > i
        ]

    return sort_perm, edges


def _build_belief_artifacts(cfg: DictConfig, out_root: str, batch_size: int) -> None:
    """Collect per-example output distributions and fit Hellinger PCA.

    Produces (idempotent — skipped if already present):
      - {out_root}/per_example_output_dists.safetensors
      - {out_root}/hellinger_pca.safetensors (+ sibling hellinger_pca.meta.json)
      - {out_root}/hellinger_pca_3d.html
    """
    import math
    from sklearn.decomposition import PCA
    from causalab.methods.metric import class_probabilities
    from causalab.io.plots.plot_3d_interactive import plot_3d

    nat_path = os.path.join(out_root, "per_example_output_dists.safetensors")
    pca_st_path = os.path.join(out_root, "hellinger_pca.safetensors")
    pca_meta_path = os.path.join(out_root, "hellinger_pca.meta.json")
    viz_3d_path = os.path.join(out_root, "hellinger_pca_3d.html")
    viz_2d_path = os.path.join(out_root, "hellinger_pca_2d.pdf")

    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    expected_dim = len(task.intervention_values) + 1  # +1 for 'other' bin

    data_present = False
    if (
        os.path.exists(nat_path)
        and os.path.exists(pca_st_path)
        and os.path.exists(pca_meta_path)
    ):
        from causalab.io.artifacts import load_tensor_results

        saved = load_tensor_results(out_root, "per_example_output_dists.safetensors")[
            "dists"
        ]
        if saved.shape[-1] == expected_dim:
            logger.info("Belief artifacts already present; skipping collection.")
            data_present = True
        else:
            logger.warning(
                "Stale belief artifacts: saved last-dim=%d, expected=%d (W+1). "
                "Re-collecting against current task config.",
                saved.shape[-1],
                expected_dim,
            )
            os.remove(nat_path)
            os.remove(pca_st_path)
            os.remove(pca_meta_path)

    train_dataset, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    if data_present:
        # Reload data to drive the visualization step.
        from causalab.io.artifacts import load_tensor_results

        per_example_dists = load_tensor_results(
            out_root,
            "per_example_output_dists.safetensors",
        )["dists"]
        hellinger_pca = load_pca(out_root, "hellinger_pca")
        sqrt_dists = torch.sqrt(per_example_dists.clamp(min=0))
        hellinger_coords = hellinger_pca.transform(sqrt_dists.numpy())
        pipeline = None
    else:
        pipeline = load_pipeline(
            model_name=cfg.model.name,
            task=task,
            max_new_tokens=cfg.task.max_new_tokens,
            device=cfg.model.device,
            dtype=cfg.model.get("dtype"),
            eager_attn=cfg.model.get("eager_attn"),
        )
        score_token_ids, _ = get_output_token_ids(task, pipeline)
        if score_token_ids is None or not task.intervention_values:
            raise RuntimeError(
                "output_manifold requires score tokens and a task intervention_variable."
            )

        output_logits: list[list[torch.Tensor]] = []
        n_batches = math.ceil(len(train_dataset) / batch_size)
        for bi in range(n_batches):
            start, end = bi * batch_size, min((bi + 1) * batch_size, len(train_dataset))
            batch_inputs = [ex["input"] for ex in train_dataset[start:end]]
            result = pipeline.generate(batch_inputs)
            scores = result["scores"]
            for k in range(len(batch_inputs)):
                output_logits.append([s[k].cpu() for s in scores])

        per_example_probs_fvs = torch.stack(
            [
                class_probabilities(
                    ol[-1], score_token_ids, full_vocab_softmax=True
                ).squeeze(0)
                for ol in output_logits
            ]
        )
        other_mass = (1.0 - per_example_probs_fvs.sum(dim=-1, keepdim=True)).clamp(
            min=0.0
        )
        per_example_dists = torch.cat([per_example_probs_fvs, other_mass], dim=-1)
        save_tensor_results(
            {"dists": per_example_dists},
            out_root,
            "per_example_output_dists.safetensors",
        )

        # Fit Hellinger PCA
        sqrt_dists = torch.sqrt(per_example_dists.clamp(min=0))
        hellinger_pca = PCA(n_components=3)
        hellinger_coords = hellinger_pca.fit_transform(sqrt_dists.numpy())
        save_pca(hellinger_pca, out_root, "hellinger_pca")
        logger.info(
            "Fit Hellinger PCA: %.1f%% var",
            hellinger_pca.explained_variance_ratio_.sum() * 100,
        )

    # 3D interactive + 2D static scatter of Hellinger PCA coordinates.
    # Always retried if the output files are missing — separately gated from
    # data collection so a previously-failed viz can be regenerated.
    if os.path.exists(viz_3d_path) and os.path.exists(viz_2d_path):
        logger.info("Hellinger PCA visualizations already present; skipping.")
    else:
        try:
            iv_values = task.intervention_values
            iv_name = task.intervention_variable
            sort_perm, edges = _build_centroid_alignment(task)
            # feature_kind='hellinger' tells plot_3d/plot_features_2d to take the
            # √ internally and use √(mean(p)) per class as centroids — drift-free.
            # Pass per-example probabilities directly; PCA fit happens inside.
            plot_3d(
                features=per_example_dists,
                output_path=viz_3d_path,
                train_dataset=train_dataset,
                intervention_variable=iv_name,
                embeddings=task.causal_model.embeddings,
                variable_values=[str(v) for v in iv_values],
                colormap=resolve_task_colormap(cfg.task, "rainbow"),
                edges=edges,
                feature_kind="hellinger",
            )
            from causalab.io.plots.pca_scatter import plot_features_2d

            plot_features_2d(
                features=per_example_dists,
                output_path=viz_2d_path,
                train_dataset=train_dataset,
                intervention_variable=task.intervention_variable,
                embeddings=task.causal_model.embeddings,
                colormap=resolve_task_colormap(cfg.task, "rainbow"),
                variable_values=[str(v) for v in iv_values],
                edges=edges,
                feature_kind="hellinger",
            )
        except Exception as e:
            logger.warning("Hellinger PCA visualization failed: %s", e, exc_info=True)

    if pipeline is not None:
        del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the output_manifold analysis: fit a TPS in Hellinger belief space."""
    from causalab.methods.spline.belief_fit import (
        fit_belief_tps_pca,
        fit_belief_tps_parameter,
    )
    from causalab.methods.pullback.geodesic import load_natural_distributions
    from causalab.io.plots.plot_3d_interactive import plot_3d
    from causalab.analyses.activation_manifold.fitting_pipeline import (
        _auto_detect_intrinsic_dim,
    )

    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root

    # Top-level output_manifold dir owns the Hellinger artifacts (shared across TPS configs)
    bm_root = os.path.join(root, "output_manifold")
    os.makedirs(bm_root, exist_ok=True)

    # Output directory for this particular TPS fit
    m_sub = f"{analysis.method}_s{analysis.smoothness}"
    out_dir = os.path.join(bm_root, m_sub)
    tv = cfg.task.get("target_variable")
    if tv:
        out_dir = os.path.join(out_dir, tv)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure Hellinger artifacts exist (collected here, not in baseline)
    _build_belief_artifacts(cfg, bm_root, batch_size=analysis.batch_size)

    # Load task (for intervention_values, causal_model, embeddings)
    task, _task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    W = len(task.intervention_values)
    if W == 0:
        raise ValueError(
            "Task has no intervention values. "
            "output_manifold requires a task with discrete concept classes."
        )

    # Regenerate the train_dataset _build_belief_artifacts produced (deterministic
    # via seed) to get TRUE class labels row-aligned with natural_dists.
    _bm_train_ds, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )
    _class_assignments = torch.tensor(
        [task.intervention_value_index(ex) for ex in _bm_train_ds],
        dtype=torch.long,
    )

    # Compute per-class centroids in probability space (grouped by TRUE class)
    natural_dists, per_class_dists = load_natural_distributions(
        root,
        W,
        class_assignments=_class_assignments,
    )
    logger.info(
        "Loaded belief data: %d examples, %d classes, %d dims (W+1)",
        natural_dists.shape[0],
        W,
        per_class_dists.shape[1],
    )

    # Load Hellinger PCA (needed for pca mode and visualization)
    try:
        hellinger_pca = load_pca(bm_root, "hellinger_pca")
    except FileNotFoundError:
        hellinger_pca = None

    # Auto-detect intrinsic_dim
    intrinsic_dim = analysis.intrinsic_dim
    intrinsic_mode = analysis.intrinsic_mode

    if intrinsic_dim is None:
        if intrinsic_mode == "parameter":
            intrinsic_dim = _auto_detect_intrinsic_dim(task.causal_model)
        else:
            raise ValueError(
                "intrinsic_dim must be set explicitly for pca intrinsic_mode"
            )
        logger.info(
            "Auto-detected intrinsic_dim=%d (mode=%s)", intrinsic_dim, intrinsic_mode
        )

    # Fit belief TPS
    if intrinsic_mode == "pca":
        if hellinger_pca is None:
            raise FileNotFoundError(f"hellinger_pca not found at {bm_root}")
        result = fit_belief_tps_pca(
            per_class_dists=per_class_dists,
            hellinger_pca=hellinger_pca,
            intrinsic_dim=intrinsic_dim,
            smoothness=analysis.smoothness,
            output_dir=out_dir,
            max_control_points=analysis.get("max_control_points", "all"),
        )
    elif intrinsic_mode == "parameter":
        result = fit_belief_tps_parameter(
            per_class_dists=per_class_dists,
            causal_model=task.causal_model,
            intervention_variable=task.intervention_variable,
            intervention_values=task.intervention_values,
            intrinsic_dim=intrinsic_dim,
            smoothness=analysis.smoothness,
            output_dir=out_dir,
            max_control_points=analysis.get("max_control_points", "all"),
        )
    else:
        raise ValueError(f"Unknown intrinsic_mode: {intrinsic_mode!r}")

    belief_spline = result["manifold"]
    centroid_hellinger = result["centroid_hellinger"]

    # ── 3D Visualization (Hellinger PCA space) ──
    vis_dir = os.path.join(out_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    if hellinger_pca is not None:
        iv = task.intervention_variable or "class"
        variable_values = [str(v) for v in task.intervention_values]

        # Identity standardization (belief space has no standardization).
        # Pass per-example PROBABILITIES with feature_kind='hellinger': plot_3d
        # √-transforms internally for the PCA fit / display and uses √(mean(p))
        # per class as centroids — drift-free.
        D = natural_dists.shape[1]
        mean = torch.zeros(D)
        std = torch.ones(D)
        # Data-driven mesh range (99.7% quantile of encoded √p) instead of
        # control-points + 5% margin — avoids spike-end artefacts where the
        # TPS spline extrapolates beyond its training extent.
        from causalab.analyses.activation_manifold.utils import (
            _compute_intrinsic_ranges as _data_intrinsic_ranges,
        )

        sqrt_nat_for_ranges = torch.sqrt(natural_dists.clamp(min=0)).float()
        ranges = _data_intrinsic_ranges(
            sqrt_nat_for_ranges,
            belief_spline,
            mean,
            std,
        )
        _, edges = _build_centroid_alignment(task)

        plot_3d(
            features=natural_dists.float(),
            output_path=os.path.join(vis_dir, "output_manifold_3d.html"),
            title="Output Manifold (Hellinger space)",
            train_dataset=_bm_train_ds[: natural_dists.shape[0]],
            intervention_variable=iv,
            embeddings=task.causal_model.embeddings,
            colormap=analysis.get("colormap", None),
            manifold_obj=belief_spline,
            mean=mean,
            std=std,
            ranges=ranges,
            variable_values=variable_values,
            edges=edges,
            feature_kind="hellinger",
        )
        logger.info("Saved 3D output manifold visualization.")
    else:
        logger.warning("Skipping visualization: no hellinger_pca available.")

    # ── Analysis metadata ──
    metadata = {
        "analysis": "output_manifold",
        "intrinsic_mode": intrinsic_mode,
        "method": analysis.method,
        "smoothness": analysis.smoothness,
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": int(per_class_dists.shape[1]),
        "n_classes": W,
        "model": cfg.model.id,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
        ),
        **result["metadata"],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("output_manifold analysis complete. Output in %s", out_dir)
    return {"output_dir": out_dir, "metadata": metadata, **result}
