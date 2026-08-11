"""Activation manifold analysis: fit a manifold in the subspace."""

from __future__ import annotations

import json
import logging
import os
import time as _time
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    resolve_task,
    prepare_datasets,
    get_output_token_ids,
    resolve_intervention_metric,
    build_targets_for_layers,
    build_targets_for_grid,
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
)
from causalab.io.pipelines import (
    load_pipeline,
    load_lite_pipeline,
    find_subspace_dirs,
    load_subspace_metadata,
)
from causalab.io.artifacts import load_tensor_results, load_tensors_with_meta
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.methods.output_tokens import form_group_values

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "activation_manifold"

# task.* keys this analysis reads with no safe default: ``intervention_metric``
# (direct attribute access) and ``colormap`` (the ``${task.colormap}``
# interpolation baked into activation_manifold.yaml). The runner validates these
# up front so a factory task missing them fails fast listing every gap, instead
# of crashing one key at a time deep in the run (#264).
REQUIRED_TASK_KEYS = ("colormap", "intervention_metric")


def _resolve_grid_cell(
    ss_meta: dict,
    explicit_layer: int | None,
    explicit_token_position: str | None,
) -> tuple[int | None, str | None]:
    """Resolve a single (layer, token_position) from grid subspace metadata.

    Priority:
    1. Explicit values from the activation_manifold config.
    2. Fill missing values from ``best_cell`` in the subspace metadata.
    3. Return ``(None, None)`` if unresolvable.
    """
    best_cell = ss_meta.get("best_cell") or {}

    layer = explicit_layer if explicit_layer is not None else best_cell.get("layer")
    token_position = (
        explicit_token_position
        if explicit_token_position is not None
        else best_cell.get("token_position")
    )

    if layer is not None:
        layer = int(layer)
    if token_position is not None:
        token_position = str(token_position)

    return layer, token_position


def _manifold_subdir(analysis: DictConfig) -> str:
    """Resolve the method subdir for the output path, honoring a CLI override.

    Reads the Hydra-resolved ``analysis._subdir`` (default
    ``${.method}_s${.smoothness}``) rather than rebuilding
    ``f"{method}_s{smoothness}"`` directly, so a CLI override of
    ``activation_manifold._subdir=...`` actually changes the write path
    (#171, sub-issue C5). The ``_shuf{seed}`` suffix is appended when
    ``embedding_shuffle_seed`` is set, matching prior behavior.
    """
    m_sub = str(analysis._subdir)
    shuffle_seed = analysis.get("embedding_shuffle_seed", None)
    if shuffle_seed is not None:
        m_sub += f"_shuf{shuffle_seed}"
    return m_sub


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the activation_manifold analysis: fit a manifold in the subspace."""
    from safetensors.torch import load_file
    from causalab.analyses.activation_manifold.fitting_pipeline import (
        ManifoldFittingConfig,
        run_manifold_fitting_pipeline,
    )
    from causalab.methods.metric import compute_reference_distributions

    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root

    # Discover subspace dirs
    subspace_sub = analysis.subspace
    if subspace_sub is not None:
        subspace_subs = [subspace_sub]
    else:
        subspace_subs = find_subspace_dirs(root)
        if not subspace_subs:
            raise ValueError(
                f"No subspace directories found in {root}/subspace/. "
                "Run the subspace analysis first."
            )

    # Load task + model (once for all subspace dirs)
    task, task_cfg_raw = resolve_task(  # pyright: ignore[reportUnusedVariable]
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    # Only ``comparison_fn`` (the distribution metric) is used here; the string
    # metric (the task's checker, #167) is resolved for signature parity.
    _string_metric, comparison_fn = resolve_intervention_metric(
        cfg.task.intervention_metric, checker=task.checker
    )

    _t = _time.time()
    train_dataset, test_dataset = prepare_datasets(  # pyright: ignore[reportUnusedVariable]
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
        filter_correct=False,
    )
    logger.info("Dataset generation: %.1fs", _time.time() - _t)

    # Only load full model weights when the decoding reconstruction test will run.
    # Manifold fitting itself works on cached features; building site specs
    # and saving the composed featurizer only need tokenizer + model config.
    skip_decoding_eval = analysis.get("skip_decoding_eval", True)

    _t = _time.time()
    if skip_decoding_eval:
        pipeline = load_lite_pipeline(
            model_name=cfg.model.name,
            max_new_tokens=cfg.task.max_new_tokens,
            use_chat_template=cfg.model.get("chat_template", False),
        )
        logger.info("Lite pipeline loading: %.1fs", _time.time() - _t)
    else:
        pipeline = load_pipeline(
            model_name=cfg.model.name,
            task=task,
            max_new_tokens=cfg.task.max_new_tokens,
            device=cfg.model.device,
            dtype=cfg.model.get("dtype"),
            eager_attn=cfg.model.get("eager_attn"),
            use_chat_template=cfg.model.get("chat_template", False),
            chat_answer_directive=cfg.model.get("chat_answer_directive"),
        )
        logger.info("Model loading: %.1fs", _time.time() - _t)

    score_token_ids, n_classes = get_output_token_ids(task, pipeline)
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    results = {}

    # Pre-loop config values
    tv = cfg.task.get("target_variable")
    explicit_layers = analysis.get("layers", None)
    explicit_token_positions = analysis.get("token_positions", None)
    explicit_layer = int(explicit_layers[0]) if explicit_layers else None
    explicit_token_position = (
        str(explicit_token_positions[0]) if explicit_token_positions else None
    )

    for ss_sub in subspace_subs:
        logger.info("=== Manifold for subspace %s ===", ss_sub)

        ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
        k_features_raw = ss_meta.get("k_features")
        if k_features_raw is None:
            logger.warning("Skipping %s: missing k_features in metadata", ss_sub)
            continue
        k_features = int(k_features_raw)
        ss_method = ss_meta.get("method", "pca")
        ss_mode = ss_meta.get("mode", "single")

        # Resolve (layer, token_position) depending on subspace mode
        if ss_mode == "grid":
            layer, token_position = _resolve_grid_cell(
                ss_meta,
                explicit_layer,
                explicit_token_position,
            )
            if layer is None or token_position is None:
                logger.warning(
                    "Skipping %s: cannot resolve grid cell (set analysis.layers "
                    "and analysis.token_positions, or ensure subspace metadata "
                    "contains best_cell)",
                    ss_sub,
                )
                continue
            logger.info(
                "Grid subspace %s: resolved cell (layer=%d, position=%s)",
                ss_sub,
                layer,
                token_position,
            )
        else:
            layer = ss_meta.get("layer")
            token_position = ss_meta.get("token_position") or explicit_token_position
            if layer is None:
                logger.warning("Skipping %s: missing layer in metadata", ss_sub)
                continue

        # Output directory — include cell info for grid subspaces
        m_sub = _manifold_subdir(analysis)
        if token_position is not None:
            out_dir = os.path.join(
                root,
                "activation_manifold",
                ss_sub,
                f"L{layer}_{token_position}",
                m_sub,
            )
        else:
            out_dir = os.path.join(root, "activation_manifold", ss_sub, m_sub)
        if tv:
            out_dir = os.path.join(out_dir, tv)
        os.makedirs(out_dir, exist_ok=True)

        # Build the cell's site spec
        if token_position is not None:
            targets, positions = build_targets_for_grid(  # pyright: ignore[reportUnusedVariable]
                pipeline,
                task,
                [layer],
                position_names=[token_position],
            )
        else:
            targets, _token_pos = build_targets_for_layers(pipeline, task, [layer])
        spec = next(iter(targets.values()))[0][0]

        # Resolve subspace artifact directories
        from causalab.analyses.subspace import load_subspace_onto_spec

        subspace_out_dir = os.path.join(root, "subspace", ss_sub)
        if tv:
            subspace_out_dir = os.path.join(subspace_out_dir, tv)

        # For grid-mode PCA, per-cell artifacts are in layer_x_pos/L{layer}_{pos}/
        # For DAS, featurizer models are at the grid root in das/models/{layer}__{pos}/
        if ss_mode == "grid" and ss_method == "pca":
            cell_dir = os.path.join(
                subspace_out_dir,
                "layer_x_pos",
                f"L{layer}_{token_position}",
            )
        else:
            cell_dir = subspace_out_dir

        spec = load_subspace_onto_spec(
            spec,
            cell_dir,
            ss_method,
            k_features,
            layer=layer,
        )

        # Load saved dataset from subspace (ensures row alignment with features)
        # Check cell dir first, fall back to grid root
        from causalab.io.counterfactuals import load_counterfactual_examples

        saved_ds_path = os.path.join(cell_dir, "train_dataset.json")
        if not os.path.exists(saved_ds_path):
            saved_ds_path = os.path.join(subspace_out_dir, "train_dataset.json")
        if os.path.exists(saved_ds_path):
            train_dataset = load_counterfactual_examples(
                saved_ds_path, task.causal_model
            )
            logger.info(
                "Loaded saved subspace dataset: %d examples", len(train_dataset)
            )

        # Load pre-computed features from subspace analysis
        features_path = os.path.join(
            cell_dir, "features", "training_features.safetensors"
        )
        if os.path.exists(features_path):
            features = load_file(features_path)["features"]
        else:
            if skip_decoding_eval:
                raise FileNotFoundError(
                    f"Cached features not found at {features_path}. "
                    "Activation manifold runs on cached features by default; "
                    "either re-run the subspace analysis to populate the cache, "
                    "or set skip_decoding_eval=False to load model weights and "
                    "collect features on the fly."
                )
            # Fallback: collect features through loaded featurizer (DAS grid case)
            from causalab.neural.activations.collect import collect_features
            from torch import Tensor

            # collect_output_logits=False (default) returns dict[str, Tensor];
            # cast tells pyright we're in that branch of the union.
            features_dict = cast(
                dict[str, Tensor],
                collect_features(
                    dataset=train_dataset,
                    pipeline=pipeline,
                    sites=[spec],
                    batch_size=analysis.batch_size,
                ),
            )
            features = features_dict[spec.key].detach()
            logger.info(
                "Collected features on-the-fly for grid cell (layer=%s, pos=%s)",
                layer,
                token_position,
            )

        # Load ref_dists (only needed for decoding reconstruction test)
        ref_dists = None
        if (
            not skip_decoding_eval
            and score_token_ids is not None
            and task.intervention_values
        ):
            # Try baseline first, then subspace cache, then compute
            beval_path = os.path.join(
                root, "baseline", "per_class_output_dists.safetensors"
            )
            subspace_ref_path = os.path.join(subspace_out_dir, "ref_dists.safetensors")
            if os.path.exists(beval_path):
                ref_dists = load_tensor_results(
                    os.path.join(root, "baseline"), "per_class_output_dists.safetensors"
                )["dists"]
                logger.info("Loaded ref_dists from baseline")
                # Normalize (baseline saves full_vocab_softmax)
                row_sums = ref_dists.sum(dim=-1)
                if (row_sums < 0.99).any():
                    ref_dists = ref_dists / row_sums.unsqueeze(-1).clamp(min=1e-10)
            elif os.path.exists(subspace_ref_path):
                ref_tensors, _ = load_tensors_with_meta(subspace_out_dir, "ref_dists")
                ref_dists = ref_tensors["value"]
                logger.info("Loaded cached ref_dists from %s", subspace_ref_path)
                row_sums = ref_dists.sum(dim=-1)
                if (row_sums < 0.99).any():
                    ref_dists = ref_dists / row_sums.unsqueeze(-1).clamp(min=1e-10)
            else:
                _t = _time.time()
                if n_classes is None:
                    raise ValueError(
                        "n_classes is None when computing ref_dists — "
                        "task lacks intervention_variable and output_tokens."
                    )
                ref_dists = compute_reference_distributions(
                    dataset=train_dataset,
                    # score_token_ids may be a Tensor from get_output_token_ids;
                    # compute_reference_distributions accepts list-typed ids.
                    score_token_ids=cast(list[int], score_token_ids),
                    n_classes=n_classes,
                    example_to_class=task.intervention_value_index,
                    pipeline=pipeline,
                    batch_size=analysis.batch_size,
                    score_token_index=0,
                )
                logger.info("Reference distributions: %.1fs", _time.time() - _t)

        # Run the manifold fitting pipeline
        fitting_config = ManifoldFittingConfig(
            pipeline=pipeline,
            sites=[[spec]],
            features=features,
            train_dataset=train_dataset,
            causal_model=task.causal_model,
            output_dir=out_dir,
            k_features=k_features,
            manifold_method=analysis.method,
            smoothness=analysis.smoothness,
            intrinsic_dim=analysis.intrinsic_dim,
            intrinsic_mode=analysis.intrinsic_mode,
            intervention_variable=task.intervention_variable,
            periodic_info=task.causal_model.periods,
            embeddings=task.causal_model.embeddings,
            n_grid=analysis.n_grid,
            # get_output_token_ids may return a Tensor; ManifoldFittingConfig
            # accepts list-typed ids. Cast for the runtime-known list path.
            score_token_ids=cast("list[int] | list[list[int]] | None", score_token_ids),
            batch_size=analysis.batch_size,
            ref_dists=ref_dists,
            n_classes=n_classes,
            comparison_fn=comparison_fn,
            seed=cfg.seed,
            skip_decoding_eval=skip_decoding_eval,
            colormap=analysis.get("colormap", None),
            embedding_shuffle_seed=analysis.get("embedding_shuffle_seed", None),
            score_variable_values=(
                # The deduplicated value list per distinct output-form group —
                # what the removed ``output_token_values`` used to hold (#291).
                {
                    task.intervention_variable: form_group_values(
                        task.causal_model.output_tokens[task.intervention_variable]
                    )
                }
                if (
                    task.intervention_variable
                    and task.causal_model.output_tokens
                    and task.causal_model.output_tokens.get(task.intervention_variable)
                )
                else None
            ),
            figure_format=figure_fmt,
            max_control_points=analysis.get("max_control_points", "all"),
        )
        pipeline_result = run_manifold_fitting_pipeline(fitting_config)

        # Save manifold analysis metadata
        manifold_meta = {
            "analysis": "activation_manifold",
            "method": analysis.method,
            "smoothness": analysis.smoothness,
            "subspace": ss_sub,
            "layer": layer,
            "token_position": token_position,
            "k_features": k_features,
            "model": cfg.model.name,
            "task": cfg.task.name,
            "task_config": _task_config_for_metadata(
                cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
            ),
            "n_train": cfg.task.n_train,
            "seed": cfg.seed,
            "embedding_shuffle_seed": analysis.get("embedding_shuffle_seed", None),
        }
        if "reconstruction_kl" in pipeline_result:
            manifold_meta["reconstruction_kl"] = pipeline_result["reconstruction_kl"]

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(manifold_meta, f, indent=2)

        results[ss_sub] = {
            "output_dir": out_dir,
            "metadata": manifold_meta,
            "reconstruction_kl": pipeline_result.get("reconstruction_kl"),
        }

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Activation manifold analysis complete.")
    return results
