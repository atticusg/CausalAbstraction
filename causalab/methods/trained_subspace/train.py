"""
Generic Intervention Training Script (DBM and DAS)

This module provides a generic function to train interventions on any nested
:class:`~causalab.neural.specs.SiteSpec` groups (or a whole
:data:`~causalab.neural.activations.site_grids.SiteGrid`). Supports both:
- DBM (Desiderata-Based Masking): Learns binary masks over sites
- DAS (Distributed Alignment Search): Learns linear subspaces containing causal variables

Feature-space management is functional (WU4, #506): featurizer initialization
and the trained feature-index readout *return new specs*
(``with_featurizer`` / ``with_feature_ids``) — nothing mutates the caller's
specs. The trained featurizer modules themselves are shared by reference, so
the returned specs carry the trained parameters.

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── models/                     # Trained models per key
│   ├── 0__first_token/
│   ├── 0__last_token/
│   └── ...
├── training/                   # Training-specific artifacts
│   └── feature_indices.json
├── train_eval/                 # Training set evaluation
│   ├── scores.json
│   └── raw_results.json
└── test_eval/                  # Test set evaluation
    ├── scores.json
    └── raw_results.json
"""

import collections
import copy
import logging
import random
from typing import Dict, Any, Callable, Mapping, Sequence, Union, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.featurizer import Featurizer
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.methods.metric import (
    causal_score_intervention_outputs,
    score_label_predictions,
)
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.activations.site_grids import SiteGrid
from causalab.neural.dataset import (
    forward_inputs,
    resolve_spec_positions,
)
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import MaskGate
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import collect_ordered, forward_key
from causalab.methods.edit_training import temperature_schedule
from causalab.neural.specs import SiteSpec
from causalab.neural.trainable import (
    concat_label_inputs,
    das_edit,
    dbm_edit,
    freeze_model_parameters,
    place_edit_parameters,
    selected_feature_ids,
    site_device,
    traced_label_loss,
)
from causalab.io.artifacts import (
    save_intervention_results,
    save_training_artifacts,
    save_aggregate_metadata,
)

logger = logging.getLogger(__name__)


def train_interventions(
    causal_model: CausalModel,
    grid: Union[SiteGrid, Sequence[Sequence[SiteSpec]]],
    train_dataset: list[CounterfactualExample],
    test_dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    metric: Callable[[Any, Any], bool],
    config: dict[str, Any],
    source_pipeline: LMPipeline | None = None,
) -> Dict[str, Any]:
    """
    Train interventions (DBM or DAS) on one or more grid cells of SiteSpecs.

    Handles featurizer initialization based on config:
    - intervention_type="mask": Uses Featurizer with tie_masks from config
    - intervention_type="interchange": Uses SubspaceFeaturizer with n_features from config

    This function trains interventions, then evaluates using run_interventions() on both
    train and test datasets.

    Args:
        causal_model: Causal model for generating expected outputs
        grid: Either a :data:`~causalab.neural.activations.site_grids.SiteGrid`
            (cell key → nested spec groups) or a single nested
            ``Sequence[Sequence[SiteSpec]]`` (wrapped as the ``("single",)``
            cell)
        train_dataset: In-memory training counterfactual examples
        test_dataset: In-memory test counterfactual examples
        pipeline: Target LMPipeline where interventions are applied
        target_variable_group: Tuple of target variable names to evaluate jointly
                              (e.g., ("answer",) or ("answer", "position"))
        metric: Function to compare neural output with expected output
        config: Fully-resolved training configuration dict. All keys must be
                present — callers merge with DEFAULT_CONFIG before calling.
                Must include: intervention_type, train_batch_size,
                evaluation_batch_size, training_epoch, init_lr, log_dir,
                featurizer_kwargs (for mask), DAS (for interchange), masking, etc.
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching where you train
            to find features in pipeline that align with activations from
            source_pipeline.

    Note:
        Specs with pre-initialized featurizers (id != "null") are preserved.
        This allows using PCA/SVD-initialized featurizers without overwriting
        them. Initialization and the trained readout are functional — the
        caller's specs are never mutated; the *returned* ``trained_specs``
        carry the trained feature spaces.

    Returns:
        Dictionary containing:
            - results_by_key: dict mapping keys to per-cell results with:
                - train_score: training accuracy
                - test_score: test accuracy
                - feature_indices: ``{spec.key: indices}`` dict
                - train_eval: full train evaluation results
                - test_eval: full test evaluation results
                - trained_specs: the trained nested spec groups (a DBM mask
                  that switched every feature off drops its spec here — the
                  no-op-by-omission contract; its ``feature_indices`` entry
                  records the empty selection)
            - avg_train_score: average training accuracy across cells
            - avg_test_score: average test accuracy across cells
            - metadata: experiment configuration and summary

    Raises:
        ValueError: If invalid intervention_type
        KeyError: If required config keys are missing
    """

    intervention_type = config["intervention_type"]

    if intervention_type not in ["mask", "interchange"]:
        raise ValueError(
            f"Invalid intervention_type: {intervention_type}. "
            f"Must be 'mask' (DBM) or 'interchange' (DAS)."
        )

    # Validate required config for DBM
    if intervention_type == "mask" and "featurizer_kwargs" not in config:
        raise ValueError(
            "config['featurizer_kwargs'] is required for mask interventions. "
            "Set config['featurizer_kwargs'] = {'tie_masks': True} for one mask per unit, "
            "or {'tie_masks': False} for separate masks per feature dimension."
        )

    # Wrap bare nested spec groups in a single-cell grid
    if not isinstance(grid, Mapping):
        grid = {("single",): [list(group) for group in grid]}

    # Initialize featurizers based on config (skips pre-initialized featurizers;
    # returns new specs — the caller's grid is untouched)
    grid = _initialize_featurizers(grid, config)

    # Label training dataset
    labeled_train_dataset = causal_model.label_counterfactual_data(
        copy.deepcopy(train_dataset), list(target_variable_group)
    )

    results_by_key = {}
    eval_batch_size = config["evaluation_batch_size"]

    # Outer progress bar for all cells
    pbar = tqdm(
        grid.items(),
        desc="Training targets",
        disable=not logger.isEnabledFor(logging.DEBUG),
        total=len(grid),
    )

    for key, groups in pbar:
        # Train this cell (inner progress bar handled by training loop)
        trained_groups, feature_indices, _summary = _run_training_loop(
            pipeline=pipeline,
            groups=groups,
            counterfactual_dataset=labeled_train_dataset,  # pyright: ignore[reportArgumentType]  # label_counterfactual_data returns plain dicts
            intervention_type=intervention_type,
            config=config,  # type: ignore[arg-type]
            checker=metric,
            source_pipeline=source_pipeline,
        )

        # Run interventions on train data
        train_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=train_dataset,
                groups=trained_groups,
                batch_size=eval_batch_size,
                output_scores=False,
                source_pipeline=source_pipeline,
            )
        }

        # Score train results
        train_eval = causal_score_intervention_outputs(
            results=train_results,
            dataset=train_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )

        # Run interventions on test data
        test_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=test_dataset,
                groups=trained_groups,
                batch_size=eval_batch_size,
                output_scores=False,
                source_pipeline=source_pipeline,
            )
        }

        # Score test results
        test_eval = causal_score_intervention_outputs(
            results=test_results,
            dataset=test_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )

        results_by_key[key] = {
            "train_score": train_eval["results_by_key"][key]["avg_score"],
            "test_score": test_eval["results_by_key"][key]["avg_score"],
            "feature_indices": feature_indices,
            "train_eval": train_eval["results_by_key"][key],
            "test_eval": test_eval["results_by_key"][key],
            "trained_specs": trained_groups,  # For model saving
        }

    pbar.close()

    # Compute averages
    avg_train = float(
        sum(r["train_score"] for r in results_by_key.values()) / len(results_by_key)
    )
    avg_test = float(
        sum(r["test_score"] for r in results_by_key.values()) / len(results_by_key)
    )

    # Count selected units/features (for DBM)
    num_selected = sum(
        1
        for result in results_by_key.values()
        for indices in result["feature_indices"].values()
        if indices and len(indices) > 0
    )

    training_config: Dict[str, Any] = {
        "train_batch_size": config["train_batch_size"],
        "evaluation_batch_size": config["evaluation_batch_size"],
        "training_epoch": config["training_epoch"],
        "init_lr": config["init_lr"],
    }
    metadata: Dict[str, Any] = {
        "intervention_type": intervention_type,
        "target_variable_group": list(target_variable_group),
        "num_train_examples": len(train_dataset),
        "num_test_examples": len(test_dataset),
        "num_targets": len(grid),
        "avg_train_score": float(avg_train),
        "avg_test_score": float(avg_test),
        "training_config": training_config,
    }

    # Add intervention-specific metadata
    if intervention_type == "mask":
        metadata["num_selected_units"] = num_selected
        training_config["regularization_coefficient"] = config["masking"][
            "regularization_coefficient"
        ]
        training_config["tie_masks"] = config["featurizer_kwargs"]["tie_masks"]
    else:
        training_config["n_features"] = config["DAS"]["n_features"]

    result = {
        "results_by_key": results_by_key,
        "avg_train_score": avg_train,
        "avg_test_score": avg_test,
        "metadata": metadata,
    }

    # Add intervention-specific results
    if intervention_type == "mask":
        result["num_selected_units"] = num_selected

    return result


def _initialize_featurizers(
    grid: Mapping[Tuple[Any, ...], Sequence[Sequence[SiteSpec]]],
    config: dict[str, Any],
) -> Dict[Tuple[Any, ...], list[list[SiteSpec]]]:
    """
    Initialize featurizers on all specs based on config — functionally.

    Returns a new grid: specs that already carry a non-placeholder featurizer
    (id != "null") pass through unchanged, allowing pre-initialized
    featurizers (e.g., from PCA/SVD) to be preserved; the rest get a fresh
    trainable featurizer via ``with_featurizer`` (the caller's specs are
    never mutated).

    For intervention_type="mask": Uses Featurizer with tie_masks
    For intervention_type="interchange": Uses SubspaceFeaturizer with n_features
    """
    intervention_type = config["intervention_type"]

    def initialize(spec: SiteSpec) -> SiteSpec:
        # Skip specs with pre-initialized featurizers
        if spec.fsite.featurizer.id != "null":
            return spec

        if spec.width is None:
            raise ValueError(f"Unit {spec.key} has no width defined")
        if intervention_type == "mask":
            tie_masks = config["featurizer_kwargs"]["tie_masks"]
            return spec.with_featurizer(
                Featurizer(
                    n_features=spec.width,
                    tie_masks=tie_masks,
                    id=f"mask_{spec.key}",
                )
            )
        else:  # "interchange" (DAS or subspace tracing)
            n_features = config["DAS"]["n_features"]
            return spec.with_featurizer(
                SubspaceFeaturizer(
                    shape=(spec.width, n_features),
                    trainable=True,
                    id=f"DAS_{spec.key}",
                )
            )

    return {
        key: [[initialize(spec) for spec in group] for group in groups]
        for key, groups in grid.items()
    }


def _collect_raw_sources(
    pipeline: LMPipeline,
    groups: Sequence[Sequence[SiteSpec]],
    dataset: Sequence[CounterfactualExample],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Pre-collect every spec's RAW source activations over the dataset.

    One early-stopped forward per (batch, counterfactual group) on
    ``pipeline``'s model (the *source* pipeline under cross-model patching),
    reading each spec's site through an identity :class:`FeaturizedSite` —
    the training shapes (:func:`das_edit` / :func:`dbm_edit`) featurize the
    raw source *live in the base trace*, so gradients reach the rotation
    through the source path too.

    Sources are constants of the optimization (the ED3
    ``source_representations`` pattern): collected once here, sliced per
    training batch. Returns ``{id(spec): (n_examples, k, d)}`` in dataset
    order, offloaded to CPU (the package's collect convention — a
    full-dataset pre-collect must not stay GPU-resident for the whole run);
    ``batch_edits`` moves each batch's slice back to its site's device.
    """
    model = pipeline.model
    chunks: dict[int, list[torch.Tensor]] = {
        id(spec): [] for group in groups for spec in group
    }
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        for g, group in enumerate(groups):
            if not group:
                continue
            cf_traces = [ex["counterfactual_inputs"][g] for ex in batch]
            cf_encoding = pipeline.load(cf_traces, return_offsets_mapping=True)
            taps = []
            for spec in group:
                rows = resolve_spec_positions(
                    spec, cf_traces, cf_encoding, is_original=False
                )
                raw_site = FeaturizedSite(spec.fsite.site)  # identity → raw
                taps.append(
                    (
                        forward_key(raw_site.site, model),
                        lambda m, s=raw_site, r=rows: s.read(m, r),
                    )
                )
            with torch.no_grad():
                values = collect_ordered(
                    model, forward_inputs(cf_encoding), taps, offload=True
                )
            for spec, value in zip(group, values):
                if value.dim() != 3:
                    raise ValueError(
                        f"unit {spec.key!r}: training requires every example to "
                        f"select the same number of positions (got a ragged "
                        f"read of shape {tuple(value.shape)})"
                    )
                chunks[id(spec)].append(value)
    return {uid: torch.cat(rows, dim=0) for uid, rows in chunks.items()}


def _run_training_loop(
    pipeline: LMPipeline,
    groups: Sequence[Sequence[SiteSpec]],
    counterfactual_dataset: list[LabeledCounterfactualExample],
    intervention_type: str,
    config: dict[str, Any],
    checker: Callable[[Dict[str, Any], Any], float | bool],
    source_pipeline: LMPipeline | None = None,
) -> tuple[list[list[SiteSpec]], dict[str, list[int] | None], str]:
    """
    Train intervention parameters on a labeled counterfactual dataset.

    The loop composes the ED3 toolkit (:mod:`causalab.neural.trainable`):
    :func:`das_edit` / :func:`dbm_edit` build each batch's edits around the
    specs' shared featurizer / :class:`MaskGate` modules, the label-concat
    forward is :func:`traced_label_loss` under the pinned saved-logits grad
    contract, and answer scoring is :func:`score_label_predictions` (MX1) —
    this function owns only the orchestration around them: the LR scheduler,
    early stopping, the DBM temperature/sparsity schedule, progress logging,
    memory hygiene, and the feature-indices readout.

    Args:
        pipeline: Target pipeline where interventions are applied
        groups: Nested :class:`SiteSpec` groups to intervene on, where groups
            share counterfactual inputs
        counterfactual_dataset: Labeled counterfactual examples (each carries
            the ``label`` the loss slice scores against)
        intervention_type: Type of intervention ("interchange" or "mask")
        config: Configuration parameters including:
            - train_batch_size: Number of examples per batch
            - training_epoch: Maximum number of training epochs
            - init_lr: Initial learning rate
            - masking.regularization_coefficient: Weight for sparsity regularization (mask only)
            - masking.temperature_schedule: Start and end temperature for mask annealing
            - masking.temperature_annealing_fraction: Fraction of training steps to anneal
            - patience: Epochs without improvement before early stopping
            - scheduler_type: Learning rate scheduler type
            - memory_cleanup_freq: Batch frequency for memory cleanup
            - shuffle: Whether to shuffle data
        checker: Answer-scoring authority for the in-loop accuracy metric,
            ``(neural_output, label) -> bool | float`` (a task checker via
            ``as_label_checker``)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline during training. Enables cross-model patching.

    Returns:
        ``(trained_groups, feature_indices, summary)``:

        - ``trained_groups`` — the trained specs, nested like ``groups``. For
          DAS these are the input specs (their shared featurizer modules now
          carry the trained rotation); for DBM each spec gets its
          hard-threshold selection via ``with_feature_ids``, and a mask that
          switched *every* feature off drops its spec entirely (the
          no-op-by-omission contract — an empty selection is not
          constructible as a spec).
        - ``feature_indices`` — ``{spec.key: indices}`` readout, recording
          the raw selection for every spec (including ``[]`` for all-off DBM
          masks and ``None`` for all-features).
        - ``summary`` — human-readable final-metrics string.
    """
    model = pipeline.model
    freeze_model_parameters(model)
    sites = [spec for group in groups for spec in group]

    # ----- Data Preparation ----- #
    train_batch_size = config["train_batch_size"]
    shuffle = config["shuffle"]
    num_batches = -(-len(counterfactual_dataset) // train_batch_size)

    # ----- Configuration ----- #
    num_epoch = config["training_epoch"]
    regularization_coefficient = config["masking"]["regularization_coefficient"]
    memory_cleanup_freq = config["memory_cleanup_freq"]
    patience = config["patience"]
    scheduler_type = config["scheduler_type"]

    # ----- Early Stopping Setup ----- #
    best_loss = float("inf")
    patience_counter = 0
    early_stopping_enabled = patience is not None

    # ----- Trainable state: gates (DBM) + the specs' featurizers (DAS) ----- #
    gates: dict[int, MaskGate] = {}
    if intervention_type == "mask":
        tie_masks = config["featurizer_kwargs"]["tie_masks"]
        for spec in sites:
            gates[id(spec)] = MaskGate(
                spec.fsite.featurizer.n_features, tie=tie_masks
            ).train()

    # ----- Pre-collected raw sources (constants of the optimization) ----- #
    raw_sources = _collect_raw_sources(
        source_pipeline if source_pipeline is not None else pipeline,
        groups,
        counterfactual_dataset,
        batch_size=train_batch_size,
    )

    def batch_edits(
        example_indices: list[int],
    ) -> tuple[tuple[Edit, ...], dict[str, Any]]:
        """One batch's edits: tokenize the base side, resolve each spec's
        positions in this batch's padded frame, slice its pre-collected raw
        source rows, and wrap them in the training shape. The state that
        trains lives in the shared featurizer/gate modules, not in the frozen
        Edit values."""
        base_traces = [counterfactual_dataset[i]["input"] for i in example_indices]
        base_encoding = pipeline.load(base_traces, return_offsets_mapping=True)
        edits = []
        for spec in sites:
            fsite = spec.fsite
            rows = resolve_spec_positions(
                spec, base_traces, base_encoding, is_original=True
            )
            raw = raw_sources[id(spec)][example_indices].to(site_device(model, fsite))
            if intervention_type == "mask":
                edits.append(dbm_edit(fsite, raw, gates[id(spec)], positions=rows))
            else:
                edits.append(das_edit(fsite, raw, positions=rows))
        return tuple(edits), base_encoding

    # ----- Optimizer Configuration ----- #
    params: dict[int, torch.nn.Parameter] = {}
    for spec in sites:
        for module in (
            spec.fsite.featurizer.featurizer,
            spec.fsite.featurizer.inverse_featurizer,
        ):
            for p in module.parameters():
                if p.requires_grad:
                    params[id(p)] = p
    for gate in gates.values():
        for p in gate.parameters():
            params[id(p)] = p
    optimizer = torch.optim.AdamW(
        list(params.values()), lr=config["init_lr"], weight_decay=0
    )

    scheduler = transformers.get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_epoch * num_batches,
    )

    # Track step count manually instead of accessing scheduler._step_count
    current_step = 0

    # ----- Temperature Scheduling for Mask Interventions ----- #
    temps = None
    mask_numel = 0
    if intervention_type == "mask":
        temperature_start, temperature_end = config["masking"]["temperature_schedule"]
        temps = temperature_schedule(
            temperature_start,
            temperature_end,
            num_epoch * num_batches,
            config["masking"]["temperature_annealing_fraction"],
        )
        mask_numel = sum(g.mask.numel() for g in gates.values())

    # ----- Training Loop ----- #
    placed = False
    postfix_dict: dict[str, str] = {}  # Initialize to avoid unbound error
    site_keys_desc = ", ".join(spec.key for spec in sites)
    train_iterator = tqdm(
        range(0, int(num_epoch)),
        desc=f"Training [{site_keys_desc}]"[:100] + "...",
        leave=False,
    )
    for epoch in train_iterator:
        # Shuffle indices for this epoch if requested
        indices = list(range(len(counterfactual_dataset)))
        if shuffle:
            random.shuffle(indices)

        aggregated_stats = collections.defaultdict(list)

        epoch_iterator = tqdm(
            range(0, len(indices), train_batch_size),
            desc=f"Epoch: {epoch}",
            position=1,
            leave=False,
        )
        for step, start in enumerate(epoch_iterator):
            example_indices = indices[start : start + train_batch_size]
            examples = [counterfactual_dataset[i] for i in example_indices]

            edits, base_encoding = batch_edits(example_indices)
            if not placed:
                place_edit_parameters(model, edits)
                placed = True

            # Anneal the gates BEFORE the forward (the train_edits step
            # order): the loss at step s uses temperature s.
            if temps is not None:
                for gate in gates.values():
                    gate.set_temperature(temps[current_step])

            labels = [ex["label"] for ex in examples]
            label_strs = [
                label["string"] if isinstance(label, dict) else label
                for label in labels
            ]
            joint, label_ids = concat_label_inputs(pipeline, base_encoding, label_strs)
            loss, pred_ids = traced_label_loss(
                model, joint, label_ids, edits, pipeline.tokenizer.pad_token_id
            )

            # Add sparsity loss for mask interventions. Normalize by total
            # mask elements so regularization_coefficient has consistent
            # meaning regardless of number of features/units.
            if gates and regularization_coefficient and mask_numel:
                total_sparsity = sum(
                    g.sparsity_loss().to(loss.device) for g in gates.values()
                )
                loss = loss + regularization_coefficient * (total_sparsity / mask_numel)

            # In-loop answer scoring — the task checker stays the single
            # match authority (score_label_predictions, MX1 #408).
            label_scores = score_label_predictions(pipeline, pred_ids, labels, checker)
            eval_metrics = {
                "accuracy": label_scores["accuracy"],
                "token_accuracy": label_scores["accuracy"],
            }

            # Update statistics
            aggregated_stats["loss"].append(loss.item())
            aggregated_stats["metrics"].append(eval_metrics)

            # Update progress bar
            postfix = {"loss": round(np.mean(aggregated_stats["loss"]), 2)}
            for k, v in eval_metrics.items():
                postfix[k] = round(np.mean(v), 2)
            epoch_iterator.set_postfix(postfix)

            # Optimization step
            loss.backward()
            optimizer.step()
            # get_scheduler's return type includes ReduceLROnPlateau which needs metrics,
            # but we only use schedulers that don't require it
            scheduler.step()  # pyright: ignore[reportCallIssue]
            current_step += 1
            optimizer.zero_grad()

            # Periodic memory cleanup
            if step % memory_cleanup_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update progress bar with epoch summary
        epoch_avg_loss = np.mean(aggregated_stats["loss"])
        postfix_dict = {"loss": f"{epoch_avg_loss:.4f}"}

        if aggregated_stats["metrics"]:
            # Aggregate metrics across all batches in the epoch
            all_metrics = {}
            for batch_metrics in aggregated_stats["metrics"]:
                for k, v in batch_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
            # Add metrics to postfix
            for k, v in all_metrics.items():
                postfix_dict[k] = f"{np.mean(v):.4f}"

        train_iterator.set_postfix(postfix_dict)

        # Early stopping check at end of epoch
        if early_stopping_enabled:
            epoch_avg_loss = np.mean(aggregated_stats["loss"])
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # ----- Feature Selection for Mask Interventions (functional readout) ----- #
    feature_indices: dict[str, list[int] | None] = {}
    if intervention_type == "mask":
        # Hard-threshold readout: tied gates keep the legacy convention
        # (None = all features when on, [] when off).
        selected: dict[int, list[int] | None] = {
            id(spec): selected_feature_ids(gates[id(spec)]) for spec in sites
        }
        feature_indices = {spec.key: selected[id(spec)] for spec in sites}
        trained_groups: list[list[SiteSpec]] = []
        for group in groups:
            trained_group: list[SiteSpec] = []
            for spec in group:
                ids = selected[id(spec)]
                if ids is not None and len(ids) == 0:
                    # A mask that switched EVERY feature off is a no-op edit.
                    # An empty selection is not constructible as a spec
                    # (FeaturizedSite refuses empty feature_ids), so the
                    # trained flow drops the spec — the dataset layer's
                    # no-op-by-omission contract. The empty selection stays
                    # recorded in feature_indices above.
                    continue
                trained_group.append(spec.with_feature_ids(ids))
            trained_groups.append(trained_group)
    else:
        feature_indices = {
            spec.key: (
                list(spec.fsite.feature_ids)
                if spec.fsite.feature_ids is not None
                else None
            )
            for spec in sites
        }
        trained_groups = [list(group) for group in groups]

    summary = f"Trained intervention for [{site_keys_desc}]"[:200]
    summary += "\nFinal metrics: " + " ".join(
        [f"{k}: {v}" for k, v in postfix_dict.items()]
    )
    return trained_groups, feature_indices, summary


def save_train_results(result: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Save train_interventions results to disk.

    ``results_by_key[key]["trained_specs"]`` holds nested :class:`SiteSpec`
    groups; ``causalab.io.artifacts.save_training_artifacts`` writes each
    cell's ``models/`` bundle from them via
    ``causalab.neural.specs.save_site_specs`` (WU5, #507). Everything else
    (feature_indices, evals, metadata) saves as before.
    """
    results_by_key = result["results_by_key"]
    metadata = result["metadata"]
    output_paths: Dict[str, str] = {}

    train_eval_paths = save_intervention_results(
        {k: v["train_eval"] for k, v in results_by_key.items()},
        output_dir=output_dir,
        prefix="train_eval",
    )
    output_paths.update({f"train_{k}": v for k, v in train_eval_paths.items()})

    test_eval_paths = save_intervention_results(
        {k: v["test_eval"] for k, v in results_by_key.items()},
        output_dir=output_dir,
        prefix="test_eval",
    )
    output_paths.update({f"test_{k}": v for k, v in test_eval_paths.items()})

    training_paths = save_training_artifacts(results_by_key, output_dir=output_dir)
    output_paths.update(training_paths)

    metadata_path = save_aggregate_metadata(metadata, output_dir)
    output_paths["metadata_path"] = metadata_path

    return output_paths
