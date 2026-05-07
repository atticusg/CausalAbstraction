"""
Train Manifold Flow - Learn a d-dimensional manifold in k-dimensional DAS feature space.

This module trains a ManifoldFlow on collected DAS features to learn a lower-dimensional
manifold structure. The trained flow partitions the latent space as [intrinsic, residual],
where intrinsic coordinates capture the on-manifold variation and residual coordinates
are regularized toward zero.

The output is designed for use with steer_manifold.py:
- Returns the inner flow for composition: SubspaceFeaturizer >> StandardizeFeaturizer >> ManifoldFeaturizer
- Returns preprocessing params (mean, std) for standardization

Output Structure:
================
output_dir/
├── metadata.json          # Config + final metrics
├── ckpt_final.pt          # Final checkpoint
├── ckpt_step_*.pt         # Intermediate checkpoints
└── metrics.jsonl          # Training metrics log
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.flow import build_manifold_flow, ManifoldFlow
from causalab.neural.units import InterchangeTarget, AtomicModelUnit
from causalab.neural.pipeline import Pipeline
from causalab.neural.activations.collect import collect_features

logger = logging.getLogger(__name__)


@dataclass
class ManifoldTrainConfig:
    """Configuration for manifold flow training."""

    # Architecture
    num_layers: int = 8
    hidden: int = 256
    depth: int = 2
    s_scale: float = 2.0

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    steps: int = 10000
    grad_clip: float | None = 5.0

    # Loss weights
    residual_weight: float = 1.0
    reconstruction_weight: float = 1.0

    # Misc
    seed: int = 0
    device: str = "auto"
    eval_every: int = 500
    save_every: int = 2000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_target(target: InterchangeTarget) -> AtomicModelUnit:
    """Validate target has single unit with SubspaceFeaturizer, return it."""
    units = target.flatten()
    if len(units) != 1:
        raise ValueError(f"Expected single unit, got {len(units)}")

    unit = units[0]
    featurizer = unit.featurizer

    # Check it's a SubspaceFeaturizer (or compatible with DAS rotation)
    if not hasattr(featurizer, "featurizer") or not hasattr(
        featurizer.featurizer, "rotate"
    ):
        raise ValueError(
            f"Expected SubspaceFeaturizer with rotation, got {type(featurizer)}"
        )

    return unit


def _build_manifold_flow(
    ambient_dim: int,
    intrinsic_dim: int,
    config: ManifoldTrainConfig,
) -> ManifoldFlow:
    """Build ManifoldFlow from config using the builder function."""
    return build_manifold_flow(
        dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        num_layers=config.num_layers,
        hidden=config.hidden,
        depth=config.depth,
        s_scale=config.s_scale,
        seed=config.seed,
    )


def _save_checkpoint(
    path: str,
    mf: ManifoldFlow,
    opt: Optimizer,
    step: int,
    config: ManifoldTrainConfig,
    mean: Tensor,
    std: Tensor,
) -> None:
    """Save checkpoint compatible with ManifoldFeaturizer loading."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save inner flow state, not ManifoldFlow wrapper
    payload = {
        "model": mf.flow.state_dict(),
        "optim": opt.state_dict(),
        "step": step,
        "config": {
            "num_layers": config.num_layers,
            "hidden": config.hidden,
            "depth": config.depth,
            "s_scale": config.s_scale,
            "intrinsic_dim": mf.k,
            "ambient_dim": mf.n,
        },
        "preprocess": {
            "mean": mean.cpu(),
            "std": std.cpu(),
            "eps": 1e-6,
        },
    }

    # Include FlowConfig if available (for robust deserialization)
    if mf.flow.config is not None:
        payload["flow_config"] = mf.flow.config.to_dict()

    torch.save(payload, path)


@torch.no_grad()  # type: ignore[reportUntypedFunctionDecorator]
def _eval_manifold(
    mf: ManifoldFlow,
    features: Tensor,
    mean: Tensor,
    std: Tensor,
    device: torch.device,
    config: ManifoldTrainConfig,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate manifold flow on a dataset."""
    mf.eval()
    total_recon = 0.0
    total_resid = 0.0
    count = 0

    loader = DataLoader(
        TensorDataset(features),
        batch_size=batch_size,
        shuffle=False,
    )

    for (x,) in loader:
        x = x.to(device)
        x_std = (x - mean) / std
        _, metrics = mf.loss(
            x_std,
            residual_weight=config.residual_weight,
            reconstruction_weight=config.reconstruction_weight,
        )
        total_recon += metrics["recon"] * x.shape[0]
        total_resid += metrics["residual"] * x.shape[0]
        count += x.shape[0]

    mf.train()
    return {
        "recon": total_recon / max(count, 1),
        "residual": total_resid / max(count, 1),
    }


def _train_loop(
    mf: ManifoldFlow,
    features: Tensor,
    mean: Tensor,
    std: Tensor,
    config: ManifoldTrainConfig,
    output_dir: str,
    verbose: bool,
) -> dict[str, float]:
    """Run training loop, return final validation metrics."""
    from causalab.neural.pipeline import resolve_device

    device = torch.device(resolve_device(config.device))
    mf = mf.to(device)
    mean = mean.to(device)
    std = std.to(device)

    # Train/val split (90/10)
    n = features.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.seed))
    n_train = int(0.9 * n)
    train_features = features[perm[:n_train]]
    val_features = features[perm[n_train:]]

    # Ensure batch size doesn't exceed training set size
    effective_batch_size = min(config.batch_size, n_train)

    # Create data loader
    train_loader = DataLoader(
        TensorDataset(train_features),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
    )

    opt = torch.optim.AdamW(mf.parameters(), lr=config.lr)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    mf.train()
    it = iter(train_loader)
    final_val_metrics: dict[str, float] = {}

    iterator = tqdm(range(config.steps), desc="Training", disable=not verbose)
    for step in iterator:
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(train_loader)
            (x,) = next(it)

        # Standardize
        x = x.to(device)
        x_std = (x - mean) / std

        loss, train_metrics = mf.loss(
            x_std,
            residual_weight=config.residual_weight,
            reconstruction_weight=config.reconstruction_weight,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(mf.parameters(), config.grad_clip)

        opt.step()

        # Update progress bar
        iterator.set_postfix(
            recon=f"{train_metrics['recon']:.4f}",
            resid=f"{train_metrics['residual']:.4f}",
        )

        # Eval
        if (step + 1) % config.eval_every == 0:
            val_metrics = _eval_manifold(
                mf, val_features, mean, std, device, config, config.batch_size
            )
            record = {
                "step": step + 1,
                "train_recon": train_metrics["recon"],
                "train_residual": train_metrics["residual"],
                "val_recon": val_metrics["recon"],
                "val_residual": val_metrics["residual"],
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if verbose:
                logger.info(
                    f"Step {step + 1}: val_recon={val_metrics['recon']:.4f}, "
                    f"val_residual={val_metrics['residual']:.4f}"
                )

            final_val_metrics = val_metrics

        # Checkpoint
        if (step + 1) % config.save_every == 0:
            _save_checkpoint(
                os.path.join(output_dir, f"ckpt_step_{step + 1}.pt"),
                mf,
                opt,
                step + 1,
                config,
                mean.cpu(),
                std.cpu(),
            )

    # Final checkpoint
    _save_checkpoint(
        os.path.join(output_dir, "ckpt_final.pt"),
        mf,
        opt,
        config.steps,
        config,
        mean.cpu(),
        std.cpu(),
    )

    # Final eval if not already done
    if config.steps % config.eval_every != 0:
        final_val_metrics = _eval_manifold(
            mf, val_features, mean, std, device, config, config.batch_size
        )

    return final_val_metrics


def train_manifold(
    interchange_target: InterchangeTarget,
    dataset_path: str,
    pipeline: Pipeline,
    intrinsic_dim: int,
    output_dir: str,
    config: ManifoldTrainConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train a ManifoldFlow on k-dimensional DAS features to learn a d-dimensional manifold.

    The target must have a single unit with a trained SubspaceFeaturizer (DAS rotation).
    Features are collected from the dataset using this featurizer, then a ManifoldFlow
    is trained to learn a lower-dimensional manifold structure.

    Args:
        interchange_target: Single target with trained DAS featurizer.
            Must have exactly one group with one unit.
        dataset_path: Path to HuggingFace dataset directory with "input" column,
            or list of CounterfactualExample dicts.
        pipeline: Model pipeline for processing inputs.
        intrinsic_dim: Dimensionality d of manifold (d < k where k is DAS dimension).
        output_dir: Directory for outputs.
        config: Training configuration. Uses defaults if None.
        verbose: Whether to show progress.

    Returns:
        {
            "manifold_flow": ManifoldFlow,  # Full ManifoldFlow wrapper
            "flow": Flow,                   # Inner flow for ManifoldFeaturizer composition
            "mean": Tensor,                 # (k,) standardization mean
            "std": Tensor,                  # (k,) standardization std
            "checkpoint_path": str,         # Path to ckpt_final.pt
            "features": Tensor,             # (n, k) collected features
            "metadata": {
                "ambient_dim": int,         # k (DAS features)
                "intrinsic_dim": int,       # d (manifold dim)
                "n_samples": int,
                "final_val_recon": float,
                "final_val_residual": float,
            },
        }

    Example:
        >>> # 1. Train manifold on DAS features
        >>> result = train_manifold(
        ...     target, dataset_path, pipeline, intrinsic_dim=2, output_dir
        ... )
        >>>
        >>> # 2. Get DAS rotation from original featurizer
        >>> das_unit = target.flatten()[0]
        >>> rotation = das_unit.featurizer.featurizer.rotate
        >>>
        >>> # 3. Compose SubspaceFeaturizer >> StandardizeFeaturizer >> ManifoldFeaturizer
        >>> from causalab.methods.spline.featurizer import ManifoldFeaturizer
        >>> from causalab.methods.standardize import StandardizeFeaturizer
        >>> standardize = StandardizeFeaturizer(result["mean"], result["std"])
        >>> manifold_feat = ManifoldFeaturizer(result["flow"], n_features=k)
        >>> composed = das_unit.featurizer >> standardize >> manifold_feat
        >>>
        >>> # 4. Update target featurizer
        >>> das_unit.set_featurizer(composed)
        >>>
        >>> # 5. Now steer_manifold.py will work
        >>> steer_result = steer_manifold(target, test_path, pipeline, intrinsic_dim, grid, out)
    """
    if config is None:
        config = ManifoldTrainConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Validate target
    unit = _validate_target(interchange_target)
    k = unit.featurizer.n_features
    if k is None:
        raise ValueError(
            "Featurizer n_features is None; cannot determine DAS dimension"
        )

    if intrinsic_dim >= k:
        raise ValueError(f"intrinsic_dim ({intrinsic_dim}) must be < n_features ({k})")

    logger.info(f"Training manifold: ambient_dim={k}, intrinsic_dim={intrinsic_dim}")

    # Load dataset
    if isinstance(dataset_path, list):
        # Already a list of examples
        dataset: list[CounterfactualExample] = dataset_path
    else:
        # Load from JSON file
        with open(dataset_path) as f:
            data = json.load(f)
        # Convert to list format if needed
        if isinstance(data, dict) and "input" in data:
            # Dict format with "input" and "counterfactual_inputs" keys
            dataset = [
                {"input": inp, "counterfactual_inputs": cf}
                for inp, cf in zip(data["input"], data["counterfactual_inputs"])
            ]
        else:
            # Already list format - cast is safe: validated by downstream usage
            dataset = cast(list[CounterfactualExample], data)

    logger.info(f"Loaded {len(dataset)} examples from dataset")

    # Collect k-dim features
    if verbose:
        logger.info("Collecting features...")

    features_dict = collect_features(
        dataset=dataset,
        pipeline=pipeline,
        model_units=[unit],
        batch_size=config.batch_size,
    )
    # Detach from computation graph - we only need the values for manifold training
    features = features_dict[unit.id].detach()  # Shape: (n_samples, k)
    logger.info(f"Collected features: {features.shape}")

    # Compute standardization on full dataset
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1e-6)

    # Build ManifoldFlow
    mf = _build_manifold_flow(k, intrinsic_dim, config)

    # Train
    if verbose:
        logger.info(f"Training for {config.steps} steps...")

    final_metrics = _train_loop(mf, features, mean, std, config, output_dir, verbose)

    # Save metadata
    checkpoint_path = os.path.join(output_dir, "ckpt_final.pt")
    metadata = {
        "ambient_dim": k,
        "intrinsic_dim": intrinsic_dim,
        "n_samples": len(dataset),
        "final_val_recon": final_metrics.get("recon", 0.0),
        "final_val_residual": final_metrics.get("residual", 0.0),
        "config": config.to_dict(),
        "unit_id": unit.id,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Checkpoint saved to {checkpoint_path}")
    logger.info(
        f"Final metrics: recon={final_metrics.get('recon', 0):.4f}, "
        f"residual={final_metrics.get('residual', 0):.4f}"
    )

    return {
        "manifold_flow": mf,
        "flow": mf.flow,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "checkpoint_path": checkpoint_path,
        "features": features,
        "metadata": {
            "ambient_dim": k,
            "intrinsic_dim": intrinsic_dim,
            "n_samples": len(dataset),
            "final_val_recon": final_metrics.get("recon", 0.0),
            "final_val_residual": final_metrics.get("residual", 0.0),
        },
    }
