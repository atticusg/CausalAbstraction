"""Loaders for pipelines and prior analysis outputs.

These functions read from disk (or hydrate objects that then do). They live
in ``io/`` so that ``runner/`` and ``analyses/`` can consume them without
pulling in orchestration.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from causalab.tasks.loader import Task

logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_pipeline(
    model_name: str,
    task: "Task",
    max_new_tokens: int,
    device: str | None = None,
    dtype: str | None = None,
    eager_attn: bool | None = None,
    enable_attention_probs: bool | None = None,
    use_chat_template: bool = False,
    chat_answer_directive: str | None = None,
):
    """Load an ``LMPipeline`` from explicit parameters.

    ``use_chat_template`` wraps every prompt in the model's chat template
    (required for instruct models, which are post-trained on the chat format and
    otherwise restate the question instead of answering). ``chat_answer_directive``,
    when set, is emitted as a system message nudging the model to reply with only
    the bare answer so a 1-token completion contract still scores the answer.

    ``eager_attn=True`` opts into eager attention (needed for
    attention-probability extraction); ``None`` defers to ``LMPipeline``'s
    default — HF's own resolution, sdpa/flash where available (SH3, #424).

    ``enable_attention_probs=True`` additionally exposes nnterp's *editable*
    ``attention_probabilities[i]`` trace target (attention knockout /
    renormalize — :mod:`causalab.neural.attention_probs`, CAP4 #457): it
    forces eager attention on its own and runs nnterp's ``check_source()``
    validation gate at load. Thread it from an analysis config as
    ``enable_attention_probs=cfg.model.get("enable_attention_probs")``.
    """
    from causalab.neural.pipeline import LMPipeline, resolve_device

    # Use device_map="auto" for multi-GPU sharding when device is unspecified or "auto"
    use_device_map = (
        device is None or device == "auto"
    ) and torch.cuda.device_count() > 1

    model_kwargs: dict[str, Any]
    if use_device_map:
        model_kwargs = {"device_map": "auto"}
        logger.info(
            "Loading model: %s (device_map=auto, %d GPUs)",
            model_name,
            torch.cuda.device_count(),
        )
    else:
        resolved_device = resolve_device(device)
        model_kwargs = {"device": resolved_device}
        logger.info("Loading model: %s (device=%s)", model_name, resolved_device)

    if dtype:
        model_kwargs["dtype"] = DTYPE_MAP.get(dtype, torch.bfloat16)
    if eager_attn is not None:
        model_kwargs["eager_attn"] = eager_attn
    if enable_attention_probs is not None:
        model_kwargs["enable_attention_probs"] = enable_attention_probs

    pipeline = LMPipeline(
        model_name,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        chat_answer_directive=chat_answer_directive,
        **model_kwargs,
    )

    if task.validate is not None:
        task.validate(pipeline)

    return pipeline


def load_lite_pipeline(
    model_name: str,
    max_new_tokens: int = 3,
    use_chat_template: bool = False,
):
    """Load an ``LMPipeline`` with tokenizer + config only (no model weights).

    For analyses that need ``pipeline.tokenizer`` and ``pipeline.model.config``
    (e.g. to build :class:`~causalab.neural.specs.SiteSpec` grids from cached
    features) but never run forward passes. Returns immediately for any model
    size; calling ``pipeline.generate`` or running the model will fail.

    ``use_chat_template`` is threaded through even though no forward pass runs:
    the token-position code paths these lite pipelines feed (site-spec grid
    construction) tokenize prompts, so they must wrap consistently with the full
    pipeline or intervention indices would be computed on the bare text.
    """
    from causalab.neural.pipeline import LMPipeline

    logger.info("Loading lite pipeline (tokenizer+config only): %s", model_name)
    return LMPipeline(
        model_name,
        max_new_tokens=max_new_tokens,
        load_weights=False,
        use_chat_template=use_chat_template,
    )


# --------------------------------------------------------------------------- #
# Discovery helpers for prior analysis runs                                   #
# --------------------------------------------------------------------------- #


def find_subspace_dirs(root: str) -> list[str]:
    """Scan ``root/subspace/`` for completed subspace runs."""
    subspace_root = os.path.join(root, "subspace")
    if not os.path.isdir(subspace_root):
        return []
    return sorted(
        d
        for d in os.listdir(subspace_root)
        if os.path.isdir(os.path.join(subspace_root, d))
    )


def find_activation_manifold_dirs(root: str, subspace_sub: str) -> list[str]:
    """Scan ``root/activation_manifold/{subspace_sub}/`` for completed manifold runs.

    Returns relative paths from ``activation_manifold/{subspace_sub}/``.
    Handles both single-cell layout (``spline_s0.0/``) and grid-cell layout
    (``L{layer}_{pos}/spline_s0.0/``).
    """
    manifold_root = os.path.join(root, "activation_manifold", subspace_sub)
    if not os.path.isdir(manifold_root):
        return []

    found: list[str] = []
    for d in sorted(os.listdir(manifold_root)):
        d_path = os.path.join(manifold_root, d)
        if not os.path.isdir(d_path):
            continue
        # Direct method dir (single-cell layout)
        if os.path.isfile(os.path.join(d_path, "metadata.json")):
            found.append(d)
            continue
        # Grid-cell layout: L{layer}_{pos}/method_dir/
        # metadata.json may be directly in method_dir/ or one level deeper
        # (e.g. method_dir/target_variable/) when target_variable nesting is used.
        for sub in sorted(os.listdir(d_path)):
            sub_path = os.path.join(d_path, sub)
            if not os.path.isdir(sub_path):
                continue
            if os.path.isfile(os.path.join(sub_path, "metadata.json")):
                found.append(os.path.join(d, sub))
            else:
                # Check one level deeper (target_variable subdirectory)
                for tv_sub in sorted(os.listdir(sub_path)):
                    tv_path = os.path.join(sub_path, tv_sub)
                    if os.path.isdir(tv_path) and os.path.isfile(
                        os.path.join(tv_path, "metadata.json")
                    ):
                        found.append(os.path.join(d, sub))
                        break
    return found


def load_locate_result(root: str) -> dict:
    """Read best layer/position from ``locate/`` analysis output.

    Prefers ``results.json``; falls back to ``metadata.json`` for
    backwards compatibility with older runs.
    """
    locate_root = os.path.join(root, "locate")
    if not os.path.isdir(locate_root):
        return {}

    for method_dir in sorted(os.listdir(locate_root)):
        dir_path = os.path.join(locate_root, method_dir)
        results_path = os.path.join(dir_path, "results.json")
        if os.path.isfile(results_path):
            with open(results_path) as f:
                return json.load(f)
        meta_path = os.path.join(dir_path, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                return json.load(f)
    return {}


def load_subspace_metadata(
    root: str,
    subspace_sub: str,
    target_variable: str | None = None,
) -> dict:
    """Read metadata from a subspace run directory."""
    parts = [root, "subspace", subspace_sub]
    if target_variable:
        parts.append(target_variable)
    meta_path = os.path.join(*parts, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def load_activation_manifold_metadata(
    root: str,
    subspace_sub: str,
    manifold_sub: str,
    target_variable: str | None = None,
) -> dict:
    """Read metadata from an activation_manifold run directory."""
    parts = [root, "activation_manifold", subspace_sub, manifold_sub]
    if target_variable:
        parts.append(target_variable)
    meta_path = os.path.join(*parts, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)
