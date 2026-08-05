"""Path-patching analysis: sweep attention-head senders and render the
(layer × head) direct-effect-on-logits heatmap.

For each ``(layer, head)`` the head's output is patched from the counterfactual
source while every other path selected by ``restore`` is frozen to the clean base
(path patching), and the edge's direct effect on the logit difference between a
correct and a distractor token is measured. This is the IOI Fig. 3 head
sweep — one intervened forward per cell. The reported grid
value is the direct effect (base minus patched logit difference); name-mover
heads score large-positive.

``restore`` selects the estimand: ``[attention, mlp]`` is a strict
residual-to-output direct effect; ``[attention]`` lets MLPs/LayerNorm recompute,
matching Wang et al. (2022) §3.1. ``corruption`` selects the source: ``abc`` (all
three names → distinct randoms, the paper's §3.1 corruption) or ``answer_flip``
(resample ``resample_variable`` only, e.g. ``name_C`` → flips the answer).

Mirrors ``causalab/analyses/ablation/main.py`` for task/pipeline/data loading,
figure-format handling, and the results/metadata disk layout.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.io.pipelines import load_pipeline
from causalab.io.plots.figure_format import (
    path_with_figure_format,
    resolve_figure_format_from_analysis,
)
from causalab.io.plots.score_heatmap import plot_attention_head_heatmap
from causalab.methods.metric import compute_base_outputs, make_logit_diff_metric
from causalab.methods.path_patching.run import (
    run_path_patching_scan,
    run_path_patching_set_scan,
)
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    deepest_receiver,
    sender_reaches_receiver,
)
from causalab.neural.activations.targets import build_attention_head_targets
from causalab.neural.token_positions import TokenPosition, get_last_token_index
from causalab.runner.helpers import (
    generate_datasets,
    resolve_task,
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "path_patching"
# The sweep derives correct/distractor entirely from this analysis's own
# ``correct_variable`` / ``distractor_variable`` config and ignores the injected
# ``task.target_variable``, so it manages its own variable selection — run once,
# not once per task target variable (which would rerun the identical sweep and
# overwrite ``results.json`` since ``_output_dir`` is not keyed by variable).
# Mirrors ``analyses/locate/main.py``; cf. the opposite note in ``ablation/main.py``.
HANDLES_MULTI_VARIABLE = True
VALID_CORRUPTIONS = ("abc", "answer_flip")


def _last_token(pipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _resolve_token_position(analysis: DictConfig, task, pipeline) -> TokenPosition:
    """Resolve the task position the sender and internal receiver read at.

    Defaults to the last token (the IOI Fig. 3/4 readout position). Set
    ``path_patching.token_position`` to a task position name to move *both* the
    sender and the receiver there — e.g. ``name_C`` (the repeated subject, "S2")
    for the Fig. 5 Duplicate-Token / Induction → S-Inhibition-value edge, whose
    "this name is duplicated" signal is computed at S2, not at the final token.

    The name is resolved against ``task.create_token_positions(pipeline)`` (the
    same lookup the attention-pattern analysis uses), raising with the available
    names if it is unknown.
    """
    name = analysis.get("token_position", None)
    if name is None or name == "last_token":
        return _last_token(pipeline)
    lookup = task.create_token_positions(pipeline)
    if name not in lookup:
        available = sorted(lookup.keys())
        raise ValueError(
            f"path_patching.token_position={name!r} is not a position of task "
            f"{task.name!r}; available: {available}"
        )
    return lookup[name]


def _build_receiver_spec(analysis: DictConfig, pos: TokenPosition) -> ReceiverSpec:
    """Resolve the configured receiver (defaults to the output logits).

    Internal receivers read at ``pos`` — the shared sender/receiver position from
    :func:`_resolve_token_position` (last token by default). ``head_value_input``
    and ``head_query_input`` need ``layer`` + ``head``; ``mlp_input`` / ``residual``
    need ``layer`` (``residual`` also honours ``residual_point``: ``block_output``
    (default) or ``block_input``).
    """
    rcfg = analysis.get("receiver", None)
    kind = "output" if rcfg is None else rcfg.get("kind", "output")
    if kind == "output":
        return OUTPUT
    layer = rcfg.get("layer")
    if kind in ("head_value_input", "head_query_input"):
        return ReceiverSpec(
            kind=kind, layer=layer, head=rcfg.get("head"), token_position=pos
        )
    if kind == "mlp_input":
        return ReceiverSpec(kind=kind, layer=layer, token_position=pos)
    if kind == "residual":
        return ReceiverSpec(
            kind=kind,
            layer=layer,
            token_position=pos,
            residual_point=rcfg.get("residual_point", "block_output"),
        )
    raise ValueError(
        f"path_patching.receiver.kind must be one of output | head_value_input | "
        f"head_query_input | mlp_input | residual, got {kind!r}"
    )


def _build_receiver_specs(
    analysis: DictConfig, pos: TokenPosition
) -> list[ReceiverSpec] | None:
    """Resolve a receiver *set* from ``receiver.heads``, or ``None`` for a single receiver.

    A set is configured by listing head coordinates, e.g.::

        receiver: {kind: head_query_input, heads: [[9, 6], [9, 9], [10, 0]]}

    and is patched simultaneously (the IOI Fig. 4 / Fig. 5 case) via
    :func:`run_path_patching_set_scan`. Only head receivers
    (``head_query_input`` / ``head_value_input``) support sets, and every member
    shares the read position ``pos``. Returns ``None`` when ``receiver.heads`` is
    absent, so the caller falls back to the single-receiver path.
    """
    rcfg = analysis.get("receiver", None)
    if rcfg is None or rcfg.get("heads", None) is None:
        return None
    kind = rcfg.get("kind", "output")
    if kind not in ("head_value_input", "head_query_input"):
        raise ValueError(
            f"path_patching.receiver.heads (a receiver set) is only valid for kind "
            f"head_value_input | head_query_input, got {kind!r}. A single head uses "
            f"receiver.head; the output sweep uses kind: output."
        )
    specs = [
        ReceiverSpec(kind=kind, layer=int(layer), head=int(head), token_position=pos)
        for layer, head in rcfg.heads
    ]
    if not specs:
        raise ValueError("path_patching.receiver.heads is empty.")
    return specs


def _pos_suffix(receiver: ReceiverSpec) -> str:
    """Output-dir/title suffix for the read position, empty for the default last token.

    Keeps sweeps at distinct positions (e.g. S2 vs last_token) in separate subdirs.
    """
    pos_id = getattr(receiver.token_position, "id", None)
    return f"_{pos_id}" if pos_id and pos_id != "last_token" else ""


def _receiver_tag(receiver: ReceiverSpec) -> str:
    """Short, path-safe label for the receiver (output dir suffix + figure title)."""
    if receiver.kind == "output":
        return "output"
    pos_suffix = _pos_suffix(receiver)
    if receiver.kind in ("head_value_input", "head_query_input"):
        return f"{receiver.kind}_L{receiver.layer}H{receiver.head}{pos_suffix}"
    if receiver.kind == "residual":
        return f"residual_{receiver.residual_point}_L{receiver.layer}{pos_suffix}"
    return f"{receiver.kind}_L{receiver.layer}{pos_suffix}"


def _receiver_set_tag(specs: list[ReceiverSpec]) -> str:
    """Path-safe label for a receiver set (output subdir + figure title).

    Lists the member heads, e.g. ``head_query_input_set_L9H6_L9H9_L10H0`` (plus a
    position suffix when not the default last token), so sets at distinct positions
    or compositions never clobber each other's ``results.json``.
    """
    heads = "_".join(f"L{s.layer}H{s.head}" for s in specs)
    return f"{specs[0].kind}_set_{heads}{_pos_suffix(specs[0])}"


def _reachable_senders(
    pipeline,
    senders: dict[tuple[Any, ...], Any],
    receiver: ReceiverSpec,
    receiver_tag: str,
) -> dict[tuple[Any, ...], Any]:
    """Drop senders downstream of an internal receiver (no forward path → wasted run).

    ``output`` receivers are always reachable, so the grid is returned unchanged.
    Otherwise keep only senders upstream of the receiver's read point, log how many
    layers were dropped, and raise if nothing can reach it (rather than emit an empty
    heatmap). Reachability is layer-based, so whole layers drop and the grid stays
    rectangular.
    """
    if receiver.kind == "output":
        return senders
    reachable = {
        key: s
        for key, s in senders.items()
        if sender_reaches_receiver(pipeline, s, receiver)
    }
    dropped = sorted({key[0] for key in senders} - {key[0] for key in reachable})
    if dropped:
        logger.info(
            "Path patching: dropped %d sender layer(s) %s downstream of receiver %s "
            "(no forward path; direct effect is structurally zero).",
            len(dropped),
            dropped,
            receiver_tag,
        )
    if not reachable:
        raise ValueError(
            f"No sender in the grid can reach receiver {receiver_tag}: every swept "
            f"layer is at or downstream of its read point. Choose a higher-layer "
            f"receiver or widen path_patching.layers to upstream layers."
        )
    return reachable


def _build_dataset(cfg: DictConfig, analysis: DictConfig, task) -> list[dict[str, Any]]:
    """Build the counterfactual dataset for the configured corruption.

    ``abc`` dynamically loads the task's ``generate_abc_dataset`` (the IOI §3.1
    all-distinct-names source); ``answer_flip`` uses the generic
    ``generate_datasets`` resampling only ``resample_variable``.
    """
    corruption = analysis.get("corruption", "abc")
    if corruption not in VALID_CORRUPTIONS:
        raise ValueError(
            f"path_patching.corruption must be one of {VALID_CORRUPTIONS}, "
            f"got {corruption!r}"
        )
    if corruption == "abc":
        mod = importlib.import_module(f"causalab.tasks.{cfg.task.name}.counterfactuals")
        if not hasattr(mod, "generate_abc_dataset"):
            raise ValueError(
                f"corruption='abc' requires task {cfg.task.name!r} to define "
                f"generate_abc_dataset in its counterfactuals module."
            )
        return mod.generate_abc_dataset(
            task.causal_model, n=cfg.task.n_test, seed=cfg.seed
        )
    _train, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=analysis.get("resample_variable", "name_C"),
    )
    return test_dataset


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the path-patching head sweep and write a direct-effect heatmap."""
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    restore = tuple(analysis.get("restore", ["attention", "mlp"]))
    correct_variable = analysis.correct_variable
    distractor_variable = analysis.distractor_variable

    # --- task, pipeline, data -------------------------------------------------
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=correct_variable,
        seed=cfg.seed,
    )
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
    test_dataset = _build_dataset(cfg, analysis, task)

    # --- receiver -------------------------------------------------------------
    # Default is the output logits (single-pass). Internal receivers use the
    # two-pass collect/inject runner and get their own output subdir so distinct
    # receivers (and the output sweep) never clobber each other's results.json.
    # Sender and internal receiver share a read position (last token by default;
    # e.g. name_C / "S2" for the Fig. 5 duplicate→S-inhibition-value edge).
    token_pos = _resolve_token_position(analysis, task, pipeline)
    # A receiver *set* (``receiver.heads``) is patched simultaneously; otherwise a
    # single receiver (``output`` by default). For a set the reachability gate and tag
    # use the deepest member (reaching it ⇔ reaching at least one), so the rest of the
    # flow is shared.
    receiver_specs = _build_receiver_specs(analysis, token_pos)
    receiver = (
        None
        if receiver_specs is not None
        else _build_receiver_spec(analysis, token_pos)
    )
    reach_receiver = (
        deepest_receiver(pipeline, receiver_specs)
        if receiver_specs is not None
        else receiver
    )
    receiver_tag = (
        _receiver_set_tag(receiver_specs)
        if receiver_specs is not None
        else _receiver_tag(receiver)
    )
    if reach_receiver.kind != "output":
        out_dir = os.path.join(out_dir, receiver_tag)
        os.makedirs(out_dir, exist_ok=True)

    # --- sender grid ----------------------------------------------------------
    config = pipeline.model.config
    layers = (
        list(analysis.layers)
        if analysis.get("layers") is not None
        else list(range(config.num_hidden_layers))
    )
    heads = (
        list(analysis.heads)
        if analysis.get("heads") is not None
        else list(range(config.num_attention_heads))
    )
    sender_pos = token_pos
    sender_targets = build_attention_head_targets(pipeline, layers, heads, sender_pos)
    senders = {key: target.flatten()[0] for key, target in sender_targets.items()}

    # An internal receiver can only be reached by senders upstream of its read point;
    # the methods runner raises on a downstream (structurally-zero) edge, so filter the
    # grid to reachable senders before sweeping. Reachability is layer-based, so the
    # grid stays rectangular; the heatmap layer axis is recomputed from the survivors.
    senders = _reachable_senders(pipeline, senders, reach_receiver, receiver_tag)
    if reach_receiver.kind != "output":
        layers = sorted({key[0] for key in senders})

    # --- metric + base outputs -----------------------------------------------
    relative_to_base = bool(analysis.get("relative_to_base", True))
    metric = make_logit_diff_metric(
        pipeline,
        test_dataset,
        correct_of=lambda ex: ex["input"][correct_variable],
        distractor_of=lambda ex: ex["input"][distractor_variable],
        relative_to_base=relative_to_base,
    )
    base_outputs = (
        compute_base_outputs(test_dataset, pipeline, batch_size=analysis.batch_size)
        if relative_to_base
        else None
    )

    # --- scan -----------------------------------------------------------------
    if receiver_specs is not None:
        effect_grid = run_path_patching_set_scan(
            pipeline,
            test_dataset,
            senders,
            receiver_specs=receiver_specs,
            metric=metric,
            restore=restore,
            batch_size=analysis.batch_size,
            output_scores=True,
            original_outputs=base_outputs,
        )
    else:
        effect_grid = run_path_patching_scan(
            pipeline,
            test_dataset,
            senders,
            metric=metric,
            receiver=receiver,
            restore=restore,
            batch_size=analysis.batch_size,
            output_scores=True,
            original_outputs=base_outputs,
        )

    # --- save -----------------------------------------------------------------
    _save_results(
        out_dir=out_dir,
        effect_grid=effect_grid,
        layers=layers,
        heads=heads,
        restore=restore,
        receiver_tag=receiver_tag,
        top_k=analysis.get("top_k", 20),
        figure_format=figure_fmt,
    )
    metadata = {
        "analysis": ANALYSIS_NAME,
        "sender_component": "attention_head",
        "receiver": receiver_tag,
        "restore": list(restore),
        "corruption": analysis.get("corruption", "abc"),
        "correct_variable": correct_variable,
        "distractor_variable": distractor_variable,
        "relative_to_base": relative_to_base,
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "layers": layers,
        "heads": heads,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Path-patching analysis complete. Output in %s", out_dir)
    return {
        "output_dir": out_dir,
        "effect_grid": {f"{k[0]}|{k[1]}": v for k, v in effect_grid.items()},
        "metadata": metadata,
    }


def _save_results(
    *,
    out_dir: str,
    effect_grid: dict[tuple[Any, ...], float],
    layers: list[int],
    heads: list[int],
    restore: tuple[str, ...],
    receiver_tag: str,
    top_k: int,
    figure_format: str,
) -> None:
    """Persist results.json and the direct-effect heatmap."""
    top_cells = sorted(effect_grid.items(), key=lambda kv: abs(kv[1]), reverse=True)[
        :top_k
    ]
    results_data = {
        "receiver": receiver_tag,
        "restore": list(restore),
        "effect_grid": {f"{k[0]}|{k[1]}": v for k, v in effect_grid.items()},
        "top_k_cells": [
            {"cell": f"{k[0]}|{k[1]}", "direct_effect": v} for k, v in top_cells
        ],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    # effect_grid holds a *signed, unbounded* direct effect on a logit difference
    # (negative name-movers score negative). create_heatmap's [0, 1] sequential
    # default would clamp every negative cell to the floor color — indistinguishable
    # from an inactive cell — hiding the strongest negative name-movers (the very
    # result an IOI direct-effect map exists to show). Use a symmetric diverging
    # scale centered at 0 so the sign is legible; the bound falls back to 1.0 when
    # the grid is empty or all-zero (a zero-width range would degenerate the cmap).
    title = (
        f"Path-patching direct effect on logit diff "
        f"(receiver={receiver_tag}, restore={'+'.join(restore)})"
    )
    finite_effects = [
        v for v in effect_grid.values() if v is not None and not math.isnan(v)
    ]
    bound = max((abs(v) for v in finite_effects), default=1.0) or 1.0
    save_path = path_with_figure_format(
        os.path.join(out_dir, "heatmap.png"), figure_format
    )
    try:
        plot_attention_head_heatmap(
            scores=effect_grid,
            layers=layers,
            heads=heads,
            title=title,
            save_path=save_path,
            figure_format=figure_format,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            cbar_label="Direct effect on logit diff",
        )
    except Exception as e:  # noqa: BLE001 — a render failure shouldn't lose results
        logger.warning("Path-patching heatmap render failed: %s", e)
