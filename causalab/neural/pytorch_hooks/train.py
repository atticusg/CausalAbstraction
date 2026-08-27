"""The train loop (spec §2.11, §8): the document declares the fit, this
backend owns the loop.

Semantics implemented exactly as declared — none of the legacy loop's
ambient state survives:

* ``seed`` drives featurizer init, data order, and every draw, so a
  ``{"sweep": [0,1,2]}`` on ``seed`` yields three genuinely different fits.
  Init reaches the featurizer as an explicit argument (``executor.seed`` →
  ``build_stack(seed=…)``, a *local* generator) rather than through the
  global RNG, because the same construction runs on apply paths where no
  loop entry ever executes; the batch order has its own local ``order_rng``;
  ``torch.manual_seed`` at loop entry — the one deliberate use of the global
  RNG — covers everything else that draws (dropout, and torch's own
  orthogonal-parametrization basis);
* ``objective`` terms are differentiable metric tensors plus regularizers
  (``l1``/``l2`` over a featurizer's params; a ``gate``'s l1 is the mean
  soft mask — the DBM sparsity semantics);
* ``anneal`` targets ``<featurizer>.<slot>.temperature`` linearly from
  start to end over the first ``frac`` of total steps, then holds;
* ``eval`` runs on the declared split in eval mode (hard gate, no grad)
  every N epochs/updates; ``early_stop`` tracks the eval metric with
  patience;
* ``batch.pairs`` counts base+counterfactual pairs; roles are sliced together
  (rows are paired by index, §2.2).

The model's weights are frozen at load; only featurizer slots (and, later,
free ``params`` tensors) optimize.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from causalab.neural.pytorch_hooks.executor import PointExecutor, document_seed
from causalab.neural.pytorch_hooks.featurizers import Gate, Stage
from causalab.neural.pytorch_hooks.metrics import column_token_ids
from causalab.protocol.backend import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    Document,
    MetricSpec,
    concrete_int,
    concrete_str,
    metric_reads_vocabulary,
)

__all__ = ["metric_tensor", "run_training"]


def metric_tensor(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """A differentiable per-example metric (the objective-side twin of
    ``metrics.compute_metric``). Only the reduction kinds with a gradient
    are usable in an objective."""
    logits = of_value[:, 0, :] if of_value.dim() == 3 else of_value
    logits = logits.float()
    kind = str(metric.kind)

    form = str(metric.token_form)  # §2.10; `auto` is the historical default

    def ids(field: str) -> torch.Tensor:
        column = concrete_str(metric.fields[field], f"metric field {field}")
        return torch.tensor(
            column_token_ids(
                tokenizer,
                [str(row[column]) for row in rows],
                token_form=form,
                where=f"metric {kind}.{field}",
            ),
            dtype=torch.long,
            device=logits.device,
        )

    if kind == "cross_entropy":
        log_probs = torch.log_softmax(logits, dim=-1)
        return -log_probs.gather(1, ids("target").unsqueeze(1)).squeeze(1)
    if kind == "logit_diff":
        return logits.gather(1, ids("a").unsqueeze(1)).squeeze(1) - logits.gather(
            1, ids("b").unsqueeze(1)
        ).squeeze(1)
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        target = target_value[:, 0, :] if target_value.dim() == 3 else target_value
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(target.float(), dim=-1)
        return (p.exp() * (p - q)).sum(dim=-1)
    raise ProtocolError(
        "P2",
        f"metric kind {kind!r} has no gradient — objectives compose from "
        "cross_entropy / logit_diff / kl (§2.11)",
    )


def _regularizer(kind: str, target: str, stages: Mapping[str, Stage]) -> torch.Tensor:
    fname = target.split(".", 1)[0]
    stage = stages[fname]
    if isinstance(stage, Gate) and kind == "l1":
        # DBM sparsity: the mean SOFT mask, not |theta| — pushing mask mass
        # toward zero features, temperature-annealed
        return torch.sigmoid(stage.theta / stage.temperature).mean()
    total = torch.zeros(())
    count = 0
    for slot, param in stage.slot_params().items():
        if "." in target and slot != target.split(".", 1)[1]:
            continue
        total = total + (param.abs().mean() if kind == "l1" else param.pow(2).mean())
        count += 1
    if count == 0:
        raise ProtocolError("P2", f"regularizer target {target!r} matches no slot")
    return total / count


def _slice_rows(
    role_rows: Mapping[str, list[dict[str, Any]]], indices: list[int]
) -> dict[str, list[dict[str, Any]]]:
    return {role: [rows[i] for i in indices] for role, rows in role_rows.items()}


def run_training(
    doc: Document, executor: PointExecutor, request: ExecutionRequest
) -> dict[str, Stage]:
    """Fit the declared params; returns the trained stages by featurizer
    name (for the save manifest). ``executor`` is the full-data executor —
    its stage cache is shared with every training minibatch, so the stages
    it later evaluates are the fitted ones."""
    train = doc.train
    assert train is not None
    # one reader of train.seed, shared with the featurizer inits the executor
    # builds below — the loop and the init cannot disagree about the seed
    seed = document_seed(doc)
    torch.manual_seed(seed)

    trained_names = sorted({p.split(".", 1)[0] for p in train.params})
    stages: dict[str, Stage] = {}
    parameters: list[torch.nn.Parameter] = []
    for pname in train.params:
        fname, _, slot = pname.partition(".")
        if fname not in doc.featurizers:
            raise NotImplementedError(
                f"free params entries ({pname!r}) are not trainable in this "
                "backend yet — featurizer slots only"
            )
        stage = executor.stage(fname)
        stages[fname] = stage
        slot_params = stage.slot_params()
        if slot:
            parameters.append(slot_params[slot])  # type: ignore[arg-type]
        else:
            parameters.extend(p for p in stage.parameters() if p.requires_grad)
    if not parameters:
        raise ProtocolError("P2", "train.params resolved to no trainable tensors")

    optimizer = _build_optimizer(train.optimizer, parameters)

    n_examples = len(executor.rows_for_metrics())
    pairs = concrete_int(train.batch["pairs"], "train.batch.pairs")
    batches = [
        list(range(start, min(start + pairs, n_examples)))
        for start in range(0, n_examples, pairs)
    ]
    if "epochs" in train.steps:
        epochs = concrete_int(train.steps["epochs"], "train.steps.epochs")
        total_steps = epochs * len(batches)
    else:
        total_steps = concrete_int(train.steps["updates"], "train.steps.updates")
        epochs = -(-total_steps // len(batches))

    anneals = _parse_anneals(train.anneal, stages)
    eval_every_epochs = None
    if train.eval is not None and "epochs" in train.eval["every"]:
        eval_every_epochs = concrete_int(
            train.eval["every"]["epochs"], "train.eval.every.epochs"
        )
    best: float | None = None
    stale = 0
    order_rng = torch.Generator().manual_seed(seed)

    minibatch_executors = [
        PointExecutor(
            doc,
            executor.bundle,
            role_rows=_slice_rows(executor.role_rows, indices),
            role_fields=executor.role_fields,
            load_tensors=executor.load_tensors,
            stage_cache=executor.stage_cache,  # shared: one stage per name
            grad_enabled=True,
            coords=executor.coords,
        )
        for indices in batches
    ]

    step = 0
    for _epoch in range(epochs):
        if step >= total_steps:
            break
        for batch_index in torch.randperm(len(batches), generator=order_rng).tolist():
            if step >= total_steps:
                break
            for stage in stages.values():
                stage.train(True)
            for dotted, (start, end, frac) in anneals.items():
                _set_anneal(stages, dotted, start, end, frac, step, total_steps)
            mb = minibatch_executors[batch_index]
            mb.reset_reads()
            loss = torch.zeros(())
            for weight, target in train.objective:
                w = float(weight) if isinstance(weight, (int, float)) else 1.0
                if isinstance(target, str):
                    metric = doc.metrics[target]
                    of_value = mb.dense_value(str(metric.of))
                    target_value = (
                        mb.dense_value(str(metric.fields["target"]))
                        if metric.kind == "kl"
                        else None
                    )
                    term = metric_tensor(
                        metric,
                        of_value,
                        mb.rows_for_metrics(),
                        mb.bundle.tokenizer,
                        target_value=target_value,
                    ).mean()
                elif isinstance(target, tuple):
                    reg_kind, reg_target = target
                    term = _regularizer(str(reg_kind), str(reg_target), stages)
                else:
                    raise ProtocolError("P2", f"unresolvable objective term {target!r}")
                loss = loss + w * term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        if eval_every_epochs is not None and (_epoch + 1) % eval_every_epochs == 0:
            assert train.eval is not None
            score = _run_eval(
                doc,
                executor,
                request,
                concrete_str(train.eval["split"], "train.eval.split"),
            )
            if train.early_stop is not None:
                metric_name = str(train.early_stop["metric"])
                mode = str(train.early_stop["mode"])
                value = score[metric_name]
                improved = (
                    best is None
                    or (mode == "max" and value > best)
                    or (mode == "min" and value < best)
                )
                if improved:
                    best, stale = value, 0
                else:
                    stale += 1
                    if stale > concrete_int(train.early_stop["patience"], "patience"):
                        break
    for stage in stages.values():
        stage.eval()
    return {name: stages[name] for name in trained_names}


def _build_optimizer(
    spec: Mapping[str, Any], parameters: list[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    name = str(spec["name"])
    lr = float(spec["lr"])
    weight_decay = float(spec.get("weight_decay", 0.0))
    if name in ("adamw", "adam"):
        raw_betas = spec.get("betas", (0.9, 0.999))
        betas = (float(raw_betas[0]), float(raw_betas[1]))
        eps = float(spec.get("eps", 1e-8))
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(spec.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    raise ProtocolError("P4", f"unknown optimizer {name!r}")


def _parse_anneals(
    anneal: Mapping[str, Any] | None, stages: Mapping[str, Stage]
) -> dict[str, tuple[float, float, float]]:
    if anneal is None:
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for dotted, schedule in anneal.items():
        fname = dotted.split(".", 1)[0]
        if fname not in stages:
            raise ProtocolError("P2", f"anneal target {dotted!r} is not being trained")
        if not isinstance(schedule, (list, tuple)) or len(schedule) != 3:
            raise ProtocolError(
                "P2", f"anneal schedule for {dotted!r} is [start, end, frac]"
            )
        start, end, frac = (float(v) for v in schedule)
        out[dotted] = (start, end, frac)
    return out


def _set_anneal(
    stages: Mapping[str, Stage],
    dotted: str,
    start: float,
    end: float,
    frac: float,
    step: int,
    total_steps: int,
) -> None:
    fname, _, tail = dotted.partition(".")
    hyper = tail.rsplit(".", 1)[-1]
    ramp_steps = max(1, int(frac * total_steps))
    progress = min(1.0, step / ramp_steps)
    value = start + (end - start) * progress
    stage = stages[fname]
    if not hasattr(stage, hyper):
        raise ProtocolError("P2", f"{fname!r} has no annealable {hyper!r}")
    setattr(stage, hyper, value)


def _run_eval(
    doc: Document,
    executor: PointExecutor,
    request: ExecutionRequest,
    split: str,
) -> dict[str, float]:
    """Evaluate the declared eval metrics on the split, hard-gate eval mode."""
    from causalab.neural.pytorch_hooks.metrics import compute_metric

    rows_of = getattr(request.env.datasets, "rows")
    split_rows = rows_of(split)
    role_rows = {role: split_rows for role in executor.role_rows}
    for stage in executor.stage_cache.values():
        stage.eval()
    eval_executor = PointExecutor(
        doc,
        executor.bundle,
        role_rows=role_rows,
        role_fields=executor.role_fields,
        load_tensors=executor.load_tensors,
        stage_cache=executor.stage_cache,
        grad_enabled=False,
        coords=executor.coords,
    )
    assert doc.train is not None and doc.train.eval is not None
    scores: dict[str, float] = {}
    for name in doc.train.eval["metrics"]:
        metric = doc.metrics[name]
        of_value = eval_executor.dense_value(str(metric.of))
        target = (
            eval_executor.dense_value(str(metric.fields["target"]))
            if metric.kind == "kl"
            else None
        )
        values = compute_metric(
            metric,
            of_value,
            eval_executor.rows_for_metrics(),
            eval_executor.bundle.tokenizer,
            target_value=target,
            vocab_axis=metric_reads_vocabulary(doc, metric),
        )
        numeric = [v for v in values if isinstance(v, (int, float))]
        scores[name] = sum(numeric) / len(numeric) if numeric else 0.0
    return scores
