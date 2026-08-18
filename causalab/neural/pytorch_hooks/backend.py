"""The reference backend: intervention protocols on native pytorch hooks.

Implements the spec §8 services for prefill-only documents on the two
supported architecture families. Capabilities: ``grad`` (the train loop,
train.py), ``paired_forward`` (cross-input operand flow via the lazy group
executor), ``full_logits`` (lm_head is an ordinary tap), and
``pytorch_fn_local`` (this backend is local). ``editable_attention_probs``
is deliberately absent — no attention-internal tap yet — so capability
routing refuses those documents before anything runs.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping

import torch

from causalab.neural.pytorch_hooks.executor import PointExecutor
from causalab.neural.pytorch_hooks.loading import load_model
from causalab.neural.pytorch_hooks.metrics import compute_metric
from causalab.neural.pytorch_hooks.outputs import (
    MetricTable,
    TensorFile,
    code_commit,
    write_outputs,
)
from causalab.protocol.backend import Backend, ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import Document, parse_document

__all__ = ["PytorchHooksBackend"]


class PytorchHooksBackend(Backend):
    name = "pytorch_hooks"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    is_local = True

    def __init__(self, *, device: str = "cpu", dtype: str = "fp32") -> None:
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        tensor_files: dict[str, TensorFile] = {}
        metric_files: dict[str, MetricTable] = {}
        summaries: list[Mapping[str, Any]] = []
        for point_raw, coords, digest in zip(
            request.points, request.coords, request.digests
        ):
            doc = parse_document(point_raw)
            summary = self._execute_point(
                doc,
                request,
                coords=coords,
                point_digest=digest,
                tensor_files=tensor_files,
                metric_files=metric_files,
            )
            summaries.append(summary)
        first_doc = parse_document(request.points[0])
        identity_base = {
            "produced_by": request.document_digest,
            "model_key": str(first_doc.model.key),
            "model_revision": str(first_doc.model.revision),
            "backend": self.name,
            "commit": code_commit(Path(__file__).resolve().parents[3]),
        }
        files = write_outputs(
            request.output_dir,
            tensor_files,
            metric_files,
            identity_base=identity_base,
        )
        return RunResult(files=files, summaries=tuple(summaries))

    # ------------------------------------------------------------------ #

    def _executor(
        self, doc: Document, request: ExecutionRequest, *, grad_enabled: bool = False
    ) -> PointExecutor:
        bundle = load_model(
            str(doc.model.key),
            str(doc.model.revision),
            dtype=self.dtype,
            device=self.device,
        )
        role_rows, role_fields = _resolve_roles(doc, request)
        return PointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(_load_tensors, request),
            grad_enabled=grad_enabled,
        )

    def _execute_point(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        coords: Mapping[str, Any],
        point_digest: str,
        tensor_files: dict[str, TensorFile],
        metric_files: dict[str, MetricTable],
    ) -> Mapping[str, Any]:
        executor = self._executor(doc, request)
        trained_stages: dict[str, Any] = {}
        if doc.train is not None:
            from causalab.neural.pytorch_hooks.train import run_training

            trained_stages = run_training(doc, executor, request)
        executor.run_all()
        metric_values: dict[str, list[Any]] = {}
        for qname, metric in doc.metrics.items():
            of_value = executor.dense_value(str(metric.of))
            target = None
            if metric.kind == "kl":
                target = executor.dense_value(str(metric.fields["target"]))
            metric_values[qname] = compute_metric(
                metric,
                of_value,
                executor.rows_for_metrics(),
                executor.bundle.tokenizer,
                target_value=target,
            )
        for entry in doc.save:
            if entry.value in doc.metrics:
                metric_files.setdefault(entry.file_path, MetricTable()).add(
                    entry.value, metric_values[entry.value], coords, point_digest
                )
            elif entry.value in doc.reads:
                tensor_files.setdefault(entry.file_path, TensorFile()).add(
                    entry.value, executor.read_value(entry.value), coords
                )
            else:  # a trained featurizer bundle
                stage = trained_stages.get(entry.value)
                if stage is None:
                    raise ProtocolError(
                        "P2", f"featurizer {entry.value!r} was not trained this run"
                    )
                bundle_file = tensor_files.setdefault(entry.file_path, TensorFile())
                for slot, param in stage.slot_params().items():
                    bundle_file.add(slot, param.detach(), coords)
                bundle_file.metadata.update(
                    _featurizer_identity(doc, entry.value, entry.site, point_digest)
                )
        return {
            "point": point_digest,
            "coords": dict(coords),
            "metrics": {
                name: _summary_stat(values) for name, values in metric_values.items()
            },
        }


def _summary_stat(values: list[Any]) -> Any:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if numeric:
        return sum(numeric) / len(numeric)
    return f"{len(values)} rows"


def _featurizer_identity(
    doc: Document, name: str, site_name: str | None, point_digest: str
) -> dict[str, str]:
    from causalab.protocol.resolve import build_artifact_identity

    spec = doc.featurizers[name]
    site: Mapping[str, Any] | None = None
    if site_name is not None:
        record = doc.sites[site_name]
        site = {
            key: value
            for key, value in {
                "component": record.component,
                "layer": record.layer,
                "head": record.head,
                "expert": record.expert,
                "stream": record.stream,
            }.items()
            if value is not None
        }
    base = doc.data["base"]
    trained_on = base.dataset if not isinstance(base, tuple) else base[0].dataset
    return build_artifact_identity(
        produced_by=point_digest,
        model_key=str(doc.model.key),
        model_revision=str(doc.model.revision),
        site=site,
        k=spec.k if isinstance(spec.k, int) else None,
        parametrization=spec.parametrization
        if isinstance(spec.parametrization, str)
        else None,
        dtype=spec.dtype if isinstance(spec.dtype, str) else "fp32",
        trained_on=str(trained_on),
    )


def _resolve_roles(
    doc: Document, request: ExecutionRequest
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Dataset rows + field selector per input role, rows paired by index."""
    datasets = request.env.datasets
    rows_of = getattr(datasets, "rows", None)
    if rows_of is None:
        raise ProtocolError(
            "P2",
            "this resolution environment's dataset resolver exposes no rows() — "
            "execution needs the table content, not just its digest",
        )
    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_fields: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        for j, role_spec in enumerate(entries):
            role_name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            role_rows[role_name] = rows_of(str(role_spec.dataset))
            role_fields[role_name] = str(role_spec.field)
            lengths[role_name] = len(role_rows[role_name])
    if len(set(lengths.values())) > 1:
        raise ProtocolError(
            "P2",
            f"input roles have unequal row counts {lengths} — rows are paired "
            "by index (§2.2)",
        )
    return role_rows, role_fields


def _load_tensors(request: ExecutionRequest, file_path: str) -> dict[str, torch.Tensor]:
    """Load a tensor bundle referenced by a featurizer/params file_path,
    relative to the artifacts root."""
    from safetensors.torch import load_file

    artifacts = request.env.artifacts
    root = getattr(artifacts, "root", None)
    if root is None:
        raise ProtocolError("P2", "artifact store exposes no filesystem root")
    return load_file(str(Path(root) / file_path))
