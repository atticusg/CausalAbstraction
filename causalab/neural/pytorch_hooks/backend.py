"""The reference backend: intervention protocols on native pytorch hooks.

Implements the spec §8 services on the two supported architecture families.
Capabilities: ``grad`` (the train loop, train.py), ``paired_forward``
(cross-input operand flow via the lazy group executor), ``full_logits``
(lm_head is an ordinary tap), ``generate`` (the greedy decode in
executor.py, which interventions reach only through the prefill), and
``pytorch_fn_local`` (this backend is local). ``writable_attention_probs``
is deliberately absent — no attention-internal tap yet — so capability
routing refuses those documents before anything runs.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping


from causalab.neural.pytorch_hooks.executor import PointExecutor
from causalab.neural.pytorch_hooks.loading import TensorBundle, load_model
from causalab.neural.pytorch_hooks.metrics import compute_metric
from causalab.neural.pytorch_hooks.outputs import (
    MetricTable,
    TensorFile,
    code_commit,
    write_outputs,
)
from causalab.protocol.backend import Backend, ExecutionRequest, RunResult
from causalab.protocol.canonical import canonical_model
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import Document, parse_document

__all__ = ["PytorchHooksBackend"]


class PytorchHooksBackend(Backend):
    name = "pytorch_hooks"
    capabilities = frozenset(
        {
            "grad",
            "paired_forward",
            "full_logits",
            "generate",
            "pytorch_fn_local",
            "quantized_weights",
        }
    )
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        # placement is execution (the backend's call, §8); precision is not —
        # dtype and quantization come from each point's own `model` section
        self.device = device

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
        first_realization = canonical_model(first_doc.raw["model"])
        identity_base = {
            "produced_by": request.document_digest,
            "model_key": str(first_doc.model.key),
            "model_revision": str(first_doc.model.revision),
            "model_dtype": str(first_realization["dtype"]),
            "model_quantization": first_realization.get("quantization"),
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
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
    ) -> PointExecutor:
        realization = canonical_model(doc.raw["model"])
        bundle = load_model(
            str(doc.model.key),
            str(doc.model.revision),
            dtype=str(realization["dtype"]),
            device=self.device,
            quantization=_quantization_key(realization),
        )
        role_rows, role_fields = _resolve_roles(doc, request)
        return PointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(_load_tensors, request),
            grad_enabled=grad_enabled,
            coords=coords,
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
        executor = self._executor(doc, request, coords=coords)
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
                # the site goes on the entry too: a harvested activation is
                # bound to where it was read, and a consumer (a transform op
                # fitting a basis on it, then a document loading that basis)
                # has no other way to prove the two agree
                read_site = _site_identity(doc, str(doc.reads[entry.value].site))
                tensor_files.setdefault(entry.file_path, TensorFile()).add(
                    entry.value,
                    executor.read_value(entry.value),
                    coords,
                    reduce=entry.reduce,
                    identity={
                        "produced_by": point_digest,
                        **(
                            {"site": json.dumps(read_site, sort_keys=True)}
                            if read_site
                            else {}
                        ),
                    },
                )
            else:  # a trained featurizer bundle
                stage = trained_stages.get(entry.value)
                if stage is None:
                    raise ProtocolError(
                        "P2", f"featurizer {entry.value!r} was not trained this run"
                    )
                bundle_file = tensor_files.setdefault(entry.file_path, TensorFile())
                identity = _featurizer_identity(
                    doc, entry.value, entry.site, point_digest
                )
                for slot, param in stage.slot_params().items():
                    # per entry, not per file: a swept fit writes one file from
                    # many points, and only the entry table can say which point
                    # produced which rotation (§8)
                    bundle_file.add(
                        slot,
                        param.detach(),
                        coords,
                        label_entry=entry.value,
                        identity=identity,
                    )
                bundle_file.record_common(identity)
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


def _site_identity(doc: Document, site_name: str | None) -> dict[str, Any] | None:
    """One site as the ArtifactIdentity records it — the non-null address
    fields only, the shape ``loader.py`` builds its expectation in."""
    if site_name is None or site_name not in doc.sites:
        return None
    record = doc.sites[site_name]
    return {
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


def _featurizer_identity(
    doc: Document, name: str, site_name: str | None, point_digest: str
) -> dict[str, str]:
    from causalab.protocol.resolve import build_artifact_identity

    spec = doc.featurizers[name]
    site = _site_identity(doc, site_name)
    base = doc.data["base"]
    trained_on = base.dataset if not isinstance(base, tuple) else base[0].dataset
    realization = canonical_model(doc.raw["model"])
    return build_artifact_identity(
        produced_by=point_digest,
        model_key=str(doc.model.key),
        model_revision=str(doc.model.revision),
        model_dtype=str(realization["dtype"]),
        model_quantization=realization.get("quantization"),
        site=site,
        k=spec.k if isinstance(spec.k, int) else None,
        parametrization=spec.parametrization
        if isinstance(spec.parametrization, str)
        else None,
        dtype=spec.dtype if isinstance(spec.dtype, str) else "fp32",
        trained_on=str(trained_on),
    )


def _quantization_key(
    realization: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...] | None:
    """The materialized ``quantization`` block as a hashable, order-free key —
    :func:`load_model` caches on it, so it must hash and compare by value."""
    quantization = realization.get("quantization")
    if quantization is None:
        return None
    return tuple(sorted(quantization.items()))


def _resolve_roles(
    doc: Document, request: ExecutionRequest
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Dataset rows + field selector per input role, rows paired by index.

    ``rows`` is part of the :class:`~causalab.protocol.resolve.DatasetResolver`
    contract, so this reads it directly — a resolver without it is a typing
    error at construction, not a surprise at run time."""
    rows_of = request.env.datasets.rows
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


@functools.lru_cache(maxsize=32)
def _read_bundle(path: str, _stamp: tuple[int, int]) -> TensorBundle:
    """One bundle, read once. The cache matters: a write operand resolves
    its ``params`` tensor on every application, so an uncached read would
    re-open the same file for every batch of every point.

    ``_stamp`` is the file's (mtime, size), so a path rewritten in the same
    process — a step re-run into an existing run tree — is a cache miss
    rather than a stale tensor."""
    from safetensors.torch import load_file

    from causalab.protocol.resolve import read_safetensors_metadata

    meta = read_safetensors_metadata(Path(path)) or {}
    raw_entries = meta.get("entries")
    entry_coords: dict[str, Any] = {}
    if isinstance(raw_entries, str):
        try:
            decoded = json.loads(raw_entries)
        except json.JSONDecodeError as err:
            raise ProtocolError(
                "P2", f"{path}: unreadable 'entries' table in the header — {err}"
            ) from err
        if isinstance(decoded, dict):
            entry_coords = decoded
    return TensorBundle(tensors=load_file(path), entry_coords=entry_coords)


def _load_tensors(request: ExecutionRequest, file_path: str) -> TensorBundle:
    """Load a tensor bundle referenced by a featurizer/params file_path,
    resolved through the artifact store (which owns the run-tree/external
    overlay inside a workflow)."""
    artifacts = request.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    stat = target.stat()
    return _read_bundle(str(target), (stat.st_mtime_ns, stat.st_size))
