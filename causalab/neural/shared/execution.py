"""The engine-neutral half of ``Engine.execute``: run every point, lower its
metrics, and fill the output tables.

Everything here consumes the *executor surface* — ``read_value`` /
``dense_value`` / ``windowed_value`` / ``run_all`` / ``rows_for_metrics`` /
``is_generated`` / ``addressed_steps`` / ``generated_ids`` / ``bundle`` — and
nothing in it knows whether a hook or a trace produced the tensors. An engine
supplies its executor factory (and, if it trains, its train runner) and keeps
only its own identity stamp.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from causalab.neural.shared.metrics import compute_metric, compute_windowed_metric
from causalab.neural.shared.outputs import (
    MetricTable,
    TensorFile,
    code_commit,
    write_outputs,
)
from causalab.neural.shared.services import site_identity
from causalab.protocol.canonical import canonical_model
from causalab.protocol.engine import ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    METRIC_DOMAINS,
    WHOLE_WINDOW_METRIC_KINDS,
    Document,
    metric_reads_vocabulary,
    parse_document,
)

__all__ = ["ExecutorSurface", "execute_request", "featurizer_identity"]


class ExecutorSurface(Protocol):
    """What :func:`execute_request` needs from a point executor."""

    bundle: Any

    def run_all(self) -> None: ...
    def read_value(self, name: str) -> Any: ...
    def dense_value(self, name: str) -> Any: ...
    def windowed_value(self, name: str) -> list[Any]: ...
    def is_generated(self, name: str) -> bool: ...
    def addressed_steps(self, name: str) -> list[list[int]]: ...
    def generated_ids(self, name: str) -> list[list[int]]: ...
    def rows_for_metrics(self) -> list[dict[str, Any]]: ...


@dataclasses.dataclass(frozen=True)
class _Windowed:
    """One continuation metric's per-example results, plus what the rows need
    to stay legible: the steps each value scored, and whether the example
    addressed anything at all."""

    values: list[list[Any]]
    steps: list[list[int]] | None
    matched: list[bool]


def execute_request(
    request: ExecutionRequest,
    *,
    engine_name: str,
    executor_factory: Callable[
        [Document, ExecutionRequest, Mapping[str, Any]], ExecutorSurface
    ],
    train_runner: Callable[[Document, Any, ExecutionRequest], dict[str, Any]]
    | None = None,
) -> RunResult:
    """Run one :class:`ExecutionRequest` through one engine's executors.

    ``train_runner`` is the engine's train loop; an engine without one (its
    ``grad`` capability absent, so routing never sends it a ``train``
    document) refuses loudly if a train document reaches it anyway.
    """
    tensor_files: dict[str, TensorFile] = {}
    metric_files: dict[str, MetricTable] = {}
    summaries: list[Mapping[str, Any]] = []
    for point_raw, coords, digest in zip(
        request.points, request.coords, request.digests
    ):
        doc = parse_document(point_raw)
        summary = _execute_point(
            doc,
            request,
            coords=coords,
            point_digest=digest,
            tensor_files=tensor_files,
            metric_files=metric_files,
            executor_factory=executor_factory,
            train_runner=train_runner,
            engine_name=engine_name,
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
        "engine": engine_name,
        "commit": code_commit(Path(__file__).resolve().parents[3]),
    }
    files = write_outputs(
        request.output_dir,
        tensor_files,
        metric_files,
        identity_base=identity_base,
    )
    return RunResult(files=files, summaries=tuple(summaries))


def _execute_point(
    doc: Document,
    request: ExecutionRequest,
    *,
    coords: Mapping[str, Any],
    point_digest: str,
    tensor_files: dict[str, TensorFile],
    metric_files: dict[str, MetricTable],
    executor_factory: Callable[
        [Document, ExecutionRequest, Mapping[str, Any]], ExecutorSurface
    ],
    train_runner: Callable[[Document, Any, ExecutionRequest], dict[str, Any]] | None,
    engine_name: str,
) -> Mapping[str, Any]:
    executor = executor_factory(doc, request, coords)
    trained_stages: dict[str, Any] = {}
    if doc.train is not None:
        if train_runner is None:
            raise ProtocolError(
                "P4",
                f"this document declares a train section, which the "
                f"{engine_name!r} engine does not implement — its 'grad' "
                "capability is absent, so routing should not have sent it "
                "here",
            )
        trained_stages = train_runner(doc, executor, request)
    executor.run_all()
    metric_values: dict[str, list[Any]] = {}
    windowed: dict[str, _Windowed] = {}
    for qname, metric in doc.metrics.items():
        of_name = str(metric.of)
        target_name = str(metric.fields["target"]) if metric.kind == "kl" else None
        if executor.is_generated(of_name):
            # a continuation read addresses as many positions as the row
            # generated, so its metric reduces per step and reports which
            # steps it saw (§2.3, §2.10)
            windowed[qname] = _Windowed(
                values=compute_windowed_metric(
                    metric,
                    executor.windowed_value(of_name),
                    executor.rows_for_metrics(),
                    executor.bundle.tokenizer,
                    target_windows=(
                        executor.windowed_value(target_name)
                        if target_name is not None
                        else None
                    ),
                    generated_ids=(
                        executor.generated_ids(of_name)
                        if METRIC_DOMAINS.get(str(metric.kind)) == "ids"
                        else None
                    ),
                    vocab_axis=metric_reads_vocabulary(doc, metric),
                ),
                steps=(
                    None
                    if str(metric.kind) in WHOLE_WINDOW_METRIC_KINDS
                    else executor.addressed_steps(of_name)
                ),
                matched=[bool(steps) for steps in executor.addressed_steps(of_name)],
            )
            continue
        metric_values[qname] = compute_metric(
            metric,
            executor.dense_value(of_name),
            executor.rows_for_metrics(),
            executor.bundle.tokenizer,
            target_value=(
                executor.dense_value(target_name) if target_name is not None else None
            ),
            vocab_axis=metric_reads_vocabulary(doc, metric),
        )
    for entry in doc.save:
        if entry.value in doc.metrics:
            table = metric_files.setdefault(entry.file_path, MetricTable())
            if entry.value in windowed:
                window = windowed[entry.value]
                table.add_windowed(
                    entry.value,
                    window.values,
                    coords,
                    point_digest,
                    steps=window.steps,
                    matched=window.matched,
                )
            else:
                table.add(entry.value, metric_values[entry.value], coords, point_digest)
        elif entry.value in doc.reads:
            # the site goes on the entry too: a harvested activation is
            # bound to where it was read, and a consumer (a script step
            # fitting a basis on it, then a document loading that basis)
            # has no other way to prove the two agree
            read_site = site_identity(doc, str(doc.reads[entry.value].site))
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
            identity = featurizer_identity(doc, entry.value, entry.site, point_digest)
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


def featurizer_identity(
    doc: Document, name: str, site_name: str | None, point_digest: str
) -> dict[str, str]:
    """The ArtifactIdentity a trained featurizer bundle stamps (§8)."""
    from causalab.protocol.resolve import build_artifact_identity

    spec = doc.featurizers[name]
    site = site_identity(doc, site_name)
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
