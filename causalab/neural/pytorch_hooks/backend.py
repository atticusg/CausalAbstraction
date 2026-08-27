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

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


from causalab.neural.pytorch_hooks.executor import (
    ForwardCache,
    Interning,
    PointExecutor,
)
from causalab.neural.pytorch_hooks.loading import TensorBundle, load_model
from causalab.neural.pytorch_hooks.metrics import (
    compute_metric,
    compute_windowed_metric,
)
from causalab.neural.pytorch_hooks.outputs import (
    MetricTable,
    TensorFile,
    code_commit,
    write_outputs,
)
from causalab.protocol.backend import Backend, ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import PointPlan, generated_budget, plan_point
from causalab.protocol.schema import (
    METRIC_DOMAINS,
    WHOLE_WINDOW_METRIC_KINDS,
    Document,
    SiteSpec,
    parse_document,
)

__all__ = ["PytorchHooksBackend", "campaign_plans"]


def campaign_plans(docs: Sequence[Document]) -> tuple[PointPlan, ...]:
    """The per-point plans this backend executes a campaign from.

    Public because the interning claim is checkable arithmetic:
    :func:`~causalab.protocol.plan.interned_groups` over these plans is how
    many forward groups a run *owes*, and
    :attr:`~causalab.protocol.backend.RunResult.forwards` is what it paid. One
    derivation, so the number a caller verifies against is the number
    execution keyed on.
    """
    return tuple(plan_point(doc, data_identity=_data_identity(doc)) for doc in docs)


@dataclasses.dataclass(frozen=True)
class _Windowed:
    """One continuation metric's per-example results, plus what the rows need
    to stay legible: the steps each value scored, and whether the example
    addressed anything at all."""

    values: list[list[Any]]
    steps: list[list[int]] | None
    matched: list[bool]


class PytorchHooksBackend(Backend):
    name = "pytorch_hooks"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "generate", "pytorch_fn_local"}
    )
    is_local = True

    def __init__(self, *, device: str = "cpu", dtype: str = "fp32") -> None:
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        # The whole campaign is planned before anything runs, because §3's
        # interning is a property of the point *set*: a forward group can only
        # be shared once you know which other points share it, and the union
        # of taps it must capture only exists across all of them.
        docs = tuple(parse_document(point_raw) for point_raw in request.points)
        plans = campaign_plans(docs)
        cache = ForwardCache(wanted=_tap_union(docs, plans))

        tensor_files: dict[str, TensorFile] = {}
        metric_files: dict[str, MetricTable] = {}
        summaries: list[Mapping[str, Any]] = []
        for doc, plan, coords, digest in zip(
            docs, plans, request.coords, request.digests
        ):
            summary = self._execute_point(
                doc,
                request,
                coords=coords,
                point_digest=digest,
                tensor_files=tensor_files,
                metric_files=metric_files,
                interning=Interning(
                    digests={
                        (group.model, group.input): group.digest
                        for group in plan.groups
                    },
                    cache=cache,
                ),
            )
            summaries.append(summary)
        identity_base = {
            "produced_by": request.document_digest,
            "model_key": str(docs[0].model.key),
            "model_revision": str(docs[0].model.revision),
            "backend": self.name,
            "commit": code_commit(Path(__file__).resolve().parents[3]),
        }
        files = write_outputs(
            request.output_dir,
            tensor_files,
            metric_files,
            identity_base=identity_base,
        )
        return RunResult(
            files=files,
            summaries=tuple(summaries),
            forwards=len(cache.executed),
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
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
            coords=coords,
            interning=interning,
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
        interning: Interning | None = None,
    ) -> Mapping[str, Any]:
        executor = self._executor(doc, request, coords=coords, interning=interning)
        trained_stages: dict[str, Any] = {}
        if doc.train is not None:
            from causalab.neural.pytorch_hooks.train import run_training

            trained_stages = run_training(doc, executor, request)
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
                    ),
                    steps=(
                        None
                        if str(metric.kind) in WHOLE_WINDOW_METRIC_KINDS
                        else executor.addressed_steps(of_name)
                    ),
                    matched=[
                        bool(steps) for steps in executor.addressed_steps(of_name)
                    ],
                )
                continue
            metric_values[qname] = compute_metric(
                metric,
                executor.dense_value(of_name),
                executor.rows_for_metrics(),
                executor.bundle.tokenizer,
                target_value=(
                    executor.dense_value(target_name)
                    if target_name is not None
                    else None
                ),
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
                    table.add(
                        entry.value, metric_values[entry.value], coords, point_digest
                    )
            elif entry.value in doc.reads:
                # the site goes on the entry too: a harvested activation is
                # bound to where it was read, and a consumer (a script step
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


def _data_identity(doc: Document) -> dict[str, str]:
    """Input role → the identity of the rows that role will be encoded from.

    Folded into every forward-group digest so two points reading *different*
    data on the same role never intern together. ``(dataset, field)`` is
    exactly what determines a role's batch — :func:`_resolve_roles` resolves
    rows by dataset name and selects one field, and the executor tokenizes
    nothing else — so this is neither coarser nor finer than the thing being
    shared. The role names mirror ``_resolve_roles`` (``counterfactual[0]``
    for a tuple-valued role) so the keys line up with the plan's ``input``.
    """
    identity: dict[str, str] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        for j, role_spec in enumerate(entries):
            role_name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            identity[role_name] = f"{role_spec.dataset}#{role_spec.field}"
    return identity


def _tap_union(
    docs: Sequence[Document], plans: Sequence[PointPlan]
) -> dict[str, tuple[SiteSpec, ...]]:
    """Forward-group digest → every site the campaign taps in that group.

    The union *is* the interning. Taps are deliberately absent from a group's
    digest, so the single pass a shared digest earns has to capture every
    address any point will ask of it — for a 32-layer scan that is one
    counterfactual forward with 32 taps instead of 32 forwards with one each.

    Continuation reads are excluded: those are served by the decode's
    per-step accumulation, not by the prefill capture this store holds, so a
    decoding group contributes only its prompt-frame taps (and can therefore
    still hand its prefill to a non-decoding point that shares the digest).
    """
    union: dict[str, dict[str, SiteSpec]] = {}
    for doc, plan in zip(docs, plans):
        for group in plan.groups:
            wanted = union.setdefault(group.digest, {})
            for tap in group.taps:
                read = doc.reads[tap.read]
                if generated_budget(doc, read.pos) is not None:
                    continue
                spec = doc.sites[tap.site]
                wanted[json.dumps(_site_identity(doc, tap.site), sort_keys=True)] = spec
    return {digest: tuple(specs.values()) for digest, specs in union.items()}


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
