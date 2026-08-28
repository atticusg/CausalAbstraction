"""The backend contract: capabilities, `requires`, and routing (spec §8).

A backend is anything that can execute point protocols — nnsight, Megatron,
SGLang, or the in-repo reference over native pytorch hooks
(:mod:`causalab.neural.pytorch_hooks`). The document never knows which one
runs it: ``requires`` derives the capability set a document needs, each
backend declares what it supports, and ``choose_backend`` picks the first
backend whose capabilities cover the requirement — with refusal messages
generated from the missing capabilities, never hand-written per case.
"""

from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ValidationError
from causalab.protocol.plan import generated_budget
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.schema import Document, MetricSpec

__all__ = [
    "Backend",
    "CAPABILITIES",
    "ExecutionRequest",
    "RunResult",
    "choose_backend",
    "requires",
    "requires_campaign",
]

#: The closed capability vocabulary (§8).
CAPABILITIES: tuple[str, ...] = (
    "grad",
    "paired_forward",
    "full_logits",
    "writable_attention_probs",
    "pytorch_fn_local",
    "generate",
    "quantized_weights",
)

#: Metric kinds that need the whole vocabulary materialized (§8) — but only
#: when their read actually taps ``lm_head``. ``class_probs`` always does
#: (validation binds it to a vocabulary projection); ``top_k`` ranks whatever
#: axis its read has, and a top-k over a 4k-wide residual stream or a 100k-wide
#: SAE code obliges no vocabulary projection at all. Charging it ``full_logits``
#: would route such a document onto a full-vocab backend for nothing.
_FULL_VOCAB_METRICS = frozenset({"top_k", "class_probs"})


def _metric_read_obliges_full_projection(doc: Document, metric: MetricSpec) -> bool:
    """Whether serving ``metric``'s read means materializing the vocabulary.

    Deliberately NOT :func:`~causalab.protocol.schema.metric_reads_vocabulary`:
    that predicate asks what the read *hands the metric* (token ids, or a
    featurizer's latents / a ``dims`` re-index), which governs softmaxing and
    decoding. This one asks what the backend must *compute upstream* — and a
    featurized ``lm_head`` read still consumes the whole projection, its
    featurizer merely re-expresses it. The two questions diverge exactly
    there. A ``dims`` slice is the one transform that needs only its named
    rows, matching the saved-read rule above.
    """
    read = doc.reads.get(str(metric.of))
    if read is None:
        return False
    site = doc.sites.get(str(read.site))
    return site is not None and site.component == "lm_head" and read.dims is None


def requires(doc: Document) -> frozenset[str]:
    """The capability set one concrete document needs — derived, never
    authored (§6)."""
    needed: set[str] = set()
    if doc.train is not None:
        needed.add("grad")
    if doc.model.quantization is not None:
        needed.add("quantized_weights")
    for im in doc.intervened_models.values():
        if not isinstance(im.writes, tuple):
            raise AssertionError(
                "requires() takes a concrete point document — expand sweeps first"
            )
        for ename in im.writes:
            write = doc.writes[ename]
            payload = write.do.payload
            operand_names = (
                [payload]
                if isinstance(payload, str)
                else [v for v in payload.values() if isinstance(v, str)]
                if isinstance(payload, Mapping)
                else []
            )
            for op in operand_names:
                read = doc.reads.get(op)
                if read is not None and str(read.input) != str(im.input):
                    needed.add("paired_forward")
            if write.do.mechanism == "pytorch_fn":
                needed.add("pytorch_fn_local")
            site = doc.sites[str(write.site)]
            if site.component == "attention_probs":
                needed.add("writable_attention_probs")
    saved = {entry.value for entry in doc.save}
    for rname, read in doc.reads.items():
        if rname in saved and read.dims is None:
            site = doc.sites[str(read.site)]
            if site.component == "lm_head":
                needed.add("full_logits")
    for metric in doc.metrics.values():
        if metric.kind in _FULL_VOCAB_METRICS and _metric_read_obliges_full_projection(
            doc, metric
        ):
            needed.add("full_logits")
    for read in doc.reads.values():
        if generated_budget(doc, read.pos) is not None:
            # a continuation to address means the backend must decode one
            needed.add("generate")
            break
    return frozenset(needed)


@dataclasses.dataclass(frozen=True)
class ExecutionRequest:
    """Everything a backend needs to run one document: the concrete points
    (raw trees, artifact fields resolved), their canonical forms and
    digests, coordinates per point, the resolution environment, and where
    outputs land."""

    points: tuple[Mapping[str, Any], ...]
    canonical: tuple[Mapping[str, Any], ...]
    digests: tuple[str, ...]
    coords: tuple[Mapping[str, Any], ...]
    document_digest: str
    env: ResolutionEnv
    output_dir: Path


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What an execution produced: saved files (save-manifest paths →
    absolute paths on disk) and per-point summaries for `explain`-style
    reporting."""

    files: Mapping[str, Path]
    summaries: tuple[Mapping[str, Any], ...] = ()


class Backend(abc.ABC):
    """One execution engine, described by data and entered through one
    method. Implementations own the §8 services (SiteResolver, position
    resolution, planning, mechanisms, featurizers, metrics, training, RNG,
    stamping) internally — the seam is the document, not the services."""

    #: Backend name, for routing messages and ArtifactIdentity stamping.
    name: str = "abstract"
    #: The §8 capability set this backend supports.
    capabilities: frozenset[str] = frozenset()
    #: Local backends may run ``pytorch_fn`` writes (§2.8).
    is_local: bool = False

    @abc.abstractmethod
    def execute(self, request: ExecutionRequest) -> RunResult:
        """Run every point and write everything the save manifests name."""


def requires_campaign(docs: Sequence[Document]) -> frozenset[str]:
    """The union of every point's capability needs — a heterogeneous sweep
    routes on the whole campaign, not its first point."""
    needed: frozenset[str] = frozenset()
    for doc in docs:
        needed |= requires(doc)
    return needed


def choose_backend(
    doc: Document | Sequence[Document], backends: Sequence[Backend]
) -> Backend:
    """The first backend whose capabilities cover the document's (or the
    whole campaign's) needs; the refusal message is generated from the
    missing capabilities (§8)."""
    needed = requires(doc) if isinstance(doc, Document) else requires_campaign(doc)
    shortfalls: list[str] = []
    for backend in backends:
        missing = needed - backend.capabilities
        if not missing:
            return backend
        shortfalls.append(f"{backend.name} lacks {sorted(missing)}")
    raise ValidationError(
        13,
        f"no backend supports this document: it requires {sorted(needed)}; "
        + "; ".join(shortfalls),
    )
