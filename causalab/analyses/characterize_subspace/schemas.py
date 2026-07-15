"""Typed shapes for the subspace-characterization analysis.

The judge-independence invariant lives partly in this file: ``Step1Summary``
intentionally carries projection statistics and example spans *without* any
``Significance`` field. The derive call in :mod:`judge` accepts only
``WebtextEvidence`` and ``Step1Summary`` as inputs, so a future caller
cannot pipe the provided hypothesis into the derive prompt without first
widening one of those types — at which point the runtime substring guard in
:mod:`judge` is the second line of defence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Verdict = Literal["confirmed", "refined", "disagreed", "unresolved"]


@dataclass(frozen=True)
class Span:
    """A single text span with its scalar projection value.

    ``text`` is the ±W-token *window* around the document's peak token, with
    the peak token wrapped ``<<…>>`` (see :func:`webtext._extract_window`).
    ``projection_value`` is the peak token's subspace-activation norm
    ``‖proj‖₂`` (Euclidean norm over all subspace dimensions, BOS excluded),
    not a per-doc mean. The judge prompt formatter reads both fields verbatim.
    """

    text: str
    projection_value: float


@dataclass(frozen=True)
class PeakRecord:
    """One document's peak-norm-token result.

    The peak ``k``-dim projection vector is **not** stored here — it lives in
    a parallel ``(N, k)`` tensor so this record (which is serialised to JSON
    for the figures and the cache) stays small.
    """

    projection_value: float
    peak_token_index: int
    window_text: str


@dataclass(frozen=True)
class QuantileBin:
    """Spans sampled from one quantile of the projection distribution."""

    quantile: float
    projection_range: tuple[float, float]
    spans: list[Span] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectionStats:
    """Distribution-level statistics for a set of projections."""

    n_samples: int
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class WebtextEvidence:
    """Broad-corpus evidence fed to the hypothesis-derivation call.

    Documents are scored by the non-negative subspace-activation norm of their
    peak token. Quantile bins span the full norm distribution; ``topk_spans``
    are the **strongest-activating** documents and ``bottomk_spans`` the
    **weakest-activating** (typically generic text). Because the score is a
    magnitude, direction is discarded — a bipolar subspace's opposite poles can
    both land in ``topk_spans``. ``stats`` summarises the norm distribution so
    the judge has scale context without seeing the raw projection vectors.
    """

    corpus: str
    quantile_bins: list[QuantileBin]
    topk_spans: list[Span]
    bottomk_spans: list[Span]
    stats: ProjectionStats


@dataclass(frozen=True)
class Step1Summary:
    """Phase-1-dataset summary fed alongside ``WebtextEvidence``.

    Contains projection moments and a small set of example spans so the
    judge can compare the phase-1 distribution against the broad corpus.
    **Does not carry any field from** ``Significance`` — this is the type-
    level half of the judge-independence invariant. Anything user-supplied
    about the subspace's purported meaning lives in ``Significance`` and
    only reaches ``reconcile_hypotheses``.
    """

    dataset_name: str
    stats: ProjectionStats
    example_spans: list[Span]


@dataclass(frozen=True)
class Significance:
    """User-supplied description of what the subspace is meant to track.

    Each field is optional; an empty ``Significance`` (all fields ``None``)
    is valid and triggers the description-absent reproduction path.
    """

    hypothesis_text: str | None = None
    figure_path: str | None = None
    topology_description: str | None = None

    def non_empty_values(self) -> list[str]:
        """Return all populated string-valued fields.

        Used by the runtime substring guard to construct the set of needles
        that must not appear in a derive-call prompt.
        """
        out: list[str] = []
        for v in (self.hypothesis_text, self.figure_path, self.topology_description):
            if v:
                out.append(v)
        return out


@dataclass(frozen=True)
class JudgeHypothesis:
    """Structured output of ``derive_hypothesis``.

    ``hypothesis_text`` is the natural-language claim about the semantic
    axis the subspace tracks. ``supporting_spans`` are short excerpts the
    judge cited as evidence. ``raw_response`` is the un-parsed LLM string,
    retained for audit and downstream debugging.
    """

    hypothesis_text: str
    confidence: float
    supporting_spans: list[str]
    framing: str
    model: str
    raw_response: str


@dataclass(frozen=True)
class ReconciliationIteration:
    """One iteration of the reconcile loop."""

    framing: str
    verdict: Verdict
    rationale: str
    raw_response: str


@dataclass(frozen=True)
class ReconciliationResult:
    """Final reconcile-call output, including the per-iteration trace.

    ``verdict`` collapses the iteration sequence to a single outcome:
    ``confirmed`` if any iteration confirmed, the last iteration's verdict
    if the loop exhausted ``max_iterations``, and ``unresolved`` as a
    catch-all for malformed responses.
    """

    verdict: Verdict
    judge_hypothesis: str
    provided_hypothesis: str | None
    rationale: str
    iterations: list[ReconciliationIteration]
    final_framing: str
    model: str
