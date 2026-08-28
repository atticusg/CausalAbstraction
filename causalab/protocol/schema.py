"""The intervention-protocol object model and its strict parser.

This module is the authoring surface of ``docs/intervention_protocol.md``:
it turns a raw JSON/YAML mapping into a typed, frozen
:class:`Document` — or refuses with a structured
:class:`~causalab.protocol.errors.ParseError` /
:class:`~causalab.protocol.errors.ValidationError`. Everything here is
backend-free and torch-free: sites, positions, featurizers and writes are
pure data records; a backend interprets them (spec §8).

Parsing owns the *shape* rules of the spec:

* strict keys — an unknown field anywhere is an error with suggestions
  (§5.1); closed enums reject with suggestions; derived fields (§6) may not
  be authored;
* section order (§1) and the ``save``-last rule;
* sugar — a bare int ``pos`` means ``{"index": n}`` and the bare string
  ``"all"`` means ``{"all": true}`` (§2.3), ``neural_model`` is an alias of
  ``model`` (§2.1); sugar is expanded here, so the object model only ever
  holds the canonical spelling;
* the two value wrappers — ``{"sweep": …}`` (§3) and
  ``{"artifact": …, "key": …}`` (§1) — are accepted anywhere a scalar-,
  list- or spec-typed *leaf* is expected and preserved as
  :class:`Sweep` / :class:`ArtifactRef` values; expansion and resolution
  happen in :mod:`causalab.protocol.sweep` / :mod:`causalab.protocol.resolve`.

Cross-reference and semantic checks (the §5 checklist items that need the
whole document) live in :mod:`causalab.protocol.validate`, not here.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union, get_args

from causalab.protocol.errors import ParseError, ValidationError, suggest

__all__ = [
    "ALL_POSITIONS",
    "ArtifactRef",
    "COMPONENTS",
    "Component",
    "DataRole",
    "Do",
    "Document",
    "WriteSpec",
    "FEATURIZER_KINDS",
    "FeaturizerKind",
    "FeaturizerSpec",
    "IMSpec",
    "MATCH_MODES",
    "MECHANISMS",
    "METRIC_FIELD_DEFAULTS",
    "METRIC_DOMAINS",
    "METRIC_KINDS",
    "Mechanism",
    "MetricKind",
    "MetricDomain",
    "MetricSpec",
    "DOCUMENT_TYPES",
    "MODEL_DTYPE_DEFAULT",
    "ModelRef",
    "NON_COLUMN_METRIC_FIELDS",
    "OPTIONAL_METRIC_FIELDS",
    "ParamSpec",
    "QUANT_METHODS",
    "QUANT_SCHEMES",
    "QuantizationSpec",
    "PositionSpec",
    "ReadSpec",
    "RESERVED_NAMES",
    "SECTION_ORDER",
    "SaveEntry",
    "SiteSpec",
    "TOKEN_COLUMN_METRIC_KINDS",
    "TOKEN_FORMS",
    "TOP_K_RANKINGS",
    "VOCAB_TOP_K_RANKING",
    "WHOLE_WINDOW_METRIC_KINDS",
    "TokenForm",
    "concrete_int",
    "concrete_str",
    "metric_reads_vocabulary",
    "Sweep",
    "TrainSpec",
    "parse_document",
    "load_raw",
]


# --------------------------------------------------------------------------- #
# closed vocabularies (spec §2.4, §2.5, §2.8, §2.10)
# --------------------------------------------------------------------------- #

Component = Literal[
    "embeddings",
    "block_input",
    "block_output",
    "attention_output",
    "attention_value",
    "attention_probs",
    "mlp_input",
    "mlp_output",
    "mlp_activation",
    "router_logits",
    "expert_output",
    "ln_final",
    "lm_head",
]

#: The site component vocabulary (§2.4), in spec order.
COMPONENTS: tuple[Component, ...] = get_args(Component)

#: Components that carry no ``layer`` field.
LAYERLESS_COMPONENTS: frozenset[str] = frozenset({"embeddings", "ln_final", "lm_head"})

FeaturizerKind = Literal["identity", "subspace", "pca", "sae", "standardize", "gate"]
FEATURIZER_KINDS: tuple[FeaturizerKind, ...] = get_args(FeaturizerKind)

#: Auto-declared param slots per featurizer kind (§2.5) — ``<name>.<slot>``.
FEATURIZER_SLOTS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "subspace": ("weight",),
    "pca": ("weight",),
    "sae": ("enc", "dec", "b_enc", "b_dec"),
    "standardize": ("mu", "sigma"),
    "gate": ("theta",),
}

#: Authorable choice fields per featurizer kind (§2.5) — everything else about
#: a featurizer (width, param shapes, slots) is derived and may not be
#: authored (§6). ``file_path`` (load a fitted artifact) and ``dtype`` are
#: legal on every kind; ``description`` is legal everywhere.
FEATURIZER_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset(),
    "subspace": frozenset({"k", "parametrization", "init"}),
    "pca": frozenset({"k"}),
    "sae": frozenset(),
    "standardize": frozenset(),
    "gate": frozenset(),
}

PARAMETRIZATIONS: tuple[str, ...] = ("cayley", "matrix_exp", "stiefel")

Mechanism = Literal[
    "swap",
    "add_scaled",
    "lerp",
    "affine",
    "gaussian",
    "renormalize",
    "clamp",
    "pytorch_fn",
]
#: The closed ``do`` mechanism set (§2.8).
MECHANISMS: tuple[Mechanism, ...] = get_args(Mechanism)

#: Mechanisms whose write is a delta added after the absolute write (§2.8).
ADDITIVE_MECHANISMS: frozenset[str] = frozenset({"add_scaled", "gaussian"})

MetricKind = Literal[
    "logit_diff",
    "token_logit",
    "cross_entropy",
    "kl",
    "class_probs",
    "top_k",
    "match",
    "decode",
]
METRIC_KINDS: tuple[MetricKind, ...] = get_args(MetricKind)

#: Value fields per metric kind beyond ``of`` (§2.10). ``kl.target`` names a
#: read; every other value field names a dataset column (checked at run time
#: by ``validate --data``, §2.2) — except the fields in
#: :data:`NON_COLUMN_METRIC_FIELDS`.
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "token_logit": ("token",),
    "cross_entropy": ("target",),
    "kl": ("target",),
    "class_probs": ("groups",),
    "top_k": ("k", "by"),
    "match": ("expected",),
    "decode": (),
}

#: Mandatory value fields that are *not* dataset column names: ``top_k.k`` is
#: an integer and ``top_k.by`` is a closed enum. Everything else in
#: :data:`METRIC_FIELDS` (bar ``kl.target``, a read) is checked against the
#: resolved datasets' columns.
NON_COLUMN_METRIC_FIELDS: frozenset[str] = frozenset({"k", "by"})

#: What each metric kind consumes from its read (§2.10). ``distribution``
#: kinds reduce the read's dense value at the addressed positions (the
#: vocabulary projection for an ``lm_head`` read, which is every kind but
#: ``top_k``); ``ids`` kinds consume only the tokens the decode produced. The
#: split is what lets the planner (§8) tell a text probe — which obliges no
#: vocabulary projection at all — from a scoring one, so it is a property of
#: the kind, never of the document.
MetricDomain = Literal["distribution", "ids"]
METRIC_DOMAINS: dict[str, MetricDomain] = {
    "logit_diff": "distribution",
    "token_logit": "distribution",
    "cross_entropy": "distribution",
    "kl": "distribution",
    "class_probs": "distribution",
    "top_k": "distribution",
    "match": "distribution",
    "decode": "ids",
}

#: Metric kinds that reduce the whole addressed window to one value per
#: example rather than one per position: ``decode`` joins its tokens into a
#: string, so a per-step row would be a per-character-ish lie.
WHOLE_WINDOW_METRIC_KINDS: frozenset[str] = frozenset({"decode"})


TokenForm = Literal["auto", "bare", "space_prefixed"]

#: How a metric's string answers become token ids (§2.10). ``auto`` is the
#: historical resolver — try ``" " + s`` first, fall back to ``s`` — which is
#: right for answers that follow a space (weekdays, names, MCQA letters) and
#: wrong for answers that do not (punctuation: gpt2 emits ``"?"`` = 30, but
#: ``" ?"`` = 5633 is also one token and wins). ``bare`` and ``space_prefixed``
#: pin one form, so a document can say which one it means.
TOKEN_FORMS: tuple[TokenForm, ...] = get_args(TokenForm)

#: Metric kinds whose value fields carry string answers that must resolve to
#: token ids — the kinds ``token_form`` applies to. ``kl`` compares two reads'
#: distributions and ``top_k`` reports indices it found (decoding them only
#: when the read happens to tap ``lm_head``), so neither resolves an authored
#: string and neither accepts the key.
TOKEN_COLUMN_METRIC_KINDS: frozenset[str] = frozenset(
    {"logit_diff", "token_logit", "cross_entropy", "class_probs", "match"}
)

#: Optional value fields per metric kind (§2.10). An omitted optional field is
#: materialized to its default in the canonical form (§7), so an authored
#: default and an omitted one digest identically — the same treatment
#: ``train.optimizer`` defaults get.
OPTIONAL_METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "match": ("mode",),
}

#: Defaults for the optional fields above, by ``(kind, field)``.
METRIC_FIELD_DEFAULTS: dict[tuple[str, str], Any] = {
    ("match", "mode"): "exact",
}

#: How ``match`` compares the argmax token to an expected form (§2.10):
#: ``exact`` needs the form to be one token; ``first_token`` credits the
#: form's first token, which is what "prefix" means with logits at one
#: position (a multi-token answer's first piece).
MATCH_MODES: tuple[str, ...] = ("exact", "first_token")

#: How ``top_k`` ranks the entries of its read's last axis (§2.10). Mandatory,
#: because the right answer depends on what the axis *is* and only the author
#: knows: a vocabulary projection has no meaningful negative entries, while a
#: residual stream and a signed feature code do — ranking a 100k-latent SAE
#: code by signed value and by magnitude give different top-k sets, and
#: silently picking one for the author is how a plot ends up meaning something
#: other than its caption.
#:
#: * ``value`` — the k largest signed entries. Any read.
#: * ``abs_value`` — the k largest by ``|x|``; the reported value stays signed.
#:   Any read.
#: * ``prob`` — softmax the last axis, then take the k largest probabilities.
#:   Legal **only** on an ``lm_head`` read: a softmax across neurons or SAE
#:   latents normalizes over an axis that is not an event space, and the
#:   resulting "probabilities" would mean nothing (validation refuses it).
TOP_K_RANKINGS: tuple[str, ...] = ("value", "abs_value", "prob")

#: The ``top_k.by`` ranking that normalizes, and so is vocabulary-only.
VOCAB_TOP_K_RANKING: str = "prob"

#: The bare-string spelling of an all-positions spec (§2.3 sugar). Reserved as
#: a name so a ``positions`` entry can never shadow the sugar.
ALL_POSITIONS: str = "all"

#: Names no section may declare (§1): the input roles, the un-intervened
#: model, the indexed-counterfactual family (checked by prefix for
#: ``counterfactual[``), and the all-positions sugar.
RESERVED_NAMES: frozenset[str] = frozenset(
    {"base", "counterfactual", "original", ALL_POSITIONS}
)

#: Top-level sections in mandatory order (§1). ``neural_model`` is accepted
#: at position 3 as an alias of ``model`` and canonicalizes away.
SECTION_ORDER: tuple[str, ...] = (
    "version",
    "type",
    "description",
    "model",
    "data",
    "positions",
    "sites",
    "featurizers",
    "params",
    "reads",
    "writes",
    "intervened_models",
    "metrics",
    "train",
    "save",
)

REQUIRED_SECTIONS: frozenset[str] = frozenset(
    {"version", "model", "data", "sites", "reads", "save"}
)

#: The name-bearing sections sharing one global namespace (§1: sections 6–13).
NAMED_SECTIONS: tuple[str, ...] = (
    "positions",
    "sites",
    "featurizers",
    "params",
    "reads",
    "writes",
    "intervened_models",
    "metrics",
)

PRECISION_DTYPES: tuple[str, ...] = ("fp32", "bf16", "fp16")

#: The compute dtype a document runs in when it authors none (§2.1). This is
#: the value canonicalization materializes, so every canonical form names a
#: dtype and every digest covers it; ``fp32`` keeps an unauthored document
#: running exactly as it did when dtype was an execution flag.
MODEL_DTYPE_DEFAULT: str = "fp32"

#: Weight-quantization schemes (§2.1). Each names one *load-time* scheme the
#: reference backend can realize through bitsandbytes: ``int8`` is LLM.int8()
#: mixed-precision decomposition, ``nf4`` / ``fp4`` are the two 4-bit
#: quantization types. There is no bare ``int4``: bitsandbytes' 4-bit is one
#: of these two datatypes, and "int4" would not say which — the whole point
#: of putting the field in the record is that it names one realization.
#: Weights quantized *ahead of time* (GPTQ, AWQ) are a property of the
#: checkpoint, so they are named by ``model.key``/``revision``, not here.
QUANT_SCHEMES: tuple[str, ...] = ("int8", "nf4", "fp4")

#: Quantizers (§2.1). One entry in v1 — the library the reference backend
#: calls; naming it keeps a document honest when a second one appears.
QUANT_METHODS: tuple[str, ...] = ("bitsandbytes",)

#: Quantization fields that only make sense for some schemes (rule 17).
_QUANT_4BIT_FIELDS: tuple[str, ...] = ("double_quant",)
_QUANT_INT8_FIELDS: tuple[str, ...] = ("int8_threshold",)

#: The document types (§1.1). A protocol document may be written flat or split
#: into ``application`` + ``method`` halves — both are ``protocol``; ``method``
#: is a reusable method *file*, the one shape structure cannot name. ``type``
#: is authoring metadata — it declares what a file is so a typo cannot
#: masquerade as a different kind — and canonicalization drops it: the
#: canonical form is the experiment, not the file.
DOCUMENT_TYPES: tuple[str, ...] = ("protocol", "method", "workflow")

#: Optimizer field vocabulary and per-name defaults, materialized into the
#: canonical form (§7: "every default (constant LR, optimizer betas, dtypes)").
#: The vocabulary is closed like every other enum here; extending it is a
#: schema change, not a free-form pass-through.
OPTIMIZER_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "momentum",
        "clip_grad_norm",
        "schedule",
    }
)
OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    # torch.optim.AdamW defaults (torch 2.x): betas=(0.9, 0.999), eps=1e-8,
    # weight_decay=1e-2 — but the protocol default is 0.0: a regularizer is an
    # objective term here (§2.11), never an optimizer side-effect.
    "adamw": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "adam": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "sgd": {"momentum": 0.0, "weight_decay": 0.0, "schedule": "constant"},
}


# --------------------------------------------------------------------------- #
# value wrappers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Sweep:
    """An explicit sweep axis (§3): ``{"sweep": [v1, v2]}`` or
    ``{"sweep": {"range": [start, stop, step?]}}``. ``values`` holds the
    expanded value list either way (a range is expanded eagerly — it is sugar
    for the list it denotes)."""

    values: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    """An artifact-valued field (§1): one value read from a prior run's
    artifact at load. Unresolved in the authored object model; resolution
    (and the missing-artifact load error, §5.15) is
    :mod:`causalab.protocol.resolve`'s job."""

    artifact: str
    key: str


#: A leaf that may still be swept or artifact-valued in the authored model.
Leaf = Union[Any, Sweep, ArtifactRef]


def concrete_int(value: Leaf, what: str) -> int:
    """Narrow a leaf that must be concrete by now (a point document) to int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ParseError("P2", f"{what} is not a concrete integer (got {value!r})")


def concrete_str(value: Leaf, what: str) -> str:
    """Narrow a leaf that must be concrete by now (a point document) to str."""
    if isinstance(value, str):
        return value
    raise ParseError("P2", f"{what} is not a concrete string (got {value!r})")


# --------------------------------------------------------------------------- #
# section records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class QuantizationSpec:
    """§2.1 — how the weights are quantized at load.

    Quantization changes what the network computes, so it is document
    vocabulary, not an execution flag: two runs of the same protocol at
    ``nf4`` and at ``bf16`` are two experiments, and their digests say so.
    Every field that moves a number is here — the scheme, the quantizer, the
    dtype the dequantized matmuls run in, and the scheme's own knobs.
    """

    scheme: Leaf
    method: Leaf = "bitsandbytes"
    compute_dtype: Leaf | None = None
    double_quant: Leaf | None = None
    int8_threshold: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """§2.1 — the network as a name, plus how it is realized numerically.

    ``revision`` defaults to ``"main"`` and ``dtype`` to
    :data:`MODEL_DTYPE_DEFAULT` (both materialized by canonicalization when
    unauthored, §7).
    """

    key: Leaf
    revision: Leaf = "main"
    dtype: Leaf | None = None
    quantization: QuantizationSpec | None = None


@dataclasses.dataclass(frozen=True)
class DataRole:
    """§2.2 — one input-row column: a dataset ref (local path or HF key, no
    digest — the content digest is stamped at load) plus the column selector
    ``field`` (``[j]`` indexes list-valued columns)."""

    dataset: Leaf
    field: Leaf


@dataclasses.dataclass(frozen=True)
class PositionSpec:
    """§2.3 — a token-position spec. Exactly one of ``index`` / ``span`` /
    ``variable`` / ``column`` / ``all`` is set; ``scope`` /
    ``relative_to`` name an anchor (a prompt variable, or a dataset
    column when ``anchor_source`` is ``"column"``), only modify
    ``index``/``span``, and are mutually exclusive. ``all`` selects every
    content token of the row and takes no modifiers. Positions are never
    resolved to integers in the document — resolution is a backend
    service against a ``PositionFrame`` (§2.3, §8).

    ``generated`` is a **frame selector**, not an anchor: it says the
    anchor resolves inside the row's greedy continuation instead of its
    prompt, and it carries the decode budget (``{"max_new_tokens": n}``).
    The anchor vocabulary is unchanged inside that frame."""

    index: Leaf | None = None
    span: Leaf | None = None
    variable: Leaf | None = None
    column: Leaf | None = None
    all: Leaf | None = None
    scope: Leaf | None = None
    relative_to: Leaf | None = None
    #: The continuation frame and its decode budget (§2.3). ``None`` is
    #: the prompt frame — where every position lived before generation.
    generated: Mapping[str, Leaf] | None = None
    #: Where ``scope``/``relative_to`` resolve from: ``"variable"`` (the
    #: role's prompt variables) or ``"column"`` (a top-level row column).
    #: Not authored on its own — it comes from the anchor's spelling.
    anchor_source: str = "variable"


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """§2.4 — a named activation address: pure data, no behavior."""

    component: Leaf
    layer: Leaf | None = None
    head: Leaf | None = None
    expert: Leaf | None = None
    stream: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class FeaturizerSpec:
    """§2.5 — a named feature-space map. Only choices are authored; widths
    and param shapes derive from (model, site). ``file_path`` loads a fitted
    artifact (its ``ArtifactIdentity`` is checked; a loaded featurizer may
    not be trained)."""

    kind: Leaf = "identity"
    k: Leaf | None = None
    parametrization: Leaf | None = None
    init: Leaf | None = None
    dtype: Leaf | None = None
    file_path: Leaf | None = None
    entry: Any = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """§2.6 — a free tensor owned by no featurizer: either a loaded constant
    (``file_path``, optionally narrowed to one bundle entry by ``entry``) or
    a trainable free tensor (``shape`` + ``init``, which must then appear in
    ``train.params``)."""

    file_path: Leaf | None = None
    entry: Any = None
    shape: Leaf | None = None
    init: Leaf | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ReadSpec:
    """§2.7 — a value producer. ``pos`` is a positions-table name or an
    inline :class:`PositionSpec` (int sugar already expanded). ``featurizer``
    is a single name or a left-to-right composition tuple."""

    site: Leaf
    pos: Leaf
    model: Leaf
    input: Leaf
    featurizer: Leaf | None = None
    dims: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class Do:
    """§2.8 — one mechanism from the closed set. ``payload`` holds the
    mechanism's single value exactly as authored (operand name / literal
    scalar for ``swap``; the option mapping for the structured mechanisms;
    ``True`` for ``renormalize``)."""

    mechanism: Leaf
    payload: Any


@dataclasses.dataclass(frozen=True)
class WriteSpec:
    """§2.8 — an inert effect definition: no model, no input, no conditions;
    it executes inside every intervened model that lists it."""

    site: Leaf
    pos: Leaf
    do: Do
    featurizer: Leaf | None = None
    dims: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class IMSpec:
    """§2.9 — an intervened model ℒ_{b∪𝕀}: a mandatory input role plus the
    writes in force (unordered; canonical form sorts)."""

    input: Leaf
    writes: tuple[str, ...] | Sweep | ArtifactRef


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """§2.10 — a closed-vocabulary reduction over one read (``of``) plus
    dataset columns. ``fields`` holds the kind's extra value fields.
    ``token_form`` says how this metric's string answers become token ids
    (``TOKEN_FORMS``); the ``auto`` default is the historical resolver, so an
    unauthored metric behaves exactly as before. ``top_k`` additionally
    carries a mandatory ``by`` (``TOP_K_RANKINGS``) in ``fields``, because a
    top-k over a signed feature code and one over a vocabulary projection are
    different questions."""

    kind: Leaf
    of: Leaf
    fields: Mapping[str, Leaf]
    token_form: Leaf = "auto"


@dataclasses.dataclass(frozen=True)
class TrainSpec:
    """§2.11 — the fit, declared. ``objective`` is a tuple of
    ``(weight, metric-name-or-regularizer)`` terms; a regularizer is
    ``("l1"|"l2", target-name)``."""

    objective: tuple[tuple[Leaf, Leaf | tuple[str, str]], ...]
    params: tuple[str, ...]
    optimizer: Mapping[str, Leaf]
    steps: Mapping[str, Leaf]
    batch: Mapping[str, Leaf]
    anneal: Mapping[str, Leaf] | None = None
    precision: Mapping[str, Leaf] | None = None
    eval: Mapping[str, Any] | None = None
    early_stop: Mapping[str, Leaf] | None = None
    checkpoint: Mapping[str, Leaf] | None = None
    seed: Leaf = 0


@dataclasses.dataclass(frozen=True)
class SaveEntry:
    """§2.12 — one manifest entry. Read/metric entries carry
    ``model``/``input``; trained-featurizer entries carry ``site``. The
    restated binding is cross-checked at validation, never trusted.
    ``reduce`` (reads only) saves a statistic over the gathered rows
    instead of the rows themselves (§2.12)."""

    value: str
    file_path: str
    model: str | None = None
    input: str | None = None
    site: str | None = None
    reduce: str | None = None


@dataclasses.dataclass(frozen=True)
class Document:
    """One parsed intervention-protocol document (authored form, sugar
    expanded, wrappers preserved). Section tables preserve authoring order;
    ``raw`` keeps the normalized mapping the parser consumed — the substrate
    sweep expansion and canonicalization operate on."""

    version: str
    model: ModelRef
    data: Mapping[str, DataRole | tuple[DataRole, ...]]
    sites: Mapping[str, SiteSpec]
    reads: Mapping[str, ReadSpec]
    save: tuple[SaveEntry, ...]
    description: str | None = None
    positions: Mapping[str, PositionSpec | Sweep | ArtifactRef] = dataclasses.field(
        default_factory=dict
    )
    featurizers: Mapping[str, FeaturizerSpec] = dataclasses.field(default_factory=dict)
    params: Mapping[str, ParamSpec] = dataclasses.field(default_factory=dict)
    writes: Mapping[str, WriteSpec] = dataclasses.field(default_factory=dict)
    intervened_models: Mapping[str, IMSpec] = dataclasses.field(default_factory=dict)
    metrics: Mapping[str, MetricSpec] = dataclasses.field(default_factory=dict)
    train: TrainSpec | None = None
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def named_entries(self) -> dict[str, str]:
        """Every declared name → the section that declares it. Duplicates are
        a validation concern (§5.3); the parser reports the first section."""
        seen: dict[str, str] = {}
        for section in NAMED_SECTIONS:
            table: Mapping[str, Any] = getattr(self, section)
            for name in table:
                seen.setdefault(name, section)
        return seen


def metric_reads_vocabulary(doc: Document, metric: MetricSpec) -> bool:
    """Whether the last axis of ``metric``'s read is the vocabulary.

    True exactly when the ``of`` read taps ``lm_head`` **and hands the
    projection on unchanged**. The site alone is not enough: a ``featurizer``
    re-expresses the value in its own latents and ``dims`` re-indexes a slice
    of it, so under either one the read's entries are no longer token ids — a
    softmax over them normalizes an axis that is not the vocabulary, and
    decoding index *j* as token *j* names the wrong token.

    Three places need the same answer and must not disagree: capability
    derivation (does this document oblige ``full_logits``?, §8), validation
    (may this ``top_k`` normalize, may a token-space kind bind here?, §2.10)
    and the reduction itself (are these indices token ids worth decoding?). A
    dangling ``of`` is validation's error to report, so it answers False here
    rather than raising."""
    read = doc.reads.get(str(metric.of))
    if read is None:
        return False
    if read.featurizer is not None or read.dims is not None:
        return False
    site = doc.sites.get(str(read.site))
    return site is not None and site.component == "lm_head"


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ParseError("P2", f"duplicate key {key!r} in one object")
        out[key] = value
    return out


def load_raw(text: str) -> dict[str, Any]:
    """Parse strict JSON text into an order-preserving mapping.

    YAML is accepted at the CLI surface (it parses to the same object model);
    this function is the JSON path and the normative behavior: duplicate keys
    and non-object top levels are errors, NaN/Infinity are rejected.
    """
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ParseError:
        raise
    except json.JSONDecodeError as err:
        raise ParseError("P1", f"not valid JSON: {err}") from err
    if not isinstance(raw, dict):
        raise ParseError("P1", "the top level must be a JSON object")
    return raw


def _reject_constant(name: str) -> Any:
    raise ParseError("P1", f"non-finite JSON constant {name!r} is not allowed")


# --------------------------------------------------------------------------- #
# parse helpers
# --------------------------------------------------------------------------- #


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(
            "P2", f"expected an object, got {type(value).__name__}", path=path
        )
    return value


def _check_keys(mapping: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_set = set(allowed)
    for key in mapping:
        if key not in allowed_set:
            raise ParseError(
                "P3",
                f"unknown key {key!r}{suggest(key, allowed_set)}",
                path=path,
            )


def _enum(value: Any, options: Sequence[str], path: str) -> str:
    if not isinstance(value, str) or value not in options:
        raise ParseError(
            "P4",
            f"{value!r} is not one of {list(options)}"
            + (suggest(value, options) if isinstance(value, str) else ""),
            path=path,
        )
    return value


def _wrapped(
    value: Any,
    elem: Callable[[Any, str], Any],
    path: str,
    *,
    allow_sweep: bool = True,
) -> Any:
    """Parse a leaf that may be a ``{"sweep": …}`` wrapper around
    ``elem``-typed values (§3). Artifact references (§1) resolve *before*
    the parse gate (loader.load), so one reaching the parser is a loader
    misuse, not an authoring surface."""
    if isinstance(value, dict) and "sweep" in value:
        if not allow_sweep:
            raise ValidationError(14, "a sweep wrapper is not allowed here", path=path)
        _check_keys(value, ("sweep",), path)
        return _parse_sweep(value["sweep"], elem, path)
    if isinstance(value, dict) and isinstance(value.get("artifact"), str):
        raise ParseError(
            "P2",
            "unresolved artifact reference reached the parser — load through "
            "causalab.protocol.loader.load, which resolves artifact fields first",
            path=path,
        )
    return elem(value, path)


def _parse_sweep(spec: Any, elem: Callable[[Any, str], Any], path: str) -> Sweep:
    if isinstance(spec, dict):
        _check_keys(spec, ("range",), f"{path}.sweep")
        rng = spec.get("range")
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                14,
                "sweep range must be [start, stop] or [start, stop, step] of integers",
                path=f"{path}.sweep",
            )
        start, stop = rng[0], rng[1]
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                14, "sweep range step must be non-zero", path=f"{path}.sweep"
            )
        if len(range(start, stop, step)) > 1_000_000:  # O(1); before materializing
            raise ValidationError(
                14,
                "sweep range denotes over 1,000,000 values — refuse before "
                "materializing (§5.14)",
                path=f"{path}.sweep",
            )
        values = list(range(start, stop, step))
    elif isinstance(spec, list):
        values = spec
    else:
        raise ValidationError(
            14,
            f"a sweep wrapper takes a list or a range object, got {type(spec).__name__}",
            path=f"{path}.sweep",
        )
    if not values:
        raise ValidationError(
            14, "a sweep axis must have at least one value", path=path
        )
    return Sweep(
        values=tuple(elem(v, f"{path}.sweep[{i}]") for i, v in enumerate(values))
    )


#: Reductions a ``save`` entry may apply to a read (§2.12). Closed, and
#: deliberately one entry wide: ``mean`` is what mean-ablation needs, and a
#: second kind should arrive with the consumer that needs it.
SAVE_REDUCTIONS: tuple[str, ...] = ("mean",)


def _entry_selector(
    value: Any, path: str, *, allow_slot: bool = False
) -> dict[str, Any]:
    """§2.5/§2.6 ``entry``: a mapping of coordinate name to scalar value,
    naming one entry inside a loaded bundle
    (:mod:`causalab.protocol.bundles`). Names are the coordinate names as
    they appear in the producer's keys (``k``, ``seed``,
    ``target.layer``) — not full axis ids, which the consuming document has
    no reason to know."""
    obj = _require_mapping(value, path)
    if not obj:
        raise ParseError(
            "P2", "an 'entry' selector names at least one coordinate", path=path
        )
    selector: dict[str, Any] = {}
    for name, coord in obj.items():
        if name == "slot":
            if not allow_slot:
                raise ParseError(
                    "P2",
                    "a featurizer bundle's slots are fixed by its kind — "
                    "'slot' selects only inside a params bundle",
                    path=f"{path}.slot",
                )
            if not isinstance(coord, str):
                raise ParseError("P2", "'slot' names one tensor", path=f"{path}.slot")
            selector[name] = coord
            continue
        if isinstance(coord, (dict, list)):
            raise ParseError(
                "P2",
                f"entry coordinate {name!r} must be a scalar — a bundle key "
                "records one value per coordinate",
                path=f"{path}.{name}",
            )
        selector[name] = coord
    return selector


def _scalar_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ParseError(
            "P2", f"expected a string, got {type(value).__name__}", path=path
        )
    return value


def _scalar_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ParseError(
            "P2", f"expected an integer, got {type(value).__name__}", path=path
        )
    return value


def _scalar_number(value: Any, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(
            "P2", f"expected a number, got {type(value).__name__}", path=path
        )
    return value


def _scalar_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ParseError("P2", f"expected true or false (got {value!r})", path=path)
    return value


def _int_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise ParseError("P2", "expected a list of integers", path=path)
    return tuple(value)


def _str_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ParseError("P2", "expected a list of strings", path=path)
    return tuple(value)


def _any_leaf(value: Any, path: str) -> Any:
    return value


# --------------------------------------------------------------------------- #
# section parsers
# --------------------------------------------------------------------------- #


def _parse_model(raw: Any, path: str) -> ModelRef:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("key", "revision", "dtype", "quantization"), path)
    if "key" not in obj:
        raise ParseError("P2", "model needs a 'key'", path=path)
    key = _wrapped(obj["key"], _scalar_str, f"{path}.key")
    revision = (
        _wrapped(obj["revision"], _scalar_str, f"{path}.revision")
        if "revision" in obj
        else "main"
    )
    dtype = (
        _wrapped(
            obj["dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.dtype",
        )
        if "dtype" in obj
        else None
    )
    quantization = (
        _parse_quantization(obj["quantization"], f"{path}.quantization")
        if "quantization" in obj
        else None
    )
    return ModelRef(key=key, revision=revision, dtype=dtype, quantization=quantization)


def _parse_quantization(raw: Any, path: str) -> QuantizationSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        ("scheme", "method", "compute_dtype", *_QUANT_4BIT_FIELDS, *_QUANT_INT8_FIELDS),
        path,
    )
    if "scheme" not in obj:
        raise ParseError(
            "P2",
            f"quantization needs a 'scheme' — one of {list(QUANT_SCHEMES)}",
            path=path,
        )
    return QuantizationSpec(
        scheme=_wrapped(
            obj["scheme"], lambda v, p: _enum(v, QUANT_SCHEMES, p), f"{path}.scheme"
        ),
        method=_wrapped(
            obj["method"], lambda v, p: _enum(v, QUANT_METHODS, p), f"{path}.method"
        )
        if "method" in obj
        else "bitsandbytes",
        compute_dtype=_wrapped(
            obj["compute_dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.compute_dtype",
        )
        if "compute_dtype" in obj
        else None,
        double_quant=_wrapped(obj["double_quant"], _scalar_bool, f"{path}.double_quant")
        if "double_quant" in obj
        else None,
        int8_threshold=_wrapped(
            obj["int8_threshold"], _scalar_number, f"{path}.int8_threshold"
        )
        if "int8_threshold" in obj
        else None,
    )


def _parse_data_role(raw: Any, path: str) -> DataRole:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("dataset", "field"), path)
    for field in ("dataset", "field"):
        if field not in obj:
            raise ParseError("P2", f"data role needs a {field!r}", path=path)
    return DataRole(
        dataset=_wrapped(obj["dataset"], _scalar_str, f"{path}.dataset"),
        field=_wrapped(obj["field"], _scalar_str, f"{path}.field"),
    )


def _parse_data(raw: Any, path: str) -> dict[str, DataRole | tuple[DataRole, ...]]:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("base", "counterfactual"), path)
    if "base" not in obj:
        raise ParseError("P2", "data needs a 'base' role", path=path)
    out: dict[str, DataRole | tuple[DataRole, ...]] = {
        "base": _parse_data_role(obj["base"], f"{path}.base")
    }
    if "counterfactual" in obj:
        cf = obj["counterfactual"]
        if isinstance(cf, list):
            out["counterfactual"] = tuple(
                _parse_data_role(s, f"{path}.counterfactual[{j}]")
                for j, s in enumerate(cf)
            )
        else:
            out["counterfactual"] = _parse_data_role(cf, f"{path}.counterfactual")
    return out


def _parse_position_spec(raw: Any, path: str) -> PositionSpec:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return PositionSpec(index=raw)  # §6.1 int sugar
    if raw == ALL_POSITIONS:
        return PositionSpec(all=True)  # §6.1 "all" sugar
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        (
            "index",
            "span",
            "variable",
            "column",
            "all",
            "scope",
            "relative_to",
            "generated",
        ),
        path,
    )
    anchors = [k for k in ("index", "span", "variable", "column", "all") if k in obj]
    if len(anchors) != 1:
        raise ParseError(
            "P2",
            "a position spec needs exactly one of "
            f"index/span/variable/column/all, got {anchors}",
            path=path,
        )
    if "all" in obj and obj["all"] is not True:
        raise ParseError(
            "P2",
            f'all is the flag {{"all": true}} — got {obj["all"]!r}; there is no '
            "other all-positions selection to spell",
            path=path,
        )
    index = (
        _wrapped(obj["index"], _scalar_int, f"{path}.index") if "index" in obj else None
    )
    span = _wrapped(obj["span"], _parse_span, f"{path}.span") if "span" in obj else None
    variable = (
        _wrapped(obj["variable"], _scalar_str, f"{path}.variable")
        if "variable" in obj
        else None
    )
    column = (
        _wrapped(obj["column"], _scalar_str, f"{path}.column")
        if "column" in obj
        else None
    )
    every = True if "all" in obj else None
    scope_ref = (
        _wrapped(obj["scope"], _parse_anchor_ref, f"{path}.scope")
        if "scope" in obj
        else None
    )
    relative_ref = (
        _wrapped(obj["relative_to"], _parse_anchor_ref, f"{path}.relative_to")
        if "relative_to" in obj
        else None
    )
    anchor_ref = scope_ref if scope_ref is not None else relative_ref
    anchor_source = anchor_ref[0] if anchor_ref is not None else "variable"
    scope = scope_ref[1] if scope_ref is not None else None
    relative_to = relative_ref[1] if relative_ref is not None else None
    if (scope is not None or relative_to is not None) and (
        variable is not None or column is not None or every is not None
    ):
        raise ParseError(
            "P2",
            "scope/relative_to modify an index or span, not a variable/column/all spec",
            path=path,
        )
    if scope is not None and relative_to is not None:
        raise ParseError(
            "P2", "scope and relative_to are mutually exclusive", path=path
        )
    generated = (
        _parse_generated(obj["generated"], f"{path}.generated")
        if "generated" in obj
        else None
    )
    if generated is not None:
        # The continuation frame carries no prompt-frame notions: a `column`
        # holds a substring of the *input* text, and scope/relative_to anchor
        # on a prompt variable's token run. Both are meaningless in a frame
        # the prompt does not contain (§2.3).
        offenders = [
            key
            for key, present in (
                ("column", column is not None),
                ("scope", scope is not None),
                ("relative_to", relative_to is not None),
            )
            if present
        ]
        if offenders:
            raise ParseError(
                "P2",
                f"{offenders} resolve against the prompt, so they cannot combine "
                "with 'generated' — anchor inside the continuation with "
                "index/span/variable/all instead",
                path=path,
            )
    if isinstance(span, tuple):
        lo, hi = span
        if scope is None and (lo < 0 or hi <= lo):
            raise ParseError(
                "P2",
                f"span [{lo}, {hi}) is not a forward window — unscoped spans are "
                "content-frame, non-negative, non-empty",
                path=path,
            )
        if (
            scope is not None
            and (lo < 0) == (hi < 0 or hi == 0 and lo < 0)
            and lo >= hi
        ):
            raise ParseError(
                "P2", f"scoped span [{lo}, {hi}) is statically empty", path=path
            )
    return PositionSpec(
        index=index,
        span=span,
        variable=variable,
        column=column,
        all=every,
        scope=scope,
        relative_to=relative_to,
        anchor_source=anchor_source,
        generated=generated,
    )


def _parse_generated(value: Any, path: str) -> dict[str, Any]:
    """§2.3 — the continuation frame selector: ``{"max_new_tokens": n}``.

    A mapping rather than a bare int on purpose: stopping conditions
    (``stop``, ``min_new_tokens``) join this object later without any
    ambiguity about what a bare number would have meant.
    """
    obj = _require_mapping(value, path)
    _check_keys(obj, ("max_new_tokens",), path)
    if "max_new_tokens" not in obj:
        raise ParseError(
            "P2", "generated needs 'max_new_tokens' — the decode budget", path=path
        )
    budget = _wrapped(obj["max_new_tokens"], _scalar_int, f"{path}.max_new_tokens")
    if isinstance(budget, int) and budget < 1:
        raise ParseError(
            "P2",
            f"max_new_tokens is {budget} — a continuation frame needs at least "
            "one generated token",
            path=f"{path}.max_new_tokens",
        )
    return {"max_new_tokens": budget}


def _parse_span(value: Any, path: str) -> tuple[int, int]:
    ints = _int_list(value, path)
    if len(ints) != 2:
        raise ParseError("P2", "a span is [a, b) — exactly two integers", path=path)
    return (ints[0], ints[1])


def _parse_anchor_ref(value: Any, path: str) -> tuple[str, str]:
    """A ``scope``/``relative_to`` anchor: ``(source, name)`` where source is
    ``"variable"`` (per-role prompt variable) or ``"column"`` (a top-level row
    column) — §2.3."""
    obj = _require_mapping(value, path)
    _check_keys(obj, ("variable", "column"), path)
    named = [key for key in ("variable", "column") if key in obj]
    if len(named) != 1 or not isinstance(obj[named[0]], str):
        raise ParseError(
            "P2", 'expected {"variable": "<name>"} or {"column": "<name>"}', path=path
        )
    return named[0], obj[named[0]]


def _parse_positions(
    raw: Any, path: str
) -> dict[str, PositionSpec | Sweep | ArtifactRef]:
    obj = _require_mapping(raw, path)
    return {
        name: _wrapped(value, _parse_position_spec, f"{path}.{name}")
        for name, value in obj.items()
    }


def _parse_site(raw: Any, path: str) -> SiteSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("component", "layer", "head", "expert", "stream"), path)
    if "component" not in obj:
        raise ParseError("P2", "a site needs a 'component'", path=path)
    component = _wrapped(
        obj["component"],
        lambda v, p: _enum(v, COMPONENTS, p),
        f"{path}.component",
    )
    layer = (
        _wrapped(obj["layer"], _scalar_int, f"{path}.layer") if "layer" in obj else None
    )
    if isinstance(component, str):  # un-swept: layer presence is checkable now
        if component in LAYERLESS_COMPONENTS and layer is not None:
            raise ParseError("P2", f"{component} is layer-less", path=path)
        if component not in LAYERLESS_COMPONENTS and layer is None:
            raise ParseError("P2", f"{component} needs a 'layer'", path=path)
    return SiteSpec(
        component=component,
        layer=layer,
        head=_wrapped(obj["head"], _scalar_int, f"{path}.head")
        if "head" in obj
        else None,
        expert=_wrapped(obj["expert"], _scalar_int, f"{path}.expert")
        if "expert" in obj
        else None,
        stream=_wrapped(obj["stream"], _scalar_int, f"{path}.stream")
        if "stream" in obj
        else None,
    )


def _parse_sites(raw: Any, path: str) -> dict[str, SiteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_site(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_featurizer(raw: Any, path: str) -> FeaturizerSpec:
    obj = _require_mapping(raw, path)
    kind_raw = obj.get("kind", "identity")
    kind = _wrapped(
        kind_raw,
        lambda v, p: _enum(v, FEATURIZER_KINDS, p),
        f"{path}.kind",
    )
    kind_key = kind if isinstance(kind, str) else "identity"
    allowed = {"kind", "file_path", "entry", "dtype", "description"} | set(
        FEATURIZER_FIELDS.get(kind_key, frozenset())
    )
    _check_keys(obj, allowed, path)
    if "entry" in obj and "file_path" not in obj:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    parametrization = None
    if "parametrization" in obj:
        parametrization = _wrapped(
            obj["parametrization"],
            lambda v, p: _enum(v, PARAMETRIZATIONS, p),
            f"{path}.parametrization",
        )
    return FeaturizerSpec(
        kind=kind,
        k=_wrapped(obj["k"], _scalar_int, f"{path}.k") if "k" in obj else None,
        parametrization=parametrization,
        init=_wrapped(obj["init"], _scalar_str, f"{path}.init")
        if "init" in obj
        else None,
        dtype=_wrapped(
            obj["dtype"], lambda v, p: _enum(v, PRECISION_DTYPES, p), f"{path}.dtype"
        )
        if "dtype" in obj
        else None,
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if "file_path" in obj
        else None,
        entry=_wrapped(obj["entry"], _entry_selector, f"{path}.entry")
        if "entry" in obj
        else None,
        description=obj.get("description"),
    )


def _parse_featurizers(raw: Any, path: str) -> dict[str, FeaturizerSpec]:
    obj = _require_mapping(raw, path)
    return {
        name: _parse_featurizer(value, f"{path}.{name}") for name, value in obj.items()
    }


def _parse_param(raw: Any, path: str) -> ParamSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("file_path", "entry", "shape", "init", "description"), path)
    loaded = "file_path" in obj
    trainable = "shape" in obj or "init" in obj
    if "entry" in obj and not loaded:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    if loaded == trainable:
        raise ParseError(
            "P2",
            "a params entry is either loaded (file_path) or trainable (shape + init)",
            path=path,
        )
    if trainable and not ("shape" in obj and "init" in obj):
        raise ParseError(
            "P2", "a trainable params entry needs both shape and init", path=path
        )
    return ParamSpec(
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if loaded
        else None,
        entry=_wrapped(
            obj["entry"],
            lambda v, p: _entry_selector(v, p, allow_slot=True),
            f"{path}.entry",
        )
        if "entry" in obj
        else None,
        shape=_wrapped(obj["shape"], _int_list, f"{path}.shape")
        if "shape" in obj
        else None,
        init=_wrapped(obj["init"], _scalar_str, f"{path}.init")
        if "init" in obj
        else None,
        description=obj.get("description"),
    )


def _parse_params(raw: Any, path: str) -> dict[str, ParamSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_param(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_pos_field(value: Any, path: str) -> Any:
    """A read/write ``pos``: a positions-table name or an inline spec. The
    bare string ``"all"`` is the all-positions sugar, never a name — it is
    reserved (§5.3), so no entry can be declared under it."""
    if isinstance(value, str) and value != ALL_POSITIONS:
        return value
    return _parse_position_spec(value, path)


def _parse_featurizer_ref(value: Any, path: str) -> Any:
    """A ``featurizer`` reference: one name or a composition list (§2.5)."""
    if isinstance(value, str):
        return value
    return _str_list(value, path)


def _parse_read(raw: Any, path: str) -> ReadSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("site", "pos", "model", "input", "featurizer", "dims"), path)
    for field in ("site", "pos", "model", "input"):
        if field not in obj:
            raise ParseError("P2", f"a read needs {field!r}", path=path)
    return ReadSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        model=_wrapped(obj["model"], _scalar_str, f"{path}.model"),
        input=_wrapped(obj["input"], _scalar_str, f"{path}.input"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
    )


def _parse_reads(raw: Any, path: str) -> dict[str, ReadSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_read(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_operand(value: Any, path: str) -> Any:
    """A write operand: a read/param name or a literal scalar (§2.8)."""
    if isinstance(value, str):
        return value
    return _scalar_number(value, path)


def _parse_do(raw: Any, path: str) -> Do:
    obj = _require_mapping(raw, path)
    if len(obj) != 1:
        raise ParseError("P2", "'do' has exactly one mechanism key", path=path)
    ((mech, payload),) = obj.items()
    if mech not in MECHANISMS:
        raise ParseError(
            "P4", f"unknown mechanism {mech!r}{suggest(mech, MECHANISMS)}", path=path
        )
    p = f"{path}.{mech}"
    if mech == "swap":
        return Do(mechanism=mech, payload=_wrapped(payload, _parse_operand, p))
    if mech in ("add_scaled", "lerp"):
        options = _require_mapping(payload, p)
        _check_keys(options, ("op", "alpha"), p)
        for field in ("op", "alpha"):
            if field not in options:
                raise ParseError("P2", f"{mech} needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "op": _wrapped(options["op"], _parse_operand, f"{p}.op"),
                "alpha": _wrapped(options["alpha"], _parse_operand, f"{p}.alpha"),
            },
        )
    if mech == "affine":
        options = _require_mapping(payload, p)
        _check_keys(options, ("A", "b"), p)
        for field in ("A", "b"):
            if field not in options:
                raise ParseError("P2", f"affine needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "A": _wrapped(options["A"], _scalar_str, f"{p}.A"),
                "b": _wrapped(options["b"], _scalar_str, f"{p}.b"),
            },
        )
    if mech == "gaussian":
        options = _require_mapping(payload, p)
        _check_keys(options, ("seed", "scale", "axis"), p)
        for field in ("seed", "scale", "axis"):
            if field not in options:
                raise ParseError("P2", f"gaussian needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "seed": _wrapped(options["seed"], _scalar_int, f"{p}.seed"),
                "scale": _wrapped(options["scale"], _scalar_number, f"{p}.scale"),
                "axis": _wrapped(
                    options["axis"],
                    lambda v, pp: _enum(v, ("tp_duplicated", "tp_split"), pp),
                    f"{p}.axis",
                ),
            },
        )
    if mech == "renormalize":
        if payload is not True:
            raise ParseError(
                "P2", 'renormalize is written {"renormalize": true}', path=p
            )
        return Do(mechanism=mech, payload=True)
    if mech == "clamp":
        options = _require_mapping(payload, p)
        _check_keys(options, ("lo", "hi"), p)
        for field in ("lo", "hi"):
            if field not in options:
                raise ParseError("P2", f"clamp needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "lo": _wrapped(options["lo"], _scalar_number, f"{p}.lo"),
                "hi": _wrapped(options["hi"], _scalar_number, f"{p}.hi"),
            },
        )
    # pytorch_fn
    options = _require_mapping(payload, p)
    _check_keys(options, ("qualname",), p)
    if "qualname" not in options:
        raise ParseError("P2", "pytorch_fn needs a 'qualname'", path=p)
    return Do(
        mechanism=mech,
        payload={
            "qualname": _wrapped(options["qualname"], _scalar_str, f"{p}.qualname")
        },
    )


def _parse_write(raw: Any, path: str) -> WriteSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("site", "pos", "featurizer", "dims", "do"), path)
    for field in ("site", "pos", "do"):
        if field not in obj:
            raise ParseError("P2", f"a write needs {field!r}", path=path)
    return WriteSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        do=_parse_do(obj["do"], f"{path}.do"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
    )


def _parse_writes(raw: Any, path: str) -> dict[str, WriteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_write(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_im(raw: Any, path: str) -> IMSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("input", "writes"), path)
    for field in ("input", "writes"):
        if field not in obj:
            raise ParseError("P2", f"an intervened_model needs {field!r}", path=path)
    return IMSpec(
        input=_wrapped(obj["input"], _scalar_str, f"{path}.input"),
        writes=_wrapped(obj["writes"], _str_list, f"{path}.writes"),
    )


def _parse_intervened_models(raw: Any, path: str) -> dict[str, IMSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_im(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_metric(raw: Any, path: str) -> MetricSpec:
    obj = _require_mapping(raw, path)
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in METRIC_KINDS:
        raise ParseError(
            "P4",
            f"unknown metric kind {kind!r}{suggest(str(kind), METRIC_KINDS)}",
            path=f"{path}.kind",
        )
    extra = METRIC_FIELDS[kind]
    takes_token_form = kind in TOKEN_COLUMN_METRIC_KINDS
    optional = OPTIONAL_METRIC_FIELDS.get(kind, ())
    _check_keys(
        obj,
        (
            "kind",
            "of",
            *extra,
            *optional,
            *(("token_form",) if takes_token_form else ()),
        ),
        path,
    )
    if "of" not in obj:
        raise ParseError("P2", "a metric needs 'of' (a read name)", path=path)
    token_form: Any = "auto"
    if "token_form" in obj:
        token_form = _wrapped(
            obj["token_form"],
            lambda v, p: _enum(v, TOKEN_FORMS, p),
            f"{path}.token_form",
            allow_sweep=False,
        )
    fields: dict[str, Any] = {}
    for field in extra:
        if field not in obj:
            hint = (
                " — the ranking rule is mandatory because the read decides it: "
                f"{VOCAB_TOP_K_RANKING!r} (softmax the vocabulary, lm_head reads "
                "only), 'value' (largest signed entries) or 'abs_value' (largest "
                "magnitude). A pre-'by' document that scored logits meant "
                f"{VOCAB_TOP_K_RANKING!r}"
                if (kind, field) == ("top_k", "by")
                else ""
            )
            raise ParseError(
                "P2", f"metric kind {kind!r} needs {field!r}{hint}", path=path
            )
        if field == "k":
            fields[field] = _wrapped(obj[field], _scalar_int, f"{path}.{field}")
        elif field == "by":
            # ranking rule, not a dataset column — and not sweepable: a sweep
            # over `by` would fork a campaign on how a plot is read rather
            # than on a research variable (same reasoning as `token_form`).
            fields[field] = _wrapped(
                obj[field],
                lambda v, p: _enum(v, TOP_K_RANKINGS, p),
                f"{path}.{field}",
                allow_sweep=False,
            )
        elif field == "groups":
            fields[field] = _wrapped(obj[field], _any_leaf, f"{path}.{field}")
        else:
            fields[field] = _wrapped(obj[field], _scalar_str, f"{path}.{field}")
    for field in optional:
        value = obj.get(field, METRIC_FIELD_DEFAULTS[(kind, field)])
        fields[field] = _wrapped(value, _scalar_str, f"{path}.{field}")
        if (kind, field) == ("match", "mode") and fields[field] not in MATCH_MODES:
            raise ParseError(
                "P4",
                f"unknown match mode {fields[field]!r} — one of "
                f"{list(MATCH_MODES)}{suggest(str(fields[field]), MATCH_MODES)}. "
                "(A task's own 'prefix' match mode is 'first_token' here: with "
                "logits at one position, a prefix is the answer's first token.)",
                path=f"{path}.{field}",
            )
    return MetricSpec(
        kind=kind,
        of=_wrapped(obj["of"], _scalar_str, f"{path}.of"),
        fields=fields,
        token_form=token_form,
    )


def _parse_metrics(raw: Any, path: str) -> dict[str, MetricSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_metric(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_objective_term(value: Any, path: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ParseError(
            "P2", "an objective term is [weight, metric-or-regularizer]", path=path
        )
    weight = _wrapped(value[0], _scalar_number, f"{path}[0]")
    target_raw = value[1]
    if isinstance(target_raw, str):
        return (weight, target_raw)
    reg = _require_mapping(target_raw, f"{path}[1]")
    if len(reg) != 1 or next(iter(reg)) not in ("l1", "l2"):
        raise ParseError(
            "P2", 'a regularizer is {"l1": name} or {"l2": name}', path=f"{path}[1]"
        )
    ((kind, target),) = reg.items()
    if not isinstance(target, str):
        raise ParseError("P2", "a regularizer target is a name", path=f"{path}[1]")
    return (weight, (kind, target))


def _parse_counter(value: Any, path: str) -> dict[str, Any]:
    obj = _require_mapping(value, path)
    _check_keys(obj, ("epochs", "updates"), path)
    if len(obj) != 1:
        raise ParseError("P2", "expected exactly one of epochs/updates", path=path)
    ((unit, count),) = obj.items()
    return {unit: _wrapped(count, _scalar_int, f"{path}.{unit}")}


def _parse_train(raw: Any, path: str) -> TrainSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        (
            "objective",
            "params",
            "optimizer",
            "steps",
            "batch",
            "anneal",
            "precision",
            "eval",
            "early_stop",
            "checkpoint",
            "seed",
        ),
        path,
    )
    for field in ("objective", "params", "optimizer", "steps", "batch"):
        if field not in obj:
            raise ParseError("P2", f"train needs {field!r}", path=path)
    objective_raw = obj["objective"]
    if not isinstance(objective_raw, list) or not objective_raw:
        raise ParseError(
            "P2",
            "train.objective is a non-empty list of terms",
            path=f"{path}.objective",
        )
    objective = tuple(
        _parse_objective_term(t, f"{path}.objective[{i}]")
        for i, t in enumerate(objective_raw)
    )
    params = _str_list(obj["params"], f"{path}.params")
    optimizer = _require_mapping(obj["optimizer"], f"{path}.optimizer")
    _check_keys(optimizer, OPTIMIZER_FIELDS, f"{path}.optimizer")
    if "name" not in optimizer or "lr" not in optimizer:
        raise ParseError(
            "P2", "train.optimizer needs 'name' and 'lr'", path=f"{path}.optimizer"
        )
    _enum(optimizer["name"], tuple(OPTIMIZER_DEFAULTS), f"{path}.optimizer.name")
    for field in ("lr", "weight_decay", "eps", "momentum", "clip_grad_norm"):
        if field in optimizer:
            _wrapped(optimizer[field], _scalar_number, f"{path}.optimizer.{field}")
    if "betas" in optimizer:
        betas = optimizer["betas"]
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or not all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in betas
            )
        ):
            raise ParseError(
                "P2",
                "optimizer betas is a two-number list",
                path=f"{path}.optimizer.betas",
            )
    if "schedule" in optimizer:
        _wrapped(optimizer["schedule"], _scalar_str, f"{path}.optimizer.schedule")
    steps = _parse_counter(obj["steps"], f"{path}.steps")
    batch = _require_mapping(obj["batch"], f"{path}.batch")
    _check_keys(batch, ("pairs",), f"{path}.batch")
    if "pairs" not in batch:
        raise ParseError(
            "P2",
            "train.batch counts base+counterfactual pairs: {'pairs': n}",
            path=f"{path}.batch",
        )
    anneal = None
    if "anneal" in obj:
        anneal_raw = _require_mapping(obj["anneal"], f"{path}.anneal")
        anneal = {
            key: _wrapped(
                value,
                lambda v, p: _parse_anneal_schedule(v, p),
                f"{path}.anneal.{key}",
            )
            for key, value in anneal_raw.items()
        }
    precision = None
    if "precision" in obj:
        precision_raw = _require_mapping(obj["precision"], f"{path}.precision")
        _check_keys(precision_raw, ("feature", "loss"), f"{path}.precision")
        precision = {
            key: _wrapped(
                value,
                lambda v, p: _enum(v, PRECISION_DTYPES, p),
                f"{path}.precision.{key}",
            )
            for key, value in precision_raw.items()
        }
    eval_spec = None
    if "eval" in obj:
        eval_raw = _require_mapping(obj["eval"], f"{path}.eval")
        _check_keys(eval_raw, ("every", "split", "metrics"), f"{path}.eval")
        for field in ("every", "split", "metrics"):
            if field not in eval_raw:
                raise ParseError(
                    "P2", f"train.eval needs {field!r}", path=f"{path}.eval"
                )
        eval_spec = {
            "every": _parse_counter(eval_raw["every"], f"{path}.eval.every"),
            "split": _wrapped(eval_raw["split"], _scalar_str, f"{path}.eval.split"),
            "metrics": _str_list(eval_raw["metrics"], f"{path}.eval.metrics"),
        }
    early_stop = None
    if "early_stop" in obj:
        es_raw = _require_mapping(obj["early_stop"], f"{path}.early_stop")
        _check_keys(es_raw, ("metric", "patience", "mode"), f"{path}.early_stop")
        for field in ("metric", "patience", "mode"):
            if field not in es_raw:
                raise ParseError(
                    "P2", f"train.early_stop needs {field!r}", path=f"{path}.early_stop"
                )
        early_stop = {
            "metric": _wrapped(
                es_raw["metric"], _scalar_str, f"{path}.early_stop.metric"
            ),
            "patience": _wrapped(
                es_raw["patience"], _scalar_int, f"{path}.early_stop.patience"
            ),
            "mode": _wrapped(
                es_raw["mode"],
                lambda v, p: _enum(v, ("min", "max"), p),
                f"{path}.early_stop.mode",
            ),
        }
    checkpoint = None
    if "checkpoint" in obj:
        ck_raw = _require_mapping(obj["checkpoint"], f"{path}.checkpoint")
        _check_keys(ck_raw, ("every", "file_path"), f"{path}.checkpoint")
        checkpoint = {
            key: (
                _parse_counter(value, f"{path}.checkpoint.every")
                if key == "every"
                else _wrapped(value, _scalar_str, f"{path}.checkpoint.file_path")
            )
            for key, value in ck_raw.items()
        }
    seed: Any = 0
    if "seed" in obj:
        seed = _wrapped(obj["seed"], _scalar_int, f"{path}.seed")
    return TrainSpec(
        objective=objective,
        params=params,
        optimizer=dict(optimizer),
        steps=steps,
        batch=dict(batch),
        anneal=anneal,
        precision=precision,
        eval=eval_spec,
        early_stop=early_stop,
        checkpoint=checkpoint,
        seed=seed,
    )


def _parse_anneal_schedule(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ParseError("P2", "an anneal schedule is [start, end, frac]", path=path)
    start, end, frac = (_scalar_number(v, f"{path}[{i}]") for i, v in enumerate(value))
    return (float(start), float(end), float(frac))


def _parse_save(raw: Any, path: str) -> tuple[SaveEntry, ...]:
    if not isinstance(raw, list):
        raise ParseError("P2", "save is a list of entries", path=path)
    entries: list[SaveEntry] = []
    for i, entry_raw in enumerate(raw):
        p = f"{path}[{i}]"
        obj = _require_mapping(entry_raw, p)
        _check_keys(obj, ("value", "model", "input", "site", "file_path", "reduce"), p)
        for field in ("value", "file_path"):
            if field not in obj:
                raise ParseError("P2", f"a save entry needs {field!r}", path=p)
        has_binding = "model" in obj or "input" in obj
        has_site = "site" in obj
        if has_binding and has_site:
            raise ValidationError(
                10, "a save entry carries model/input or site, never both", path=p
            )
        if has_binding and not ("model" in obj and "input" in obj):
            raise ValidationError(
                10, "a read/metric save entry needs both model and input", path=p
            )
        if not has_binding and not has_site:
            raise ValidationError(
                10,
                "a save entry needs its binding: model+input (read/metric) or site (featurizer)",
                path=p,
            )
        entries.append(
            SaveEntry(
                value=_scalar_str(obj["value"], f"{p}.value"),
                file_path=_scalar_str(obj["file_path"], f"{p}.file_path"),
                model=_scalar_str(obj["model"], f"{p}.model")
                if "model" in obj
                else None,
                input=_scalar_str(obj["input"], f"{p}.input")
                if "input" in obj
                else None,
                site=_scalar_str(obj["site"], f"{p}.site") if "site" in obj else None,
                reduce=_enum(obj["reduce"], SAVE_REDUCTIONS, f"{p}.reduce")
                if "reduce" in obj
                else None,
            )
        )
    return tuple(entries)


# --------------------------------------------------------------------------- #
# the document parser
# --------------------------------------------------------------------------- #


def _normalize_top_level(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the top-level alias (``neural_model`` → ``model``, §2.1) and
    reject unknown sections."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = "model" if key == "neural_model" else key
        if name in out:
            raise ParseError("P2", "both 'model' and 'neural_model' are present")
        if name not in SECTION_ORDER:
            raise ParseError(
                "P3", f"unknown section {key!r}{suggest(key, SECTION_ORDER)}", path=key
            )
        out[name] = value
    return out


def _check_section_order(sections: Sequence[str]) -> None:
    """§5.2 — sections appear in the §1 order; ``save`` last."""
    ranks = {name: i for i, name in enumerate(SECTION_ORDER)}
    order = [ranks[s] for s in sections]
    if order != sorted(order):
        raise ValidationError(
            2,
            f"sections out of order: got {list(sections)}, expected the "
            f"docs/intervention_protocol.md §1 order {list(SECTION_ORDER)}",
        )
    if sections and sections[-1] != "save":
        raise ValidationError(2, "'save' must be the last section")


def parse_document(raw: Mapping[str, Any]) -> Document:
    """Strict-parse a raw mapping into a :class:`Document`.

    Owns §5.1 (strict keys, closed enums, no authored derived fields) and
    §5.2 (section order); cross-reference rules are
    :func:`causalab.protocol.validate.validate_document`'s job. ``raw`` must
    be order-preserving (any ``dict`` from :func:`load_raw` / ``yaml.safe_load``).
    """
    normalized = _normalize_top_level(raw)
    _check_section_order(list(normalized))
    declared_type = normalized.get("type")
    if declared_type is not None:
        declared_type = _enum(declared_type, DOCUMENT_TYPES, "type")
        if declared_type != "protocol":
            raise ParseError(
                "P2",
                f"this file declares type {declared_type!r}, but it is being read "
                "as a protocol document — a method binds to an application and "
                "the composition is what runs (§1.1)",
                path="type",
            )
    for section in REQUIRED_SECTIONS:
        if section not in normalized:
            raise ParseError("P2", f"missing required section {section!r}")
    version = normalized["version"]
    if version != "1":
        raise ParseError(
            "P2", f'unsupported version {version!r}; this loader reads version "1"'
        )
    description = normalized.get("description")
    if description is not None and not isinstance(description, str):
        raise ParseError("P2", "description is free text", path="description")
    save = _parse_save(normalized["save"], "save")
    if not save:
        raise ValidationError(10, "save must be non-empty", path="save")
    return Document(
        version=version,
        description=description,
        model=_parse_model(normalized["model"], "model"),
        data=_parse_data(normalized["data"], "data"),
        positions=_parse_positions(normalized.get("positions", {}), "positions"),
        sites=_parse_sites(normalized["sites"], "sites"),
        featurizers=_parse_featurizers(
            normalized.get("featurizers", {}), "featurizers"
        ),
        params=_parse_params(normalized.get("params", {}), "params"),
        reads=_parse_reads(normalized["reads"], "reads"),
        writes=_parse_writes(normalized.get("writes", {}), "writes"),
        intervened_models=_parse_intervened_models(
            normalized.get("intervened_models", {}), "intervened_models"
        ),
        metrics=_parse_metrics(normalized.get("metrics", {}), "metrics"),
        train=_parse_train(normalized["train"], "train")
        if "train" in normalized
        else None,
        save=save,
        raw=normalized,
    )
