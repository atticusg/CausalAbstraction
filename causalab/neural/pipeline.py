from __future__ import annotations

import dataclasses
import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel
from nnterp import StandardizedTransformer
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline",
    "LMPipeline",
    "GenerationResult",
    "compress_scores_top_k",
    "UnsupportedArchitectureError",
    "assert_architecture_supported",
    "device_for_layer",
    "resolve_device",
    "left_pad_position_ids",
    "ensure_position_ids",
    "right_pad_sequences",
]


# ---------------------------------------------------------------------------
# Position ids — single source of truth for the left-pad convention
# ---------------------------------------------------------------------------


def left_pad_position_ids(attention_mask: Tensor) -> Tensor:
    """``position_ids`` for a (possibly padded) ``attention_mask``.

    Uses HF's convention — ``cumsum(mask) - 1`` over the real tokens, with pad
    slots pinned to 1 — so each real token is numbered from 0 regardless of how
    many pad tokens precede it. Reduces to ``arange`` on an unpadded mask.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def ensure_position_ids(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return ``inputs`` carrying ``position_ids`` for a plain (non-generate) forward.

    A plain forward with no ``position_ids`` lets the model default to
    ``arange(seq_len)``, which under the pipeline's default left padding starts
    numbering at the PAD tokens — corrupting every activation in a padded row for
    models with **absolute / learned** position embeddings (GPT-2, GPT-Neo, OPT).
    Rotary models are immune (RoPE is relative; a uniform left-pad shift cancels),
    so this is a no-op for them.

    For plain (non-generate) forwards, call this directly. The generate paths
    handle it as follows (see ``dataset.run_intervened_generation`` and the
    engine's ``plan._check_generate_inputs``):

    * **Multi-step** generation numbers its own per-step ``position_ids``;
      feeding it a prompt-shaped ``position_ids`` is wrong across decode steps
      — measured to shift GPT-2 multi-step logits by ~0.26 — so it is left
      alone. This is also why ``load`` does not emit ``position_ids`` by
      default (its output feeds both paths).
    * **Single-step** generation (``max_new_tokens == 1`` — the path-patching
      case) has only the prompt prefill forward, so a prompt-shaped
      ``position_ids`` is exactly correct; the engine's generate path applies
      this to its base (and, always, to its source-collection forwards)
      internally. Callers of that path need not wrap inputs themselves.

    No-op if ``position_ids`` is already present (e.g. the pipeline was built with
    ``position_ids=True``) or if there is no ``attention_mask`` to derive from.
    Returns a new dict; the input is not mutated.
    """
    if "position_ids" in inputs or "attention_mask" not in inputs:
        return inputs
    return {
        **inputs,
        "position_ids": left_pad_position_ids(inputs["attention_mask"]),
    }


# ---------------------------------------------------------------------------
# Generated-sequence width — single source of truth for the fixed-width
# contract on generated-token blocks
# ---------------------------------------------------------------------------


def right_pad_sequences(sequences: Tensor, width: int, pad_token_id: int) -> Tensor:
    """Pin a generated-tokens block ``(batch, n_generated)`` to a fixed
    ``(batch, width)`` shape.

    Right-pads with ``pad_token_id`` (= EOS under the pipeline convention,
    dropped on decode) when early EOS ended generation short of the budget,
    and truncates when a run generated past it — so downstream consumers that
    concatenate sequences across batches always see the same width.
    Extracted from :meth:`LMPipeline._generated_tokens` (EU4, #485) so the
    engine's plain ``[:, prompt_len:]`` slice
    (:attr:`causalab.neural.plan.PlanResult.sequences`) gets the same
    stable-shape contract on the dataset generation path.
    """
    deficit = width - sequences.shape[1]
    if deficit > 0:
        pad_block = sequences.new_full((sequences.shape[0], deficit), pad_token_id)
        return torch.cat([sequences, pad_block], dim=1)
    if deficit < 0:
        return sequences[:, :width]
    return sequences


# ---------------------------------------------------------------------------
# GenerationResult — THE generation output shape (EU5a, #486)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GenerationResult:
    """One flat result per generation run — the single output shape every
    generation producer emits (EU5a, #486).

    Replaces the three divergent legacy shapes (``LMPipeline.generate``'s
    ``{"scores", "sequences", "string"}`` dict, ``run_intervened_generation``'s
    batch-nested lists, and the per-example dicts consumers built from them) —
    and with them the ``['string']`` vs ``[0]['string']`` vs ``['string'][0]``
    access-pattern divergence. Producers flatten across their internal batches;
    the batch split is an execution detail and never appears in the result.

    Attributes:
        sequences: ``(n_examples, max_new_tokens)`` generated tokens only
            (prompt stripped), CPU, right-padded with the pipeline's
            ``pad_token_id`` (:func:`right_pad_sequences`); the width is the
            **pipeline's** ``max_new_tokens`` budget even under a per-call
            ``max_new_tokens`` override — the deliberate legacy width contract
            (see :meth:`LMPipeline._generated_tokens`).
        strings: ``pipeline.dump(sequences)`` per example — ALWAYS a list of
            ``n_examples`` strings, including for a single example (unlike
            ``LMPipeline.dump``'s bare-``str``-for-one collapse).
        scores: per-step full-vocabulary logits ``(n_examples, vocab)``, CPU —
            one entry per actually generated step (early EOS can stop short of
            the budget, so ``len(scores)`` may be < ``sequences.shape[1]``).
            ``None`` when scores were not requested. Exclusive with
            ``scores_top_k``.
        scores_top_k: memory-compressed per-step top-k structures
            (:func:`compress_scores_top_k`); ``None`` unless compressed.
            Exclusive with ``scores``.
    """

    sequences: torch.Tensor
    strings: list[str]
    scores: list[torch.Tensor] | None = None
    scores_top_k: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.scores is not None and self.scores_top_k is not None:
            raise ValueError(
                "GenerationResult carries either full-vocabulary scores or "
                "top-k compressed scores, never both — compress_scores_top_k "
                "replaces scores with scores_top_k."
            )

    def to_raw_results(self) -> dict[str, list[Any]]:
        """The legacy ``raw_results`` dict — ONE synthetic batch.

        The io/artifact boundary consumes the pre-EU5a batch-nested contract
        (``{"sequences": [per-batch (b, W)], "string": [per-batch list],
        "scores": [per-batch per-step]}``); this adapter wraps the flat
        result as a single synthetic batch of that schema — exactly what the
        legacy path produced at ``batch_size >= n_examples`` — so the
        stored-artifact schema (``io.artifacts.save_intervention_results``,
        ``raw_results.json``) is unchanged. Since EU5b (#487) this view is
        io-only: the scorers (``score_intervention_outputs``) consume the
        flat :class:`GenerationResult` directly. ``scores`` carries whichever
        score form the result holds (full-vocab or top-k) and is omitted when
        neither was requested, matching the legacy key set.
        """
        raw: dict[str, list[Any]] = {
            "sequences": [self.sequences],
            "string": [self.strings],
        }
        if self.scores is not None:
            raw["scores"] = [self.scores]
        elif self.scores_top_k is not None:
            raw["scores"] = [self.scores_top_k]
        return raw


def compress_scores_top_k(
    result: GenerationResult, pipeline: "Pipeline", k: int
) -> GenerationResult:
    """Compress a result's full-vocabulary ``scores`` to per-step top-k.

    The memory-efficiency tail the wrappers apply for ``output_scores=int``
    (replaces the retired legacy ``postprocess_batch_outputs`` /
    ``convert_to_top_k`` passes, operating on the flat shape): each per-step
    ``(n_examples, vocab)`` tensor becomes ``{"top_k_logits": (n_examples,
    k), "top_k_indices": (n_examples, k), "top_k_tokens":
    list[n_examples][k]}`` — the exact structure the legacy pass produced,
    so stored top-k artifacts are value-identical.

    Returns a new :class:`GenerationResult` with ``scores_top_k`` set and
    ``scores`` dropped (the two are exclusive). Requires ``result.scores``;
    a scores-less result has nothing to compress and is refused loudly.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if result.scores is None:
        raise ValueError(
            "compress_scores_top_k needs full-vocabulary scores; this result "
            "carries none (generate with output_scores=True, and compress "
            "at most once)."
        )
    top_k_scores: list[dict[str, Any]] = []
    for step_logits in result.scores:
        # step_logits: (n_examples, vocab)
        n_examples, vocab_size = step_logits.shape
        k_actual = min(k, vocab_size)
        top_k_values, top_k_indices = torch.topk(step_logits, k=k_actual, dim=1)
        flat_indices = top_k_indices.flatten().tolist()
        flat_tokens = pipeline.tokenizer.batch_decode(
            [[idx] for idx in flat_indices], skip_special_tokens=False
        )
        top_k_tokens = [
            flat_tokens[i * k_actual : (i + 1) * k_actual] for i in range(n_examples)
        ]
        top_k_scores.append(
            {
                "top_k_logits": top_k_values,
                "top_k_indices": top_k_indices,
                "top_k_tokens": top_k_tokens,
            }
        )
    return dataclasses.replace(result, scores=None, scores_top_k=top_k_scores)


# ---------------------------------------------------------------------------
# Device resolution for sharded (hf_device_map) models
# ---------------------------------------------------------------------------


def _device_for_key(key: str, hf_device_map: dict) -> str:
    """Resolve the GPU device for a dotted module path.

    Walks up the dotted path (``"model.layers.77"`` → ``"model.layers"`` → …)
    until it finds a match in ``hf_device_map``.
    """
    path = key.split("#")[0]
    while path:
        if path in hf_device_map:
            return hf_device_map[path]
        path = path.rsplit(".", 1)[0] if "." in path else ""
    return next(iter(hf_device_map.values()))


def device_for_layer(pipeline: "Pipeline", layer: int) -> torch.device:
    """Resolve the device a given transformer layer lives on.

    For models loaded with ``device_map="auto"``, different layers can live
    on different GPUs. Tensors that participate in operations at that layer
    (steering vectors, featurizers, etc.) must be on the same device. For
    single-device models this returns ``model.device``.
    """
    if hasattr(pipeline.hf_model, "hf_device_map"):
        device_map = pipeline.hf_model.hf_device_map
        key = f"model.layers.{layer}"
        if key in device_map:
            return torch.device(device_map[key])
        # Fallback: look up nearest ancestor in the map
        return torch.device(_device_for_key(key, device_map))
    return pipeline.hf_model.device


# ---------------------------------------------------------------------------
# Compat patches
# ---------------------------------------------------------------------------

_patched_extra_special_tokens = False


def _patch_extra_special_tokens() -> None:
    """Work around transformers <5 bug with list-valued extra_special_tokens.

    Some newer tokenizer configs (e.g. Gemma 4) ship extra_special_tokens as a
    list, but ``_set_model_specific_special_tokens`` in transformers 4.57.x
    unconditionally calls ``.keys()`` on the value, raising AttributeError.
    See https://github.com/huggingface/transformers/issues/45376
    """
    global _patched_extra_special_tokens
    if _patched_extra_special_tokens:
        return
    _patched_extra_special_tokens = True

    from transformers import tokenization_utils_base as _tub

    _orig = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens  # pyright: ignore[reportPrivateUsage]

    def _safe_set_model_specific_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        # Monkey-patch shim; the underlying API accepts either list or dict at
        # runtime even though the stub narrows to list[str].
        return _orig(self, special_tokens)  # pyright: ignore[reportArgumentType]

    _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = (  # pyright: ignore[reportPrivateUsage]
        _safe_set_model_specific_special_tokens
    )


# ---------------------------------------------------------------------------
# Architecture preflight — fail fast, with the actual cause, on model types
# the installed transformers cannot load
# ---------------------------------------------------------------------------


class UnsupportedArchitectureError(ValueError):
    """A checkpoint's ``model_type`` is unknown to the installed transformers.

    Subclasses ``ValueError`` so :func:`causalab.neural.validate.validate_model_load`
    reports it as a failed gate verdict rather than a crash.
    """


def assert_architecture_supported(model_name: str, token: str | None = None) -> None:
    """Fail fast when the installed transformers cannot load ``model_name``.

    A checkpoint whose ``model_type`` postdates the pinned transformers (e.g.
    ``gemma4`` under transformers 4.57.x) otherwise dies deep inside
    ``AutoConfig`` with a generic "update Transformers" message that names
    neither the architecture gap nor the version the checkpoint needs. This
    preflight reads the checkpoint's **raw** config dict — via
    ``PretrainedConfig.get_config_dict``, which unlike ``AutoConfig`` works for
    model types the installed transformers doesn't know — and checks its
    ``model_type`` against transformers' ``CONFIG_MAPPING``.

    Deliberately best-effort and out of the way:

    * unreadable config (offline without cache, gated repo, malformed JSON) →
      return and let the normal load path fail as before;
    * no ``model_type`` in the config → return (nothing to check);
    * repos shipping their own code (``auto_map`` present) → return; they
      resolve their classes via ``trust_remote_code``, not ``CONFIG_MAPPING``.

    Raises:
        UnsupportedArchitectureError: naming the checkpoint, its ``model_type``,
            the installed transformers version, and (when the config records it)
            the transformers version the checkpoint was saved with.
    """
    from transformers import PretrainedConfig

    try:
        config_dict, _ = PretrainedConfig.get_config_dict(model_name, token=token)
    except Exception as exc:
        logger.debug("architecture preflight: no config for %s (%s)", model_name, exc)
        return
    model_type = config_dict.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        return
    if "auto_map" in config_dict:
        return

    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type in CONFIG_MAPPING:
        return
    saved_with = config_dict.get("transformers_version")
    saved_with_note = (
        f" (the checkpoint was saved with transformers {saved_with})"
        if saved_with
        else ""
    )
    raise UnsupportedArchitectureError(
        f"{model_name!r} has model_type {model_type!r}, which the installed "
        f"transformers {transformers.__version__} does not know{saved_with_note}. "
        f"Loading this architecture requires a newer transformers release than "
        f"the installed/pinned one."
    )


# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------


def resolve_device(device: str | None = None) -> str:
    """Resolve a device string, supporting ``"auto"`` for platform detection.

    Priority for ``"auto"`` (or *None*): **cuda → mps → cpu**.
    Explicit values (``"cuda"``, ``"mps"``, ``"cpu"``) are returned as-is.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _infer_device_and_dtype(
    requested_device: str | torch.device | None = None,
    requested_dtype: torch.dtype | str | None = None,
) -> tuple[str | torch.device, torch.dtype | str]:
    """Return a sensible `(device, dtype)` pair when not fully specified.

    If dtype is None, defaults to "auto" which tells transformers to use the
    dtype from the model's config (e.g., bfloat16 if the model was saved that way).
    """
    if requested_device is None or requested_device == "auto":
        requested_device = resolve_device()
    if requested_dtype is None:
        requested_dtype = "auto"
    return requested_device, requested_dtype


#: safetensors header dtype strings → torch floating dtypes (the dtypes HF's
#: ``dtype="auto"`` weights-fallback can resolve to; integer/quantized entries
#: are deliberately absent — they never decide a model's compute dtype).
_SAFETENSORS_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def _checkpoint_weights_dtype(
    name: str, token: str | None = None
) -> torch.dtype | None:
    """The dominant floating dtype of a checkpoint's safetensors weights.

    This is HF ``dtype="auto"``'s *second* resolution step — used when the
    config carries no ``dtype``/``torch_dtype`` (#449 below-cut 1). A local
    directory reads the first shard's JSON header directly (8-byte length
    prefix + header; no tensor data touched); a hub id reads the repo's
    safetensors metadata via range requests (no weight download). Returns
    ``None`` when it can't be determined — the caller then falls back to the
    torch default (fp32), the pre-existing behavior."""
    try:
        counts: dict[str, int] = {}
        if os.path.isdir(name):
            import glob
            import json
            import struct

            shards = sorted(glob.glob(os.path.join(name, "*.safetensors")))
            if not shards:
                return None
            with open(shards[0], "rb") as fh:
                (header_len,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(header_len))
            for tensor_name, info in header.items():
                if tensor_name == "__metadata__":
                    continue
                numel = 1
                for dim in info.get("shape") or [1]:
                    numel *= int(dim)
                counts[info["dtype"]] = counts.get(info["dtype"], 0) + numel
        else:
            from huggingface_hub import get_safetensors_metadata

            meta = get_safetensors_metadata(name, token=token)
            counts = {str(k): int(v) for k, v in meta.parameter_count.items()}
        float_counts = {
            k: v for k, v in counts.items() if k in _SAFETENSORS_FLOAT_DTYPES
        }
        if not float_counts:
            return None
        dominant = max(float_counts, key=lambda k: float_counts[k])
        return _SAFETENSORS_FLOAT_DTYPES[dominant]
    except Exception as exc:  # noqa: BLE001 — best-effort fallback, never fatal
        logger.debug("could not read weights dtype for %s: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Base pipeline – minimal signatures (no *args / **kwargs)
# ---------------------------------------------------------------------------


class Pipeline(ABC):
    """Abstract base pipeline.

    Subclasses must implement the hooks below. The base class deliberately
    avoids variadic parameters so implementers have full freedom to define
    their own concrete signatures.
    """

    model: Any
    tokenizer: Any
    model_or_name: Any

    def __init__(self, model_or_name: Any) -> None:
        self.model_or_name = model_or_name
        self._setup_model()

    @property
    def hf_model(self) -> Any:
        """The raw HuggingFace ``PreTrainedModel`` backing this pipeline.

        ``self.model`` is an nnterp ``StandardizedTransformer`` (the standardized
        accessor surface the site layer is built on); its underlying HF module lives
        at ``._model``. A handful of consumers need that raw module rather than the
        standardized wrapper: ``hf_device_map`` sharding lookups and plain
        HF ``generate``. Everything else (``.config`` / ``.device`` / ``.dtype`` /
        ``.name_or_path`` / ``.generation_config`` / ``.eval()``) is proxied by the
        standardized wrapper, so callers keep using ``pipeline.model`` for those.

        Passthrough when ``self.model`` is not a ``StandardizedTransformer`` — a raw
        HF model, or the ``load_weights=False`` ``SimpleNamespace`` (config only).
        """
        return getattr(self.model, "_model", self.model)

    # ------------------------------------------------------------------
    # Abstract hooks – simple signatures only
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_model(self) -> None:
        pass

    @abstractmethod
    def load(self, raw_input: Any) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def dump(self, model_output: Any) -> str | List[str]:
        pass

    @abstractmethod
    def generate(self, prompt: Any) -> GenerationResult:
        pass


# ---------------------------------------------------------------------------
# Language‑model pipeline (typed; unchanged implementation)
# ---------------------------------------------------------------------------


class LMPipeline(Pipeline):
    """Pipeline for autoregressive HuggingFace causal‑LMs."""

    # Content-independent sentinel used to locate where user content begins in
    # the chat-wrapped prompt. ``wrapped.find(content)`` is unsafe because task
    # text can collide with role markers (e.g. content ``"INST"`` vs the
    # ``[INST]`` marker); the record-separator glyphs (U+241E) never appear in
    # task text or chat templates, so the sentinel resolves unambiguously.
    _CHAT_CONTENT_SENTINEL = "␞CAUSALAB_CONTENT␞"

    def __init__(
        self,
        model_or_name: str | PreTrainedModel,
        *,
        max_new_tokens: int = 3,
        max_length: int | None = None,
        logit_labels: bool = False,
        position_ids: bool = False,
        use_chat_template: bool = False,
        chat_answer_directive: str | None = None,
        padding_side: str | None = "left",
        load_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.logit_labels = logit_labels
        self.position_ids = position_ids
        self.use_chat_template = use_chat_template
        self.chat_answer_directive = chat_answer_directive
        self.padding_side = padding_side
        self.load_weights = load_weights
        self._init_extra_kwargs = kwargs
        # Lazily-computed chat-prefix metadata (see _chat_prefix_* helpers).
        # Cached per instance: the prefix is fixed once tokenizer + directive
        # are bound, so the wrap-and-tokenize cost is paid at most once each.
        self._chat_prefix_char_offset_cache: int | None = None
        self._chat_prefix_token_count_cache: int | None = None
        super().__init__(model_or_name)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        _patch_extra_special_tokens()

        if self._init_extra_kwargs.get("enable_attention_probs", False) and (
            self._init_extra_kwargs.get("check_renaming", True) is False
        ):
            # nnterp validates-and-enables the attention-probability accessor
            # only under its load-time checks (check_source runs iff
            # check_renaming); with check_renaming=False it silently disables
            # the accessor, and the failure would only surface at first read
            # as a misleading "load it with enable_attention_probs=True" —
            # even though the caller did. Name the real cause at load instead.
            raise ValueError(
                "enable_attention_probs=True is incompatible with "
                "check_renaming=False: nnterp only enables (and check_source-"
                "validates) the attention-probability accessor under its "
                "load-time checks, and silently disables it when they are "
                "skipped. Drop check_renaming=False."
            )

        device, dtype = _infer_device_and_dtype(
            self._init_extra_kwargs.get("device"), self._init_extra_kwargs.get("dtype")
        )
        hf_token = (
            self._init_extra_kwargs.get("hf_token", None)
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        if isinstance(self.model_or_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_or_name, token=hf_token
            )
            if self.load_weights:
                self.model = self._load_standardized_from_name(
                    self.model_or_name, device=device, dtype=dtype, hf_token=hf_token
                )
                self._apply_model_conventions()
            else:
                # Tokenizer + config only: skip weight load. Forward passes will fail;
                # this mode is for code paths that only need hidden_size + tokenization
                # (e.g. building site grids for cached-feature manifold fitting).
                from types import SimpleNamespace
                from transformers import AutoConfig

                hf_config = AutoConfig.from_pretrained(
                    self.model_or_name,
                    token=hf_token,
                )
                self.model = SimpleNamespace(config=hf_config)
        else:
            # Pre-loaded model instance: wrap it in a StandardizedTransformer as-is.
            # Respect the caller's placement — relocate only on an *explicit* device
            # request. A bare/``"auto"`` device must NOT force-move the model onto
            # CUDA just because a GPU is visible: production always loads by *name*
            # (the branch above resolves ``"auto" → cuda`` there, which is correct
            # for a hub load), so this pre-loaded branch is the test-fixture /
            # bring-your-own-model entry point. Silently relocating here is wrong in
            # two ways: (a) it drags a CPU test model onto a visible GPU while the
            # rest of the CPU tier stays on CPU — the #471 CPU-tier device mismatch;
            # and (b) it would collapse an already-sharded ``device_map`` model onto
            # a single device. dtype is converted only for an explicit ``torch.dtype``;
            # the user's config (attn impl / use_cache / generation_config) is left
            # untouched.
            requested_device = self._init_extra_kwargs.get("device")
            module = self.model_or_name
            if requested_device is not None and requested_device != "auto":
                module = module.to(requested_device)
            if isinstance(dtype, torch.dtype):
                module = module.to(dtype)
            wrap_kwargs = self._nnterp_validation_kwargs()
            if self._init_extra_kwargs.get("enable_attention_probs", False):
                # A pre-loaded module keeps its own attention implementation
                # (this path never rewrites the caller's config), and the
                # attention-probability tap only exists under the eager kernel
                # — fail fast with the remedy instead of letting nnterp's
                # check_source() die mid-trace on a missing source node.
                impl = getattr(module.config, "_attn_implementation", None)
                if impl != "eager":
                    raise ValueError(
                        f"enable_attention_probs=True needs eager attention, but "
                        f"this pre-loaded model uses attn_implementation="
                        f"{impl!r}. Reload it with attn_implementation='eager' "
                        f"(or pass the model name so the pipeline loads it "
                        f"eagerly itself)."
                    )
                wrap_kwargs["enable_attention_probs"] = True
            self.tokenizer = AutoTokenizer.from_pretrained(module.config.name_or_path)
            self.model = StandardizedTransformer(
                module, tokenizer=self.tokenizer, **wrap_kwargs
            )
            self.model.dispatch()
            if getattr(self.model, "config", None) is None:
                # nnsight populates ``.config`` only on name-loads; wrapping a
                # pre-loaded module leaves it None. Consumers read
                # ``pipeline.model.config`` (hidden_size, num_hidden_layers, …), so
                # mirror the HF config onto the standardized wrapper.
                self.model.config = module.config

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )
        if self.padding_side is not None:
            self.tokenizer.padding_side = self.padding_side

    def _load_standardized_from_name(
        self,
        name: str,
        *,
        device: str | torch.device,
        dtype: torch.dtype | str,
        hf_token: str | None,
    ) -> StandardizedTransformer:
        """Load ``name`` as a **dispatched** nnterp ``StandardizedTransformer``.

        nnterp standardizes the module tree across architectures — the uniform
        ``layers_output[i]`` / ``attentions_output[i]`` accessors the site layer is
        built on — and validates the rename/IO shapes at load. Model-load kwargs
        (``dtype``, ``device_map``, ``config``, ``token``, attention
        implementation) flow
        through nnsight's ``LanguageModel`` to HF ``from_pretrained``. We
        ``dispatch()`` immediately so the weights are materialised rather than lazy
        (meta) tensors: plain HF generation needs real tensors.
        """
        # Fail fast — with the transformers-version cause — on architectures the
        # installed transformers cannot load (see assert_architecture_supported).
        assert_architecture_supported(name, token=hf_token)
        st_kwargs: dict[str, Any] = dict(
            tokenizer=self.tokenizer,
            # Single device by default; a caller-supplied ``device_map`` ("auto" for
            # multi-GPU sharding, via io/pipelines.py) takes precedence.
            device_map=self._init_extra_kwargs.get("device_map") or device,
        )
        # Resolve dtype to a concrete torch.dtype. HF ``from_pretrained`` accepts the
        # string "auto" (load in the checkpoint's saved dtype), but nnsight's meta
        # load path passes it to ``getattr(torch, "auto")`` and crashes — so read the
        # config's dtype ourselves, which is exactly what "auto" means. When the
        # config carries no dtype, mirror HF's second "auto" step: the weights'
        # own dtype from the safetensors headers (#449 below-cut 1) — otherwise a
        # config-less non-fp32 checkpoint silently loads fp32.
        resolved_dtype = dtype
        if resolved_dtype == "auto":
            cfg = self._init_extra_kwargs.get("config")
            if cfg is None:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(name, token=hf_token)
            resolved_dtype = getattr(cfg, "dtype", None) or getattr(
                cfg, "torch_dtype", None
            )
            if isinstance(resolved_dtype, str):
                resolved_dtype = getattr(torch, resolved_dtype, None)
            if resolved_dtype is None:
                resolved_dtype = _checkpoint_weights_dtype(name, token=hf_token)
        if isinstance(resolved_dtype, torch.dtype):
            st_kwargs["dtype"] = resolved_dtype
        if self._init_extra_kwargs.get("config") is not None:
            st_kwargs["config"] = self._init_extra_kwargs["config"]
        if hf_token is not None:
            st_kwargs["token"] = hf_token
        if self._init_extra_kwargs.get("eager_attn", False):
            # Opt-in eager attention (``eager_attn=True``): required for
            # attention-probability work — under sdpa, ``output_attentions=True``
            # yields no weights (transformers' capture flags support only
            # eager/eager_paged/flex_attention). By default we pass nothing and
            # let HF resolve the implementation (sdpa/flash where available) —
            # the deliberate post-cutover SH3 flip (#424); goldens are pinned
            # under this default, while the migration parity pins force eager
            # themselves (tests/neural/parity/cases.py).
            st_kwargs["attn_implementation"] = "eager"
        if self._init_extra_kwargs.get("enable_attention_probs", False):
            # Attention-probability editing (CAP4, #457): nnterp exposes
            # ``attention_probabilities[i]`` as an editable trace target only
            # when loaded with this flag, which forces eager attention itself
            # (compatible with an explicit ``eager_attn=True``; nnterp rejects
            # any non-eager ``attn_implementation``). Under the default
            # ``check_renaming=True`` nnterp also runs its
            # ``attention_probabilities.check_source()`` causal-validation
            # gate at load — probs have the right shape, sum to 1, and
            # modifying them changes the logits (the F2-deferred adoption,
            # #393); with ``check_renaming=False`` nnterp disables the
            # accessor outright.
            st_kwargs["enable_attention_probs"] = True
        st_kwargs.update(self._nnterp_validation_kwargs())

        model = StandardizedTransformer(name, **st_kwargs)
        model.dispatch()
        return model

    def _nnterp_validation_kwargs(self) -> dict[str, Any]:
        """nnterp load-time validation passthrough (mirrors ``causalab.neural.validate``
        — see F2 / #393).

        Honors an explicit ``check_renaming`` and a custom-architecture
        ``rename_config`` from the constructor kwargs; otherwise nnterp's defaults
        apply. ``check_renaming=False`` is the escape hatch for a model that can't be
        re-validated in place (e.g. a deep copy of an already-dispatched model).
        """
        kw: dict[str, Any] = {}
        if "check_renaming" in self._init_extra_kwargs:
            kw["check_renaming"] = self._init_extra_kwargs["check_renaming"]
        if self._init_extra_kwargs.get("rename_config") is not None:
            kw["rename_config"] = self._init_extra_kwargs["rename_config"]
        return kw

    def _apply_model_conventions(self) -> None:
        """Pin the pipeline's greedy conventions on the loaded model.

        Applied to freshly *loaded* models only (never a caller's pre-loaded
        instance): strip sampling-only generation fields so transformers doesn't
        warn on every greedy ``generate`` call, and freeze the model parameters
        once — trainable edits (ED3) optimize featurizer/gate parameters only,
        and freezing here replaces the retired backbone's per-model
        ``disable_model_gradients`` dance (see ``causalab.neural.trainable``;
        gradients w.r.t. activations still flow from any trainable leaf onward).

        ``config.use_cache`` is left at the HF default (enabled). The retired
        pyvene backbone forced it off (its decode-step hooks mis-addressed
        cached generation); nnsight edits are cache-safe — prefill edits
        persist through the decode *because* generation is cached — and both
        generate paths pass ``use_cache=True`` explicitly (SH3, #424).
        """
        hf_model = self.hf_model
        for p in hf_model.parameters():
            p.requires_grad_(False)
        gen_cfg = getattr(hf_model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.do_sample = False
            gen_cfg.temperature = None
            gen_cfg.top_p = None
            gen_cfg.top_k = None

    # ------------------------------------------------------------------
    # Chat-template prefix metadata
    # ------------------------------------------------------------------

    def _chat_messages(self, content: str) -> list[dict[str, str]]:
        """Build the message list ``load`` wraps in chat mode.

        Emits an optional ``chat_answer_directive`` as a **system** message so
        it lands entirely inside the chat *prefix* (it never perturbs task-content
        char offsets), followed by the task text as the single user turn.
        """
        messages: list[dict[str, str]] = []
        if self.chat_answer_directive:
            messages.append({"role": "system", "content": self.chat_answer_directive})
        messages.append({"role": "user", "content": content})
        return messages

    def _chat_prefix_char_offset(self) -> int:
        """Character offset where user content begins in the chat-wrapped prompt.

        Wraps a content-independent sentinel through the same message structure
        :meth:`load` uses (so any system directive is accounted for) and returns
        ``wrapped.find(SENTINEL)``. Returns ``0`` when chat templating is off.
        Cached per instance.
        """
        if not self.use_chat_template:
            return 0
        offset = self._chat_prefix_char_offset_cache
        if offset is None:
            wrapped = self.tokenizer.apply_chat_template(
                self._chat_messages(self._CHAT_CONTENT_SENTINEL),
                tokenize=False,
                add_generation_prompt=True,
            )
            offset = wrapped.find(self._CHAT_CONTENT_SENTINEL)
            self._chat_prefix_char_offset_cache = offset
        return offset

    def _chat_prefix_token_count(self) -> int:
        """Number of tokens before user content in the chat-wrapped prompt.

        Needed to rebase **non-negative** absolute token indices so ``position=0``
        means the first *content* token rather than BOS. Computed by tokenizing
        the prefix slice with ``add_special_tokens=False`` (chat wrapping already
        embeds the specials — see :meth:`load`). Returns ``0`` when chat templating
        is off. Cached per instance.
        """
        if not self.use_chat_template:
            return 0
        count = self._chat_prefix_token_count_cache
        if count is None:
            wrapped = self.tokenizer.apply_chat_template(
                self._chat_messages(self._CHAT_CONTENT_SENTINEL),
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = wrapped[: self._chat_prefix_char_offset()]
            count = len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])
            self._chat_prefix_token_count_cache = count
        return count

    def _batch_device(self) -> torch.device:
        """Device for batched token tensors returned by :meth:`load`.

        Indexing-only pipelines (``load_weights=False``) have no model
        parameters; keep token batches on CPU rather than requiring
        ``self.model.device``.
        """
        if not self.load_weights:
            return torch.device("cpu")
        return self.model.device

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def load(
        self,
        input: list[CausalTrace],
        *,
        max_length: int | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = True,
        use_chat_template: bool | None = None,
        no_padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        if use_chat_template is None:
            use_chat_template = self.use_chat_template

        raw_input = [item["raw_input"] for item in input]

        # Apply chat template if requested. The wrapped string already embeds
        # the model's special tokens (BOS, role markers), so we tokenize it with
        # ``add_special_tokens=False`` below to avoid prepending a *second* BOS
        # (double-BOS bug) — that also keeps generation and offset math correct.
        wrapped_input: list[str] | None = None
        if use_chat_template:
            wrapped_input = [
                self.tokenizer.apply_chat_template(
                    self._chat_messages(text),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for text in raw_input
            ]
            raw_input = wrapped_input
            add_special_tokens = False

        if max_length is None and not no_padding:
            max_length = self.max_length

        if padding_side is not None:
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = padding_side

        # LMPipeline only wraps AutoModelForCausalLM (see import above), so the
        # downstream model never consumes ``token_type_ids``. Disabling it keeps
        # ``enc`` to the keys the rest of this method handles (input_ids,
        # attention_mask, optional offset_mapping, optional position_ids) and
        # avoids passing unexpected kwargs to model.forward. Encoder models
        # (BERT/RoBERTa) are out of scope for this pipeline class.
        enc = self.tokenizer(
            raw_input,
            padding=False if no_padding else ("max_length" if max_length else True),
            max_length=max_length,
            truncation=max_length is not None,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
            return_token_type_ids=False,
        )
        if self.position_ids:
            # Opt-in, NOT the default — deliberately. This dict feeds both plain
            # forwards (which need position_ids under left padding) and multi-step
            # generate (which numbers its own per-step position_ids). Emitting a
            # prompt-shaped position_ids unconditionally would regress multi-step
            # decoding on absolute-position models (measured ~0.26 logit shift on
            # GPT-2), so we leave it off here; plain-forward sites opt in via
            # ensure_position_ids, and the engine's single-step generate path
            # applies it internally. Same left-pad convention either way; set in place so enc
            # stays a BatchEncoding.
            enc["position_ids"] = left_pad_position_ids(enc["attention_mask"])
        # Pop offset_mapping if present - it's a list of tuples, not a tensor
        offset_mapping = enc.pop("offset_mapping", None)

        batch_device = self._batch_device()
        for k, v in enc.items():
            enc[k] = v.to(batch_device)

        # Add back offset_mapping if it was present
        if offset_mapping is not None:
            enc["offset_mapping"] = offset_mapping
            # Under a chat template the offsets index into the *wrapped* string,
            # so token-position resolution needs to know where the bare task
            # content starts. Attach the (scalar) content char offset plus the
            # wrapped strings so callers can rebase bare char ranges and verify
            # the template preserved the content verbatim. These are scalars /
            # lists (not tensors): attach *after* the .to(device) loop and ignore
            # them in any tensor-only consumer (existing callers already do).
            if use_chat_template:
                enc["content_char_offset"] = self._chat_prefix_char_offset()
                enc["wrapped_text"] = wrapped_input

        if padding_side is not None:
            self.tokenizer.padding_side = prev_padding_side
        return enc

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def dump(
        self,
        model_output: Tensor | list[Tensor] | tuple[Tensor, ...] | dict[str, Any],
        *,
        is_logits: bool = True,
    ) -> str | list[str]:
        if isinstance(model_output, dict):
            model_output = model_output.get("sequences", model_output.get("scores"))
            if isinstance(model_output, torch.Tensor):
                is_logits = model_output.dim() >= 3

        if isinstance(model_output, (list, tuple)):
            model_output = (
                model_output[0].unsqueeze(1)
                if len(model_output) == 1
                else torch.stack(model_output, dim=1)
            )

        if isinstance(model_output, torch.Tensor):
            if model_output.dim() >= 3 and is_logits:
                token_ids = model_output.argmax(dim=-1)
            elif model_output.dim() == 2:
                token_ids = model_output
            elif model_output.dim() == 1:
                token_ids = model_output.unsqueeze(0)
            else:
                raise ValueError("Unexpected output shape for dump().")
        else:
            raise TypeError("model_output must be Tensor / list / tuple / dict")

        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generated_tokens(self, sequences: Tensor, prompt_len: int) -> Tensor:
        """Slice the model-produced tokens out of a ``[prompt | generated]`` sequence.

        ``model.generate`` returns the prompt followed by the generated tokens, so
        the generated region is ``sequences[:, prompt_len:]``. Slicing the *last*
        ``max_new_tokens`` tokens instead (the previous behaviour) is correct only
        when generation consumes the full budget — on an early EOS stop the window
        reaches back into the prompt and leaks trailing prompt tokens into the
        decoded string. Under a chat template those trailing tokens are the
        ``assistant`` role header (plain text, not stripped by
        ``skip_special_tokens``), which corrupts multi-token outputs; at
        ``max_new_tokens == 1`` the bug is masked because exactly one token is
        always emitted.

        The slice is right-padded back to a fixed ``max_new_tokens`` width with
        ``pad_token_id`` (= EOS, dropped on decode) so the returned tensor keeps a
        stable shape contract for downstream consumers that concatenate sequences
        across batches (:func:`right_pad_sequences` owns the width contract).
        """
        return right_pad_sequences(
            sequences[:, prompt_len:], self.max_new_tokens, self.tokenizer.pad_token_id
        )

    def generate(
        self,
        input: list[CausalTrace],
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Plain (un-intervened, un-traced) greedy HF generation over a batch.

        Returns the unified :class:`GenerationResult` (EU5a, #486): CPU
        ``sequences`` right-padded to the pipeline's ``max_new_tokens``
        budget, ``strings`` always a list, per-step CPU ``scores`` (``None``
        when the caller passes ``output_scores=False``). Values are exactly
        the legacy ``{"scores", "sequences", "string"}`` dict's — only the
        shape changed.
        """
        # Persistent edits live in nnsight's tracing layer; the plain HF
        # generate below bypasses them entirely, so a steered eval would
        # silently produce UNsteered outputs. Refuse loudly instead (the
        # compose-or-refuse contract, causalab.neural.persistent).
        from causalab.neural.site import backbone_has_edits

        if backbone_has_edits(self.model):
            from causalab.neural.persistent import PersistentEditError

            raise PersistentEditError(
                "this pipeline's model carries persistent edits (model.edit(), "
                "causalab.neural.persistent), but LMPipeline.generate runs "
                "plain HF generation on pipeline.hf_model, which bypasses "
                "nnsight edits — the output would silently ignore them. Run "
                "generation through the traced path "
                "(causalab.neural.dataset.run_intervened_generation) or "
                "uninstall_edits(pipeline.model) first."
            )
        inputs = self.load(input)
        defaults: dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            use_cache=True,
        )
        defaults.update(gen_kwargs)
        with torch.no_grad():
            # Plain generation runs on the underlying HF model (nnterp's own
            # ``generate`` is a deferred nnsight trace with a different return shape);
            # this preserves the {scores, sequences, string} contract exactly.
            out = self.hf_model.generate(**inputs, **defaults)
        # HF returns scores=None iff output_scores=False was requested; keep
        # that as scores=None (vs an actual per-step list) on the result.
        scores = (
            [s.detach().cpu() for s in out.scores] if out.scores is not None else None
        )
        prompt_len = inputs["input_ids"].shape[1]
        seq = self._generated_tokens(out.sequences, prompt_len).detach().cpu()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        decoded = self.dump(seq, is_logits=False)
        return GenerationResult(
            sequences=seq,
            strings=[decoded] if isinstance(decoded, str) else decoded,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def compute_outputs(
        self,
        dataset: list[CounterfactualExample],
        batch_size: int = 32,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compute outputs for base inputs and counterfactual inputs from a dataset.

        Processes all base inputs and all counterfactual inputs through the model
        without interventions, returning the raw generation outputs.

        Args:
            dataset: List of CounterfactualExample with base inputs and counterfactual inputs
            batch_size: Batch size for processing

        Returns:
            Dictionary with:
                - "base_outputs": List of per-example output dicts (one per base input)
                - "counterfactual_outputs": List of per-example output dicts (flattened)

            Each output dict contains:
                - "sequences": Tensor of shape (1, seq_len)
                - "scores": List of score tensors (if available)
                - "string": String output
        """

        def per_example_outputs(inputs: list, desc: str) -> list[dict[str, Any]]:
            """Generate over ``inputs`` in batches; slice the flat
            :class:`GenerationResult` into the per-example dicts the
            checker/metric protocols consume (EU5b, #487)."""
            outputs: list[dict[str, Any]] = []
            for start in tqdm(
                range(0, len(inputs), batch_size),
                desc=desc,
                disable=not logger.isEnabledFor(logging.DEBUG),
                leave=False,
            ):
                batch_inputs = inputs[start : start + batch_size]
                with torch.no_grad():
                    result = self.generate(batch_inputs)
                for i in range(len(batch_inputs)):
                    example_output: dict[str, Any] = {
                        "sequences": result.sequences[i : i + 1],
                    }
                    if result.scores:
                        example_output["scores"] = [
                            score[i : i + 1] for score in result.scores
                        ]
                    example_output["string"] = result.strings[i]
                    outputs.append(example_output)
            return outputs

        base_inputs = [example["input"] for example in dataset]
        base_outputs = per_example_outputs(base_inputs, "Computing base outputs")

        # Extract counterfactual inputs (flattened)
        counterfactual_inputs = []
        for example in dataset:
            counterfactual_inputs.extend(example["counterfactual_inputs"])

        counterfactual_outputs = per_example_outputs(
            counterfactual_inputs, "Computing counterfactual outputs"
        )

        return {
            "base_outputs": base_outputs,
            "counterfactual_outputs": counterfactual_outputs,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_num_layers(self) -> int:
        # Prefer nnterp's standardized introspection; fall back to the HF config for
        # the load_weights=False (SimpleNamespace) path. Values are identical.
        n = getattr(self.model, "num_layers", None)
        return int(n) if n is not None else int(self.model.config.num_hidden_layers)

    def get_num_attention_heads(self) -> int:
        n = getattr(self.model, "num_heads", None)
        return int(n) if n is not None else int(self.model.config.num_attention_heads)
