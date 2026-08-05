from __future__ import annotations

import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel
from nnsight import TransformersModel  # type: ignore[import-untyped]
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline",
    "LMPipeline",
    "resolve_device",
    "left_pad_position_ids",
    "ensure_position_ids",
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

    Call this for a plain forward you drive yourself. The intervention engine
    does not need it — nnsight derives mask-based ``position_ids`` for any
    left-padded text batch — and ``generate`` numbers its own per-step positions,
    so a prompt-shaped ``position_ids`` must not be forced on it (it cannot
    extend across decode steps; measured to shift GPT-2 multi-step logits by
    ~0.26). That is why ``load`` does not emit ``position_ids`` by default.

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


# ---------------------------------------------------------------------------
# Base pipeline – minimal signatures (no *args / **kwargs)
# ---------------------------------------------------------------------------


class Pipeline(ABC):
    """Abstract base pipeline.

    Subclasses must implement the hooks below. The base class deliberately
    avoids variadic parameters so implementers have full freedom to define
    their own concrete signatures.

    Two handles on the same network, deliberately:

    * ``model`` — the raw ``torch.nn.Module`` (a HuggingFace model). Everything
      that reads config, moves devices, or installs its own hooks uses this.
    * ``nnsight`` — the :class:`nnsight.TransformersModel` envoy wrapping *that
      same module*. The intervention engine traces through this.

    They are never two copies: ``model is nnsight._module``.

    **Do not ``copy.deepcopy(pipeline.model)``.** Wrapping installs a controller
    ``forward`` on every submodule that closes over a *weakref* to that module.
    Deepcopy duplicates the closure but the weakref still resolves to the
    original, so the copy runs the original model's layers — wrong activations,
    wrong logits, and no error to notice it by. To build a second model from an
    existing one, rebuild it from its config and state dict::

        clone = type(pipeline.model)(pipeline.model.config)
        clone.load_state_dict(pipeline.model.state_dict())
    """

    model: Any
    nnsight: TransformersModel
    tokenizer: Any
    model_or_name: Any

    def __init__(self, model_or_name: Any) -> None:
        self.model_or_name = model_or_name
        self._setup_model()

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
    def generate(self, prompt: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def format_generation(
        self,
        result: Any,
        base_encoding: dict[str, Any],
        output_scores: bool | int = True,
    ) -> Dict[str, Any]:
        """A backbone ``generate`` result -> the dict every scorer consumes.

        Decoding the generated region is pipeline-specific (which tokens are the
        completion, how they decode), so the intervention engine hands the raw
        result back here rather than interpreting it.
        """
        pass


# ---------------------------------------------------------------------------
# Language‑model pipeline (typed; unchanged implementation)
# ---------------------------------------------------------------------------


class LMPipeline(Pipeline):
    """Pipeline for autoregressive HuggingFace causal‑LMs.

    Extra keyword arguments (via ``**kwargs``) that affect loading:
    ``device``, ``dtype``, ``device_map``, ``config``, ``hf_token``,
    ``eager_attn``, and ``tokenizer`` — pass the last one when wrapping a model
    built in-process, which has no ``config.name_or_path`` to resolve one from.
    """

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
        """Build ``self.nnsight`` (the envoy) and ``self.model`` (the raw module).

        The tokenizer is loaded here rather than left to nnsight's pipeline: the
        token-position machinery depends on this exact instance's
        ``padding_side`` / ``pad_token`` / ``offset_mapping`` behaviour, so we
        configure it and hand the *same object* to ``TransformersModel`` — the
        pipeline and ``pipeline.load`` must never disagree about tokenization.
        """
        _patch_extra_special_tokens()

        device, dtype = _infer_device_and_dtype(
            self._init_extra_kwargs.get("device"), self._init_extra_kwargs.get("dtype")
        )

        if isinstance(self.model_or_name, str):
            hf_token = (
                self._init_extra_kwargs.get("hf_token", None)
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_or_name, token=hf_token
            )
            device_map = self._init_extra_kwargs.get("device_map")
            load_kwargs: dict[str, Any] = {"token": hf_token}
            # `dtype="auto"` means "read it off the checkpoint", which only
            # `from_pretrained` understands. The meta build goes through
            # `from_config`, where the string reaches `getattr(torch, "auto")`
            # and raises — and there are no weights whose dtype to adopt anyway.
            if self.load_weights or isinstance(dtype, torch.dtype):
                load_kwargs["dtype"] = dtype
            config = self._init_extra_kwargs.get("config")
            if config is not None:
                load_kwargs["config"] = config
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            elif self.load_weights:
                load_kwargs["device"] = device
            if self._init_extra_kwargs.get("eager_attn", True):
                load_kwargs["attn_implementation"] = "eager"
            # dispatch=False builds the architecture on the meta device: a full,
            # traceable envoy tree (and `nnsight.scan()` for shapes) without
            # paying the weight load. That is the `load_weights=False` mode —
            # strictly more capable than the config-only stub it replaces.
            self.nnsight = TransformersModel(
                self.model_or_name,
                task="text-generation",
                tokenizer=self.tokenizer,
                dispatch=self.load_weights,
                **load_kwargs,
            )
        else:
            # Pre-loaded model: move to device, and only convert dtype if explicit
            module = self.model_or_name.to(device)
            if isinstance(dtype, torch.dtype):
                module = module.to(dtype)
            # If dtype is "auto", keep the model's existing dtype
            tokenizer = self._init_extra_kwargs.get("tokenizer")
            if tokenizer is None:
                # A model built in-process (`LlamaForCausalLM(config)`) has an
                # empty `name_or_path`, so there is nothing to resolve — say so
                # instead of failing inside the Hub client's repo-id validator.
                name_or_path = getattr(module.config, "name_or_path", "")
                if not name_or_path:
                    raise ValueError(
                        "A pre-loaded model with no `config.name_or_path` gives "
                        "LMPipeline no way to find its tokenizer. Pass one "
                        "explicitly: LMPipeline(model, tokenizer=<tokenizer>)."
                    )
                tokenizer = AutoTokenizer.from_pretrained(name_or_path)
            self.tokenizer = tokenizer
            self.nnsight = TransformersModel(
                module,
                task="text-generation",
                tokenizer=self.tokenizer,
                dispatch=True,
            )

        # The raw module the envoy wraps — the handle for config reads, device
        # moves, and any code that installs its own PyTorch hooks.
        self.model = self.nnsight._module

        if self.load_weights:
            if self._init_extra_kwargs.get("eager_attn", True):
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = "eager"
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
            # We always greedy-decode (do_sample=False); strip sampling-only
            # fields from generation_config so transformers doesn't warn that
            # temperature/top_p are being ignored on every generate() call.
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.do_sample = False
                gen_cfg.temperature = None
                gen_cfg.top_p = None
                gen_cfg.top_k = None

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )
        if self.padding_side is not None:
            self.tokenizer.padding_side = self.padding_side

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
            # ensure_position_ids. Same left-pad convention either way; set in
            # place so enc stays a BatchEncoding.
            enc["position_ids"] = left_pad_position_ids(enc["attention_mask"])
        # Pop offset_mapping if present - it's a list of tuples, not a tensor
        offset_mapping = enc.pop("offset_mapping", None)

        # A `load_weights=False` pipeline lives on the meta device — it exists to
        # tokenize and read config, never to run a forward. Moving real ids onto
        # meta would silently discard them, so leave them on CPU.
        target_device = self.model.device
        if target_device.type != "meta":
            for k, v in enc.items():
                enc[k] = v.to(target_device)

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
        across batches.
        """
        gen = sequences[:, prompt_len:]
        deficit = self.max_new_tokens - gen.shape[1]
        if deficit > 0:
            pad_block = gen.new_full(
                (gen.shape[0], deficit), self.tokenizer.pad_token_id
            )
            gen = torch.cat([gen, pad_block], dim=1)
        elif deficit < 0:
            gen = gen[:, : self.max_new_tokens]
        return gen

    def generate(
        self,
        input: list[CausalTrace],
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
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
            out = self.model.generate(**inputs, **defaults)
        scores = [s.detach().cpu() for s in (out.scores or [])]
        prompt_len = inputs["input_ids"].shape[1]
        seq = self._generated_tokens(out.sequences, prompt_len).detach().cpu()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "scores": scores,
            "sequences": seq,
            "string": self.dump(seq, is_logits=False),
        }

    def format_generation(
        self,
        result: Any,
        base_encoding: dict[str, Any],
        output_scores: bool | int = True,
    ) -> dict[str, Any]:
        """A ``generate`` result → the ``{sequences, scores, string}`` dict callers expect.

        ``result`` is a HuggingFace ``GenerateDecoderOnlyOutput`` (what
        ``tracer.result`` carries when generation ran with
        ``return_dict_in_generate=True``). The generated region is sliced from the
        prompt end — see :meth:`_generated_tokens` for why the last
        ``max_new_tokens`` is the wrong window.
        """
        prompt_len = base_encoding["input_ids"].shape[1]
        sequences = self._generated_tokens(result.sequences, prompt_len).detach().cpu()
        formatted: dict[str, Any] = {"sequences": sequences}
        if output_scores:
            formatted["scores"] = [s.detach().cpu() for s in (result.scores or [])]
        formatted["string"] = self.dump(sequences, is_logits=False)
        return formatted

    # ------------------------------------------------------------------
    # Intervention generation
    # ------------------------------------------------------------------

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
        base_inputs = [example["input"] for example in dataset]

        base_outputs = []

        # Process base inputs in batches
        for start in tqdm(
            range(0, len(base_inputs), batch_size),
            desc="Computing base outputs",
            disable=not logger.isEnabledFor(logging.DEBUG),
            leave=False,
        ):
            batch_inputs = base_inputs[start : start + batch_size]
            with torch.no_grad():
                # Generate outputs
                output_dict = self.generate(batch_inputs)

                # Flatten batch outputs into individual examples
                for i in range(len(batch_inputs)):
                    example_output = {
                        "sequences": output_dict["sequences"][i : i + 1],
                    }
                    if "scores" in output_dict and output_dict["scores"]:
                        example_output["scores"] = [
                            score[i : i + 1] for score in output_dict["scores"]
                        ]
                    if "string" in output_dict:
                        example_output["string"] = (
                            output_dict["string"][i]
                            if isinstance(output_dict["string"], list)
                            else output_dict["string"]
                        )
                    base_outputs.append(example_output)

        # Extract counterfactual inputs (flattened)
        counterfactual_inputs = []
        for example in dataset:
            counterfactual_inputs.extend(example["counterfactual_inputs"])

        # Process counterfactuals if they exist
        counterfactual_outputs = []
        if counterfactual_inputs:
            for start in tqdm(
                range(0, len(counterfactual_inputs), batch_size),
                desc="Computing counterfactual outputs",
                disable=not logger.isEnabledFor(logging.DEBUG),
                leave=False,
            ):
                batch_inputs = counterfactual_inputs[start : start + batch_size]
                with torch.no_grad():
                    # Generate outputs
                    output_dict = self.generate(batch_inputs)

                    # Flatten batch outputs into individual examples
                    for i in range(len(batch_inputs)):
                        example_output = {
                            "sequences": output_dict["sequences"][i : i + 1],
                        }
                        if "scores" in output_dict and output_dict["scores"]:
                            example_output["scores"] = [
                                score[i : i + 1] for score in output_dict["scores"]
                            ]
                        if "string" in output_dict:
                            example_output["string"] = (
                                output_dict["string"][i]
                                if isinstance(output_dict["string"], list)
                                else output_dict["string"]
                            )
                        counterfactual_outputs.append(example_output)

        return {
            "base_outputs": base_outputs,
            "counterfactual_outputs": counterfactual_outputs,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def get_num_attention_heads(self) -> int:
        return int(self.model.config.num_attention_heads)
