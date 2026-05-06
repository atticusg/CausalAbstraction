from __future__ import annotations

import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from pyvene import IntervenableModel  # type: ignore[import-untyped]
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "LMPipeline", "resolve_device"]

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

    _orig = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def _safe_set_model_specific_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return _orig(self, special_tokens)

    _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = (
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
    """

    model: Any
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
    def intervenable_generate(
        self,
        intervenable_model: IntervenableModel,
        base: Any,
        sources: Any,
        map: Any,  # noqa: A002 – intentional name
        feature_indices: Any,
        source_representations: Any = None,
    ) -> Dict[str, Any]:
        pass


# ---------------------------------------------------------------------------
# Language‑model pipeline (typed; unchanged implementation)
# ---------------------------------------------------------------------------


class LMPipeline(Pipeline):
    """Pipeline for autoregressive HuggingFace causal‑LMs."""

    def __init__(
        self,
        model_or_name: str | PreTrainedModel,
        *,
        max_new_tokens: int = 3,
        max_length: int | None = None,
        logit_labels: bool = False,
        position_ids: bool = False,
        use_chat_template: bool = False,
        padding_side: str | None = "left",
        load_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.logit_labels = logit_labels
        self.position_ids = position_ids
        self.use_chat_template = use_chat_template
        self.padding_side = padding_side
        self.load_weights = load_weights
        self._init_extra_kwargs = kwargs
        super().__init__(model_or_name)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
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
            pretrained_kwargs: dict[str, Any] = dict(
                config=self._init_extra_kwargs.get("config"),
                token=hf_token,
                dtype=dtype,
            )
            if device_map is not None:
                pretrained_kwargs["device_map"] = device_map
            if self.load_weights:
                self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore[call-arg]
                    self.model_or_name, **pretrained_kwargs
                )
                if device_map is None:
                    self.model = self.model.to(device=device)
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
            else:
                # Tokenizer + config only: skip weight load. Forward passes will fail;
                # this mode is for code paths that only need hidden_size + tokenization
                # (e.g. building InterchangeTargets for cached-feature manifold fitting).
                from types import SimpleNamespace
                from transformers import AutoConfig

                hf_config = AutoConfig.from_pretrained(
                    self.model_or_name,
                    token=hf_token,
                )
                self.model = SimpleNamespace(config=hf_config)
        else:
            # Pre-loaded model: move to device, and only convert dtype if explicit
            self.model = self.model_or_name.to(device)
            if isinstance(dtype, torch.dtype):
                self.model = self.model.to(dtype)
            # If dtype is "auto", keep the model's existing dtype
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.name_or_path
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )
        if self.padding_side is not None:
            self.tokenizer.padding_side = self.padding_side

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

        # Apply chat template if requested
        if use_chat_template:
            processed_input = []
            for text in raw_input:
                # Convert to messages format and apply chat template
                messages = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed_input.append(formatted)
            raw_input = processed_input

        if max_length is None and not no_padding:
            max_length = self.max_length

        if padding_side is not None:
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = padding_side

        enc = self.tokenizer(
            raw_input,
            padding=False if no_padding else ("max_length" if max_length else True),
            max_length=max_length,
            truncation=max_length is not None,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        if self.position_ids:
            enc["position_ids"] = self.model.prepare_inputs_for_generation(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )["position_ids"]
        # Pop offset_mapping if present - it's a list of tuples, not a tensor
        offset_mapping = enc.pop("offset_mapping", None)

        for k, v in enc.items():
            enc[k] = v.to(self.model.device)

        # Add back offset_mapping if it was present
        if offset_mapping is not None:
            enc["offset_mapping"] = offset_mapping

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
        seq = out.sequences[:, -self.max_new_tokens :].detach().cpu()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "scores": scores,
            "sequences": seq,
            "string": self.dump(seq, is_logits=False),
        }

    # ------------------------------------------------------------------
    # Intervention generation
    # ------------------------------------------------------------------

    def intervenable_generate(
        self,
        intervenable_model: IntervenableModel,
        base: dict[str, Tensor],
        sources: list[dict[str, Tensor]] | None,
        map: dict[str, Any],
        feature_indices: list[list[int]] | None,
        source_representations: list[Tensor] | None = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate with interventions applied.

        Args:
            intervenable_model: PyVENE model with preset intervention locations
            base: Tokenized base inputs
            sources: Tokenized counterfactual inputs. Can be None if source_representations
                is provided (cross-model patching case).
            map: Unit locations mapping {"sources->base": (source_indices, base_indices)}
            feature_indices: Feature subspace indices for each unit
            source_representations: Pre-collected activations to use instead of computing
                from sources. When provided, sources should be None. This enables cross-model
                patching where activations are collected from a different model.
                Format: List of tensors, one per intervention location.
            **gen_kwargs: Additional generation kwargs
        """
        defaults = dict(
            unit_locations=map,
            subspaces=feature_indices,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            intervene_on_prompt=True,
            do_sample=False,
            use_cache=True,
        )
        defaults.update(gen_kwargs)
        with torch.no_grad():
            # pyvene type stubs are incomplete - source_representations accepts list or dict
            out = intervenable_model.generate(
                base,
                sources=sources,
                source_representations=source_representations,  # type: ignore[reportArgumentType]
                **defaults,  # type: ignore[reportArgumentType]
            )  # type: ignore[reportOptionalMemberAccess]

        # Return dictionary like HuggingFace models
        sequences = out[-1].sequences[:, -self.max_new_tokens :].detach().cpu()
        result = {"sequences": sequences}

        if gen_kwargs.get("output_scores", True):
            scores = [s.detach().cpu() for s in (out[-1].scores or [])]
            result["scores"] = scores

        result["string"] = self.dump(sequences, is_logits=False)

        return result

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
