from __future__ import annotations

import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from datasets import Dataset
from pyvene import IntervenableModel
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "LMPipeline"]

# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------


def _infer_device_and_dtype(
    requested_device: str | torch.device | None = None,
    requested_dtype: torch.dtype | None = None,
) -> tuple[str | torch.device, torch.dtype]:
    """Return a sensible `(device, dtype)` pair when not fully specified."""
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_dtype is None:
        requested_dtype = torch.float32
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
        **kwargs: Any,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.logit_labels = logit_labels
        self.position_ids = position_ids
        self.use_chat_template = use_chat_template
        self.padding_side = padding_side
        self._init_extra_kwargs = kwargs
        super().__init__(model_or_name)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_or_name,
                config=self._init_extra_kwargs.get("config"),
                token=hf_token,
            ).to(device=device, dtype=dtype)
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = "eager"
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        else:
            self.model = self.model_or_name.to(device).to(dtype)
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
        input: dict[str, Any] | list[dict[str, Any]] | str | list[str],
        *,
        max_length: int | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = True,
        use_chat_template: bool | None = None,
        no_padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Tensor]:
        if use_chat_template is None:
            use_chat_template = self.use_chat_template

        if isinstance(input, str):
            input = [{"raw_input": input}]
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            input = [{"raw_input": p} for p in input]

        if isinstance(input, dict):
            assert "raw_input" in input, (
                "Input dictionary must contain 'raw_input' key."
            )
            raw_input = [input["raw_input"]]
        else:
            assert isinstance(input, list) or isinstance(input, tuple), (
                "Input must be a dictionary or a list/tuple of dictionaries."
            )
            assert all("raw_input" in item for item in input), (
                "Each input dictionary must contain 'raw_input' key."
            )
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
        for k, v in enc.items():
            if isinstance(v, torch.Tensor):
                enc[k] = v.to(self.model.device)

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
        input: dict[str, Any] | list[dict[str, Any]] | str | list[str],
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        # Handle backward compatibility for raw strings
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
        sources: list[dict[str, Tensor]],
        map: dict[str, Any],
        feature_indices: list[list[int]] | None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
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
            # pyvene type stubs are incomplete
            out = intervenable_model.generate(base, sources=sources, **defaults)  # type: ignore[reportOptionalMemberAccess]

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
        dataset: CounterfactualDataset,
        batch_size: int = 32,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compute outputs for base inputs and counterfactual inputs from a dataset.

        Processes all base inputs and all counterfactual inputs through the model
        without interventions, returning the raw generation outputs.

        Args:
            dataset: CounterfactualDataset with base inputs and counterfactual inputs
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

        # Extract base inputs
        base_inputs = [example["input"] for example in dataset]
        base_dataset = Dataset.from_list(base_inputs)

        # Create dataloader for base inputs
        def shallow_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
            return {key: [item[key] for item in batch] for key in batch[0].keys()}

        base_dataloader = DataLoader(
            base_dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=False,
            collate_fn=shallow_collate_fn,
        )

        base_outputs = []

        # Process base inputs
        for batch in tqdm(
            base_dataloader,
            desc="Computing base outputs",
            disable=not logger.isEnabledFor(logging.DEBUG),
            leave=False,
        ):
            with torch.no_grad():
                # Reconstruct batch as list of dicts for pipeline.generate
                batch_inputs = []
                batch_size_actual = len(batch["raw_input"])
                for i in range(batch_size_actual):
                    example_dict = {key: batch[key][i] for key in batch.keys()}
                    batch_inputs.append(example_dict)

                # Generate outputs
                output_dict = self.generate(batch_inputs)

                # Flatten batch outputs into individual examples
                for i in range(batch_size_actual):
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
            cf_dataset = Dataset.from_list(counterfactual_inputs)
            cf_dataloader = DataLoader(
                cf_dataset,  # type: ignore[arg-type]
                batch_size=batch_size,
                shuffle=False,
                collate_fn=shallow_collate_fn,
            )

            for batch in tqdm(
                cf_dataloader,
                desc="Computing counterfactual outputs",
                disable=not logger.isEnabledFor(logging.DEBUG),
                leave=False,
            ):
                with torch.no_grad():
                    # Reconstruct batch as list of dicts
                    batch_inputs = []
                    batch_size_actual = len(batch["raw_input"])
                    for i in range(batch_size_actual):
                        example_dict = {key: batch[key][i] for key in batch.keys()}
                        batch_inputs.append(example_dict)

                    # Generate outputs
                    output_dict = self.generate(batch_inputs)

                    # Flatten batch outputs into individual examples
                    for i in range(batch_size_actual):
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
