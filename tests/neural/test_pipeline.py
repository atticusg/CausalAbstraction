"""Direct tests for ``causalab.neural.pipeline``.

``LMPipeline`` is the model-wiring boundary between the symbolic causal
layer and HuggingFace: it loads the tokenizer/model on the resolved device,
turns a ``list[CausalTrace]`` into batched ``input_ids`` / ``attention_mask``
/ optionally ``position_ids`` (``load``), and runs greedy generation that
always returns the unified :class:`GenerationResult` (EU5a, #486 — flat
``sequences`` / ``strings`` / per-step ``scores``; the engine's generate
path emits the same shape). Every analysis under
``causalab/analyses/subspace/*``, the manifold fitting pipeline, the
metric/filter methods, the flow/pca trainers, ``io/pipelines.py``, and the
runner smoke tier all instantiate ``LMPipeline`` — if ``load`` returns
mis-padded ids or ``generate`` swallows scores, every intervention
experiment silently produces garbage.

``resolve_device`` is the single source of truth for
``auto → cuda/mps/cpu`` fallback and is reused directly by
``analyses/path_steering/main.py`` and ``io/pipelines.py``.

The tests in this file use the session-scoped ``tiny_pipeline`` fixture
from :mod:`tests.neural.conftest` (which wraps the real
``hf-internal-testing/tiny-random-LlamaForCausalLM`` stub from
:mod:`tests._helpers.tiny` — i.e. ``tiny_random_model``). Per docs/TESTS.md's mocking-policy appendix, the
tiny real model is the canonical alternative to the previous file's
``DummyTokenizer`` / ``DummyModel`` stack — mocks of internal numerical
code silently drift away from production.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.pipeline import (
    GenerationResult,
    LMPipeline,
    Pipeline,
    UnsupportedArchitectureError,
    compress_scores_top_k,
    assert_architecture_supported,
    device_for_layer,
    ensure_position_ids,
    left_pad_position_ids,
    resolve_device,
)
from tests._helpers.tiny import (
    TINY_RANDOM_MODEL_NAME,
    fresh_tiny_random_llama,
    tiny_random_model,
)


# --------------------------------------------------------------------------- #
#  Shared helpers                                                             #
# --------------------------------------------------------------------------- #
def _make_trace(text: str) -> CausalTrace:
    """Build a minimal :class:`CausalTrace` whose only mechanism is ``raw_input``.

    ``LMPipeline.load`` only reads ``item["raw_input"]`` so this is the
    smallest viable input — no need to instantiate a full ``CausalModel``.
    """
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _cf_example(base: str, *counterfactuals: str) -> dict[str, Any]:
    """Construct a ``CounterfactualExample``-shaped dict for ``compute_outputs``."""
    return {
        "input": _make_trace(base),
        "counterfactual_inputs": [_make_trace(t) for t in counterfactuals],
    }


# --------------------------------------------------------------------------- #
#  position_ids helpers — single source of truth for the left-pad convention  #
# --------------------------------------------------------------------------- #
class TestPositionIdsHelpersUnit:
    """``left_pad_position_ids`` / ``ensure_position_ids`` are the one place the
    left-pad ``position_ids`` convention lives. Every non-generate forward
    (``collect``, path-patching pass 1, the IIA metric, pullback) routes through
    ``ensure_position_ids`` so a left-padded batch is not mis-encoded on
    absolute-position models; ``load`` reuses ``left_pad_position_ids`` when
    ``position_ids=True``. These pin the formula directly — the RoPE
    ``tiny_pipeline`` cannot surface the numeric symptom (RoPE is relative).
    """

    pytestmark = pytest.mark.unit

    def test_left_pad_mask_numbers_real_tokens_from_zero(self) -> None:
        # 2 pads then 2 real tokens; the real tokens must be 0, 1 (pads pinned to 1).
        am = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        expected = torch.tensor([[1, 1, 0, 1], [0, 1, 2, 3]])
        assert torch.equal(left_pad_position_ids(am), expected)

    def test_unpadded_mask_reduces_to_arange(self) -> None:
        am = torch.ones(2, 5, dtype=torch.long)
        expected = torch.arange(5).unsqueeze(0).expand(2, -1)
        assert torch.equal(left_pad_position_ids(am), expected)

    def test_numbers_continuously_across_a_left_then_right_pad_join(self) -> None:
        # The IIA metric concatenates a left-padded base with a right-padded label
        # (metric.py): base real tokens are right-aligned, label real tokens
        # left-aligned, so the real span is contiguous across the join and must be
        # numbered continuously (0,1,2,3) regardless of the padding-side switch.
        am = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])  # 2 base real | 2 label real
        out = left_pad_position_ids(am)
        assert torch.equal(out[0, 2:6], torch.tensor([0, 1, 2, 3]))

    def test_ensure_adds_position_ids_from_mask(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[0, 5, 6]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
        }
        out = ensure_position_ids(inputs)
        assert torch.equal(out["position_ids"], torch.tensor([[1, 0, 1]]))
        # input is not mutated
        assert "position_ids" not in inputs

    def test_ensure_is_noop_when_position_ids_present(self) -> None:
        preset = torch.tensor([[7, 8, 9]])
        inputs = {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "position_ids": preset,
        }
        out = ensure_position_ids(inputs)
        assert out["position_ids"] is preset

    def test_ensure_is_noop_without_attention_mask(self) -> None:
        inputs = {"input_ids": torch.tensor([[5, 6, 7]])}
        out = ensure_position_ids(inputs)
        assert "position_ids" not in out


# --------------------------------------------------------------------------- #
#  resolve_device                                                             #
# --------------------------------------------------------------------------- #
class TestResolveDeviceUnit:
    """``resolve_device`` is the ``"auto" → cuda/mps/cpu`` fallback used by
    every ``LMPipeline`` constructor and by ``analyses/path_steering/main.py``
    / ``io/pipelines.py`` directly.
    """

    pytestmark = pytest.mark.unit

    def test_explicit_cpu_passthrough(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_passthrough(self) -> None:
        # Explicit values are returned unchanged regardless of availability.
        assert resolve_device("cuda") == "cuda"

    def test_explicit_mps_passthrough(self) -> None:
        assert resolve_device("mps") == "mps"

    def test_none_picks_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device(None) == "cuda"

    def test_auto_picks_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"

    def test_auto_picks_mps_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_device("auto") == "mps"

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # backends.mps exists on all modern torch builds — patch its is_available.
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"


# --------------------------------------------------------------------------- #
#  Pipeline ABC                                                               #
# --------------------------------------------------------------------------- #
class TestPipelineABCUnit:
    """``Pipeline`` is the ABC every concrete pipeline subclass implements.

    The base ``__init__`` stores ``model_or_name`` and invokes
    ``_setup_model`` — this contract is what lets ``LMPipeline`` and any
    future image/multimodal pipeline reuse the same downstream callers in
    ``analyses/*``.
    """

    pytestmark = pytest.mark.unit

    def test_direct_instantiation_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            # Abstract methods (_setup_model, load, dump, generate)
            # make direct construction illegal.
            Pipeline("name")  # type: ignore[abstract]

    def test_subclass_with_stubs_can_be_constructed(self) -> None:
        calls: list[str] = []

        class _Stub(Pipeline):
            def _setup_model(self) -> None:  # type: ignore[override]
                calls.append("setup")
                self.model = SimpleNamespace()
                self.tokenizer = SimpleNamespace()

            def load(self, raw_input: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
                return {}

            def dump(self, model_output: Any) -> str:  # type: ignore[override]
                return ""

            def generate(self, prompt: Any) -> dict[str, Any]:  # type: ignore[override]
                return {}

        stub = _Stub("name")
        assert stub.model_or_name == "name"
        assert calls == ["setup"]


# --------------------------------------------------------------------------- #
#  dtype="auto" weights-header fallback (#449 below-cut 1)                     #
# --------------------------------------------------------------------------- #
class TestCheckpointWeightsDtypeUnit:
    """`_checkpoint_weights_dtype` — the weights-derived half of HF's
    ``dtype="auto"`` resolution, used when the config carries no dtype."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _write(dirpath, tensors) -> str:
        from safetensors.torch import save_file

        save_file(tensors, str(dirpath / "model.safetensors"))
        return str(dirpath)

    def test_reads_bf16_from_local_header(self, tmp_path) -> None:
        from causalab.neural.pipeline import _checkpoint_weights_dtype

        path = self._write(tmp_path, {"w": torch.zeros(4, 4, dtype=torch.bfloat16)})
        assert _checkpoint_weights_dtype(path) is torch.bfloat16

    def test_dominant_float_dtype_wins(self, tmp_path) -> None:
        from causalab.neural.pipeline import _checkpoint_weights_dtype

        path = self._write(
            tmp_path,
            {
                "big": torch.zeros(32, 32, dtype=torch.float16),
                "small": torch.zeros(2, dtype=torch.float32),
                "ids": torch.zeros(64, 64, dtype=torch.int64),  # never decides
            },
        )
        assert _checkpoint_weights_dtype(path) is torch.float16

    def test_no_shards_or_no_floats_is_none(self, tmp_path) -> None:
        from causalab.neural.pipeline import _checkpoint_weights_dtype

        empty = tmp_path / "empty"
        empty.mkdir()
        assert _checkpoint_weights_dtype(str(empty)) is None
        int_only = tmp_path / "ints"
        int_only.mkdir()
        self._write(int_only, {"ids": torch.zeros(4, dtype=torch.int64)})
        assert _checkpoint_weights_dtype(str(int_only)) is None

    def test_auto_dtype_falls_back_to_weights_header(self, monkeypatch) -> None:
        """The load path: config resolves no dtype → the safetensors-header
        dtype is what reaches the model constructor (previously: silent fp32)."""
        import causalab.neural.pipeline as pl

        captured: dict = {}

        class FakeST:
            def __init__(self, name, **kwargs):
                captured.update(kwargs)

            def dispatch(self) -> None:
                pass

        class DtypelessCfg:
            dtype = None
            torch_dtype = None

        monkeypatch.setattr(pl, "StandardizedTransformer", FakeST)
        monkeypatch.setattr(
            pl, "_checkpoint_weights_dtype", lambda name, token=None: torch.bfloat16
        )
        pipe = object.__new__(pl.LMPipeline)
        pipe._init_extra_kwargs = {"config": DtypelessCfg()}
        pipe.tokenizer = None
        pipe._load_standardized_from_name(
            "some/config-less-model", device="cpu", dtype="auto", hf_token=None
        )
        assert captured["dtype"] is torch.bfloat16


# --------------------------------------------------------------------------- #
#  LMPipeline.__init__ / _setup_model / get_num_*                             #
# --------------------------------------------------------------------------- #
class TestLMPipelineInitUnit:
    """``LMPipeline.__init__`` wires the model + tokenizer onto the resolved
    device and normalises ``pad_token`` / ``generation_config`` so downstream
    ``generate`` calls don't warn about ignored sampling fields.

    Since the nnterp rebase (F3, #394) ``self.model`` is an nnterp
    ``StandardizedTransformer`` — the standardized accessor surface — with the raw
    HF module reachable via ``self.hf_model``; ``.config`` / ``.device`` /
    ``.generation_config`` are proxied by the wrapper so these tests read them off
    ``pipeline.model`` unchanged.
    """

    pytestmark = pytest.mark.unit

    def test_constructor_attributes_pinned(self, tiny_pipeline: LMPipeline) -> None:
        # The session-scoped fixture uses max_new_tokens=1, padding_side="left".
        assert tiny_pipeline.max_new_tokens == 1
        assert tiny_pipeline.padding_side == "left"
        assert tiny_pipeline.use_chat_template is False
        assert tiny_pipeline.logit_labels is False
        assert tiny_pipeline.position_ids is False
        assert tiny_pipeline.load_weights is True

    def test_model_and_tokenizer_are_bound(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.model is not None
        assert tiny_pipeline.tokenizer is not None

    def test_model_is_standardized_transformer_hf_model_is_raw(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        # F3 (#394): pipeline.model is the nnterp standardized wrapper (the accessor
        # surface the site layer builds on); pipeline.hf_model unwraps it to the raw
        # HF module that pyvene and plain generation need.
        from nnterp import StandardizedTransformer
        from transformers import PreTrainedModel

        assert isinstance(tiny_pipeline.model, StandardizedTransformer)
        assert isinstance(tiny_pipeline.hf_model, PreTrainedModel)
        assert tiny_pipeline.hf_model is tiny_pipeline.model._model

    def test_pad_token_is_eos(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.tokenizer.pad_token == tiny_pipeline.tokenizer.eos_token

    def test_padding_side_set_from_kwarg(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.tokenizer.padding_side == "left"

    def test_use_cache_left_enabled_on_config_for_string_model(self) -> None:
        # SH3 (#424): the retired pyvene backbone forced use_cache=False on the
        # config; _apply_model_conventions no longer touches it, so a name-load
        # keeps the HF default (enabled — generation relies on it, and both
        # generate paths pass use_cache=True explicitly anyway).
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left")
        assert pipe.model.config.use_cache is True

    def test_attention_defaults_to_sdpa_for_string_model(self) -> None:
        # SH3 (#424): the pipeline no longer forces eager attention at load —
        # HF resolves the implementation itself (sdpa on this stack).
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left")
        assert pipe.model.config._attn_implementation != "eager"

    def test_eager_attn_opt_in_forces_eager(self) -> None:
        # eager_attn=True stays available for attention-probability work
        # (output_attentions=True yields no weights under sdpa).
        pipe = LMPipeline(
            TINY_RANDOM_MODEL_NAME,
            max_new_tokens=1,
            padding_side="left",
            eager_attn=True,
        )
        assert pipe.model.config._attn_implementation == "eager"

    def test_generation_config_sampling_fields_scrubbed_for_string_model(
        self,
    ) -> None:
        # _setup_model removes do_sample/temperature/top_p/top_k from
        # generation_config when loading from a model name so transformers
        # doesn't warn on every generate() call. We always greedy-decode.
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left")
        cfg = pipe.model.generation_config
        assert cfg.do_sample is False
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.top_k is None

    def test_get_num_layers_matches_config(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.get_num_layers() == int(
            tiny_pipeline.model.config.num_hidden_layers
        )

    def test_get_num_attention_heads_matches_config(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        assert tiny_pipeline.get_num_attention_heads() == int(
            tiny_pipeline.model.config.num_attention_heads
        )

    def test_pre_loaded_model_path_skips_from_pretrained(self) -> None:
        # When passed a HF model instance (not a string), _setup_model
        # should reuse it directly rather than re-downloading weights.
        m = tiny_random_model()
        pipe = LMPipeline(m, max_new_tokens=1, padding_side="left", device="cpu")
        assert pipe.model.config.name_or_path == TINY_RANDOM_MODEL_NAME
        # Object identity isn't guaranteed because .to() may return a new
        # module, but the underlying config must be the same.
        assert pipe.tokenizer is not None

    def test_pre_loaded_model_not_relocated_when_cuda_visible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #471: a pre-loaded model with no explicit device must NOT be force-moved
        # onto CUDA just because a GPU is visible. Production always loads by
        # *name* (that branch resolves "auto" -> cuda, correct for a hub load);
        # the pre-loaded branch is the CPU-tier / bring-your-own-model entry point
        # and must respect the model's existing placement -- otherwise a CPU test
        # model lands on a visible GPU while the hand-built comparison tensors
        # stay on CPU (the "Expected all tensors to be on the same device"
        # failures this issue is about), and an already-sharded device_map model
        # would collapse onto one device.
        #
        # Per-PR CI is CPU-only, so this pins the placement *decision* rather than
        # the end device: fake cuda-availability, then stub the nnterp wrapper
        # (its load-time scan itself needs a real device, so it can't run under a
        # faked GPU on a CPU box) and record which device the module is on when it
        # reaches the wrapper. Pre-fix, ``module.to(resolve_device())`` would have
        # moved it to "cuda" before the wrapper saw it (raising here, as there is
        # no real GPU); the fixed path leaves it on CPU.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device() == "cuda"  # sanity: auto-resolution prefers cuda now

        captured: dict[str, str] = {}

        class _RecordingStub:
            """Minimal stand-in that records the wrapped module's device."""

            def __init__(self, module: Any, **_kwargs: Any) -> None:
                captured["device"] = next(module.parameters()).device.type
                self.config = getattr(module, "config", None)

            def dispatch(self) -> None:  # LMPipeline calls this after wrapping
                pass

        monkeypatch.setattr(
            "causalab.neural.pipeline.StandardizedTransformer", _RecordingStub
        )

        model, _tok = fresh_tiny_random_llama()
        assert next(model.parameters()).device.type == "cpu"  # precondition
        LMPipeline(model, max_new_tokens=1, padding_side="left")
        assert captured["device"] == "cpu"

    def test_load_weights_false_yields_simplenamespace_config(self) -> None:
        # load_weights=False is the tokenizer+config-only path: build
        # site grids / token positions without paying for weight load.
        # Forward passes and generate will fail; load() is supported for
        # indexing-only tokenization (batches stay on CPU).
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, load_weights=False)
        assert isinstance(pipe.model, SimpleNamespace)
        assert hasattr(pipe.model, "config")
        # Hidden size is still readable so site-grid builders can size
        # activation buffers without forward.
        assert pipe.model.config.hidden_size > 0
        # No StandardizedTransformer in this mode, so hf_model passes through to the
        # SimpleNamespace unchanged (nothing to unwrap).
        assert pipe.hf_model is pipe.model

    def test_load_weights_false_load_tokenizes_on_cpu(self) -> None:
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, load_weights=False)
        batch = pipe.load([_make_trace("hello world")])
        assert batch["input_ids"].device.type == "cpu"
        assert batch["attention_mask"].device.type == "cpu"


# --------------------------------------------------------------------------- #
#  LMPipeline.load                                                            #
# --------------------------------------------------------------------------- #
class TestLMPipelineLoadUnit:
    """``LMPipeline.load`` is the only place ``CausalTrace`` →
    ``tokenizer(...)`` conversion happens — its return dict is what the
    PyVENE intervention pipeline (and ``analyses/manifold/*``) feed
    directly into ``model.generate`` or ``intervenable_model.generate``.
    """

    pytestmark = pytest.mark.unit

    def test_returns_input_ids_and_attention_mask(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load([_make_trace("hello world")])
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)

    def test_batch_size_preserved(self, tiny_pipeline: LMPipeline) -> None:
        batch = tiny_pipeline.load([_make_trace("ab"), _make_trace("cd ef")])
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2

    def test_attention_mask_aligned_to_input_ids(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load(
            [_make_trace("a"), _make_trace("a longer prompt here")]
        )
        # Padded tokens get attention_mask = 0; non-padded sum equals total
        # number of real tokens across the batch.
        assert batch["input_ids"].shape == batch["attention_mask"].shape

    def test_no_padding_keeps_tensor_shape_for_single_example(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        # With a single example, no_padding=True and the default both produce
        # an attention_mask of all-ones (nothing to pad).
        batch = tiny_pipeline.load([_make_trace("hello")], no_padding=True)
        assert int(batch["attention_mask"].sum()) == batch["input_ids"].numel()

    def test_padding_side_override_restored(self, tiny_pipeline: LMPipeline) -> None:
        prev = tiny_pipeline.tokenizer.padding_side
        _ = tiny_pipeline.load(
            [_make_trace("a"), _make_trace("a longer one")],
            padding_side="right",
        )
        assert tiny_pipeline.tokenizer.padding_side == prev

    def test_chat_template_path_runs(self, tiny_pipeline: LMPipeline) -> None:
        # The tiny Llama stub ships a chat_template. Apply it and assert
        # the result is still a tensor batch (the chat template adds
        # control tokens so the sequence length will be larger than the
        # plain-text version).
        plain = tiny_pipeline.load([_make_trace("hello")], use_chat_template=False)
        chatted = tiny_pipeline.load([_make_trace("hello")], use_chat_template=True)
        assert chatted["input_ids"].shape[0] == 1
        assert chatted["input_ids"].shape[1] >= plain["input_ids"].shape[1]

    def test_return_offsets_mapping_preserved(self, tiny_pipeline: LMPipeline) -> None:
        # offset_mapping is popped before .to(device) because it's a list,
        # not a tensor — then re-attached. Confirm it survives.
        batch = tiny_pipeline.load([_make_trace("hello")], return_offsets_mapping=True)
        assert "offset_mapping" in batch


# --------------------------------------------------------------------------- #
#  LMPipeline.generate                                                        #
# --------------------------------------------------------------------------- #
class TestLMPipelineGenerateUnit:
    """``LMPipeline.generate`` wraps ``model.generate`` with greedy decoding
    and always returns the unified :class:`GenerationResult` (EU5a, #486) —
    every downstream baseline runner depends on this contract.
    """

    pytestmark = pytest.mark.unit

    def test_returns_generation_result(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("hello"), _make_trace("world")])
        assert isinstance(out, GenerationResult)
        assert out.scores is not None
        assert out.scores_top_k is None

    def test_sequences_shape_equals_batch_x_max_new_tokens(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        out = tiny_pipeline.generate([_make_trace("a"), _make_trace("b")])
        assert out.sequences.shape == (2, tiny_pipeline.max_new_tokens)

    def test_scores_length_equals_max_new_tokens(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")])
        assert isinstance(out.scores, list)
        assert len(out.scores) == tiny_pipeline.max_new_tokens

    def test_output_scores_false_gives_none_scores(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")], output_scores=False)
        assert out.scores is None
        assert out.scores_top_k is None

    def test_sequences_returned_on_cpu(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")])
        assert out.sequences.device.type == "cpu"
        for s in out.scores:
            assert s.device.type == "cpu"

    def test_strings_is_list_for_multi_batch(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("a"), _make_trace("b")])
        assert isinstance(out.strings, list)
        assert len(out.strings) == 2

    def test_strings_is_list_for_single_example(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        # Unlike dump()'s bare-str-for-one collapse, strings is ALWAYS a list.
        out = tiny_pipeline.generate([_make_trace("hi")])
        assert isinstance(out.strings, list)
        assert len(out.strings) == 1
        assert out.strings[0] == tiny_pipeline.dump(out.sequences, is_logits=False)


# --------------------------------------------------------------------------- #
#  GenerationResult (EU5a, #486)                                              #
# --------------------------------------------------------------------------- #
class TestGenerationResultUnit:
    """The unified generation output shape: score-form exclusivity, the
    io-boundary single-synthetic-batch ``to_raw_results`` view, and the
    ``compress_scores_top_k`` pass (structures identical to the retired
    per-batch ``convert_to_top_k``)."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _result(n: int = 3, vocab: int = 11, steps: int = 2) -> GenerationResult:
        torch.manual_seed(0)
        return GenerationResult(
            sequences=torch.arange(n * steps, dtype=torch.long).reshape(n, steps),
            strings=[f"s{i}" for i in range(n)],
            scores=[torch.randn(n, vocab) for _ in range(steps)],
        )

    def test_scores_and_top_k_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="never both"):
            GenerationResult(
                sequences=torch.zeros(1, 1, dtype=torch.long),
                strings=["a"],
                scores=[torch.zeros(1, 3)],
                scores_top_k=[{}],
            )

    def test_to_raw_results_is_one_synthetic_batch(self) -> None:
        result = self._result()
        raw = result.to_raw_results()
        assert set(raw.keys()) == {"sequences", "string", "scores"}
        assert raw["sequences"] == [result.sequences]
        assert raw["string"] == [result.strings]
        assert raw["scores"] == [result.scores]

    def test_to_raw_results_omits_scores_when_absent(self) -> None:
        result = GenerationResult(
            sequences=torch.zeros(2, 1, dtype=torch.long), strings=["a", "b"]
        )
        assert set(result.to_raw_results().keys()) == {"sequences", "string"}

    def test_to_raw_results_carries_top_k_under_scores(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        compressed = compress_scores_top_k(self._result(), tiny_pipeline, k=4)
        raw = compressed.to_raw_results()
        assert raw["scores"] == [compressed.scores_top_k]

    def test_compress_scores_top_k_structures(self, tiny_pipeline: LMPipeline) -> None:
        """Value pin of the per-step top-k structures against ``torch.topk`` +
        the tokenizer directly — the exact structures the retired legacy
        ``convert_to_top_k`` produced (byte-equivalence was pinned against it
        until EU5b, #487, deleted the legacy pass)."""
        result = self._result()
        k = 4
        compressed = compress_scores_top_k(result, tiny_pipeline, k=k)
        assert compressed.scores is None
        assert compressed.scores_top_k is not None
        # The untouched fields carry over.
        assert compressed.strings == result.strings
        assert torch.equal(compressed.sequences, result.sequences)
        assert result.scores is not None  # _result always sets full scores
        assert len(compressed.scores_top_k) == len(result.scores)
        for got, step_logits in zip(compressed.scores_top_k, result.scores):
            want_vals, want_idx = torch.topk(step_logits, k=k, dim=1)
            assert torch.equal(got["top_k_logits"], want_vals)
            assert torch.equal(got["top_k_indices"], want_idx)
            assert got["top_k_tokens"] == [
                [
                    tiny_pipeline.tokenizer.decode([idx], skip_special_tokens=False)
                    for idx in row
                ]
                for row in want_idx.tolist()
            ]

    def test_compress_scores_top_k_clamps_k_to_vocab(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        result = self._result(vocab=5)
        compressed = compress_scores_top_k(result, tiny_pipeline, k=99)
        assert compressed.scores_top_k[0]["top_k_logits"].shape == (3, 5)

    def test_compress_scores_top_k_refuses_nonpositive_k(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        for k in (0, -3):
            with pytest.raises(ValueError, match="k must be positive"):
                compress_scores_top_k(self._result(), tiny_pipeline, k=k)

    def test_compress_scores_top_k_refuses_scoreless_result(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        result = GenerationResult(
            sequences=torch.zeros(1, 1, dtype=torch.long), strings=["a"]
        )
        with pytest.raises(ValueError, match="carries none"):
            compress_scores_top_k(result, tiny_pipeline, k=5)


# --------------------------------------------------------------------------- #
#  LMPipeline.dump                                                            #
# --------------------------------------------------------------------------- #
class TestLMPipelineDumpUnit:
    """``LMPipeline.dump`` decodes whatever ``generate`` /
    ``intervenable_generate`` produced (logits-3D, ids-2D, dict, list /
    tuple) into a string or list of strings. Every analysis that prints
    "prediction" lines goes through here.
    """

    pytestmark = pytest.mark.unit

    def test_2d_ids_decoded_directly(self, tiny_pipeline: LMPipeline) -> None:
        ids = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)
        decoded = tiny_pipeline.dump(ids, is_logits=False)
        assert isinstance(decoded, list)
        assert len(decoded) == 2

    def test_1d_ids_upgraded_to_batch(self, tiny_pipeline: LMPipeline) -> None:
        ids_1d = torch.tensor([5, 6, 7], dtype=torch.long)
        decoded = tiny_pipeline.dump(ids_1d, is_logits=False)
        # Single example → returns str, not list.
        assert isinstance(decoded, str)

    def test_3d_logits_argmax_then_decode(self, tiny_pipeline: LMPipeline) -> None:
        vocab_size = tiny_pipeline.model.config.vocab_size
        logits = torch.randn(1, 2, vocab_size)
        decoded = tiny_pipeline.dump(logits, is_logits=True)
        assert isinstance(decoded, str)  # batch=1 → str

    def test_dict_with_sequences_key(self, tiny_pipeline: LMPipeline) -> None:
        ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        decoded = tiny_pipeline.dump({"sequences": ids, "scores": None})
        assert isinstance(decoded, str)

    def test_dict_falls_back_to_scores_when_no_sequences(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        vocab_size = tiny_pipeline.model.config.vocab_size
        scores = torch.randn(1, 2, vocab_size)
        # dict.get falls through to scores; with dim()==3, is_logits is set
        # to True automatically.
        decoded = tiny_pipeline.dump({"scores": scores})
        assert isinstance(decoded, str)

    def test_list_of_tensors_stacked(self, tiny_pipeline: LMPipeline) -> None:
        # A list of (batch, vocab) logit slices → stacked to (batch, T, vocab).
        vocab_size = tiny_pipeline.model.config.vocab_size
        per_step = [torch.randn(2, vocab_size) for _ in range(3)]
        decoded = tiny_pipeline.dump(per_step)
        assert isinstance(decoded, list) and len(decoded) == 2

    def test_tuple_of_tensors_stacked(self, tiny_pipeline: LMPipeline) -> None:
        vocab_size = tiny_pipeline.model.config.vocab_size
        per_step = tuple(torch.randn(2, vocab_size) for _ in range(3))
        decoded = tiny_pipeline.dump(per_step)
        assert isinstance(decoded, list) and len(decoded) == 2

    def test_single_element_list_unsqueezed(self, tiny_pipeline: LMPipeline) -> None:
        # Single-element list takes the ``.unsqueeze(1)`` branch instead of
        # ``torch.stack``.
        vocab_size = tiny_pipeline.model.config.vocab_size
        decoded = tiny_pipeline.dump([torch.randn(1, vocab_size)])
        assert isinstance(decoded, str)

    def test_unexpected_shape_raises_value_error(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        # Zero-dim tensor → not 1/2/3-D → ValueError.
        with pytest.raises(ValueError, match="Unexpected output shape"):
            tiny_pipeline.dump(torch.tensor(5))

    def test_non_tensor_raises_type_error(self, tiny_pipeline: LMPipeline) -> None:
        with pytest.raises(TypeError, match="model_output must be"):
            tiny_pipeline.dump("not a tensor")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
class TestLMPipelineComputeOutputsUnit:
    """``LMPipeline.compute_outputs`` runs the base + counterfactual halves
    of a ``list[CounterfactualExample]`` through ``generate`` without
    interventions. ``analyses/manifold/fitting_pipeline.py`` and the
    runner's "compute baseline outputs" stage consume the returned per-
    example dicts directly — the flatten/index logic is error-prone, so
    every branch needs direct coverage.
    """

    pytestmark = pytest.mark.unit

    def test_returns_expected_keys(self, tiny_pipeline: LMPipeline) -> None:
        ds = [_cf_example("a", "a1"), _cf_example("b", "b1", "b2")]
        out = tiny_pipeline.compute_outputs(ds, batch_size=1)
        assert set(out.keys()) == {"base_outputs", "counterfactual_outputs"}

    def test_base_outputs_length_equals_dataset_length(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        ds = [_cf_example("a", "a1"), _cf_example("b", "b1", "b2")]
        out = tiny_pipeline.compute_outputs(ds, batch_size=1)
        assert len(out["base_outputs"]) == len(ds)

    def test_counterfactual_outputs_flatten_count(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        ds = [_cf_example("a", "a1"), _cf_example("b", "b1", "b2")]
        expected = sum(len(ex["counterfactual_inputs"]) for ex in ds)
        out = tiny_pipeline.compute_outputs(ds, batch_size=1)
        assert len(out["counterfactual_outputs"]) == expected

    def test_per_example_dict_carries_sequences_scores_string(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        ds = [_cf_example("a", "a1")]
        out = tiny_pipeline.compute_outputs(ds, batch_size=1)
        assert set(out["base_outputs"][0].keys()) >= {"sequences", "scores", "string"}
        assert set(out["counterfactual_outputs"][0].keys()) >= {
            "sequences",
            "scores",
            "string",
        }
        # Sequences are sliced per-example to (1, max_new_tokens).
        assert out["base_outputs"][0]["sequences"].shape == (
            1,
            tiny_pipeline.max_new_tokens,
        )

    def test_empty_counterfactuals_returns_empty_list(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        ds = [_cf_example("a"), _cf_example("b")]  # no counterfactuals
        out = tiny_pipeline.compute_outputs(ds, batch_size=2)
        assert out["counterfactual_outputs"] == []
        assert len(out["base_outputs"]) == 2

    def test_multi_batch_path(self, tiny_pipeline: LMPipeline) -> None:
        # batch_size=1 forces three forward passes for the bases; counts
        # must still match.
        ds = [_cf_example(f"prompt {i}", f"cf {i}") for i in range(3)]
        out = tiny_pipeline.compute_outputs(ds, batch_size=1)
        assert len(out["base_outputs"]) == 3
        assert len(out["counterfactual_outputs"]) == 3


# --------------------------------------------------------------------------- #
#  Property tier — invariants across the public surface                       #
# --------------------------------------------------------------------------- #
class TestLMPipelineProperty:
    """Cross-cutting invariants of ``LMPipeline``'s public surface.

    These hold for any non-empty trace batch against the tiny Llama stub
    (and, by the API contract documented in ``load`` / ``generate`` /
    ``dump``, against any HF causal-LM the pipeline supports).
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_load_preserves_first_dim(
        self, tiny_pipeline: LMPipeline, batch_size: int
    ) -> None:
        traces = [_make_trace(f"prompt {i}") for i in range(batch_size)]
        batch = tiny_pipeline.load(traces)
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["attention_mask"].shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_generate_fields_always_complete(
        self, tiny_pipeline: LMPipeline, batch_size: int
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        assert isinstance(out, GenerationResult)
        assert out.scores is not None
        assert len(out.strings) == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_generate_sequences_shape(
        self, tiny_pipeline: LMPipeline, batch_size: int
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        assert out.sequences.shape == (batch_size, tiny_pipeline.max_new_tokens)

    @pytest.mark.parametrize("batch_size,is_list", [(1, False), (2, True), (4, True)])
    def test_dump_returns_list_iff_batch_greater_than_one(
        self,
        tiny_pipeline: LMPipeline,
        batch_size: int,
        is_list: bool,
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        decoded = tiny_pipeline.dump(out.sequences, is_logits=False)
        assert isinstance(decoded, list) is is_list

    @pytest.mark.parametrize("explicit", ["cpu", "cuda", "mps"])
    def test_resolve_device_idempotent_on_explicit(
        self, explicit: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Toggling cuda/mps availability never overrides an explicit choice.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_device(explicit) == explicit

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device(explicit) == explicit

    @pytest.mark.parametrize("n_base,n_cf_per_base", [(1, 1), (2, 1), (2, 3)])
    def test_compute_outputs_base_length_equals_dataset_length(
        self,
        tiny_pipeline: LMPipeline,
        n_base: int,
        n_cf_per_base: int,
    ) -> None:
        ds = [
            _cf_example(f"b{i}", *[f"c{i}{j}" for j in range(n_cf_per_base)])
            for i in range(n_base)
        ]
        out = tiny_pipeline.compute_outputs(ds, batch_size=2)
        assert len(out["base_outputs"]) == n_base
        assert len(out["counterfactual_outputs"]) == n_base * n_cf_per_base


# --------------------------------------------------------------------------- #
#  device_for_layer (relocated from the retired pyvene module at SH2, #411)   #
# --------------------------------------------------------------------------- #
class TestDeviceForLayerUnit:
    """Direct tests for ``device_for_layer``.

    On a single-device pipeline this returns ``pipeline.model.device``; on a
    sharded pipeline it returns the mapped device for the requested layer.
    Consumers (steering vectors, featurizers) rely on this to keep tensors
    on the same device as the layer's residual stream.
    """

    pytestmark = pytest.mark.unit

    def test_single_device_returns_model_device(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        result = device_for_layer(tiny_pipeline, 0)
        assert isinstance(result, torch.device)
        assert result == tiny_pipeline.model.device

    def test_sharded_returns_mapped_device_per_layer(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synthetic_map = {
            "model.layers.3": "cpu",
            "model.layers.7": "meta",
        }
        monkeypatch.setattr(
            tiny_pipeline.hf_model, "hf_device_map", synthetic_map, raising=False
        )
        assert device_for_layer(tiny_pipeline, 3) == torch.device("cpu")
        assert device_for_layer(tiny_pipeline, 7) == torch.device("meta")


# --------------------------------------------------------------------------- #
#  device_for_layer — property tier                                           #
# --------------------------------------------------------------------------- #
class TestDeviceForLayerProperty:
    """Invariants for ``device_for_layer``.

    * Return type is always ``torch.device`` (never ``str``).
    * Ancestor-path fallback succeeds for any layer when only a coarser key
      (``"model"``) is present in ``hf_device_map`` — the dotted-path walk
      must never raise.
    """

    pytestmark = pytest.mark.property

    def test_return_type_is_torch_device(self, tiny_pipeline: LMPipeline) -> None:
        assert isinstance(device_for_layer(tiny_pipeline, 0), torch.device)

    def test_return_type_is_torch_device_sharded(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tiny_pipeline.hf_model,
            "hf_device_map",
            {"model.layers.0": "cpu"},
            raising=False,
        )
        result = device_for_layer(tiny_pipeline, 0)
        assert isinstance(result, torch.device)

    @pytest.mark.parametrize("layer", [0, 3, 31, 77])
    def test_ancestor_fallback_returns_coarse_key_device(
        self,
        tiny_pipeline: LMPipeline,
        monkeypatch: pytest.MonkeyPatch,
        layer: int,
    ) -> None:
        """When ``hf_device_map`` only has the ancestor key ``"model"``
        (no per-layer keys), the function must walk up the dotted path and
        return the ancestor's device for any layer index, without raising.
        """
        monkeypatch.setattr(
            tiny_pipeline.hf_model, "hf_device_map", {"model": "cpu"}, raising=False
        )
        assert device_for_layer(tiny_pipeline, layer) == torch.device("cpu")


# --------------------------------------------------------------------------- #
#  assert_architecture_supported — fail-fast preflight (#412)                 #
# --------------------------------------------------------------------------- #
class TestArchitecturePreflightUnit:
    """The load-by-name preflight must turn the generic AutoConfig failure on
    too-new architectures into an actionable error (checkpoint, model_type,
    installed vs. required transformers) — and must never get in the way when
    the config is unreadable or the architecture is fine.

    Mocked at the hub boundary (``PretrainedConfig.get_config_dict``), per the
    mocking policy.
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _mock_config_dict(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
        monkeypatch.setattr(
            "transformers.PretrainedConfig.get_config_dict",
            classmethod(lambda cls, name, **kw: (config, {})),
        )

    def test_unknown_model_type_raises_with_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import transformers

        self._mock_config_dict(
            monkeypatch,
            {"model_type": "gemma4", "transformers_version": "5.5.0.dev0"},
        )
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            assert_architecture_supported("google/gemma-4-E2B-it")
        message = str(excinfo.value)
        # The actionable facts: checkpoint, its model_type, both versions.
        assert "google/gemma-4-E2B-it" in message
        assert "gemma4" in message
        assert transformers.__version__ in message
        assert "5.5.0.dev0" in message
        assert "newer transformers" in message

    def test_saved_with_version_omitted_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_config_dict(monkeypatch, {"model_type": "gemma4"})
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            assert_architecture_supported("google/gemma-4-E2B-it")
        assert "saved with" not in str(excinfo.value)

    def test_known_model_type_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # "llama" is in the real CONFIG_MAPPING of any pinned transformers.
        self._mock_config_dict(monkeypatch, {"model_type": "llama"})
        assert_architecture_supported("meta-llama/Llama-3.1-8B")  # no raise

    def test_unreadable_config_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Offline / gated / missing repo: don't get in the way — the normal
        # load path fails downstream exactly as before.
        def _raise(cls, name, **kw):
            raise OSError("offline and not cached")

        monkeypatch.setattr(
            "transformers.PretrainedConfig.get_config_dict", classmethod(_raise)
        )
        assert_architecture_supported("some/unreachable-model")  # no raise

    def test_missing_model_type_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._mock_config_dict(monkeypatch, {"hidden_size": 8})
        assert_architecture_supported("some/model")  # no raise

    def test_remote_code_repo_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # trust_remote_code repos resolve their classes via auto_map, not
        # CONFIG_MAPPING — an unknown model_type there is not our verdict.
        self._mock_config_dict(
            monkeypatch,
            {
                "model_type": "somebrandnewarch",
                "auto_map": {"AutoModelForCausalLM": "modeling.MyModel"},
            },
        )
        assert_architecture_supported("some/remote-code-model")  # no raise

    def test_load_by_name_chokepoint_fails_before_model_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_config_dict(
            monkeypatch,
            {"model_type": "gemma4", "transformers_version": "5.5.0.dev0"},
        )

        def _boom(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "StandardizedTransformer must not be constructed when the "
                "preflight rejects the architecture"
            )

        monkeypatch.setattr("causalab.neural.pipeline.StandardizedTransformer", _boom)
        pipeline = LMPipeline.__new__(LMPipeline)
        pipeline._init_extra_kwargs = {}
        pipeline.tokenizer = object()
        with pytest.raises(UnsupportedArchitectureError):
            pipeline._load_standardized_from_name(
                "google/gemma-4-E2B-it",
                device="cpu",
                dtype=torch.float32,
                hf_token=None,
            )


class TestModelReclamationUnit:
    """Dead backbones must be *gc-reclaimable* — the golden tier's VRAM
    safety net (#440 / #442).

    nnterp's ``StandardizedTransformer.__init__`` stores accessor objects
    (``layers_output``, ``attentions_output``, …) that each hold a strong
    back-reference to the transformer (``LayerAccessor.model``), so dropping
    the last user reference never frees the model by refcount alone —
    reclamation relies entirely on a cycle-collector pass. (Bare nnsight
    ``LanguageModel`` and raw HF models free on refcount; the cycle is
    nnterp-specific.) The per-golden-test hook in
    ``tests/conftest.py::pytest_runtest_teardown`` calls ``gc.collect()`` +
    ``torch.cuda.empty_cache()`` on exactly that assumption. If a future
    nnsight/nnterp bump adds a *global* pin (module-level registry, cache,
    atexit closure), gc would silently stop reclaiming dead models and the
    serial single-process golden tier would OOM again — a nightly-only, GPU
    failure. This CPU test turns that regression into a per-PR failure.
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _load_use_and_drop() -> list[weakref.ref]:
        """Load, exercise, and drop a backbone; return only weakrefs.

        Lives in its own frame on purpose: reference temporaries linger on a
        *live* frame (value stack / hidden locals around the ``with`` tracer)
        and would pin the model from the test frame itself, turning the check
        into a false positive. The frame exiting is the "user dropped their
        last reference" moment the golden tier's hook runs after.
        """
        # Fresh name-based load (never the cached tiny_random_model()
        # instance — the helper's functools.cache would pin it globally and
        # this test is about what pins a model *besides* real user
        # references).
        pipeline = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, device="cpu")
        st = pipeline.model
        # Exercise the tracing path golden tests use, so whatever state a
        # real trace registers is included in the reclamation check.
        with st.trace("hello world"):
            acts = st.layers_output[0].save()
        assert acts.shape[-1] == st.config.hidden_size
        return [weakref.ref(st), weakref.ref(st._model)]

    def test_dropped_backbone_is_reclaimed_by_gc_collect(self) -> None:
        gc.collect()
        gc.disable()  # assert *collectability*, independent of gc timing
        try:
            refs = self._load_use_and_drop()
            gc.collect()
            alive = [r() is not None for r in refs]
            assert alive == [False, False], (
                f"dead backbone still pinned after gc.collect() — "
                f"[StandardizedTransformer, raw HF model] alive={alive}. "
                f"A live global reference now defeats the golden tier's "
                f"per-test GC hook (tests/conftest.py); a serial "
                f"single-process `pytest -m golden` run will re-accumulate "
                f"VRAM until it OOMs (#440)."
            )
        finally:
            gc.enable()
