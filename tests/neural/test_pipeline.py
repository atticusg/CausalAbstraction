"""Direct tests for ``causalab.neural.pipeline``.

``LMPipeline`` is the model-wiring boundary between the symbolic causal
layer and HuggingFace: it loads the tokenizer/model on the resolved device,
turns a ``list[CausalTrace]`` into batched ``input_ids`` / ``attention_mask``
/ optionally ``position_ids`` (``load``), runs greedy generation that always
returns a ``{scores, sequences, string}`` dict (``generate``), and runs
PyVENE interventions with the same return contract
(``intervenable_generate``). Every analysis under
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

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.pipeline import (
    LMPipeline,
    Pipeline,
    ensure_position_ids,
    left_pad_position_ids,
    resolve_device,
)
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME, tiny_random_model


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
            # Abstract methods (_setup_model, load, dump, generate,
            # intervenable_generate) make direct construction illegal.
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

            def intervenable_generate(  # type: ignore[override]
                self,
                intervenable_model: Any,
                base: Any,
                sources: Any,
                map: Any,  # noqa: A002
                feature_indices: Any,
                source_representations: Any = None,
            ) -> dict[str, Any]:
                return {}

        stub = _Stub("name")
        assert stub.model_or_name == "name"
        assert calls == ["setup"]


# --------------------------------------------------------------------------- #
#  LMPipeline.__init__ / _setup_model / get_num_*                             #
# --------------------------------------------------------------------------- #
class TestLMPipelineInitUnit:
    """``LMPipeline.__init__`` wires the HF model + tokenizer onto the
    resolved device and normalises ``pad_token`` / ``generation_config`` so
    downstream ``generate`` calls don't warn about ignored sampling fields.
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

    def test_pad_token_is_eos(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.tokenizer.pad_token == tiny_pipeline.tokenizer.eos_token

    def test_padding_side_set_from_kwarg(self, tiny_pipeline: LMPipeline) -> None:
        assert tiny_pipeline.tokenizer.padding_side == "left"

    def test_use_cache_disabled_on_config_for_string_model(self) -> None:
        # _setup_model sets use_cache=False on the model config when loading
        # from a model name (string path). The pre-loaded-model path
        # deliberately leaves the user's config untouched. (generate()
        # passes use_cache=True explicitly via the defaults dict regardless.)
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left")
        assert pipe.model.config.use_cache is False

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

    def test_load_weights_false_yields_simplenamespace_config(self) -> None:
        # load_weights=False is the InterchangeTargets-only path: build the
        # tokenizer + config without paying for weight load. Forward passes
        # will fail (no .device, no parameters), so the constraint is
        # documented here: don't call generate/load in this mode.
        pipe = LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, load_weights=False)
        assert isinstance(pipe.model, SimpleNamespace)
        assert hasattr(pipe.model, "config")
        # Hidden size is still readable so InterchangeTargets can size
        # activation buffers without forward.
        assert pipe.model.config.hidden_size > 0


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
    and always returns ``{"scores", "sequences", "string"}`` — every
    downstream baseline runner depends on this contract.
    """

    pytestmark = pytest.mark.unit

    def test_returns_expected_keys(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("hello"), _make_trace("world")])
        assert set(out.keys()) == {"scores", "sequences", "string"}

    def test_sequences_shape_equals_batch_x_max_new_tokens(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        out = tiny_pipeline.generate([_make_trace("a"), _make_trace("b")])
        assert out["sequences"].shape == (2, tiny_pipeline.max_new_tokens)

    def test_scores_length_equals_max_new_tokens(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")])
        assert isinstance(out["scores"], list)
        assert len(out["scores"]) == tiny_pipeline.max_new_tokens

    def test_sequences_returned_on_cpu(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")])
        assert out["sequences"].device.type == "cpu"
        for s in out["scores"]:
            assert s.device.type == "cpu"

    def test_string_is_list_for_multi_batch(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("a"), _make_trace("b")])
        assert isinstance(out["string"], list)
        assert len(out["string"]) == 2

    def test_string_is_str_for_single_example(self, tiny_pipeline: LMPipeline) -> None:
        out = tiny_pipeline.generate([_make_trace("hi")])
        # dump() returns a bare str when len(decoded) == 1.
        assert isinstance(out["string"], str)


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
#  LMPipeline.intervenable_generate                                           #
# --------------------------------------------------------------------------- #
class _IntervenableModelStub:
    """Pyvene-shaped stub recording ``.generate`` call args.

    Not a mock of internal numerical code: PyVENE is an external glue
    library, not a causalab numerical primitive, so this is within the
    create-tests "no mocks of internal numerical code" rule. The
    real ``IntervenableModel.generate`` returns a tuple whose last element
    has ``.sequences`` and ``.scores`` (HF-style); we mirror that exactly.
    """

    def __init__(self, batch_size: int, max_new_tokens: int, vocab_size: int) -> None:
        self.last_call: dict[str, Any] = {}
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        self._vocab_size = vocab_size

    def generate(self, base: Any, **kwargs: Any) -> Any:
        self.last_call = dict(kwargs)
        self.last_call["base"] = base
        seqs = torch.zeros(
            self._batch_size,
            base["input_ids"].shape[1] + self._max_new_tokens,
            dtype=torch.long,
        )
        scores = [
            torch.randn(self._batch_size, self._vocab_size)
            for _ in range(self._max_new_tokens)
        ]
        last = SimpleNamespace(sequences=seqs, scores=scores)
        return (None, last)


class TestLMPipelineIntervenableGenerateUnit:
    """``LMPipeline.intervenable_generate`` is the PyVENE-shaped wrapper
    every DAS / boundless / grid / pca / dbm / manifold-fitting analysis
    invokes. Its return contract must match ``generate`` (``sequences``
    always, ``scores`` when ``output_scores=True``, ``string`` from
    ``dump``) — drift here silently breaks every intervention experiment.
    """

    pytestmark = pytest.mark.unit

    def test_default_keys_include_sequences_scores_string(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        out = tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map={"sources->base": ([[0]], [[0]])},
            feature_indices=[[0]],
        )
        assert set(out.keys()) == {"sequences", "scores", "string"}

    def test_passes_unit_locations_and_subspaces(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        location_map = {"sources->base": ([[0]], [[0]])}
        feature_indices = [[1, 2]]
        tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map=location_map,
            feature_indices=feature_indices,
        )
        assert stub.last_call["unit_locations"] == location_map
        assert stub.last_call["subspaces"] == feature_indices

    def test_intervene_on_prompt_default_true(self, tiny_pipeline: LMPipeline) -> None:
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map={"sources->base": ([[0]], [[0]])},
            feature_indices=[[0]],
        )
        assert stub.last_call["intervene_on_prompt"] is True
        assert stub.last_call["do_sample"] is False

    def test_source_representations_passed_through(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        # Cross-model patching path: sources=None and pre-computed
        # source_representations.
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        reps = [torch.randn(1, 4)]
        tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map={"sources->base": ([[0]], [[0]])},
            feature_indices=[[0]],
            source_representations=reps,
        )
        assert stub.last_call["source_representations"] is reps
        assert stub.last_call["sources"] is None

    def test_output_scores_false_drops_scores_key(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        out = tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map={"sources->base": ([[0]], [[0]])},
            feature_indices=[[0]],
            output_scores=False,
        )
        assert "scores" not in out
        assert "sequences" in out
        assert "string" in out


# --------------------------------------------------------------------------- #
#  LMPipeline.compute_outputs                                                 #
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
    def test_generate_keys_always_complete(
        self, tiny_pipeline: LMPipeline, batch_size: int
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        assert set(out.keys()) == {"scores", "sequences", "string"}

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_generate_sequences_shape(
        self, tiny_pipeline: LMPipeline, batch_size: int
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        assert out["sequences"].shape == (batch_size, tiny_pipeline.max_new_tokens)

    @pytest.mark.parametrize("batch_size,is_list", [(1, False), (2, True), (4, True)])
    def test_dump_returns_list_iff_batch_greater_than_one(
        self,
        tiny_pipeline: LMPipeline,
        batch_size: int,
        is_list: bool,
    ) -> None:
        traces = [_make_trace(f"p {i}") for i in range(batch_size)]
        out = tiny_pipeline.generate(traces)
        decoded = tiny_pipeline.dump(out["sequences"], is_logits=False)
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

    def test_intervenable_generate_keys_subset_of_generate_keys(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        batch = tiny_pipeline.load([_make_trace("hi")])
        stub = _IntervenableModelStub(
            batch_size=1,
            max_new_tokens=tiny_pipeline.max_new_tokens,
            vocab_size=tiny_pipeline.model.config.vocab_size,
        )
        out = tiny_pipeline.intervenable_generate(
            intervenable_model=stub,  # type: ignore[arg-type]
            base=batch,
            sources=None,
            map={"sources->base": ([[0]], [[0]])},
            feature_indices=[[0]],
        )
        assert set(out.keys()) <= {"scores", "sequences", "string"}

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
