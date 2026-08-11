"""Direct tests for ``causalab.neural.activations.collect``.

This module hosts the dataset-wide activation harvesting primitive that feeds
every featurizer learner (``methods/pca``, ``methods/spline``,
``methods/flow``) and every interchange / path-steering analysis.
``collect_features`` produces a ``{spec.key -> (n_samples, hidden)}`` dict on
CPU.

If it returns the wrong shape, ordering, or device, every downstream
interchange / featurizer training run silently consumes corrupted
activations — so this module is the canonical place to pin those contracts.

The *unit* class keeps the historical mocked scaffolding (it exists to assert
delegation / dict-keying / guard behaviour, not numerics). The *property*
class uses the real ``tiny_pipeline`` fixture from
``tests/neural/conftest.py`` so the engine path is exercised end-to-end
against a real (tiny) model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.activations.collect import collect_features
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition


# --------------------------------------------------------------------------- #
#  Local RNG helper                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Module-level fixtures shared across the mocked unit-tier classes           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_dataset() -> list[dict]:
    """Tiny mock dataset matching the ``CounterfactualExample`` schema."""
    return [
        {"input": "input_1"},
        {"input": "input_2"},
        {"input": "input_3"},
    ]


@pytest.fixture
def mock_loaded_inputs() -> dict[str, torch.Tensor]:
    """Mock output of ``pipeline.load`` — a batch of tokenized inputs."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }


@pytest.fixture
def residual_sites() -> list[SiteSpec]:
    """Two distinct residual-stream ``SiteSpec``s (distinct ``key``s)."""
    return [
        SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=[0, 1],
            key="resid_L0",
        ),
        SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 2)),
            positions=[0, 1],
            key="resid_L2",
        ),
    ]


@pytest.fixture
def mock_pipeline(mock_loaded_inputs) -> MagicMock:
    """Lightweight ``Pipeline`` stand-in: just needs ``model.eval()`` + ``load()``.

    Avoids ``mock_pipeline`` (deprecated; tries to ``AutoTokenizer.from_pretrained``
    the literal string ``"mock_model"``). ``collect_features`` only touches
    ``pipeline.model.eval()`` and ``pipeline.load(...)`` on the pipeline
    itself — everything else flows through the mocked engine.
    """
    pipeline = MagicMock()
    pipeline.load = MagicMock(return_value=mock_loaded_inputs)
    pipeline.model = MagicMock()
    return pipeline


# --------------------------------------------------------------------------- #
#  collect_features                                                           #
# --------------------------------------------------------------------------- #
class TestCollectFeaturesUnit:
    """``collect_features`` is a thin public wrapper since the PL3 reroute
    (#405): it owns the duplicate-site-key guard (with the message downstream
    tooling greps for) and delegates verbatim to
    ``causalab.neural.dataset.collect_dataset_features`` — whose behaviour
    (shapes, ordering, ragged spans, logits arm) is oracle-pinned in
    ``tests/neural/test_dataset.py``. These units pin the wrapper contract."""

    pytestmark = pytest.mark.unit

    def _spy(self, monkeypatch, result):
        calls: list[dict] = []

        def fake(pipeline, dataset, sites, batch_size, collect_output_logits):
            calls.append(
                dict(
                    pipeline=pipeline,
                    dataset=dataset,
                    sites=sites,
                    batch_size=batch_size,
                    collect_output_logits=collect_output_logits,
                )
            )
            return result

        monkeypatch.setattr("causalab.neural.dataset.collect_dataset_features", fake)
        return calls

    def test_delegates_to_engine_and_returns_result(
        self, monkeypatch, mock_pipeline, residual_sites, mock_dataset
    ) -> None:
        sentinel = {"site": torch.zeros(3, 4)}
        calls = self._spy(monkeypatch, sentinel)
        result = collect_features(
            mock_dataset, mock_pipeline, residual_sites, batch_size=2
        )
        assert result is sentinel
        assert calls == [
            dict(
                pipeline=mock_pipeline,
                dataset=mock_dataset,
                sites=residual_sites,
                batch_size=2,
                collect_output_logits=False,
            )
        ]

    def test_output_logits_flag_threads_through(
        self, monkeypatch, mock_pipeline, residual_sites, mock_dataset
    ) -> None:
        sentinel = ({"site": torch.zeros(3, 4)}, [torch.zeros(3, 8)])
        calls = self._spy(monkeypatch, sentinel)
        result = collect_features(
            mock_dataset,
            mock_pipeline,
            residual_sites,
            collect_output_logits=True,
        )
        assert result is sentinel
        assert calls[0]["collect_output_logits"] is True

    def test_duplicate_site_keys_raise_before_any_execution(
        self, monkeypatch, mock_pipeline, mock_dataset
    ) -> None:
        # The guard fires before the engine import/delegation; downstream
        # consumers assume one entry per site key.
        def boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("engine must not run on duplicate keys")

        monkeypatch.setattr("causalab.neural.dataset.collect_dataset_features", boom)
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=[0],
            key="dup",
        )
        with pytest.raises(ValueError, match="Duplicate site key"):
            collect_features(mock_dataset, mock_pipeline, [spec, spec])


def _trace(text: str) -> CausalTrace:
    """Helper: build a minimal ``CausalTrace`` carrying just ``raw_input``."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _make_residual_spec(pipeline, layer: int) -> SiteSpec:
    """Helper: build a residual-stream ``SiteSpec`` at ``layer``, last token."""

    def last_token(trace):
        return [len(pipeline.load([trace])["input_ids"][0]) - 1]

    tp = TokenPosition(last_token, pipeline, id="last_token")
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", layer)),
        positions=tp,
        key=f"resid_L{layer}_last_token",
        width=pipeline.model.config.hidden_size,
    )


def _make_residual_spec_at_token(pipeline, layer: int, token_index: int) -> SiteSpec:
    """``SiteSpec`` reading a FIXED token index (unpadded-frame; the engine's
    batched position resolution rebases it into the padded batch). Reading a
    *non-final* token is what surfaces the left-pad ``position_ids`` bug."""
    tp = TokenPosition(lambda trace: [token_index], pipeline, id=f"tok{token_index}")
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", layer)),
        positions=tp,
        key=f"resid_L{layer}_tok{token_index}",
        width=pipeline.model.config.hidden_size,
    )


class TestCollectFeaturesProperty:
    """Tier-property invariants for ``collect_features`` on a real (tiny) model.

    These tests exercise the engine's collection path end-to-end against
    ``tests/neural/conftest.py::tiny_pipeline`` — the same shape contract the
    production runners hit. ``collect_features`` now pins ``model.eval()`` +
    ``torch.no_grad()`` for the duration of collection; these tests assert
    the resulting tensors have no autograd graph.
    """

    pytestmark = pytest.mark.property

    def test_sample_count_matches_dataset_length(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace(f"hello {i}")} for i in range(5)]
        spec = _make_residual_spec(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [spec], batch_size=2)
        # One scalar position per example × 5 examples → (5, hidden).
        assert result[spec.key].shape[0] == len(dataset)

    def test_tensors_are_on_cpu(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace("hello world")}]
        spec = _make_residual_spec(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [spec], batch_size=1)
        for t in result.values():
            assert t.device.type == "cpu"

    def test_deterministic_under_fixed_seed(self, tiny_pipeline) -> None:
        """Calling twice on the same dataset → byte-identical tensors."""
        dataset = [{"input": _trace(f"hello {i}")} for i in range(3)]
        spec_a = _make_residual_spec(tiny_pipeline, layer=0)
        out_a = collect_features(dataset, tiny_pipeline, [spec_a], batch_size=2)
        spec_b = _make_residual_spec(tiny_pipeline, layer=0)
        out_b = collect_features(dataset, tiny_pipeline, [spec_b], batch_size=2)
        # Same key-template since both specs target layer 0 / last_token.
        (key_a,) = out_a.keys()
        (key_b,) = out_b.keys()
        assert torch.equal(out_a[key_a], out_b[key_b])

    def test_no_grad_is_pinned_during_collection(self, tiny_pipeline) -> None:
        """Collected tensors must not carry an autograd graph (forward-only)."""
        dataset = [{"input": _trace("hello world")}]
        spec = _make_residual_spec(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [spec], batch_size=1)
        for t in result.values():
            assert t.requires_grad is False
            assert t.grad_fn is None

    def test_dict_keys_equal_spec_keys(self, tiny_pipeline) -> None:
        spec0 = _make_residual_spec(tiny_pipeline, layer=0)
        spec1 = _make_residual_spec(tiny_pipeline, layer=1)
        dataset = [{"input": _trace("hello world")}]
        result = collect_features(dataset, tiny_pipeline, [spec0, spec1], batch_size=1)
        assert set(result.keys()) == {spec0.key, spec1.key}

    @pytest.mark.parametrize(
        "pipeline_fixture", ["tiny_pipeline", "tiny_gpt2_pipeline"]
    )
    def test_left_padded_collection_matches_unpadded(
        self, pipeline_fixture, request
    ) -> None:
        """End-to-end guard for the position_ids fix on a real model.

        Collecting a non-final token must be identical whether each example is
        processed alone (``batch_size=1`` → no padding, the reference) or together
        in one left-padded batch. On an **absolute-position** model
        (``tiny_gpt2_pipeline``) this fails without ``position_ids`` on the
        collection forward — the padded row is mis-encoded (the ROME-replication
        bug). On a **RoPE** model (``tiny_pipeline``) it holds either way, since
        relative positions are invariant to a uniform left-pad shift.
        Parametrizing both pins "GPT-2 is fixed" and "Llama is unchanged" together.
        """
        pipeline = request.getfixturevalue(pipeline_fixture)
        dataset = [
            {"input": _trace("the cat sat quietly on the warm windowsill")},
            {
                "input": _trace(
                    "the quick brown fox jumps over the lazy dog again and again today"
                )
            },
        ]
        # Guards so the test actually exercises the bug: the short row must be
        # left-padded in the batch, and long enough to read a non-final token.
        short_len = int(pipeline.load([dataset[0]["input"]])["input_ids"].shape[1])
        batch_mask = pipeline.load([ex["input"] for ex in dataset])["attention_mask"]
        assert (batch_mask == 0).any(), "batch did not introduce left padding"
        assert short_len >= 3, f"short prompt too short ({short_len} tokens)"

        spec_solo = _make_residual_spec_at_token(pipeline, layer=1, token_index=1)
        spec_batched = _make_residual_spec_at_token(pipeline, layer=1, token_index=1)
        solo = collect_features(dataset, pipeline, [spec_solo], batch_size=1)
        batched = collect_features(dataset, pipeline, [spec_batched], batch_size=2)

        assert torch.allclose(
            solo[spec_solo.key], batched[spec_batched.key], atol=1e-4
        ), (
            f"[{pipeline_fixture}] left-padded collection diverged from unpadded at "
            "a non-final token — position_ids on the collection forward is what "
            "makes these equal on an absolute-position model"
        )

    def test_duplicate_site_key_raises_on_real_pipeline(self, tiny_pipeline) -> None:
        """Duplicate-key contract must hold against the real pipeline too."""
        spec_a = _make_residual_spec(tiny_pipeline, layer=0)
        spec_b = _make_residual_spec(tiny_pipeline, layer=0)
        # Both specs are layer 0 + last_token → identical ``key``.
        assert spec_a.key == spec_b.key
        dataset = [{"input": _trace("hello")}]
        with pytest.raises(ValueError, match="Duplicate site key"):
            collect_features(dataset, tiny_pipeline, [spec_a, spec_b], batch_size=1)

    def test_multitoken_span_collects_each_position_independently(
        self, tiny_pipeline
    ) -> None:
        """A uniform multi-token span collects one d-vector PER position.

        With ``keep_last_dim=True`` (PR #334) the collect intervention gathers
        ``(b, num_pos, d)`` and ``collect_features`` flattens it to
        ``(b*num_pos, d)`` — so the span's per-position rows must equal collecting
        each position on its own. The old folded ``(b, num_pos*d)`` reshape
        interleaved positions and features and would not match. This is the one
        consumer of the collect change (review obs #2) the single-token path
        leaves byte-identical; here we lock the *multi-token* path.
        """
        hidden = tiny_pipeline.model.config.hidden_size
        dataset = [{"input": _trace(f"hello world {i}")} for i in range(3)]

        def _spec(positions: list[int], key: str) -> SiteSpec:
            tp = TokenPosition(lambda _x, p=positions: p, tiny_pipeline, id=key)
            return SiteSpec(
                fsite=FeaturizedSite(Site("block_output", 0)),
                positions=tp,
                key=key,
                width=hidden,
            )

        span = _spec([0, 1], "span01")
        pos0 = _spec([0], "pos0")
        pos1 = _spec([1], "pos1")

        out_span = collect_features(dataset, tiny_pipeline, [span], batch_size=2)[
            span.key
        ]
        out_p0 = collect_features(dataset, tiny_pipeline, [pos0], batch_size=2)[
            pos0.key
        ]
        out_p1 = collect_features(dataset, tiny_pipeline, [pos1], batch_size=2)[
            pos1.key
        ]

        # (b, num_pos, d) flattened row-major: rows are [ex0-pos0, ex0-pos1, ...].
        assert out_span.shape == (len(dataset) * 2, hidden)
        assert torch.allclose(out_span[0::2], out_p0, atol=1e-5)
        assert torch.allclose(out_span[1::2], out_p1, atol=1e-5)
