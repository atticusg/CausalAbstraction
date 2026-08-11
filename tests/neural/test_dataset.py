"""Tests for :mod:`causalab.neural.dataset` — PL3 batched dataset execution
(#405) on the native spec surface (WU3 #505; the pyvene-era unit vocabulary
was deleted by the WU6 sweep, #508).

Tiers (``causalab/neural`` owes ``unit`` + ``property``; the golden GPU pin
for the full task-config → engine → scoring path is
``tests/neural/test_walking_skeleton.py`` and, transitively, the runner
goldens that gate the wrapper reroutes):

* ``unit`` — per-example vector shaping (broadcast / equal-width / ragged),
  the engine's eval-mode reassert, the construction-time feature-width
  guard, the interchange pairwise-width guard, and the no-oracle
  spec-surface contracts (the exported grouping key, the positions=None
  refusal, duplicate-key refusal, the never-raw-int noise lowering rule).
* ``property`` — against the raw-hook oracle on a fresh tiny Llama, through
  real ``pipeline.load`` batches and batch-first-resolved positions:
  ``collect_dataset_features`` equals per-position capture (equal-width,
  ragged, and rotated-featurizer paths; example-major stacking);
  ``run_intervened_generation`` equals HF ``generate`` over a prefill-hooked
  model for interchange (multi-token, batched, ragged), steer add/replace
  (broadcast + per-example vectors), and seeded noise (no-op at scale 0,
  reproducible across engine runs, seed-sensitive); and the rerouted
  generation path composes with installed persistent edits (EU4, #485 — the
  force-staged collect stage pays the full forward instead of stranding a
  deeper persistent mediator, and the terminal generate trace matches the
  prefill-hooked oracle with the persistent steer included).

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
— never the session-cached singletons (leftover pyvene hooks break nnsight
traces).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.dataset import (
    _values_for_rows,
    cf_input_key,
    collect_dataset_features,
    resolve_spec_positions,
    run_intervened_generation,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import Featurizer
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.positions import resolve_positions_batched
from causalab.neural.site import Site
from causalab.neural.specs import EditSpec, SiteSpec
from causalab.neural.token_positions import build_token_positions

from tests._helpers.tiny import fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import capture_component, component_module

_TEMPLATE = "The sum of {x} and {y} is "


def _sample(x: str, y: str) -> CausalTrace:
    text = f"The sum of {x} and {y} is "
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"]),
            "x": Mechanism(parents=[], compute=lambda t: t["x"]),
            "y": Mechanism(parents=[], compute=lambda t: t["y"]),
        },
        inputs={"raw_input": text, "x": x, "y": y},
    )


def _dataset(pairs: list[tuple[tuple[str, str], tuple[str, str]]]) -> list[dict]:
    return [
        {"input": _sample(*base), "counterfactual_inputs": [_sample(*cf)]}
        for base, cf in pairs
    ]


#: base/CF value pairs; y widths differ across examples so batches are ragged
#: in *length* (left-padding exercised) while x stays single-token.
_PAIRS = [
    (("5", "7"), ("9", "8")),
    (("3", "7777777"), ("4", "1111111")),
    (("2", "12345"), ("8", "54321")),
]


@dataclasses.dataclass
class EngineCase:
    pipeline: LMPipeline

    @property
    def oracle(self) -> Any:
        import types

        return types.SimpleNamespace(hf_model=self.pipeline.hf_model)

    def positions(self, name: str):
        specs = {
            "last": {"type": "index", "position": -1},
            "x": {"type": "variable", "name": "x"},
            "y": {"type": "variable", "name": "y"},
        }
        return build_token_positions(specs, _TEMPLATE, self.pipeline)[name]

    def spec(
        self,
        layer: int,
        position: str,
        *,
        component: str = "block_output",
        key: str | None = None,
        featurizer: Featurizer | None = None,
    ) -> SiteSpec:
        """A native :class:`SiteSpec` at ``(component, layer)``, positioned by
        the named resolver — the tests' one spec factory."""
        return SiteSpec(
            fsite=FeaturizedSite(Site(component, layer), featurizer or Featurizer()),
            positions=self.positions(position),
            key=key or f"{component}.L{layer}.{position}",
        )


@pytest.fixture(scope="module")
def case() -> EngineCase:
    raw, _tok = fresh_tiny_random_llama()
    return EngineCase(pipeline=LMPipeline(raw, max_new_tokens=3, padding_side="left"))


def _prefill_patch_generate(
    case: EngineCase,
    inputs: dict[str, torch.Tensor],
    edits: list[tuple[int, list[list[int]], torch.Tensor | None, Any]],
    **gen_kwargs: Any,
) -> Any:
    """Oracle: HF ``generate`` with hooks editing ONLY the prefill forward.

    ``edits``: ``(layer, rows, values, fn)`` — with ``values`` set, row ``i``
    of the batch gets ``values``' example-``i`` slice written at its own
    positions; otherwise ``fn(h, i, row)`` mutates in place. Cached decode
    steps see single-position forwards and are left untouched — the
    prompt-intervention semantics under test.
    """
    handles = []
    for layer, rows, values, fn in edits:
        module = case.pipeline.hf_model.model.layers[layer]

        def hook(m, i, o, rows=rows, values=values, fn=fn):
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] > 1:  # prefill only
                for r, row in enumerate(rows):
                    if fn is not None:
                        fn(h, r, row)
                    else:
                        h[r, row, :] = values[r].to(h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h

        handles.append(module.register_forward_hook(hook))
    defaults = dict(
        max_new_tokens=case.pipeline.max_new_tokens,
        pad_token_id=case.pipeline.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        use_cache=True,
    )
    defaults.update(gen_kwargs)
    try:
        with torch.no_grad():
            return case.pipeline.hf_model.generate(**inputs, **defaults)
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------------------- #
#  unit — vector shaping + engine guards                                       #
# --------------------------------------------------------------------------- #
class TestDatasetUnit:
    pytestmark = pytest.mark.unit

    def test_vector_shaping_broadcast_and_per_example(self) -> None:
        flat = torch.randn(4)
        assert _values_for_rows(flat, [[1], [2]], 0, 2) is flat
        per_example = torch.arange(12.0).reshape(3, 4)
        equal = _values_for_rows(per_example, [[1], [2]], 1, 3)
        assert equal.shape == (2, 1, 4)
        torch.testing.assert_close(equal[:, 0], per_example[1:3])
        ragged = _values_for_rows(per_example, [[1], [2, 3]], 1, 3)
        assert ragged.shape == (3, 4)
        torch.testing.assert_close(ragged[1], ragged[2])  # example 2 repeated

    def test_engine_reasserts_eval_mode(self, case: EngineCase) -> None:
        """#449 finding 5: the engine pins eval mode itself (the dropout
        leakage guard the old collect_features carried) — a model arriving in
        train mode is flipped back before any forward."""
        spec = case.spec(0, "last", component="block_input")
        dataset = _dataset(_PAIRS[:1])

        case.pipeline.hf_model.train()
        collect_dataset_features(case.pipeline, dataset, [spec])
        assert not case.pipeline.hf_model.training

        case.pipeline.hf_model.train()
        run_intervened_generation(case.pipeline, dataset, [[EditSpec(spec)]])
        assert not case.pipeline.hf_model.training

    def test_vector_feature_width_mismatch_raises(self, case: EngineCase) -> None:
        """#449 finding 2 / below-cut 2: the vector-fed edit modes go through
        the ED2 constructors, so a feature-width mismatch fails with the
        legible construction-time ``_check_width`` error instead of a
        scatter/matmul error mid-trace."""
        torch.manual_seed(0)
        feat = SubspaceFeaturizer(
            shape=(case.pipeline.hf_model.config.hidden_size, 3), trainable=False
        )
        dataset = _dataset(_PAIRS[:1])
        for mode in ("replace", "add"):
            spec = case.spec(0, "last", component="block_input", featurizer=feat)
            with pytest.raises(ValueError, match="feature width"):
                run_intervened_generation(
                    case.pipeline,
                    dataset,
                    [[EditSpec(spec, mode=mode, vector=torch.zeros(5))]],
                )

    def test_interchange_width_mismatch_raises(self, case: EngineCase) -> None:
        # x is one token on the base side; pair it against the multi-token y
        # on the counterfactual side via a paired position.
        from causalab.neural.token_positions import paired_token_position

        paired = paired_token_position(
            case.positions("x"), case.positions("y"), id="x<-y"
        )
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=paired,
            key="x<-y",
        )
        dataset = _dataset(_PAIRS)
        with pytest.raises(ValueError, match="widths differ"):
            run_intervened_generation(case.pipeline, dataset, [[EditSpec(spec)]])


# --------------------------------------------------------------------------- #
#  property — collect at scale matches per-position capture                    #
# --------------------------------------------------------------------------- #
class TestCollectDatasetProperty:
    pytestmark = pytest.mark.property

    def _expected_rows(
        self, case: EngineCase, spec: SiteSpec, dataset, batch_size: int
    ) -> torch.Tensor:
        """Per-position capture through the raw-hook oracle, example-major,
        batched exactly like the engine so padded frames line up."""
        site = spec.fsite.site
        module, kind = component_module(case.oracle, site.layer, site.component)
        chunks = []
        for lo in range(0, len(dataset), batch_size):
            traces = [ex["input"] for ex in dataset[lo : lo + batch_size]]
            enc = case.pipeline.load(traces, return_offsets_mapping=True)
            rows = resolve_positions_batched(
                spec.positions, traces, enc, is_original=True
            )
            inputs = {k: enc[k] for k in ("input_ids", "attention_mask")}
            full = capture_component(case.oracle, module, kind, inputs)
            for i, row in enumerate(rows):
                chunks.append(full[i, row, :])
        return torch.cat(chunks, dim=0)

    def test_single_position_collect_matches_capture(self, case: EngineCase) -> None:
        spec = case.spec(1, "last")
        dataset = _dataset(_PAIRS)
        got = collect_dataset_features(case.pipeline, dataset, [spec], batch_size=2)
        expected = self._expected_rows(case, spec, dataset, batch_size=2)
        assert got[spec.key].shape == (3, expected.shape[-1])
        torch.testing.assert_close(got[spec.key], expected, atol=1e-5, rtol=1e-4)

    def test_ragged_collect_matches_capture_example_major(
        self, case: EngineCase
    ) -> None:
        # y spans 1 / 4 / 3 tokens across examples — genuinely ragged rows.
        spec = case.spec(0, "y")
        dataset = _dataset(_PAIRS)
        got = collect_dataset_features(case.pipeline, dataset, [spec], batch_size=2)
        expected = self._expected_rows(case, spec, dataset, batch_size=2)
        assert got[spec.key].shape == expected.shape
        torch.testing.assert_close(got[spec.key], expected, atol=1e-5, rtol=1e-4)

    def test_featurized_collect_applies_rotation(self, case: EngineCase) -> None:
        hidden = int(case.pipeline.hf_model.config.hidden_size)
        torch.manual_seed(0)
        feat = SubspaceFeaturizer(shape=(hidden, 4), trainable=False)
        spec = case.spec(1, "last", featurizer=feat)
        dataset = _dataset(_PAIRS)
        got = collect_dataset_features(case.pipeline, dataset, [spec], batch_size=3)
        raw_spec = case.spec(1, "last")
        raw = self._expected_rows(case, raw_spec, dataset, batch_size=3)
        # Offline featurize of the raw capture is the ground truth; the
        # featurizer math itself is oracle-pinned in test_featurized_site.
        expected, _ = feat.featurize(raw)
        torch.testing.assert_close(
            got[spec.key], expected.to(got[spec.key].dtype), atol=1e-4, rtol=1e-3
        )

    def test_output_logits_contract(self, case: EngineCase) -> None:
        spec = case.spec(0, "last")
        dataset = _dataset(_PAIRS)
        features, logits = collect_dataset_features(
            case.pipeline, dataset, [spec], batch_size=2, collect_output_logits=True
        )
        assert spec.key in features and len(logits) == len(dataset)
        vocab = int(case.pipeline.hf_model.config.vocab_size)
        assert all(row.dim() == 2 and row.shape[-1] == vocab for row in logits)

    def test_duplicate_spec_keys_raise(self, case: EngineCase) -> None:
        spec = case.spec(0, "last")
        with pytest.raises(ValueError, match="duplicate site keys"):
            collect_dataset_features(case.pipeline, _dataset(_PAIRS), [spec, spec])


# --------------------------------------------------------------------------- #
#  property — intervened generation matches the prefill-hooked oracle          #
# --------------------------------------------------------------------------- #
class TestIntervenedGenerationProperty:
    pytestmark = pytest.mark.property

    def _base_inputs_and_rows(self, case: EngineCase, dataset, positions):
        traces = [ex["input"] for ex in dataset]
        enc = case.pipeline.load(traces, return_offsets_mapping=True)
        rows = resolve_positions_batched(positions, traces, enc, is_original=True)
        return {k: enc[k] for k in ("input_ids", "attention_mask")}, rows

    def test_interchange_matches_oracle_generate(self, case: EngineCase) -> None:
        spec = case.spec(1, "last")
        site = spec.fsite.site
        dataset = _dataset(_PAIRS)
        got = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec)]],
            batch_size=len(dataset),
        )

        # Oracle: capture the CF activation at the CF rows, patch it into the
        # base prefill at the base rows, generate with plain HF.
        cf_traces = [ex["counterfactual_inputs"][0] for ex in dataset]
        cf_enc = case.pipeline.load(cf_traces, return_offsets_mapping=True)
        cf_rows = resolve_positions_batched(
            spec.positions, cf_traces, cf_enc, is_original=False
        )
        module, kind = component_module(case.oracle, site.layer, site.component)
        cf_full = capture_component(
            case.oracle,
            module,
            kind,
            {k: cf_enc[k] for k in ("input_ids", "attention_mask")},
        )
        src = torch.stack([cf_full[i, row, :] for i, row in enumerate(cf_rows)])

        inputs, base_rows = self._base_inputs_and_rows(case, dataset, spec.positions)
        manual = _prefill_patch_generate(
            case, inputs, [(site.layer, base_rows, src, None)]
        )
        prompt_len = inputs["input_ids"].shape[1]
        expected_seq = case.pipeline._generated_tokens(manual.sequences, prompt_len)

        assert torch.equal(got.sequences, expected_seq.cpu())
        for step, (a, b) in enumerate(zip(got.scores, manual.scores)):
            (
                torch.testing.assert_close(
                    a.float(), b.float().cpu(), atol=1e-4, rtol=1e-3
                ),
                f"score step {step}",
            )
        # Non-vacuity: the interchange changed the generation vs clean.
        clean = _prefill_patch_generate(case, inputs, [])
        assert not torch.allclose(
            got.scores[0].float(), clean.scores[0].float().cpu(), atol=1e-5
        )

    def test_ragged_interchange_matches_oracle(self, case: EngineCase) -> None:
        # The y variable spans 1/4/3 tokens — ragged on BOTH sides, pairwise
        # width-matched per example by construction (same y width per pair).
        pairs = [
            (("5", "7"), ("9", "8")),
            (("3", "7777777"), ("4", "9999999")),
            (("2", "12345"), ("8", "54321")),
        ]
        spec = case.spec(0, "y")
        site = spec.fsite.site
        dataset = _dataset(pairs)
        got = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec)]],
            batch_size=len(dataset),
            output_scores=True,
        )

        cf_traces = [ex["counterfactual_inputs"][0] for ex in dataset]
        cf_enc = case.pipeline.load(cf_traces, return_offsets_mapping=True)
        cf_rows = resolve_positions_batched(
            spec.positions, cf_traces, cf_enc, is_original=False
        )
        module, kind = component_module(case.oracle, site.layer, site.component)
        cf_full = capture_component(
            case.oracle,
            module,
            kind,
            {k: cf_enc[k] for k in ("input_ids", "attention_mask")},
        )
        per_example_src = [cf_full[i, row, :] for i, row in enumerate(cf_rows)]

        inputs, base_rows = self._base_inputs_and_rows(case, dataset, spec.positions)

        def patch(h: torch.Tensor, i: int, row: list[int]) -> None:
            h[i, row, :] = per_example_src[i].to(h.dtype)

        manual = _prefill_patch_generate(
            case, inputs, [(site.layer, base_rows, None, patch)]
        )
        for a, b in zip(got.scores, manual.scores):
            torch.testing.assert_close(a.float(), b.float().cpu(), atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("mode", ["add", "replace"])
    def test_steer_matches_oracle(self, case: EngineCase, mode: str) -> None:
        hidden = int(case.pipeline.hf_model.config.hidden_size)
        vector = torch.linspace(-3.0, 3.0, hidden)
        scale = 2.0
        spec = case.spec(1, "last")
        site = spec.fsite.site
        dataset = _dataset(_PAIRS)
        got = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec, mode=mode, vector=vector, scale=scale)]],
            batch_size=len(dataset),
        )

        inputs, base_rows = self._base_inputs_and_rows(case, dataset, spec.positions)

        def patch(h: torch.Tensor, i: int, row: list[int]) -> None:
            v = (scale * vector).to(h.dtype)
            h[i, row, :] = h[i, row, :] + v if mode == "add" else v

        manual = _prefill_patch_generate(
            case, inputs, [(site.layer, base_rows, None, patch)]
        )
        for a, b in zip(got.scores, manual.scores):
            torch.testing.assert_close(a.float(), b.float().cpu(), atol=1e-4, rtol=1e-3)

    def test_per_example_vectors_route_per_row(self, case: EngineCase) -> None:
        hidden = int(case.pipeline.hf_model.config.hidden_size)
        vectors = torch.randn(3, hidden)
        spec = case.spec(0, "last")
        site = spec.fsite.site
        dataset = _dataset(_PAIRS)
        got = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec, mode="replace", vector=vectors)]],
            batch_size=2,  # split mid-dataset: slicing must follow the batch
        )

        for lo, hi, batch_idx in ((0, 2, 0), (2, 3, 1)):
            sub = dataset[lo:hi]
            inputs, base_rows = self._base_inputs_and_rows(case, sub, spec.positions)

            def patch(h, i, row, lo=lo):
                h[i, row, :] = vectors[lo + i].to(h.dtype)

            manual = _prefill_patch_generate(
                case, inputs, [(site.layer, base_rows, None, patch)]
            )
            # The flat result concatenates the internal batches in order, so
            # this batch's rows are the [lo:hi] slice of every step tensor.
            for a, b in zip(got.scores, manual.scores):
                torch.testing.assert_close(
                    a[lo:hi].float(), b.float().cpu(), atol=1e-4, rtol=1e-3
                )

    def test_noise_zero_scale_is_no_op_and_seed_reproduces(
        self, case: EngineCase
    ) -> None:
        spec = case.spec(0, "last")
        dataset = _dataset(_PAIRS)

        zero = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec, mode="noise", scale=0.0, seed=3)]],
            batch_size=len(dataset),
        )
        clean = run_intervened_generation(
            case.pipeline, dataset, [[]], batch_size=len(dataset)
        )
        torch.testing.assert_close(zero.scores[0], clean.scores[0])

        first = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec, mode="noise", scale=5.0, seed=3)]],
            batch_size=len(dataset),
        )
        second = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec, mode="noise", scale=5.0, seed=3)]],
            batch_size=len(dataset),
        )
        torch.testing.assert_close(first.scores[0], second.scores[0])
        assert not torch.allclose(first.scores[0], clean.scores[0], atol=1e-5)

    def test_native_noise_reproduces_and_is_seed_sensitive(
        self, case: EngineCase
    ) -> None:
        spec = case.spec(0, "last")
        dataset = _dataset(_PAIRS)

        def run(seed: int) -> torch.Tensor:
            result = run_intervened_generation(
                case.pipeline,
                dataset,
                [[EditSpec(spec, mode="noise", scale=5.0, seed=seed)]],
                batch_size=2,
            )
            assert result.scores is not None
            return result.scores[0]

        torch.testing.assert_close(run(3), run(3), atol=0.0, rtol=0.0)
        assert not torch.allclose(run(3), run(4), atol=1e-5)

    def test_noise_draws_advance_across_batches_like_unbatched(
        self, case: EngineCase
    ) -> None:
        """Batching-invisibility of noise — the one-stream-per-call rule's
        end-to-end guard (replacing the retired legacy-comparison pin): the
        SAME seeded noise run split into internal batches (2 + 1 at
        ``batch_size=2``) must match the single-batch run. ONE
        :class:`SeededNoise` per distinct seed per call is built *before*
        the batch loop, so draws advance across batch boundaries exactly as
        if unbatched (torch's CPU normal fill is chunk-consistent for the
        >= 16-element chunks drawn here). The regression this catches: a
        per-batch re-seed restarts the stream, handing the second batch's
        example the FIRST draws again instead of the stream's later ones —
        its noise then differs from the unbatched run's at the seed-change
        scale, which the tolerance rejects (the neighboring
        seed-sensitivity pin shows same-scale draw mismatches exceed
        ``atol=1e-5``).

        Not bitwise: re-padding an example into a different batch
        composition perturbs its logits at the float32 ulp level (the
        noise-free control measures ~6e-8 on this fixture) — the tolerance
        sits above that jitter floor and far below the draw-mismatch
        scale, so a failure implicates the stream."""
        spec = case.spec(0, "last")
        dataset = _dataset(_PAIRS)

        def run(batch_size: int, *, scale: float) -> torch.Tensor:
            result = run_intervened_generation(
                case.pipeline,
                dataset,
                [[EditSpec(spec, mode="noise", scale=scale, seed=3)]],
                batch_size=batch_size,
            )
            assert result.scores is not None
            return result.scores[0]

        # Control: with the noise zeroed, the batch split contributes only
        # ulp-level padding-composition jitter — the tolerance below is
        # orders of magnitude above it, so the real assertion isolates the
        # noise stream.
        torch.testing.assert_close(
            run(2, scale=0.0), run(len(dataset), scale=0.0), atol=1e-5, rtol=0.0
        )
        torch.testing.assert_close(
            run(2, scale=5.0), run(len(dataset), scale=5.0), atol=1e-5, rtol=0.0
        )

    def test_flat_output_contract(self, case: EngineCase) -> None:
        """EU5a (#486): ONE flat :class:`GenerationResult` across the internal
        batches — the batch split (2 + 1 examples at ``batch_size=2``) never
        leaks into the result. Rewritten from the pre-EU5a batch-nested
        contract pin; values are the same tensors, concatenated."""
        spec = case.spec(0, "last")
        dataset = _dataset(_PAIRS)
        got = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec)]],
            batch_size=2,
        )
        assert isinstance(got, GenerationResult)
        assert got.sequences.shape == (3, case.pipeline.max_new_tokens)
        assert got.sequences.device.type == "cpu"
        assert isinstance(got.strings, list)
        assert len(got.strings) == 3
        assert got.scores is not None
        assert got.scores_top_k is None
        assert len(got.scores) <= case.pipeline.max_new_tokens
        for step in got.scores:
            assert step.shape[0] == 3
            assert step.device.type == "cpu"
        no_scores = run_intervened_generation(
            case.pipeline,
            dataset,
            [[EditSpec(spec)]],
            batch_size=2,
            output_scores=False,
        )
        assert no_scores.scores is None
        assert no_scores.scores_top_k is None

    def test_ragged_step_counts_refuse_with_escape_hatches(
        self, case: EngineCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flatten tail refuses ragged per-batch step counts LOUDLY
        (EU5a #486; the deliberate EU5b #487 ragged-scores decision rides on
        it). Early EOS in exactly one internal batch is not constructible
        deterministically on the tiny-random stub, but the flatten step is
        directly testable: hand ``run_intervened_generation`` hand-built
        ragged per-batch ``(gen, step_scores)`` through the op-less
        ``_plain_generate`` arm — both arms share the ONE flatten tail, so
        the ``ValueError`` raised here is the real refusal, message and both
        escape hatches included."""
        dataset = _dataset(_PAIRS)  # 3 examples; batch_size=2 → 2 + 1 split
        steps_per_call = iter([2, 1])  # ragged: batch 0 → 2 steps, batch 1 → 1

        def fake_plain_generate(pipeline, inputs, *, output_scores=True, **kw):
            n = inputs["input_ids"].shape[0]
            gen = torch.zeros((n, 1), dtype=torch.long)
            return gen, [torch.zeros(n, 7) for _ in range(next(steps_per_call))]

        monkeypatch.setattr(
            "causalab.neural.dataset._plain_generate", fake_plain_generate
        )
        with pytest.raises(
            ValueError,
            match=(
                r"cannot flatten per-step scores.*"
                r"batch_size >= len\(dataset\).*"
                r"min_new_tokens=max_new_tokens"
            ),
        ):
            run_intervened_generation(
                case.pipeline,
                dataset,
                [[]],  # all groups empty → the op-less _plain_generate arm
                batch_size=2,
                output_scores=True,
            )


# --------------------------------------------------------------------------- #
#  property — the terminal generate stage (EU3, #484) vs the same oracle       #
# --------------------------------------------------------------------------- #
class TestGenerateStagingProperty:
    """Cross-input generation through the Plan engine, pinned against the
    SAME prefill-hooked HF-``generate`` oracle ``run_intervened_generation``
    is pinned to (:func:`_prefill_patch_generate`): a generation Plan whose
    edit reads ANOTHER plan input schedules the source read into an earlier
    collect stage and consumes it in the ONE terminal generate trace as a
    constant — the split-forward layout the engine now *derives* (EU4 #485
    reroutes ``run_intervened_generation`` onto exactly this path)."""

    pytestmark = pytest.mark.property

    def test_cross_input_generation_matches_prefill_patch_oracle(
        self, case: EngineCase
    ) -> None:
        from causalab.neural.edit import Edit, ReadSource
        from causalab.neural.plan import EditOp, GenerateSpec, Plan, run_plan

        layer = 1
        dataset = _dataset(_PAIRS)
        base_enc = case.pipeline.load([ex["input"] for ex in dataset])
        cf_enc = case.pipeline.load([ex["counterfactual_inputs"][0] for ex in dataset])
        base = {k: base_enc[k] for k in ("input_ids", "attention_mask")}
        source = {k: cf_enc[k] for k in ("input_ids", "attention_mask")}

        site = FeaturizedSite(Site("block_output", layer))
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        site,
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(site, positions=[-1], input="source"),
                        ),
                        positions=[-1],
                    ),
                ),
            ),
            generate=GenerateSpec(max_new_tokens=3, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.pipeline.model, plan)

        # Oracle: raw-hook capture of the source batch's layer-L activation
        # at its own last token, patched into the base prefill, HF generate.
        module, kind = component_module(case.oracle, layer, "block_output")
        src_vals = capture_component(case.oracle, module, kind, source)[:, [-1], :]
        rows = [[-1]] * len(dataset)
        manual = _prefill_patch_generate(
            case, base, [(layer, rows, src_vals, None)], max_new_tokens=3
        )
        prompt_len = base["input_ids"].shape[1]
        assert torch.equal(
            result.sequences["base"], manual.sequences[:, prompt_len:].cpu()
        )
        for step, (a, b) in enumerate(zip(result.scores["base"], manual.scores)):
            torch.testing.assert_close(
                a.float(),
                b.float().cpu(),
                atol=1e-4,
                rtol=1e-3,
                msg=lambda m, step=step: f"score step {step}: {m}",
            )
        # Non-vacuity: the staged constant changed the generation vs clean.
        clean = _prefill_patch_generate(case, base, [], max_new_tokens=3)
        assert not torch.allclose(
            result.scores["base"][0].float(), clean.scores[0].float().cpu(), atol=1e-5
        ), "inert cross-input patch"


# --------------------------------------------------------------------------- #
#  property — persistent edits × the rerouted generation path (EU4, #485)      #
# --------------------------------------------------------------------------- #
class TestPersistentGenerationProperty:
    """Persistent edits compose with ``run_intervened_generation``'s Plan
    reroute: the force-staged collect stage runs under the installed edits
    and pays the FULL forward (``_stop_carrier`` withholds the CAP6 early
    stop on an edited model — a stop after the shallow produce tap would
    strand the deeper persistent mediator mid-wait, the measured
    ``MissedProviderError``), and the terminal generate trace — which never
    early-stops by construction — applies the installed edit once, to the
    prefill, alongside the plan's own interchange."""

    pytestmark = pytest.mark.property

    def test_generation_with_installed_edit_matches_oracle(
        self, case: EngineCase
    ) -> None:
        from causalab.neural.modes import steer
        from causalab.neural.persistent import persistent_edits

        hidden = int(case.pipeline.hf_model.config.hidden_size)
        deep_layer = int(case.pipeline.hf_model.config.num_hidden_layers) - 1
        vector = torch.linspace(-2.0, 2.0, hidden)
        # Interchange at layer 0 — strictly below the persistent site, so its
        # collect stage's deepest tap sits before the persistent mediator's
        # module: an early stop there would crash the stage (CAP7 pins the
        # guard for plain plans; this pins it for a generation plan's
        # force-staged collect stage).
        spec = case.spec(0, "last")
        site = spec.fsite.site
        assert deep_layer > site.layer
        dataset = _dataset(_PAIRS)

        with persistent_edits(
            case.pipeline.model, steer(Site("block_output", deep_layer), vector)
        ):
            got = run_intervened_generation(
                case.pipeline,
                dataset,
                [[EditSpec(spec)]],
                batch_size=len(dataset),
            )

        # Oracle: raw-hook capture of the CF activation at layer 0 (below the
        # persistent site, hence unaffected by it), patched into the base
        # prefill, plus the persistent steer applied to the prefill only (an
        # installed edit under a traced generate fires once, on the prefill —
        # pinned in test_persistent.py), plain HF generate. Runs after the
        # context exits: raw-HF forwards bypass nnsight edits either way.
        cf_traces = [ex["counterfactual_inputs"][0] for ex in dataset]
        cf_enc = case.pipeline.load(cf_traces, return_offsets_mapping=True)
        cf_rows = resolve_positions_batched(
            spec.positions, cf_traces, cf_enc, is_original=False
        )
        module, kind = component_module(case.oracle, site.layer, site.component)
        cf_full = capture_component(
            case.oracle,
            module,
            kind,
            {k: cf_enc[k] for k in ("input_ids", "attention_mask")},
        )
        src = torch.stack([cf_full[i, row, :] for i, row in enumerate(cf_rows)])

        base_traces = [ex["input"] for ex in dataset]
        enc = case.pipeline.load(base_traces, return_offsets_mapping=True)
        base_rows = resolve_positions_batched(
            spec.positions, base_traces, enc, is_original=True
        )
        inputs = {k: enc[k] for k in ("input_ids", "attention_mask")}

        def steer_all(h: torch.Tensor, i: int, row: list[int]) -> None:
            # The installed steer has positions=None: every prefill position.
            h[i, :, :] = h[i, :, :] + vector.to(h.device, h.dtype)

        one_call_per_row = [[0]] * len(dataset)  # rows unused by steer_all
        manual = _prefill_patch_generate(
            case,
            inputs,
            [
                (site.layer, base_rows, src, None),
                (deep_layer, one_call_per_row, None, steer_all),
            ],
        )
        prompt_len = inputs["input_ids"].shape[1]
        expected_seq = case.pipeline._generated_tokens(manual.sequences, prompt_len)
        assert torch.equal(got.sequences, expected_seq.cpu())
        for a, b in zip(got.scores, manual.scores):
            torch.testing.assert_close(a.float(), b.float().cpu(), atol=1e-4, rtol=1e-3)

        # Non-vacuity: the persistent steer moved the output vs the same
        # interchange without it — the traced path really carried the edit.
        without = _prefill_patch_generate(
            case, inputs, [(site.layer, base_rows, src, None)]
        )
        assert not torch.allclose(
            got.scores[0].float(), without.scores[0].float().cpu(), atol=1e-5
        )


# --------------------------------------------------------------------------- #
#  WU3 (#505) — the spec-typed surface: keys, positions, noise lowering        #
# --------------------------------------------------------------------------- #
class TestSpecSurfaceUnit:
    """WU3 (#505) contracts that need no oracle: the exported grouping key,
    the positions=None refusal, duplicate-key refusal, and the never-raw-int
    noise lowering rule."""

    pytestmark = pytest.mark.unit

    def test_cf_input_key(self) -> None:
        # The grouping contract's naming half: group g reads
        # example["counterfactual_inputs"][g] under plan input cf_input_key(g).
        assert cf_input_key(0) == "cf_0"
        assert cf_input_key(3) == "cf_3"

    def test_positions_none_refused_at_resolution(self) -> None:
        # The amended #505 rule: an unbound spec (positions=None) is refused
        # loudly — None is NOT "the whole sequence" (padded batches would
        # silently include pad positions). The refusal fires before traces or
        # encoding are touched.
        spec = SiteSpec(FeaturizedSite(Site("block_output", 0)), None, key="unbound")
        with pytest.raises(ValueError, match="positions=None"):
            resolve_spec_positions(spec, [], None)

    def test_positions_none_refused_in_both_entries(self, case: EngineCase) -> None:
        spec = SiteSpec(FeaturizedSite(Site("block_output", 0)), None, key="unbound")
        dataset = _dataset(_PAIRS[:1])
        with pytest.raises(ValueError, match="positions=None"):
            collect_dataset_features(case.pipeline, dataset, [spec])
        with pytest.raises(ValueError, match="positions=None"):
            run_intervened_generation(case.pipeline, dataset, [[EditSpec(spec)]])

    def test_noise_lowering_refuses_raw_seed_without_stream(self) -> None:
        # The structural half of the noise rule: THE spec->engine conversion
        # point refuses to lower a noise edit without a run-shared stream —
        # re-seeding from the raw int per batch would repeat the same noise
        # across batch boundaries (modes.SeededNoise's documented hazard).
        from causalab.neural.dataset import _edit_spec_to_edit

        spec = SiteSpec(
            FeaturizedSite(Site("block_output", 0)), positions=[0], key="resid.L0"
        )
        edit = EditSpec(spec, mode="noise", scale=1.0)
        with pytest.raises(ValueError, match="run-shared SeededNoise"):
            _edit_spec_to_edit(edit, edit.site.fsite, [[0]], None, None, 0, 1)

    def test_native_duplicate_keys_refused(self, case: EngineCase) -> None:
        spec = case.spec(0, "last", key="k")
        with pytest.raises(ValueError, match="duplicate site keys"):
            collect_dataset_features(case.pipeline, _dataset(_PAIRS), [spec, spec])
