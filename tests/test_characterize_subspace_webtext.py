"""Unit tests for the max-token webtext representation and figure helpers.

Covers the pure logic introduced in issue #210 — per-token argmax selection,
context-window extraction (markers, boundaries, padding, sanitisation), the
PCA k<3 fallback, decile assignment / embedding cap, and the v2 cache
round-trip — all on tiny CPU stubs with no model download or network.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os

import pytest
import torch

from causalab.analyses.characterize_subspace import webtext as wt
from causalab.analyses.characterize_subspace.bundle import _decile_stratified_indices
from causalab.analyses.characterize_subspace.figures import (
    _hover_window,
    _pca_coords,
)
from causalab.analyses.characterize_subspace.loading import SubspaceProjector
from causalab.analyses.characterize_subspace.reproduction import run_gate
from causalab.analyses.characterize_subspace.schemas import Significance
from causalab.analyses.characterize_subspace.webtext import (
    _extract_window,
    collect_text_projections,
    collect_webtext_evidence,
)

pytestmark = pytest.mark.unit


class _CharTokenizer:
    """Maps each character to its ``ord`` as a token id; ``0`` is padding."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        texts,
        *,
        return_tensors: str | None = None,
        padding: bool | str = False,  # noqa: ARG002
        truncation: bool = False,  # noqa: ARG002
        max_length: int | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        max_len = max_length or 64
        encoded = [[ord(c) for c in t][:max_len] for t in texts]
        width = max((len(e) for e in encoded), default=0)
        encoded = [e + [0] * (width - len(e)) for e in encoded]
        if return_tensors == "pt":
            ids = torch.tensor(encoded, dtype=torch.long)
            attn = (ids != 0).long()
            return {"input_ids": ids, "attention_mask": attn}
        return {"input_ids": encoded}

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(int(i)) for i in ids)


class _BosCharTokenizer(_CharTokenizer):
    """Like :class:`_CharTokenizer` but prepends a BOS token (id 1) per sequence."""

    bos_token_id = 1

    def __call__(self, texts, *, return_tensors=None, **kw):
        out = super().__call__(texts, return_tensors=return_tensors, **kw)
        if return_tensors == "pt":
            ids = out["input_ids"]
            bos = torch.full((ids.shape[0], 1), self.bos_token_id, dtype=torch.long)
            ids = torch.cat([bos, ids], dim=1)
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}
        return out


class _IdHiddenModel:
    """Stub HF model whose residual dim-0 equals ``token_id - offset``.

    Returns ``n_layers + 1`` hidden-state tensors of shape ``(B, T, d_model)``
    with channel 0 set to the (float) token id minus ``offset`` and the rest
    zero, so a standard-basis projector reads dim-0 = token id - offset.
    A non-zero ``offset`` lets a test produce signed (negative) projections.
    """

    def __init__(
        self, d_model: int = 4, n_layers: int = 2, offset: float = 0.0
    ) -> None:
        self.d_model = d_model
        self.n_layers = n_layers
        self.offset = offset

    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # noqa: ARG002
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        output_hidden_states: bool = False,  # noqa: ARG002
        return_dict: bool = True,  # noqa: ARG002
    ):
        b, t = input_ids.shape
        hs = torch.zeros(b, t, self.d_model)
        hs[..., 0] = input_ids.float() - self.offset
        states = tuple(hs.clone() for _ in range(self.n_layers + 1))
        return type("_Out", (), {"hidden_states": states})()


class _DictModel:
    """Stub model that sets (dim0, dim1) of the residual per token id from a map.

    Tokens not in ``mapping`` get ``(0, 0)``. With the standard-basis projector
    this lets a test give a token a small dim-0 but a large dim-1 so that the
    full-subspace norm (not the dim-0 coordinate) decides the peak.
    """

    def __init__(self, mapping, d_model: int = 4, n_layers: int = 2) -> None:
        self.mapping = mapping
        self.d_model = d_model
        self.n_layers = n_layers

    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # noqa: ARG002
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        output_hidden_states: bool = False,  # noqa: ARG002
        return_dict: bool = True,  # noqa: ARG002
    ):
        b, t = input_ids.shape
        hs = torch.zeros(b, t, self.d_model)
        for bi in range(b):
            for ti in range(t):
                v0, v1 = self.mapping.get(int(input_ids[bi, ti].item()), (0.0, 0.0))
                hs[bi, ti, 0] = v0
                hs[bi, ti, 1] = v1
        states = tuple(hs.clone() for _ in range(self.n_layers + 1))
        return type("_Out", (), {"hidden_states": states})()


def _basis_projector() -> SubspaceProjector:
    """A ``(4, 2)`` standard-basis projector: dim-0 of the projection == hidden[...,0]."""
    rot = torch.zeros(4, 2)
    rot[0, 0] = 1.0
    rot[1, 1] = 1.0
    return SubspaceProjector(rotation=rot, k=2, d_model=4, source={})


class TestPeakNormSelection:
    def test_picks_largest_norm_token(self) -> None:
        # dim1 == 0 ⇒ per-token norm == |dim0| == token id; '~' (126) wins.
        peak_kdim, peak_value, records = collect_text_projections(
            ["aaa~aaa"],
            projector=_basis_projector(),
            model=_IdHiddenModel(),
            tokenizer=_CharTokenizer(),
            layer=0,
            site="residual",
            batch_size=4,
            max_seq_len=16,
            window=2,
            device="cpu",
        )
        assert peak_kdim.shape == (1, 2)
        assert peak_value.shape == (1,)
        assert records[0].peak_token_index == 3
        assert peak_value[0].item() == pytest.approx(126.0)
        # peak_value is the Euclidean norm of the peak token's k-dim vector.
        assert peak_value[0].item() == pytest.approx(float(peak_kdim[0].norm()))
        assert records[0].window_text == "aa<<~>>aa"

    def test_norm_uses_all_subspace_dims(self) -> None:
        # 'Y' has the larger dim-0 (5 vs 1) but 'X' has the larger *norm*
        # (sqrt(1+100) ≈ 10.05 vs 5) thanks to dim-1 — the norm must decide.
        mapping = {ord("Y"): (5.0, 0.0), ord("X"): (1.0, 10.0)}
        peak_kdim, peak_value, records = collect_text_projections(
            ["YX"],
            projector=_basis_projector(),
            model=_DictModel(mapping),
            tokenizer=_CharTokenizer(),
            layer=0,
            site="residual",
            batch_size=4,
            max_seq_len=16,
            window=2,
            device="cpu",
        )
        assert records[0].peak_token_index == 1  # 'X'
        assert peak_value[0].item() == pytest.approx((1.0**2 + 10.0**2) ** 0.5)

    def test_peak_value_is_nonnegative_norm(self) -> None:
        # offset=100 ⇒ dim0 = id - 100; dim1 == 0. 'A'(65)→|−35|=35 > 'a'(97)→3.
        # The retained scalar is the (non-negative) norm, 35 — not the signed −35.
        peak_kdim, peak_value, records = collect_text_projections(
            ["Aa"],
            projector=_basis_projector(),
            model=_IdHiddenModel(offset=100.0),
            tokenizer=_CharTokenizer(),
            layer=0,
            site="residual",
            batch_size=4,
            max_seq_len=16,
            window=2,
            device="cpu",
        )
        assert records[0].peak_token_index == 0
        assert peak_value[0].item() == pytest.approx(35.0)
        assert peak_value[0].item() >= 0.0

    def test_bos_token_is_excluded(self) -> None:
        # BOS (id 1) with offset=100 has |dim0| = 99 — the largest norm — but
        # must be skipped, so 'A'(65)→35 wins among the real tokens.
        peak_kdim, peak_value, records = collect_text_projections(
            ["Aa"],
            projector=_basis_projector(),
            model=_IdHiddenModel(offset=100.0),
            tokenizer=_BosCharTokenizer(),
            layer=0,
            site="residual",
            batch_size=4,
            max_seq_len=16,
            window=2,
            device="cpu",
        )
        # Sequence is [BOS, 'A', 'a']; BOS at index 0 is excluded → 'A' at index 1.
        assert records[0].peak_token_index == 1
        assert peak_value[0].item() == pytest.approx(35.0)


class TestExtractWindow:
    def _tok(self) -> _CharTokenizer:
        return _CharTokenizer()

    def test_marks_peak_and_respects_window(self) -> None:
        ids = torch.tensor([65, 66, 67, 68, 69, 70, 71])  # A B C D E F G
        attn = torch.ones(7, dtype=torch.long)
        out = _extract_window(ids, attn, 3, tokenizer=self._tok(), window=2)
        assert out == "BC<<D>>EF"

    def test_clamps_at_left_boundary(self) -> None:
        ids = torch.tensor([65, 66, 67, 68])
        attn = torch.ones(4, dtype=torch.long)
        out = _extract_window(ids, attn, 0, tokenizer=self._tok(), window=2)
        assert out == "<<A>>BC"

    def test_excludes_padding(self) -> None:
        ids = torch.tensor([65, 66, 67, 68, 69, 70, 71, 0, 0, 0])
        attn = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        out = _extract_window(ids, attn, 6, tokenizer=self._tok(), window=5)
        assert out == "BCDEF<<G>>"
        assert "\x00" not in out

    def test_sanitises_literal_markers(self) -> None:
        # chr(60) == '<', chr(62) == '>' → left context decodes to '<<', '>>'.
        ids = torch.tensor([60, 60, 68, 62, 62])  # '<' '<' 'D' '>' '>'
        attn = torch.ones(5, dtype=torch.long)
        out = _extract_window(ids, attn, 2, tokenizer=self._tok(), window=4)
        # The only real markers are the ones wrapping the peak token.
        assert out.count("<<") == 1
        assert out.count(">>") == 1
        assert "‹‹" in out and "››" in out


class TestPcaFallback:
    def test_k2_pads_to_three_axes_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kdim = torch.randn(20, 2)
        with caplog.at_level(logging.WARNING):
            coords, ncomp, labels = _pca_coords(kdim)
        assert coords.shape == (20, 3)
        # compute_svd caps components at min(N, k) - 1 = 1 for k=2.
        assert ncomp == 1
        assert torch.allclose(coords[:, 1], torch.zeros(20))
        assert torch.allclose(coords[:, 2], torch.zeros(20))
        assert labels[1].endswith("(n/a)") and labels[2].endswith("(n/a)")
        assert any("PCA produced only" in r.getMessage() for r in caplog.records)

    def test_full_rank_yields_three_components(self) -> None:
        coords, ncomp, labels = _pca_coords(torch.randn(30, 6))
        assert coords.shape == (30, 3)
        assert ncomp == 3
        assert all("n/a" not in lab for lab in labels)


class TestHoverWindow:
    def test_marker_becomes_bold_underline(self) -> None:
        out = _hover_window("the committee <<approved>> the motion", wrap_every=10)
        assert '<b><span style="text-decoration:underline">approved</span></b>' in out
        # No raw markers remain.
        assert "<<" not in out and ">>" not in out

    def test_wraps_every_n_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(25))  # 25 tokens
        out = _hover_window(text, wrap_every=10)
        assert out.count("<br>") == 2  # 25 tokens → lines of 10/10/5
        first_line = out.split("<br>")[0]
        assert len(first_line.split(" ")) == 10

    def test_escapes_source_angle_brackets(self) -> None:
        out = _hover_window("a < b and c > d <<peak>>", wrap_every=10)
        assert "&lt;" in out and "&gt;" in out
        # The escaped single brackets are not turned into markup.
        assert "<b><span" in out  # the real marker still renders


class TestGateStatsUseNorm:
    def test_step1_stats_are_nonnegative_norm(self) -> None:
        # peak_kdim with negative dim-0 entries but non-negative norms. The gate
        # must report the norm distribution (matching the span scalar), so the
        # stats are never negative even though dim-0 is.
        kdim = torch.tensor(
            [
                [-1.0, 0.0],
                [-2.0, 1.0],
                [-3.0, 0.0],
                [0.0, -4.0],
                [1.0, 1.0],
                [-2.0, -2.0],
                [5.0, 0.0],
                [-1.0, 3.0],
                [2.0, -1.0],
                [0.0, 2.0],
            ]
        )
        report = run_gate(
            projections=kdim,
            example_spans=[],
            significance=Significance(hypothesis_text="x"),
            dataset_name="d",
        )
        norms = kdim.norm(dim=1)
        stats = report.step1_summary.stats
        assert stats.min >= 0.0
        assert stats.min == pytest.approx(float(norms.min()))
        assert stats.mean == pytest.approx(float(norms.mean()))
        assert stats.max == pytest.approx(float(norms.max()))


class TestDecileSampling:
    def test_stratified_cap_passes_through_when_small(self) -> None:
        vals = torch.arange(20).float()
        keep = _decile_stratified_indices(vals, cap=100, n_bins=10)
        assert keep.tolist() == list(range(20))

    def test_stratified_cap_covers_all_deciles(self) -> None:
        vals = torch.arange(100).float()
        keep = _decile_stratified_indices(vals, cap=10, n_bins=10)
        assert keep.shape[0] <= 10
        # One (lowest) index per decile → spans the whole range.
        assert keep.tolist() == [i * 10 for i in range(10)]


class TestCacheV2:
    def test_roundtrip_reconstructs_records(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wt, "_cache_root", lambda: str(tmp_path / "cache"))
        texts = ["aaa~aaa", "bbbXbbb"]
        monkeypatch.setattr(wt, "_stream_documents", lambda *a, **k: iter(texts))

        projector = _basis_projector()
        common = dict(
            projector=projector,
            tokenizer=_CharTokenizer(),
            model_name="m",
            layer=0,
            site="residual",
            corpus="c",
            split="train",
            n_tokens=10,
            max_seq_len=16,
            batch_size=4,
            window=2,
            n_quantile_bins=2,
            samples_per_bin=2,
            topk=2,
            bottomk=2,
            device="cpu",
            use_cache=True,
        )
        _, kdim1, val1, rec1 = collect_webtext_evidence(
            model=_IdHiddenModel(), **common
        )

        cache_dirs = glob.glob(str(tmp_path / "cache" / "*"))
        assert cache_dirs, "cache dir should be written"
        assert os.path.isfile(os.path.join(cache_dirs[0], "peak_kdim.safetensors"))
        assert os.path.isfile(os.path.join(cache_dirs[0], "peaks.json"))

        class _Boom:
            def to(self, *_a, **_k):
                return self

            def eval(self):
                return self

            def __call__(self, *_a, **_k):
                raise AssertionError("model must not run on a cache hit")

        _, kdim2, val2, rec2 = collect_webtext_evidence(model=_Boom(), **common)
        assert [r.window_text for r in rec2] == [r.window_text for r in rec1]
        assert [r.peak_token_index for r in rec2] == [r.peak_token_index for r in rec1]
        assert torch.allclose(kdim2, kdim1)
        assert torch.allclose(val2, val1)

    def test_v2_key_differs_from_v1(self) -> None:
        projector = _basis_projector()
        fp = wt._projector_fingerprint(projector)
        v1_parts = f"c|train|10|m|0|residual|16|{fp}"
        v1_key = hashlib.sha256(v1_parts.encode("utf-8")).hexdigest()[:24]
        v2_key = wt._cache_key(
            corpus="c",
            split="train",
            n_tokens=10,
            model_name="m",
            layer=0,
            site="residual",
            max_seq_len=16,
            projector=projector,
        )
        assert v2_key != v1_key
