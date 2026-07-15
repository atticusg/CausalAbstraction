"""End-to-end smoke test for the characterize_subspace analysis.

Bypasses HuggingFace model loading and FineWeb streaming via monkeypatching
so the test stays in the smoke tier (tiny random tensors, no network).
The three parametrised cases cover the three terminal states the bundle
documents:

- ``confirmed`` → ``refined_hypothesis.json`` written with the expected schema.
- ``disagreed`` → reconcile loop runs to completion, verdict surfaces.
- gate failure → ``more_info_needed.json`` written; ``refined_hypothesis.json`` absent.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from causalab.analyses.characterize_subspace import main as main_module
from causalab.analyses.characterize_subspace import webtext as webtext_module
from causalab.analyses.characterize_subspace import judge as judge_module
from causalab.analyses.characterize_subspace.schemas import (
    PeakRecord,
    ProjectionStats,
    QuantileBin,
    Span,
    WebtextEvidence,
)

pytestmark = pytest.mark.smoke


D_MODEL = 32
K = 2


class _StubModelOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _StubModel:
    """Minimal stand-in for an HF causal LM exposing ``output_hidden_states``.

    Forward pass returns ``L + 1`` random hidden-state tensors of shape
    ``(B, T, D_MODEL)`` so the residual-stream indexing in webtext.py works.
    """

    def __init__(self, n_layers: int = 4) -> None:
        self.config = type(
            "cfg", (), {"hidden_size": D_MODEL, "num_hidden_layers": n_layers}
        )
        self.n_layers = n_layers

    def to(self, *_args, **_kwargs):  # noqa: D401 - mimic torch API
        return self

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # noqa: ARG002
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        output_hidden_states: bool = False,
        return_dict: bool = True,  # noqa: ARG002
    ) -> _StubModelOutput:
        assert output_hidden_states
        b, t = input_ids.shape
        # Hidden states correlate with token id sum so projections vary across docs.
        bias = input_ids.sum(dim=1, keepdim=True).unsqueeze(-1).float() / 100.0
        hs = tuple(torch.randn(b, t, D_MODEL) + bias for _ in range(self.n_layers + 1))
        return _StubModelOutput(hidden_states=hs)


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        texts,
        *,
        return_tensors: str | None = None,  # noqa: ARG002
        padding: bool | str = False,  # noqa: ARG002
        truncation: bool = False,  # noqa: ARG002
        max_length: int | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        max_len = max_length or 32
        encoded = []
        for t in texts:
            ids = [(ord(c) % 250) + 2 for c in t][:max_len]
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            encoded.append(ids)
        if return_tensors == "pt":
            ids = torch.tensor(encoded, dtype=torch.long)
            attn = (ids != 0).long()
            return {"input_ids": ids, "attention_mask": attn}
        return {"input_ids": encoded}

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            ids = [ids]
        # Deterministic printable stand-in; exact round-trip is not needed.
        return "".join(chr(32 + (int(i) % 94)) for i in ids)


def _write_rotation(
    path: str, *, d_model: int = D_MODEL, k: int = K, zero: bool = False
) -> None:
    if zero:
        rot = torch.zeros(d_model, k)
    else:
        rot = torch.randn(d_model, k)
        # Light orthonormalisation so the rotation looks well-formed.
        q, _ = torch.linalg.qr(rot)
        rot = q[:, :k]
    save_file({"rotation_matrix": rot.contiguous()}, path)


def _write_step1_dataset(path: str) -> None:
    texts = [
        "The committee approved the motion unanimously today.",
        "Heavy rain pooled in the street before midnight.",
        "A landmark ruling reshaped policy for years.",
        "The river ran cold and clear past the bridge.",
        "Procedural rules govern how amendments are filed.",
        "Wet leaves clogged the gutter near the door.",
        "Members voted to advance the proposal to the floor.",
        "The garden flooded after the storm last week.",
        "The motion carried by a wide margin.",
        "Streetlights flickered in the heavy downpour.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(texts, fh)


def _stub_evidence(
    stub_proj_offset: float = 0.0,
) -> tuple[WebtextEvidence, torch.Tensor, torch.Tensor, list[PeakRecord]]:
    """Return a small synthetic WebtextEvidence + peak_kdim, peak_value, records."""
    values = [float(i - 5) + stub_proj_offset for i in range(11)]
    records = [
        PeakRecord(
            projection_value=v,
            peak_token_index=0,
            window_text=f"some context <<doc{i}>> trailing context",
        )
        for i, v in enumerate(values)
    ]
    spans = [
        Span(text=records[i].window_text, projection_value=values[i]) for i in range(11)
    ]
    peak_value = torch.tensor(values)
    # (N, 2); column 0 == peak_value by construction (the leading-direction value).
    peak_kdim = torch.stack([peak_value, peak_value * 0.5], dim=1)
    evidence = WebtextEvidence(
        corpus="stub-corpus",
        quantile_bins=[
            QuantileBin(
                quantile=0.25,
                projection_range=(values[0], values[3]),
                spans=spans[:4],
            ),
            QuantileBin(
                quantile=0.75,
                projection_range=(values[7], values[-1]),
                spans=spans[7:],
            ),
        ],
        topk_spans=spans[-3:],
        bottomk_spans=spans[:3],
        stats=ProjectionStats(
            n_samples=len(values),
            mean=float(peak_value.mean().item()),
            std=float(peak_value.std().item()),
            min=float(peak_value.min().item()),
            max=float(peak_value.max().item()),
        ),
    )
    return evidence, peak_kdim, peak_value, records


def _build_cfg(
    *, artifact: str, step1: str, out_dir: str, model_name: str = "stub/model"
) -> Any:
    return OmegaConf.create(
        {
            "model": {"device": "cpu", "dtype": None},
            "characterize_subspace": {
                "_output_dir": out_dir,
                "subspace": {
                    "artifact": artifact,
                    "model": model_name,
                    "layer": 1,
                    "site": "residual",
                    "k_features_hint": "auto",
                    "step1_dataset": step1,
                },
                "significance": {
                    "hypothesis_text": "tracks formal procedural language",
                    "figure_path": None,
                    "topology_description": None,
                },
                "webtext": {
                    "corpus": "stub-corpus",
                    "split": "train",
                    "n_tokens": 1000,
                    "max_seq_len": 16,
                    "batch_size": 4,
                    "n_quantile_bins": 2,
                    "samples_per_bin": 3,
                    "topk": 3,
                    "bottomk": 3,
                    "cache": False,
                },
                "metrics": {"reproduction_pass_threshold": 0.5},
                "judge": {
                    "provider": "openrouter",
                    "model": "stub/model",
                    "framings": ["default", "broad", "contrastive"],
                    "max_reconciliation_iterations": 3,
                    "max_tokens": 256,
                },
                "output": {"emit_evidence_bundle": True},
            },
        }
    )


@pytest.fixture
def stub_model_loader(monkeypatch: pytest.MonkeyPatch):
    """Replace HF model loading with the synthetic stubs."""

    def _fake_loader(_model_name: str, *, device: str | None, dtype: str | None) -> Any:
        del device, dtype
        return _StubModel(), _StubTokenizer(), "cpu"

    monkeypatch.setattr(main_module, "_load_hf_model", _fake_loader)


@pytest.fixture
def stub_webtext(monkeypatch: pytest.MonkeyPatch):
    """Replace FineWeb streaming with synthetic spans."""

    def _fake_evidence(
        **_kwargs: Any,
    ) -> tuple[WebtextEvidence, torch.Tensor, torch.Tensor, list[PeakRecord]]:
        return _stub_evidence()

    monkeypatch.setattr(main_module, "collect_webtext_evidence", _fake_evidence)
    # webtext_module is imported elsewhere; patch for completeness even though
    # main imports from it directly.
    monkeypatch.setattr(webtext_module, "collect_webtext_evidence", _fake_evidence)


@pytest.fixture(autouse=True)
def _judge_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a judge API key for every smoke test.

    The fail-fast credential pre-flight (issue #221) runs at the very top of
    ``main``, before model load — so even tests that stub ``call_llm`` need a
    key present to get past it. ``test_missing_credentials_fail_fast_*``
    deletes it again in its own body to exercise the unset path.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-smoke-test")


class TestCharacterizeSubspaceSmoke:
    def test_confirmed_verdict_writes_refined_hypothesis(
        self,
        tmp_path,
        stub_model_loader,  # noqa: ARG002
        stub_webtext,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifact = str(tmp_path / "subspace.safetensors")
        step1 = str(tmp_path / "step1.json")
        out_dir = str(tmp_path / "out")
        _write_rotation(artifact)
        _write_step1_dataset(step1)
        cfg = _build_cfg(artifact=artifact, step1=step1, out_dir=out_dir)

        # Sequence of judge responses: 1 derive + 1 reconcile = 2 calls; the
        # reconcile short-circuits on first 'confirmed'.
        responses = iter(
            [
                json.dumps(
                    {
                        "hypothesis": "tracks formal procedural language",
                        "confidence": 0.78,
                        "supporting_spans": ["committee approved", "landmark ruling"],
                    }
                ),
                json.dumps(
                    {
                        "verdict": "confirmed",
                        "rationale": "Same axis as the provided hypothesis.",
                    }
                ),
            ]
        )

        def _judge_stub(
            messages, *, model, max_tokens, provider="openrouter", max_retries=3
        ):  # noqa: ARG001
            return next(responses)

        monkeypatch.setattr(judge_module, "call_llm", _judge_stub)

        result = main_module.main(cfg)

        assert result["passed"] is True
        assert result["verdict"] == "confirmed"
        refined_path = result["refined_hypothesis_path"]
        assert os.path.isfile(refined_path)
        with open(refined_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        expected_keys = {
            "hypothesis_text",
            "confidence",
            "verdict",
            "provided_significance",
            "evidence_summary",
            "step1_summary",
            "judge",
            "reconciliation",
        }
        assert expected_keys.issubset(payload.keys())
        assert payload["verdict"] == "confirmed"
        assert payload["hypothesis_text"] == "tracks formal procedural language"

        # Sibling bundle artifacts present.
        bundle_dir = os.path.dirname(refined_path)
        for fname in (
            "evidence.safetensors",
            "evidence.meta.json",
            "reconciliation_trace.json",
            "metadata.json",
        ):
            assert os.path.isfile(os.path.join(bundle_dir, fname)), fname
        for fname in (
            "projection_distribution.html",
            "step1_vs_webtext.html",
            "projection_explorer.html",
        ):
            assert os.path.isfile(os.path.join(bundle_dir, "figures", fname)), fname

    def test_persistent_disagreement_exhausts_reconcile_loop(
        self,
        tmp_path,
        stub_model_loader,  # noqa: ARG002
        stub_webtext,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifact = str(tmp_path / "subspace.safetensors")
        step1 = str(tmp_path / "step1.json")
        out_dir = str(tmp_path / "out")
        _write_rotation(artifact)
        _write_step1_dataset(step1)
        cfg = _build_cfg(artifact=artifact, step1=step1, out_dir=out_dir)

        responses = iter(
            [
                # Derive
                json.dumps(
                    {
                        "hypothesis": "tracks weather descriptions",
                        "confidence": 0.62,
                        "supporting_spans": ["heavy rain", "wet leaves"],
                    }
                ),
                # Three reconcile attempts, all disagreed
                json.dumps({"verdict": "disagreed", "rationale": "different axis"}),
                json.dumps({"verdict": "disagreed", "rationale": "still different"}),
                json.dumps({"verdict": "disagreed", "rationale": "still different"}),
            ]
        )

        def _judge_stub(
            messages, *, model, max_tokens, provider="openrouter", max_retries=3
        ):  # noqa: ARG001
            return next(responses)

        monkeypatch.setattr(judge_module, "call_llm", _judge_stub)

        result = main_module.main(cfg)
        assert result["passed"] is True
        assert result["verdict"] == "disagreed"

        trace_path = os.path.join(
            os.path.dirname(result["refined_hypothesis_path"]),
            "reconciliation_trace.json",
        )
        with open(trace_path, encoding="utf-8") as fh:
            trace = json.load(fh)
        assert len(trace["iterations"]) == 3
        assert [it["framing"] for it in trace["iterations"]] == [
            "default",
            "broad",
            "contrastive",
        ]

    def test_zero_rank_rotation_writes_more_info_needed(
        self,
        tmp_path,
        stub_model_loader,  # noqa: ARG002
        stub_webtext,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gate failure path. Judge must never be called."""
        artifact = str(tmp_path / "subspace.safetensors")
        step1 = str(tmp_path / "step1.json")
        out_dir = str(tmp_path / "out")
        _write_rotation(artifact, zero=True)
        _write_step1_dataset(step1)
        cfg = _build_cfg(artifact=artifact, step1=step1, out_dir=out_dir)

        def _judge_stub(*_args, **_kwargs):
            raise AssertionError("Judge should not be called when the gate fails.")

        monkeypatch.setattr(judge_module, "call_llm", _judge_stub)

        result = main_module.main(cfg)
        assert result["passed"] is False
        assert "more_info_needed" in result
        more_info_path = result["more_info_needed"]
        assert os.path.isfile(more_info_path)
        with open(more_info_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["status"] == "insufficient_handoff"
        assert payload["failed_metrics"], "At least one failed metric must be recorded."

        # No refined-hypothesis artifact on the failure path.
        bundle_dir = os.path.dirname(more_info_path)
        assert not os.path.isfile(os.path.join(bundle_dir, "refined_hypothesis.json"))

    def test_missing_credentials_fail_fast_before_model_load(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Issue #221: a missing judge key must abort BEFORE the model loads.

        Deletes the key the autouse fixture set, then makes model loading a
        tripwire: if the credential check were still placed after the model
        load, ``main`` would raise the sentinel ``AssertionError`` instead of
        the credential ``RuntimeError``.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        def _boom(*_args, **_kwargs):
            raise AssertionError(
                "model load reached before the credential check (issue #221)"
            )

        monkeypatch.setattr(main_module, "_load_hf_model", _boom)

        artifact = str(tmp_path / "subspace.safetensors")
        step1 = str(tmp_path / "step1.json")
        out_dir = str(tmp_path / "out")
        _write_rotation(artifact)
        _write_step1_dataset(step1)
        cfg = _build_cfg(artifact=artifact, step1=step1, out_dir=out_dir)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is unset"):
            main_module.main(cfg)
