"""Read routing vs the raw-hook oracle — the collect contract re-driven
through protocol documents (assertions from
tests/neural/activations/test_collect_hook_oracle.py, verbatim tolerances:
stack-vs-oracle atol=1e-5 rtol=1e-4)."""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pytorch_hooks.encoding import encode

from tests.neural.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.pytorch_hooks.conftest import BASE_TEXT, OracleShim

pytestmark = pytest.mark.unit

TOL = dict(atol=1e-5, rtol=1e-4)


def _inputs(bundle):
    batch = encode(bundle.tokenizer, [BASE_TEXT])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def harvest_doc() -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {
            "s_in0": {"component": "block_input", "layer": 0},
            "s_out0": {"component": "block_output", "layer": 0},
            "s_out1": {"component": "block_output", "layer": 1},
        },
        "reads": {
            "r_in0": {
                "site": "s_in0",
                "pos": {"index": 0},
                "model": "original",
                "input": "base",
            },
            "r_out0": {
                "site": "s_out0",
                "pos": {"index": 1},
                "model": "original",
                "input": "base",
            },
            "r_out1": {
                "site": "s_out1",
                "pos": {"index": 2},
                "model": "original",
                "input": "base",
            },
        },
        "save": [
            {
                "value": "r_in0",
                "model": "original",
                "input": "base",
                "file_path": "a.safetensors",
            },
            {
                "value": "r_out0",
                "model": "original",
                "input": "base",
                "file_path": "b.safetensors",
            },
            {
                "value": "r_out1",
                "model": "original",
                "input": "base",
                "file_path": "c.safetensors",
            },
        ],
    }


def test_reads_match_oracle_captures(bundle, oracle: OracleShim):
    executor = executor_for(harvest_doc(), bundle, base_texts=[BASE_TEXT])
    inputs = _inputs(bundle)

    block0 = oracle_lib.decoder_block(oracle, 0)
    want_in0 = oracle_lib.capture_component(oracle, block0, "in", inputs)[:, 0, :]
    want_out0 = oracle_lib.capture_residual(oracle, 0, inputs)[:, 1, :]
    want_out1 = oracle_lib.capture_residual(oracle, 1, inputs)[:, 2, :]

    # anti-vacuity: the three captures genuinely differ (oracle contract)
    assert not torch.allclose(want_in0, want_out0, atol=1e-4)
    assert not torch.allclose(want_out0, want_out1, atol=1e-4)

    torch.testing.assert_close(executor.read_value("r_in0")[:, 0, :], want_in0, **TOL)
    torch.testing.assert_close(executor.read_value("r_out0")[:, 0, :], want_out0, **TOL)
    torch.testing.assert_close(executor.read_value("r_out1")[:, 0, :], want_out1, **TOL)


@pytest.mark.parametrize(
    "component",
    ["embeddings", "attention_output", "mlp_input", "mlp_output", "mlp_activation"],
)
def test_every_component_matches_oracle(bundle, oracle: OracleShim, component: str):
    site: dict = {"component": component}
    if component != "embeddings":
        site["layer"] = 0
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"tap": site},
        "reads": {
            "r": {
                "site": "tap",
                "pos": {"index": 1},
                "model": "original",
                "input": "base",
            }
        },
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "r.safetensors",
            }
        ],
    }
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    module, kind = oracle_lib.component_module(oracle, 0, component)
    want = oracle_lib.capture_component(oracle, module, kind, _inputs(bundle))[:, 1, :]
    torch.testing.assert_close(executor.read_value("r")[:, 0, :], want, **TOL)


def test_head_value_read_matches_oracle(llama_bundle):
    """attention_value head H == the oracle's o_proj-input column slice
    (test_collect_hook_oracle.py's head case)."""
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"h": {"component": "attention_value", "layer": 0, "head": 1}},
        "reads": {
            "r": {
                "site": "h",
                "pos": {"index": 1},
                "model": "original",
                "input": "base",
            }
        },
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "r.safetensors",
            }
        ],
    }
    shim = OracleShim(hf_model=llama_bundle.model)
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    batch = encode(llama_bundle.tokenizer, [BASE_TEXT])
    want = oracle_lib.capture_head_value(
        shim,
        0,
        1,
        {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask},
    )[:, 1, :]
    torch.testing.assert_close(executor.read_value("r")[:, 0, :], want, **TOL)


def test_lm_head_read_is_the_model_logits(bundle, oracle: OracleShim):
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"lm_head": {"component": "lm_head"}},
        "reads": {
            "logits": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "original",
                "input": "base",
            }
        },
        "save": [
            {
                "value": "logits",
                "model": "original",
                "input": "base",
                "file_path": "l.safetensors",
            }
        ],
    }
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    want = oracle_lib.next_token_logits(oracle, _inputs(bundle))
    torch.testing.assert_close(executor.read_value("logits")[:, 0, :], want, **TOL)


def test_variable_position_reads_the_substring_tokens(llama_bundle):
    """A {"variable": x} position addresses exactly the tokens of the row's
    value for x — decoded back, the gathered ids spell the value."""
    text = "the quick brown fox jumps"
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "positions": {"v": {"variable": "animal"}},
        "sites": {"emb": {"component": "embeddings"}},
        "reads": {
            "r": {"site": "emb", "pos": "v", "model": "original", "input": "base"}
        },
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "r.safetensors",
            }
        ],
    }
    executor = executor_for(
        doc, llama_bundle, base_texts=[text], extra_columns={"animal": ["brown fox"]}
    )
    batch = encode(llama_bundle.tokenizer, [text])
    from causalab.neural.pytorch_hooks.encoding import resolve_position
    from causalab.protocol.schema import PositionSpec

    positions = resolve_position(
        PositionSpec(variable="animal"),
        batch,
        0,
        dataset_row={"input": text, "animal": "brown fox"},
        field="input",
    )
    decoded = llama_bundle.tokenizer.decode(batch.input_ids[0, positions])
    assert "brown fox" in decoded
    value = executor.read_value("r")
    embeddings = llama_bundle.model.model.embed_tokens(batch.input_ids)
    torch.testing.assert_close(value[:, :, :], embeddings[:, positions, :], **TOL)
