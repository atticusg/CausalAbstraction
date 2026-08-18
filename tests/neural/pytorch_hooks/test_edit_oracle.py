"""Edit semantics vs the raw-hook oracle — interchange, feature space +
error term, additive steer, gaussian determinism, cross-model two-pass
path patching. Assertions and tolerances follow the oracle suites
(stack-vs-oracle atol=1e-5 rtol=1e-4; wrapper-level cases atol=1e-4
rtol=1e-3; determinism byte-identical)."""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pytorch_hooks.encoding import encode

from tests.neural.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.pytorch_hooks.conftest import BASE_TEXT, SOURCE_TEXT, OracleShim

pytestmark = pytest.mark.unit

TOL = dict(atol=1e-5, rtol=1e-4)


def _inputs(bundle, text: str):
    batch = encode(bundle.tokenizer, [text])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def interchange_doc(pos: int = 1) -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_source=True),
        "sites": {
            "tgt": {"component": "block_output", "layer": 0},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "v_src": {
                "site": "tgt",
                "pos": {"index": pos},
                "model": "original",
                "input": "source",
            },
            "logits": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "patched",
                "input": "base",
            },
        },
        "edits": {
            "patch": {"site": "tgt", "pos": {"index": pos}, "do": {"swap": "v_src"}}
        },
        "intervened_models": {"patched": {"input": "base", "edits": ["patch"]}},
        "save": [
            {
                "value": "logits",
                "model": "patched",
                "input": "base",
                "file_path": "l.safetensors",
            }
        ],
    }


def test_interchange_matches_oracle(bundle, oracle: OracleShim):
    """The corpus-02 shape: swap the source residual into base at (L0, p1);
    the patched logits equal the oracle's hand-rolled patch, and differ
    from clean (non-vacuity)."""
    executor = executor_for(
        interchange_doc(), bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)
    src_resid = oracle_lib.capture_residual(oracle, 0, _inputs(bundle, SOURCE_TEXT))
    want = oracle_lib.next_token_logits(
        oracle, base_inputs, layer=0, positions=[1], patch_values=src_resid[:, 1:2, :]
    )
    clean = oracle_lib.next_token_logits(oracle, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)  # non-vacuity
    torch.testing.assert_close(have, want, **TOL)


def test_feature_space_swap_keeps_the_complement(llama_bundle):
    """The error-term contract (§2.5): a swap through a k=d/2 subspace
    replaces only the in-subspace coordinates; the complement of the BASE
    value survives (the oracle's lossy-split case 3)."""
    d = llama_bundle.info.hidden_size
    k = d // 2
    doc = interchange_doc()
    doc["featurizers"] = {
        "rot": {"kind": "subspace", "k": k, "parametrization": "cayley"}
    }
    doc["reads"]["v_src"]["featurizer"] = "rot"
    doc["edits"]["patch"]["featurizer"] = "rot"
    executor = executor_for(
        doc, llama_bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    have = executor.read_value("logits")[:, 0, :]

    # the SAME Q both sides: extract the parametrized weight from the stage
    q = executor.stage("rot").slot_params()["weight"].detach()
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    src_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, SOURCE_TEXT)
    )[:, 1:2, :]
    patch = base_resid - (base_resid @ q) @ q.T + (src_resid @ q) @ q.T
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    clean = oracle_lib.next_token_logits(shim, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)
    torch.testing.assert_close(have, want, **TOL)


def test_dims_swap_is_a_subspace_swap(llama_bundle):
    """A dims selection swaps only those feature coordinates — raw space,
    dims [0, 2]: untouched dims come from the pre-edit value."""
    doc = interchange_doc()
    doc["edits"]["patch"]["dims"] = [0, 2]
    doc["reads"]["v_src"]["dims"] = [0, 2]
    executor = executor_for(
        doc, llama_bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    have = executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :].clone()
    src_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, SOURCE_TEXT)
    )[:, 1:2, :]
    patch = base_resid.clone()
    patch[..., [0, 2]] = src_resid[..., [0, 2]]
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    torch.testing.assert_close(have, want, **TOL)


def test_add_scaled_matches_oracle_steer(bundle, oracle: OracleShim):
    """add_scaled with a literal-alpha scalar operand == the oracle's
    additive steer of a constant (broadcast scalar) vector."""
    doc = interchange_doc()
    doc["edits"]["patch"]["do"] = {"add_scaled": {"op": 2.5, "alpha": 1.0}}
    del doc["reads"]["v_src"]
    doc["data"] = base_data_section(with_source=False)
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)

    def edit(hidden: torch.Tensor) -> None:
        hidden[:, 1, :] = hidden[:, 1, :] + 2.5

    block = oracle_lib.decoder_block(oracle, 0)
    want = oracle_lib.component_edited_logits(oracle, base_inputs, block, "out", edit)
    torch.testing.assert_close(have, want, **TOL)


def test_gaussian_contract(llama_bundle):
    """The noise-oracle contract: scale 0 == clean; same seed byte-identical
    across runs; different seed differs."""

    def run(seed: int, scale: float) -> torch.Tensor:
        doc = interchange_doc()
        doc["edits"]["patch"]["do"] = {
            "gaussian": {"seed": seed, "scale": scale, "axis": "tp_duplicated"}
        }
        del doc["reads"]["v_src"]
        doc["data"] = base_data_section(with_source=False)
        executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
        return executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    clean = oracle_lib.next_token_logits(shim, _inputs(llama_bundle, BASE_TEXT))
    torch.testing.assert_close(run(7, 0.0), clean, **TOL)  # scale 0 == clean
    noisy = run(7, 3.0)
    assert not torch.allclose(noisy, clean, atol=1e-4)  # noise moves logits
    torch.testing.assert_close(run(7, 3.0), noisy, atol=0.0, rtol=0.0)  # same seed
    assert not torch.allclose(run(8, 3.0), noisy, atol=1e-4)  # different seed


def test_gaussian_draw_realization(llama_bundle):
    """The RNG realization the parity goldens pin: the draw is
    Generator().manual_seed(seed) → randn((batch, n_pos, width)), made
    outside the model."""
    doc = interchange_doc()
    doc["edits"]["patch"]["do"] = {
        "gaussian": {"seed": 7, "scale": 3.0, "axis": "tp_duplicated"}
    }
    del doc["reads"]["v_src"]
    doc["data"] = base_data_section(with_source=False)
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    d = llama_bundle.info.hidden_size
    draw = torch.randn((1, 1, d), generator=torch.Generator().manual_seed(7))
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=base_resid + 3.0 * draw
    )
    torch.testing.assert_close(have, want, **TOL)


def test_renormalize_restores_the_pre_edit_norm(llama_bundle):
    """add_scaled + renormalize at one address: the delta applies, then the
    feature vector is rescaled to the pre-edit norm."""
    doc = interchange_doc()
    doc["edits"] = {
        "nudge": {
            "site": "tgt",
            "pos": {"index": 1},
            "do": {"add_scaled": {"op": 2.5, "alpha": 1.0}},
        },
        "renorm": {"site": "tgt", "pos": {"index": 1}, "do": {"renormalize": True}},
    }
    doc["intervened_models"]["patched"]["edits"] = ["nudge", "renorm"]
    del doc["reads"]["v_src"]
    doc["data"] = base_data_section(with_source=False)
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    bumped = base_resid + 2.5
    rescaled = bumped * (
        base_resid.norm(dim=-1, keepdim=True) / bumped.norm(dim=-1, keepdim=True)
    )
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=rescaled
    )
    torch.testing.assert_close(have, want, **TOL)


def test_two_pass_path_patching_matches_oracle(bundle, oracle: OracleShim):
    """The spec's worked example (corpus 03) at tiny scale: sender L0 →
    receiver L1, v* read under the sender swap in 'patched', injected clean
    in 'final' — vs the two-pass hook oracle (wrapper tolerance atol=1e-4
    rtol=1e-3)."""
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_source=True),
        "sites": {
            "sender": {"component": "block_output", "layer": 0},
            "receiver": {"component": "block_output", "layer": 1},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "v_sender": {
                "site": "sender",
                "pos": {"index": -1},
                "model": "original",
                "input": "source",
            },
            "v_receiver": {
                "site": "receiver",
                "pos": {"index": -1},
                "model": "patched",
                "input": "base",
            },
            "logits": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "final",
                "input": "base",
            },
        },
        "edits": {
            "swap_sender": {
                "site": "sender",
                "pos": {"index": -1},
                "do": {"swap": "v_sender"},
            },
            "inject": {
                "site": "receiver",
                "pos": {"index": -1},
                "do": {"swap": "v_receiver"},
            },
        },
        "intervened_models": {
            "patched": {"input": "base", "edits": ["swap_sender"]},
            "final": {"input": "base", "edits": ["inject"]},
        },
        "save": [
            {
                "value": "logits",
                "model": "final",
                "input": "base",
                "file_path": "l.safetensors",
            }
        ],
    }
    executor = executor_for(
        doc, bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)
    src_resid = oracle_lib.capture_residual(oracle, 0, _inputs(bundle, SOURCE_TEXT))
    sender_block = oracle_lib.decoder_block(oracle, 0)
    receiver_block = oracle_lib.decoder_block(oracle, 1)
    patch_last = src_resid[:, -1, :]

    def sender_edit(hidden: torch.Tensor) -> None:
        hidden[:, -1, :] = patch_last

    v_star = oracle_lib.capture_with_edits(
        oracle, base_inputs, receiver_block, "out", [(sender_block, "out", sender_edit)]
    )[:, -1:, :]
    want = oracle_lib.next_token_logits(
        oracle, base_inputs, layer=1, positions=[-1], patch_values=v_star
    )
    clean = oracle_lib.next_token_logits(oracle, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)
    torch.testing.assert_close(have, want, atol=1e-4, rtol=1e-3)


def test_reads_see_the_fully_edited_state(bundle, oracle: OracleShim):
    """§2.7: a read in model M at the edited address sees the write."""
    doc = interchange_doc()
    doc["reads"]["v_at"] = {
        "site": "tgt",
        "pos": {"index": 1},
        "model": "patched",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "v_at",
            "model": "patched",
            "input": "base",
            "file_path": "v.safetensors",
        }
    )
    executor = executor_for(
        doc, bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    src_resid = oracle_lib.capture_residual(oracle, 0, _inputs(bundle, SOURCE_TEXT))[
        :, 1, :
    ]
    torch.testing.assert_close(executor.read_value("v_at")[:, 0, :], src_resid, **TOL)


def test_composed_chain_sizes_stages_by_chain_width(llama_bundle):
    """§2.5 composition: a gate after a k=3 rotation is a 3-wide gate, and
    the composed swap equals the hand-rolled two-stage math — the source's
    on-features replace the base's inside the rotated subspace, everything
    else (the gate's off-features AND the rotation's complement) survives
    from the base."""
    d = llama_bundle.info.hidden_size
    doc = interchange_doc()
    doc["featurizers"] = {
        "rot": {"kind": "subspace", "k": 3, "parametrization": "cayley"},
        "gate": {"kind": "gate"},
    }
    doc["reads"]["v_src"]["featurizer"] = ["rot", "gate"]
    doc["edits"]["patch"]["featurizer"] = ["rot", "gate"]
    executor = executor_for(
        doc, llama_bundle, base_texts=[BASE_TEXT], source_texts=[SOURCE_TEXT]
    )
    gate = executor.stage("gate")
    assert tuple(gate.theta.shape) == (3,)  # chain width, not the site width d
    with torch.no_grad():  # a non-trivial hard-eval mask: on for the last dim
        gate.theta.copy_(torch.tensor([-2.0, -1.0, 2.0]))
    rot_q = executor.stage("rot").slot_params()["weight"].detach()
    assert tuple(rot_q.shape) == (d, 3)

    have = executor.read_value("logits")[:, 0, :]
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    src_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, SOURCE_TEXT)
    )[:, 1:2, :]
    mask = (gate.theta > 0).float()
    z_base, z_src = base_resid @ rot_q, src_resid @ rot_q
    z_new = mask * z_src + (1.0 - mask) * z_base  # gate swap keeps off-features
    patch = base_resid - z_base @ rot_q.T + z_new @ rot_q.T  # rot err survives
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    clean = oracle_lib.next_token_logits(shim, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)  # non-vacuity
    torch.testing.assert_close(have, want, **TOL)
