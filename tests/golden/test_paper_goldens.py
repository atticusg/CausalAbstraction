"""Paper-golden tier: real-model runs asserted against paper-provenance values.

Every asserted number in tests/golden/paper_goldens.json traces to a
published paper figure/table or to the VeriFires task package encoding it
(tasks/) — never to a pinning run of this
stack. Documents live in tests/golden/protocols/ (identity pinned in
golden_digests.json); fixture datasets are seeded, committed JSON produced
by tests/golden/fixtures/generators/.

Run on a GPU box:

    uv run pytest tests/golden -m golden

Gated models (meta-llama/Llama-3.1-8B, google/gemma-2-2b-it) need a
licensed ``HF_TOKEN`` in the environment — without one the load 401s with
nothing naming the cause. CUDA is preferred; Apple MPS works for the
smaller documents (the 1,152-row hours document wants ~35GB free).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from causalab.protocol.cli import main

from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS, GOLDENS_FILE

pytestmark = pytest.mark.golden

GOLDENS = json.loads(GOLDENS_FILE.read_text())["goldens"]

# golden ids each test claims; test_structural cross-checks this registry
# against paper_goldens.json so no value can go silently unasserted
COVERED: dict[str, tuple[str, ...]] = {
    "test_ioi_clean_logit_diff": ("ioi.clean_logit_diff", "ioi.io_over_s_rate"),
    "test_hours_baseline_accuracy": ("arithmetic.hours_baseline_acc",),
    "test_rome_average_total_effect": ("rome.ate",),
    "test_rome_hidden_state_aie_peak": ("rome.hidden_aie_peak",),
    "test_rome_mlp_window_aie_peak": ("rome.mlp_window_aie_peak",),
    "test_hydra_compensation_r2": ("hydra.compensation_r2",),
    "test_mixing_positional_shares": (
        "mixing.positional_share_edges",
        "mixing.positional_share_middle",
    ),
    "test_arithmetic_steering_diagonal": ("arithmetic.steering_top1_fraction",),
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("golden tier needs an accelerator (cuda or mps)")


def skip_if_pending(golden_id: str) -> None:
    """A pending goldens entry has open provenance/calibration questions
    (recorded in its notes); its test stays wired but skips."""
    if GOLDENS[golden_id].get("pending"):
        pytest.skip(f"{golden_id} is pending: {GOLDENS[golden_id]['notes'][:160]}…")


def assert_golden(golden_id: str, measured: float) -> None:
    """One assertion path for every golden: prints measured next to the
    band so a near-miss is diagnosable straight from the log."""
    entry = GOLDENS[golden_id]
    if entry["sidedness"] == "at_least":
        floor = entry["floor"]
        print(f"{golden_id}: measured {measured:.4f}, floor {floor}")
        assert measured >= floor, f"{golden_id}: {measured:.4f} < floor {floor}"
    else:
        lo, hi = entry["band"]
        print(f"{golden_id}: measured {measured:.4f}, band [{lo}, {hi}]")
        assert lo <= measured <= hi, (
            f"{golden_id}: {measured:.4f} outside [{lo}, {hi}] "
            f"(paper value {entry['value']})"
        )


def run_document(name: str, out: Path, *, dtype: str = "fp32", **extra: str) -> None:
    argv = [
        "run",
        str(GOLDEN_PROTOCOLS / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(out / "artifacts"),
        "--out",
        str(out),
        "--device",
        _device(),
        "--dtype",
        dtype,
    ]
    for key, value in extra.items():
        argv += ["--set", f"{key}={value}"]
    assert main(argv) == 0


def test_ioi_clean_logit_diff(tmp_path):
    run_document("ioi_clean_logit_diff_im.json", tmp_path)
    ld = pd.read_parquet(tmp_path / "logit_diff.parquet")["value"]
    assert len(ld) == 512
    assert_golden("ioi.clean_logit_diff", float(ld.mean()))
    assert_golden("ioi.io_over_s_rate", float((ld > 0).mean()))


def test_hours_baseline_accuracy(tmp_path):
    run_document("hours_baseline_im.json", tmp_path, dtype="bf16")
    acc = pd.read_parquet(tmp_path / "acc.parquet")["value"]
    assert len(acc) == 1152
    assert_golden("arithmetic.hours_baseline_acc", float(acc.mean()))


def test_rome_average_total_effect(tmp_path):
    """ATE = mean over facts and noise seeds of P_clean - P_corrupted,
    recovered from the cross_entropy parquets (p = exp(-ce)); the clean
    read is off the seed axis, so its per-seed rows are identical and the
    per-example mean collapses them.

    One document per subject-token width (the backend refuses ragged
    edits); pooling the per-fact effects across the width shards restores
    the loose-filter width distribution — a single-width sample biases the
    ATE (diagnosed +16pts on the width-4 bucket alone)."""
    import numpy as np

    effects = []
    clean_all, corr_all = [], []
    for width in (2, 3, 4, 5):
        out = tmp_path / f"w{width}"
        run_document(f"rome_ate_w{width}_im.json", out)
        ce_clean = pd.read_parquet(out / "ce_clean.parquet")
        ce_corr = pd.read_parquet(out / "ce_corr.parquet")
        p_clean = np.exp(-ce_clean.groupby("example")["value"].mean())
        p_corr = (
            ce_corr.assign(p=np.exp(-ce_corr["value"])).groupby("example")["p"].mean()
        )
        effects.append(p_clean - p_corr)
        clean_all.append(p_clean)
        corr_all.append(p_corr)
    pooled = pd.concat(effects)
    print(
        f"n={len(pooled)}; clean baseline {pd.concat(clean_all).mean():.4f}, "
        f"corrupted {pd.concat(corr_all).mean():.4f} (paper: 0.270 / 0.0847)"
    )
    assert_golden("rome.ate", float(pooled.mean() * 100))


def _axis_column(frame: pd.DataFrame, needle: str) -> str:
    matches = [c for c in frame.columns if needle in c]
    assert len(matches) == 1, f"axis column {needle!r}: {list(frame.columns)}"
    return matches[0]


def test_rome_hidden_state_aie_peak(tmp_path):
    """AIE(l) = mean over facts and seeds of P_restored(l) - P_corrupted,
    pooled over the width shards; peak asserted against Fig 2a."""
    import numpy as np

    per_layer: dict[int, list[pd.Series]] = {}
    for width in (2, 3, 4, 5):
        out = tmp_path / f"w{width}"
        run_document(f"rome_aie_w{width}_im.json", out)
        ce_corr = pd.read_parquet(out / "ce_corr.parquet")
        ce_rest = pd.read_parquet(out / "ce_rest.parquet")
        p_corr = (
            ce_corr.assign(p=np.exp(-ce_corr["value"])).groupby("example")["p"].mean()
        )
        layer_col = _axis_column(ce_rest, "hidden.layer")
        rest = ce_rest.assign(p=np.exp(-ce_rest["value"]))
        for layer, group in rest.groupby(layer_col):
            p_rest = group.groupby("example")["p"].mean()
            per_layer.setdefault(int(layer), []).append(p_rest - p_corr)
    aie = {layer: pd.concat(parts).mean() * 100 for layer, parts in per_layer.items()}
    peak_layer = max(aie, key=aie.get)
    print(f"AIE by layer: { {k: round(v, 2) for k, v in sorted(aie.items())} }")
    print(f"peak layer {peak_layer} (paper: 15)")
    assert peak_layer < 32, f"peak at layer {peak_layer} — not early-to-middle"
    assert_golden("rome.hidden_aie_peak", float(aie[peak_layer]))


def test_rome_mlp_window_aie_peak(tmp_path):
    """Fig 2b: restore a 10-layer MLP window [l-4, l+5] (clipped) at the
    last subject token in the corrupted run. Ten simultaneous swap sites
    cannot ride one sweep axis, so one document per window center is
    generated here (the parity-goldens build-doc-per-case pattern) —
    generated documents are not digest-pinned, unlike the authored ones."""
    import json as json_lib

    import numpy as np

    centers = list(range(4, 48, 5))
    aie: dict[int, list[pd.Series]] = {}
    for width in (2, 3, 4, 5):
        for center in centers:
            layers = [l for l in range(center - 4, center + 6) if 0 <= l < 48]
            doc = {
                "version": "1",
                "description": f"generated: ROME MLP window restore, center {center}, width-{width} shard",
                "model": {"key": "gpt2-xl", "revision": "main"},
                "data": {"base": {"dataset": f"counterfact/facts_w{width}", "field": "input"}},
                "positions": {"last_subject": {"index": -1, "scope": {"variable": "subject"}}},
                "sites": {
                    "emb": {"component": "embeddings"},
                    "lm_head": {"component": "lm_head"},
                    **{f"mlp{l}": {"component": "mlp_output", "layer": l} for l in layers},
                },
                "reads": {
                    "logits_corr": {"site": "lm_head", "pos": -1, "model": "corrupted", "input": "base"},
                    "logits_rest": {"site": "lm_head", "pos": -1, "model": "restored", "input": "base"},
                    **{
                        f"v{l}": {"site": f"mlp{l}", "pos": "last_subject", "model": "original", "input": "base"}
                        for l in layers
                    },
                },
                "edits": {
                    "noise": {
                        "site": "emb",
                        "pos": {"variable": "subject"},
                        "do": {"gaussian": {"seed": 7, "scale": 0.144681, "axis": "tp_duplicated"}},
                    },
                    **{
                        f"rest{l}": {"site": f"mlp{l}", "pos": "last_subject", "do": {"swap": f"v{l}"}}
                        for l in layers
                    },
                },
                "intervened_models": {
                    "corrupted": {"input": "base", "edits": ["noise"]},
                    "restored": {"input": "base", "edits": ["noise"] + [f"rest{l}" for l in layers]},
                },
                "metrics": {
                    "ce_corr": {"kind": "cross_entropy", "of": "logits_corr", "target": "answer"},
                    "ce_rest": {"kind": "cross_entropy", "of": "logits_rest", "target": "answer"},
                },
                "save": [
                    {"value": "ce_corr", "model": "corrupted", "input": "base", "file_path": "ce_corr.parquet"},
                    {"value": "ce_rest", "model": "restored", "input": "base", "file_path": "ce_rest.parquet"},
                ],
            }
            doc_path = tmp_path / f"mlp_c{center}_w{width}.json"
            doc_path.write_text(json_lib.dumps(doc))
            out = tmp_path / f"out_c{center}_w{width}"
            argv = [
                "run",
                str(doc_path),
                "--data-root",
                str(FIXTURES / "data"),
                "--artifacts-root",
                str(out / "artifacts"),
                "--out",
                str(out),
                "--device",
                _device(),
                "--dtype",
                "fp32",
            ]
            assert main(argv) == 0
            ce_corr = pd.read_parquet(out / "ce_corr.parquet")
            ce_rest = pd.read_parquet(out / "ce_rest.parquet")
            diff = np.exp(-ce_rest["value"].to_numpy()) - np.exp(
                -ce_corr["value"].to_numpy()
            )
            aie.setdefault(center, []).append(pd.Series(diff))
    pooled = {c: pd.concat(parts).mean() * 100 for c, parts in aie.items()}
    peak = max(pooled, key=pooled.get)
    print(f"MLP-window AIE by center: { {k: round(v, 2) for k, v in sorted(pooled.items())} }")
    print(f"peak center {peak}")
    assert_golden("rome.mlp_window_aie_peak", float(pooled[peak]))


def test_hydra_compensation_r2(tmp_path):
    """Fig 7: per ablation layer, regress the summed downstream
    compensatory effect (sum over probe > ablation of de_abl - de_clean)
    on the ablated layer's own clean direct effect, per prompt; assert
    the best layer's R^2 against the model-transferred floor."""
    skip_if_pending("hydra.compensation_r2")
    import numpy as np

    import json as json_lib
    import re

    import torch
    from safetensors.torch import load_file

    from causalab.neural.pytorch_hooks.loading import load_model
    from causalab.neural.pytorch_hooks.metrics import column_token_id

    run_document("hydra_grid_im.json", tmp_path, dtype="bf16")

    # frozen-norm linearized unembedding attribution (the paper's DE):
    # DE_r(l) = (gamma * a_l / rms(h_final_r))^T W_U[ml_token], with each
    # run's normalizer held fixed — in-document subtraction/injection edits
    # let RMSNorm renormalize and measured R^2 0.09 / 0.30
    bundle = load_model("meta-llama/Llama-3.1-8B", dtype="bf16", device=_device())
    gamma = bundle.model.model.norm.weight.detach().float().cpu()
    w_u = bundle.model.lm_head.weight.detach().float().cpu()
    eps = float(bundle.model.config.rms_norm_eps)
    rows = json_lib.loads((FIXTURES / "data" / "hydra" / "facts.json").read_text())
    ml_ids = torch.tensor(
        [column_token_id(bundle.tokenizer, r["ml_token"]) for r in rows]
    )
    u_ml = w_u[ml_ids] * gamma  # (n, d): the per-prompt readout direction

    def tensor_map(path):
        """{coordinate-label: (n, d) float tensor} from one save file."""
        return {
            key[key.find("[") :] if "[" in key else "": tensor.squeeze(1).float()
            for key, tensor in load_file(str(path)).items()
        }

    def abl_of(label: str) -> int:
        return int(re.search(r"abl\.layer=(\d+)", label).group(1))

    def rms(h):
        return torch.sqrt((h * h).mean(dim=-1, keepdim=True) + eps)

    hf_clean = next(iter(tensor_map(tmp_path / "hf_clean.safetensors").values()))
    hf_abl = tensor_map(tmp_path / "hf_abl.safetensors")  # per (abl, draw) label
    inv_clean = 1.0 / rms(hf_clean)
    de_clean = {}
    draws: dict[tuple[int, int], list] = {}  # (abl_layer, probe) -> per-draw DEs
    for layer in range(32):
        a_clean = next(iter(tensor_map(tmp_path / f"acts_c_{layer}.safetensors").values()))
        de_clean[layer] = ((a_clean * inv_clean) * u_ml).sum(-1)
        for label, a in tensor_map(tmp_path / f"acts_a_{layer}.safetensors").items():
            de = ((a / rms(hf_abl[label])) * u_ml).sum(-1)
            draws.setdefault((abl_of(label), layer), []).append(de)
    # per-prompt DEs averaged over the resample draws (the paper averages ~15)
    de_abl = {key: torch.stack(parts).mean(0) for key, parts in draws.items()}

    te_clean = pd.read_parquet(tmp_path / "te_clean.parquet")
    te_abl = pd.read_parquet(tmp_path / "te_abl.parquet")
    te_c = te_clean.groupby("example")["value"].first().to_numpy()
    abl_col = _axis_column(te_abl, "abl.layer")

    r2_by_layer = {}
    for abl_layer in sorted({k[0] for k in de_abl}):
        downstream = range(abl_layer + 1, 32)
        y = sum(de_abl[(abl_layer, l)] - de_clean[l] for l in downstream).numpy()
        x = de_clean[abl_layer].numpy()
        r2_by_layer[abl_layer] = float(np.corrcoef(x, y)[0, 1] ** 2)
        # diagnostic: how much of the total-effect change the compensation explains
        te_a = (
            te_abl[te_abl[abl_col] == abl_layer]
            .groupby("example")["value"]
            .mean()
            .to_numpy()
        )
        r2_change = float(np.corrcoef(y, te_c - te_a)[0, 1] ** 2)
        print(
            f"abl {abl_layer}: R^2(compensation ~ DE) {r2_by_layer[abl_layer]:.3f}, "
            f"R^2(compensation ~ TE change) {r2_change:.3f}"
        )
    assert_golden("hydra.compensation_r2", max(r2_by_layer.values()))


def test_mixing_positional_shares(tmp_path):
    """Fig 2: positional share by query-position bucket at the most-mixed
    layer. Share = rows whose post-patch argmax equals the positional
    prediction, normalized among rows attributed to any of the three
    mechanisms; the most-mixed layer maximizes total attribution pooled
    over the three buckets (one document per bucket keeps the
    single-batch lm_head forward within GPU memory)."""
    merged_parts = []
    for bucket in ("first", "middle", "last"):
        out = tmp_path / bucket
        run_document(f"mixing_scan_{bucket}_im.json", out, dtype="bf16")
        frames = {}
        for mech in ("pos", "lex", "ref"):
            frame = pd.read_parquet(out / f"match_{mech}.parquet")
            layer_col = _axis_column(frame, "target.layer")
            frames[mech] = frame.rename(columns={layer_col: "layer"})
        part = frames["pos"][["example", "layer"]].copy()
        for mech in ("pos", "lex", "ref"):
            part[mech] = frames[mech]["value"].to_numpy()
        part["bucket"] = bucket
        merged_parts.append(part)
    merged = pd.concat(merged_parts, ignore_index=True)
    merged["attributed"] = merged[["pos", "lex", "ref"]].sum(axis=1)

    by_layer = merged.groupby("layer")["attributed"].mean()
    best_layer = by_layer.idxmax()
    print(f"attribution by layer: { {int(k): round(v, 3) for k, v in by_layer.items()} }")
    print(f"most-mixed layer {best_layer} (paper: ~18 of 26)")

    at_best = merged[merged["layer"] == best_layer]
    shares = {}
    for bucket, group in at_best.groupby("bucket"):
        attributed = group["attributed"].sum()
        assert attributed > 0, f"no attributed rows in bucket {bucket!r}"
        shares[bucket] = 100 * group["pos"].sum() / attributed
    print(f"positional shares: { {k: round(v, 1) for k, v in shares.items()} }")
    edge_rows = at_best[at_best["bucket"] != "middle"]
    edges = 100 * edge_rows["pos"].sum() / edge_rows["attributed"].sum()
    assert_golden("mixing.positional_share_edges", float(edges))
    assert_golden("mixing.positional_share_middle", float(shares["middle"]))


def test_arithmetic_steering_diagonal(tmp_path):
    """Fig 7 diagonal, the task author's operationalization: steer every
    baseline-correct hours prompt toward each of the 24 targets (layer-18
    residual, last token, alpha=10, periods {2,5,10,20,50}); a target
    counts when its hour token has the highest prompt-averaged probability
    over the 24 hour tokens. Anti-gaming constraints from the VeriFires
    judge notes are structural here: all 24 targets run, the period set is
    fixed in tests/golden/_steering.py, and the only prompt filter is
    baseline correctness.

    Hybrid by necessity: affine Fourier probes are not a v1 train
    objective, so the pinned harvest document collects the residuals and
    the probes are fitted here (the package-pinned recipe: Adam, lr 1e-3,
    500 epochs); the per-prompt Eq. 4 correction is a pytorch_fn edit
    (local-only) in per-target generated documents."""
    import json as json_lib

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from tests.golden import _steering

    # 1. baseline: keep the prompts the model answers correctly
    base_out = tmp_path / "baseline"
    run_document("hours_baseline_im.json", base_out, dtype="bf16")
    acc = pd.read_parquet(base_out / "acc.parquet")
    rows = json_lib.loads((FIXTURES / "data" / "hours" / "all.json").read_text())
    correct = [rows[int(e)] for e, v in zip(acc["example"], acc["value"]) if v == 1.0]
    print(f"baseline-correct prompts: {len(correct)}/1152")

    # 2. harvest addition residuals (pinned document)
    harvest_out = tmp_path / "harvest"
    run_document("addition_harvest_im.json", harvest_out, dtype="bf16")
    acts = load_file(str(harvest_out / "acts_l18.safetensors"))["acts"].squeeze(1).float()
    sums = torch.tensor(
        [r["sum"] for r in json_lib.loads((FIXTURES / "data" / "addition" / "pairs.json").read_text())],
        dtype=torch.float32,
    )
    assert acts.shape[0] == len(sums)

    # 3. fit the affine sine/cosine probes (package-pinned recipe)
    probes: dict[int, dict] = {}
    for period in _steering.PERIODS:
        entry = {}
        for kind, fn in (("sin", torch.sin), ("cos", torch.cos)):
            target = fn(2 * torch.pi * sums / period)
            w = torch.zeros(acts.shape[1], requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=1e-3)
            for _ in range(500):
                opt.zero_grad()
                loss = ((acts @ w + b - target) ** 2).mean()
                loss.backward()
                opt.step()
            entry[f"w_{kind}"] = w.detach()
            entry[f"b_{kind}"] = float(b.detach())
            print(f"T={period} {kind}: final MSE {float(loss):.4f}")
        probes[period] = entry
    _steering.configure(probes)

    # 4. per-target steering documents over the correct prompts
    data_root = tmp_path / "data" / "hours"
    data_root.mkdir(parents=True)
    (data_root / "correct.json").write_text(json_lib.dumps(correct))
    groups = {f"{h:02d}": [f"{h:02d}"] for h in range(24)}
    hits = 0
    for target in range(24):
        doc = {
            "version": "1",
            "description": f"generated: hours steering toward target {target:02d}",
            "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
            "data": {"base": {"dataset": "hours/correct", "field": "input"}},
            "sites": {
                "l18": {"component": "block_output", "layer": 18},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "logits": {"site": "lm_head", "pos": -1, "model": "steered", "input": "base"}
            },
            "edits": {
                "steer": {
                    "site": "l18",
                    "pos": -1,
                    "do": {
                        "pytorch_fn": {
                            "qualname": f"tests.golden._steering.apply_target_{target}"
                        }
                    },
                }
            },
            "intervened_models": {"steered": {"input": "base", "edits": ["steer"]}},
            "metrics": {
                "hour_probs": {"kind": "class_probs", "of": "logits", "groups": groups}
            },
            "save": [
                {
                    "value": "hour_probs",
                    "model": "steered",
                    "input": "base",
                    "file_path": "hour_probs.parquet",
                }
            ],
        }
        doc_path = tmp_path / f"steer_{target:02d}.json"
        doc_path.write_text(json_lib.dumps(doc))
        out = tmp_path / f"steer_out_{target:02d}"
        argv = [
            "run",
            str(doc_path),
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-root",
            str(out / "artifacts"),
            "--out",
            str(out),
            "--device",
            _device(),
            "--dtype",
            "bf16",
        ]
        assert main(argv) == 0
        table = pd.read_parquet(out / "hour_probs.parquet")
        means = {
            name: float(np.mean([json_lib.loads(v)[name] if isinstance(v, str) else v[name] for v in table["value"]]))
            for name in groups
        }
        top = max(means, key=means.get)
        hits += top == f"{target:02d}"
        print(f"target {target:02d}: top {top} (p={means[top]:.3f})")
    assert_golden("arithmetic.steering_top1_fraction", hits / 24)
