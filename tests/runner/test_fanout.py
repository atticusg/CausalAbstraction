"""Unit tests for the cluster-agnostic fan-out orchestrator.

Covers the pure, no-GPU logic: spec/axis parsing, cartesian expansion, manifest
round-trip (including list-literal overrides), scan-result merging, and per-site
cluster-directive resolution precedence. The SLURM/local submission paths are
exercised end-to-end separately (they need a scheduler / GPUs).
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from causalab.runner import fanout

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Axis / spec parsing
# --------------------------------------------------------------------------- #
def test_split_top_level_respects_brackets():
    assert fanout._split_top_level("8,16,32") == ["8", "16", "32"]
    assert fanout._split_top_level("[subject],[verb]") == ["[subject]", "[verb]"]
    # Inner commas inside a single list value are preserved.
    assert fanout._split_top_level("[a,b],[c,d]") == ["[a,b]", "[c,d]"]


def test_expand_scalar_spec_range_and_list():
    assert fanout._expand_scalar_spec("range(0,4)") == ["0", "1", "2", "3"]
    assert fanout._expand_scalar_spec("range(0,6,2)") == ["0", "2", "4"]
    assert fanout._expand_scalar_spec("3,5,7") == ["3", "5", "7"]


def test_expand_scalar_spec_zero_step_rejected():
    with pytest.raises(ValueError):
        fanout._expand_scalar_spec("range(0,4,0)")


def test_parse_axis_verbatim():
    key, values = fanout._parse_axis("subspace.k_features=8,16,32")
    assert key == "subspace.k_features"
    assert values == ["8", "16", "32"]


def test_parse_axis_expands_range_to_scalars():
    # --axis with range() yields scalars (not singleton lists) for scalar fields
    # like exploration.pair.index.
    key, values = fanout._parse_axis("exploration.pair.index=range(0,3)")
    assert key == "exploration.pair.index"
    assert values == ["0", "1", "2"]


def test_parse_axis_each_wraps_singletons():
    key, values = fanout._parse_axis_each("locate.layers=range(0,3)")
    assert key == "locate.layers"
    # Each scalar becomes a singleton list so the analysis runs one cell per shard.
    assert values == ["[0]", "[1]", "[2]"]


def test_parse_kv_requires_equals():
    with pytest.raises(ValueError):
        fanout._parse_kv("noequals", flag="--const")


# --------------------------------------------------------------------------- #
# Cartesian expansion
# --------------------------------------------------------------------------- #
def test_build_shards_cartesian_product_with_consts():
    axes = [
        ("locate.layers", ["[0]", "[1]"]),
        ("locate.token_positions", ["[s]", "[v]", "[o]"]),
    ]
    shards = fanout.build_shards(axes, {"locate.method": "interchange"})
    assert len(shards) == 6  # 2 layers x 3 tokens
    # Every shard carries the constant override.
    assert all(s["locate.method"] == "interchange" for s in shards)
    # All combinations are distinct.
    combos = {(s["locate.layers"], s["locate.token_positions"]) for s in shards}
    assert len(combos) == 6


def test_build_shards_single_axis():
    shards = fanout.build_shards([("model.id", ["a", "b"])], {})
    assert [s["model.id"] for s in shards] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_roundtrip_preserves_list_literals(tmp_path):
    shards = [
        {"locate.layers": "[0]", "locate.token_positions": "[subject,verb]"},
        {"locate.layers": "[1]", "locate.token_positions": "[subject,verb]"},
    ]
    manifest = fanout.write_manifest(tmp_path, shards, {"runner": "demo"})
    rows = fanout.read_manifest(manifest)
    assert [r["shard_id"] for r in rows] == ["00000", "00001"]
    assert rows[0]["overrides"]["locate.layers"] == "[0]"
    # The two-element list value survives intact (no comma splitting).
    assert rows[1]["overrides"]["locate.token_positions"] == "[subject,verb]"
    # spec.resolved.yaml is written alongside.
    assert (tmp_path / fanout.FANOUT_DIR_NAME / "spec.resolved.yaml").is_file()


def test_emit_row_emits_id_then_tokens(tmp_path, capsys):
    shards = [{"a": "1", "b": "[x,y]"}]
    manifest = fanout.write_manifest(tmp_path, shards, {})
    fanout.emit_row(str(manifest), 0)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "00000"
    assert set(lines[1:]) == {"a=1", "b=[x,y]"}


# --------------------------------------------------------------------------- #
# Scan-result merge
# --------------------------------------------------------------------------- #
def test_merge_scores_results_unions_disjoint_cells():
    # Two layer-shards over the same token position; cell keys are disjoint.
    shard0 = {
        "scores_per_cell": {"0|subject": 0.2, "0|verb": 0.9},
        "scores_per_layer": {"0": 0.9},
        "token_position_ids": ["subject", "verb"],
        "best_cell": {"layer": 0, "token_position": "verb"},
        "best_layer": 0,
    }
    shard1 = {
        "scores_per_cell": {"1|subject": 0.5, "1|verb": 0.95},
        "scores_per_layer": {"1": 0.95},
        "token_position_ids": ["subject", "verb"],
        "best_cell": {"layer": 1, "token_position": "verb"},
        "best_layer": 1,
    }
    merged = fanout._merge_scores_results([shard0, shard1])
    assert merged["scores_per_cell"] == {
        "0|subject": 0.2,
        "0|verb": 0.9,
        "1|subject": 0.5,
        "1|verb": 0.95,
    }
    # best recomputed across the union.
    assert merged["best_cell"] == {"layer": 1, "token_position": "verb"}
    assert merged["best_layer"] == 1
    # per-layer = max cell within each layer.
    assert merged["scores_per_layer"] == {"0": 0.9, "1": 0.95}
    # token ids unioned preserving first-seen order.
    assert merged["token_position_ids"] == ["subject", "verb"]


def test_varied_keys_and_suffix():
    rows = [
        {"overrides": {"locate.method": "interchange", "locate.layers": "[0]"}},
        {"overrides": {"locate.method": "interchange", "locate.layers": "[1]"}},
    ]
    varied = fanout._varied_keys(rows)
    assert varied == ["locate.layers"]  # method is constant, layers vary
    suffix = fanout._suffix_for({"locate.layers": "[0]"}, varied)
    assert suffix == "layers-0"  # brackets sanitized


def test_suffix_collapses_path_values():
    # A path-valued axis collapses to its basename stem (not the whole path).
    suffix = fanout._suffix_for(
        {"exploration.probe.prompts": "/tmp/sess/run/p0.json"},
        ["exploration.probe.prompts"],
    )
    assert suffix == "prompts-p0"


# --------------------------------------------------------------------------- #
# Collect (structural overlay)
# --------------------------------------------------------------------------- #
def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_collect_merges_results_and_copies_singletons(tmp_path):
    base = tmp_path
    # Two layer-shards, each a single-cell locate result + a unique file.
    fanout.write_manifest(
        base,
        [{"locate.layers": "[0]"}, {"locate.layers": "[1]"}],
        {"runner": "demo"},
    )
    for sid, layer, score in (("00000", 0, 0.3), ("00001", 1, 0.8)):
        rdir = base / "shards" / sid / "locate" / "interchange" / "var"
        _write(
            rdir / "results.json",
            {
                "scores_per_cell": {f"{layer}|subject": score},
                "scores_per_layer": {str(layer): score},
                "token_position_ids": ["subject"],
                "best_cell": {"layer": layer, "token_position": "subject"},
                "best_layer": layer,
            },
        )
    rc = fanout.collect(base)
    assert rc == 0
    merged = json.loads(
        (base / "locate" / "interchange" / "var" / "results.json").read_text()
    )
    assert merged["scores_per_cell"] == {"0|subject": 0.3, "1|subject": 0.8}
    assert merged["best_layer"] == 1


# --------------------------------------------------------------------------- #
# Cluster directive resolution
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(partition=None, account=None, qos=None, cluster=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cluster_directives_empty_by_default(monkeypatch):
    for var in (
        "CAUSALAB_SLURM_PARTITION",
        "CAUSALAB_SLURM_ACCOUNT",
        "CAUSALAB_SLURM_QOS",
    ):
        monkeypatch.delenv(var, raising=False)
    # default.yaml is all-null -> no directives.
    assert fanout.resolve_cluster_directives(_args()) == []


def test_cluster_directives_cli_flag(monkeypatch):
    for var in (
        "CAUSALAB_SLURM_PARTITION",
        "CAUSALAB_SLURM_ACCOUNT",
        "CAUSALAB_SLURM_QOS",
    ):
        monkeypatch.delenv(var, raising=False)
    assert fanout.resolve_cluster_directives(_args(partition="gpu")) == [
        "--partition=gpu"
    ]


def test_cluster_directives_env_overrides_cli(monkeypatch):
    monkeypatch.setenv("CAUSALAB_SLURM_PARTITION", "from_env")
    monkeypatch.delenv("CAUSALAB_SLURM_ACCOUNT", raising=False)
    monkeypatch.delenv("CAUSALAB_SLURM_QOS", raising=False)
    # env wins over the CLI flag.
    assert fanout.resolve_cluster_directives(_args(partition="from_cli")) == [
        "--partition=from_env"
    ]


# --------------------------------------------------------------------------- #
# Collect — heatmap dedup is order-independent (#326 review bug 2)
# --------------------------------------------------------------------------- #
def test_collect_skips_per_shard_heatmaps_when_results_merged(tmp_path, monkeypatch):
    # Each shard emits results.json (scores) AND its own heatmap.png at the same
    # relative path. The merged results.json re-renders one heatmap, so the
    # per-shard ones must be dropped — regardless of filename sort order
    # ('heatmap.png' sorts before 'results.json').
    base = tmp_path
    fanout.write_manifest(
        base,
        [{"locate.layers": "[0]"}, {"locate.layers": "[1]"}],
        {"runner": "demo"},
    )
    for sid, layer, score in (("00000", 0, 0.3), ("00001", 1, 0.8)):
        rdir = base / "shards" / sid / "locate" / "var"
        _write(
            rdir / "results.json",
            {
                "scores_per_cell": {f"{layer}|subject": score},
                "scores_per_layer": {str(layer): score},
                "token_position_ids": ["subject"],
                "best_cell": {"layer": layer, "token_position": "subject"},
                "best_layer": layer,
            },
        )
        (rdir / "heatmap.png").write_bytes(b"shard-png")

    # Stub the re-render so the test doesn't depend on matplotlib; it writes the
    # single merged heatmap the collect step is expected to leave behind.
    def _fake_render(merged, dest_dir, title):
        (dest_dir / "heatmap.png").write_bytes(b"merged-png")

    monkeypatch.setattr(fanout, "_render_locate_heatmap", _fake_render)

    assert fanout.collect(base) == 0
    var = base / "locate" / "var"
    # Exactly the single merged heatmap — no stray per-shard suffixed copies.
    assert sorted(p.name for p in var.glob("*.png")) == ["heatmap.png"]
    assert (var / "heatmap.png").read_bytes() == b"merged-png"


# --------------------------------------------------------------------------- #
# SLURM submit — does not claim to have waited when it couldn't (#326 review bug 1)
# --------------------------------------------------------------------------- #
def test_submit_slurm_reports_not_waited_without_monitor_jobs(tmp_path, monkeypatch):
    base = tmp_path
    monkeypatch.setattr(fanout, "REPO_ROOT", base)  # keep slurm_logs/ hermetic
    shards = [{"a": "1"}, {"a": "2"}]
    manifest = fanout.write_manifest(base, shards, {"runner": "demo"})

    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="Submitted batch job 777\n", stderr=""
        )

    monkeypatch.setattr(fanout.subprocess, "run", fake_run)
    # sbatch present, monitor_jobs absent -> cannot actually block.
    monkeypatch.setattr(
        fanout.shutil,
        "which",
        lambda name: "/usr/bin/sbatch" if name == "sbatch" else None,
    )

    rc, array_id, waited = fanout.submit_slurm(
        "demo",
        base,
        shards,
        manifest,
        cap=2,
        gpus=1,
        time="00:10:00",
        job_name="t",
        cluster_directives=[],
        session_code=None,
        wait=True,
    )
    assert rc == 0
    assert array_id == "777"
    # The fix: waited is False so main() falls back to --collect-only instead of
    # collecting against shards the (still queued) array job hasn't populated.
    assert waited is False
    # Only the sbatch submit ran — no monitor_jobs subprocess was spawned.
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Local backend — a launch failure is recorded, not silently dropped (#326 review)
# --------------------------------------------------------------------------- #
def test_submit_local_records_launch_failure(tmp_path, monkeypatch):
    base = tmp_path
    monkeypatch.setattr(fanout, "REPO_ROOT", base)
    monkeypatch.setattr(fanout, "_visible_gpu_ids", lambda: ["0"])
    shards = [{"a": "1"}, {"a": "2"}]
    manifest = fanout.write_manifest(base, shards, {"runner": "demo"})

    def boom(*a, **kw):
        raise FileNotFoundError("run_exp.sh")

    monkeypatch.setattr(fanout.subprocess, "run", boom)
    # Without the guard the worker thread would die on the first exception, the
    # second shard would never run, and submit_local would wrongly report success.
    assert fanout.submit_local("demo", base, shards, manifest, cap=1) == 1


# --------------------------------------------------------------------------- #
# Shared GPU/time resolution (#326 review robustness 3 & 5)
# --------------------------------------------------------------------------- #
def test_resolve_gpus_time_defaults_and_strict():
    from omegaconf import OmegaConf

    from causalab.runner.slurm_args import resolve_gpus_time

    full = OmegaConf.create(
        {"model": {"slurm": {"gpus": 4}}, "slurm": {"time": "01:00:00"}}
    )
    assert resolve_gpus_time(full) == (4, "01:00:00")

    # A slurm block without gpus no longer KeyErrors when a default is supplied.
    no_gpus = OmegaConf.create({"model": {"slurm": {}}, "slurm": {"time": "02:00:00"}})
    assert resolve_gpus_time(no_gpus, gpus_default=1, time_default="04:00:00") == (
        1,
        "02:00:00",
    )

    # No value and no default -> loud failure (preserves slurm_args strictness).
    with pytest.raises(KeyError):
        resolve_gpus_time(OmegaConf.create({"model": {}}))
