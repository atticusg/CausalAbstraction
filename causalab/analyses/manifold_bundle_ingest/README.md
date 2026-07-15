# Manifold Bundle Ingest

manifold_bundle_ingest answers: *which `characterize_subspace` inputs does a given manifold-SAE-autointerp community correspond to?* It reads one community's record from a bundle (`hypotheses.jsonl`, keyed by `comm`) and writes the rotation safetensors, an exemplar `step1_dataset.json`, and a `subspace_manifest.json` — with no hand-editing (issue #265). It is **task-less and model-less** (a format adapter, not an experiment). The artifacts produced here are prerequisites for `characterize_subspace`: point that analysis's `subspace.artifact` / `step1_dataset` / `significance` at the manifest emitted here.

## Configuration

- **Root config** (`causalab/configs/config.yaml`) — `experiment_root`: the run root; the analysis writes under `${experiment_root}/manifold_bundle_ingest/comm<id>/`. (No `task` or `model` section is read.)
- **Module config** (`causalab/configs/analysis/manifold_bundle_ingest.yaml`):
  ```yaml
  analysis:
    _name_: manifold_bundle_ingest
    _subdir: comm${.community_id}                                   # output subdir
    _output_dir: ${experiment_root}/manifold_bundle_ingest/${._subdir}
    bundle: ???        # path to the manifold-SAE-autointerp bundle dir (has hypotheses.jsonl)
    community_id: ???  # the `comm` field of the community to ingest
  ```

A shipped runner (`causalab/configs/runners/manifold_bundle_ingest.yaml`) mounts `/base` + this analysis; pass the bundle and community as overrides:

```bash
scripts/run_exp.sh --experiment-root <root> manifold_bundle_ingest \
    manifold_bundle_ingest.bundle=<bundle_dir> manifold_bundle_ingest.community_id=<id>
```

## Outputs

### Interpretation

- **`subspace_manifest.json`** — the manifest `characterize_subspace` consumes: `subspace_artifact`, `model`/`layer`/`site` (from the community's `config`), `k_features_hint` (= `n_dims`), `step1_dataset`, and `significance` (`hypothesis_text` from `hypothesis.theme`, `topology_description` from `topology_characterization`). A good result maps every field with no `???`; check `provenance.warnings` — a non-empty warning means a `picks/comm<ID>_<label>.json` filename disagreed with the record (e.g. `comm396_finance.json` is actually about *ease*), so trust the record content over the filename. `provenance.n_exemplars` below ~8 means the reproduction gate downstream may fail.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `subspace.safetensors` | `rotation_matrix` `[n_dims, d_model]` tensor | `characterize_subspace` (`subspace.artifact`) |
| `step1_dataset.json` | `list[str]` of exemplar spans | `characterize_subspace` (`subspace.step1_dataset`) |
| `subspace_manifest.json` | manifest dict (fields above) + `provenance` | the downstream `characterize_subspace` run |

The rotation is saved in the bundle's on-disk `(n_dims, d_model)` orientation under the `rotation_matrix` key; `characterize_subspace`'s `loading.load_subspace` auto-transposes it to `(d_model, k)`.
