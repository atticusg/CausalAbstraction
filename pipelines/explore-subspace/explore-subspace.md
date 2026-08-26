# Explore a subspace

> **Stale — predates the protocol refactor.** This document was written against
> the Hydra runner (`scripts/run_exp.sh`), the `causalab/analyses/` tree,
> `methods/` as Python, and SLURM dispatch, all of which were retired in PR #20.
> The science in it is still correct; the invocations, config formats, and code
> paths are not. Rewriting it is tracked as follow-up work. For the current
> protocol, start at [`../answer-research-question/answer-research-question.md`](../answer-research-question/answer-research-question.md).

State the stage, result, and next decision in plain sentences. Omit internal
mechanics such as defaults, step labels, and variable names.

Accept one of three inputs: a raw subspace `.safetensors`, a manifest `.json`, or a
manifold-SAE bundle pointer `.json` (`{manifold_bundle, community_id}`). Describe
the supplied input in prose; do not ask the researcher to classify it.

The skill produces a refined-hypothesis bundle in a working directory under your experiment's output (referred to below as `${WORKDIR}`, with `plan/`, `artifacts/`, and `code/` subdirectories). Under `${WORKDIR}/plan/`:

- `refined_hypothesis.json` — the file downstream causal experiment design consumes.
- `evidence.safetensors` + `evidence.meta.json` — projections and spans.
- `reconciliation_trace.json` — per-iteration judge framing trail.
- `figures/` — Plotly HTML: two static distributions (webtext max-token projection, step-1 vs webtext overlay) plus `projection_explorer.html`, a linked app where clicking a histogram bin reveals that bin's marked context windows and a 3D PCA scatter of its documents (coloured by activation magnitude; hover shows the window + magnitude).

If the reproduction gate fails, `more_info_needed.json` is written instead, and the skill stops with the actionable items printed rather than producing a refined-hypothesis bundle.

## Required Reading

Before running:

1. The output layout conventions in the run guide (`../implementation/running-experiments.md`) — where experiment outputs live and how the working directory is organised.
2. `causalab/analyses/characterize_subspace/README.md` — what the analysis does, the four reconcile verdicts, the bundle layout, and known limitations. (Resolves under `~/.silico/libraries/causalab-internal/`. The analysis package keeps its underscore name `characterize_subspace`; only the workflow reference was renamed to `explore-subspace`.)

## Step 1: Parse the input

The input is a `.safetensors` path (raw artifact), a manifest `.json`, or a manifold-SAE bundle pointer `.json`. Resolve it to a manifest dict with the keys the analysis needs.

**Manifold-SAE bundle (`manifold_bundle` + `community_id`)** — when the input `.json` carries `manifold_bundle` and `community_id` keys (a pointer like `{"manifold_bundle": "/path/to/run", "community_id": 396}`), run the `manifold_bundle_ingest` analysis to build the manifest. Do not hand-edit it:

```bash
cd ~/.silico/libraries/causalab-internal
BUNDLE=$(jq -r '.manifold_bundle' "$INPUT_JSON")
COMM=$(jq -r '.community_id' "$INPUT_JSON")
scripts/run_exp.sh --experiment-root "${WORKDIR}/artifacts" manifold_bundle_ingest \
    manifold_bundle_ingest.bundle="$BUNDLE" manifold_bundle_ingest.community_id="$COMM"
MANIFEST="${WORKDIR}/artifacts/manifold_bundle_ingest/comm${COMM}/subspace_manifest.json"
```

The analysis pulls the community's record from the bundle's `hypotheses.jsonl` (the rotation from `projection_matrix`, model/layer/site from `config`, the exemplar spans from `significance_description`, and the significance from `hypothesis.theme` + topology) and writes `subspace.safetensors`, `step1_dataset.json`, and `subspace_manifest.json` under `${WORKDIR}/artifacts/manifold_bundle_ingest/comm${COMM}/`. Use that emitted `${MANIFEST}` as the manifest for the rest of the skill (its paths are absolute). If a `picks/comm<ID>_<label>.json` filename disagrees with the community's content it warns and trusts the record — the pick filenames are advisory and have been wrong (e.g. `comm396_finance.json` is about *ease*). See `causalab/analyses/manifold_bundle_ingest/README.md`.

**Raw `.safetensors`** — ask the researcher for the missing pieces:

> "I have the subspace artifact. To run the analysis I need a few more things: which model and layer was it fit on, which site (residual / attn-out / mlp-out), the path to the phase-1 dataset, and anything you know about what the subspace is supposed to track (a one-line hypothesis, a point-cloud figure, or a topology description — any combination, or none of those)."

A raw `.safetensors` with no manifest can't proceed — a manifest is required. A manifold-SAE bundle pointer (above) does not need one, so prefer that when ingesting a community.

**Manifest `.json`** — expected shape:

```json
{
  "subspace_artifact": "path/to/subspace.safetensors",
  "model": "huggingface/model-id",
  "layer": 12,
  "site": "residual",
  "k_features_hint": "auto",
  "step1_dataset": "path/to/step1.json",
  "significance": {
    "hypothesis_text": "...",
    "figure_path": null,
    "topology_description": null
  }
}
```

The subspace can be supplied two ways — provide exactly one:

- `"subspace_artifact": "path/to/subspace.safetensors"` — a ready rotation matrix.
- `"subspace_source": {"sae_checkpoint": "...", "clusters_path": "...", "cluster_id": "950", "orthonormalize": true}` — an SAE feature cluster. The analysis builds the rotation itself (orthonormal basis of the cluster's decoder directions) via `methods.sae.decoder_subspace`, so no manual `.safetensors` prep is needed.

Map the supplied input into the runner config's `characterize_subspace.subspace.artifact` **or** `characterize_subspace.subspace.source` (leave the other `null`). Write the resolved manifest to `${WORKDIR}/plan/subspace_manifest.json` for provenance.

## Step 2: Materialise the runner config

Generate a runner config at `${WORKDIR}/code/configs/runners/characterize_subspace.yaml` mounting `analysis/characterize_subspace` plus the model section. Use the manifest values verbatim. Mirror the conventions of other runner configs in the repo (the easiest model is to copy a small existing runner and swap the `defaults:` list).

## Step 3: Run the analysis

Pin the experiment root into the working directory so the output location is deterministic (and the run stays local). The analysis writes under `${EXP_ROOT}/characterize_subspace/${_subdir}`, where `_subdir = ${site}_L${layer}_k${k_features_hint}` (see `causalab/configs/analysis/characterize_subspace.yaml`). Read the three subdir fields back from the manifest you wrote in Step 1 so the `cp` paths below resolve to a real directory rather than a literal:

```bash
EXP_ROOT="${WORKDIR}/artifacts"
MANIFEST="${WORKDIR}/plan/subspace_manifest.json"
SITE=$(jq -r '.site' "$MANIFEST")
LAYER=$(jq -r '.layer' "$MANIFEST")
KHINT=$(jq -r '.k_features_hint' "$MANIFEST")
OUT_DIR="${EXP_ROOT}/characterize_subspace/${SITE}_L${LAYER}_k${KHINT}"

cd ~/.silico/libraries/causalab-internal
CAUSALAB_SESSION_CODE="${WORKDIR}" scripts/run_exp.sh --experiment-root "${EXP_ROOT}" characterize_subspace
```

The `CAUSALAB_SESSION_CODE=${WORKDIR}` prefix is what lets the wrapper discover the session-local config Step 2 wrote — without it the run fails config-not-found (or silently resolves a same-named shipped preset).

On a successful run, copy the bundle into the working directory's `plan/` from the concrete `OUT_DIR`:

```bash
cp "${OUT_DIR}/refined_hypothesis.json"   "${WORKDIR}/plan/"
cp "${OUT_DIR}/evidence.safetensors"      "${WORKDIR}/plan/"
cp "${OUT_DIR}/evidence.meta.json"        "${WORKDIR}/plan/"
cp "${OUT_DIR}/reconciliation_trace.json" "${WORKDIR}/plan/"
cp -r "${OUT_DIR}/figures"                "${WORKDIR}/plan/"
```

If `more_info_needed.json` is present (and `refined_hypothesis.json` is not), copy that one instead (`cp "${OUT_DIR}/more_info_needed.json" "${WORKDIR}/plan/"`), print its `actionable_requests` to the researcher, and stop — there is no refined hypothesis to hand off.

## Step 4: Summarise and hand off

Give the researcher a one-paragraph summary: the verdict (`confirmed` / `refined` / `disagreed` / `unresolved`), the refined hypothesis text, and the path to the bundle.

The refined hypothesis (`${WORKDIR}/plan/refined_hypothesis.json`) is ready for causal experiment design. Hand it to `../answer-research-question/hypothesis-generation/hypothesis-generation.md` to build the causal model, or run the full `../subspace-causal-analysis-pipeline/subspace-causal-analysis-pipeline.md`. Check with the researcher before proceeding.

For `disagreed` / `unresolved` verdicts, stop and offer to revise the hypothesis
before committing more compute.

## Restrictions

- Only write under `${WORKDIR}/`. Do not edit the causalab library (`~/.silico/libraries/causalab-internal/`) from this workflow.
- Do not hand off to causal experiment design when the reproduction gate failed — there is no refined hypothesis to plan against.
- Do not retry the judge calls inside this workflow. The reconcile loop already iterates over framings; if the verdict is `unresolved`, report it and let the researcher decide whether to rerun with different webtext settings.
