# Subspace causal-analysis pipeline — conventions

Reference material for [`subspace-causal-analysis-pipeline.md`](subspace-causal-analysis-pipeline.md). The workflow lives there; this file holds the lifecycle map, the gate-read contracts, the viewer-merge fragment, and the restrictions.

## Input and posture

The pipeline ingests **one input: a subspace bundle** — a rotation artifact, a manifest, or a `{manifold_bundle, community_id}` pointer. The headline question is always *is this **given** subspace causal?* (interchange-IIA on the threaded `fixed` rotation). There is no input classifier and no mode switch. (To adjudicate a task/behavior instead — discovering the mediating subspace with a learned rotation — use the sibling `../answer-research-question/answer-research-question.md`.)

Fixed knobs on the path:

| Knob | Value |
|---|---|
| Headline question | *is this **given** subspace causal?* |
| `../explore-subspace/explore-subspace.md` + **G1** | run (first gate) |
| `../answer-research-question/hypothesis-generation/hypothesis-generation.md` seed | the characterized concept (G1 `refined_hypothesis`) |
| `../answer-research-question/hypothesis-generation/hypothesis-generation.md` posture | author the causal model **and** counterfactuals from the concept |
| `subspace.method` (in the plan phase) | `fixed`, `score: true` — thread the given rotation, single `layers:[L]`, `k_features` = rotation `k` |
| Subspace output dir (auto-discovered) | `subspace/fixed_k*/` |
| Viewer fragment | + global *Characterization evidence* block |

**Fixed order — `../explore-subspace/explore-subspace.md` → `../answer-research-question/hypothesis-generation/hypothesis-generation.md` → task setup.** `../answer-research-question/hypothesis-generation/hypothesis-generation.md` runs **before** the setup-task step (the setup-task guide, `../implementation/setup-task/setup-task.md`), seeded from the characterized concept. There is no second develop-causal-model pass.

**Accepted trade — design-vs-as-built drift.** With a single `../answer-research-question/hypothesis-generation/hypothesis-generation.md` pass before the setup-task step, **G2 certifies the design** (`code/hypotheses/`), not the **as-built** package the setup-task step emits. If setup-task restructures variables or coarsens the output, a certified-distinguishable design could become confounded as-built and G2 won't catch it. This is **backstopped downstream**: the §D.5 baseline gate and the IIA gate in the run phase test the as-built task on GPU, so realization-drift that kills distinguishability surfaces as a null IIA result — later and more expensive, but not silently. Acceptable **if the setup-task step faithfully realizes the design**; revisit (e.g. add a cheap as-built re-check) only if it routinely re-authors.

## Lifecycle map

| Artifact | Location | Written by |
|---|---|---|
| Resolved subspace manifest (provenance) | `$WORKDIR/plan/subspace_manifest.json` | this pipeline |
| Refined-hypothesis bundle (`refined_hypothesis.json`, evidence, figures/) | `$WORKDIR/plan/` (figures under `plan/figures/`) | `../explore-subspace/explore-subspace.md` |
| Failure artifact (`more_info_needed.json`) | `$WORKDIR/plan/` | `../explore-subspace/explore-subspace.md` |
| Hypotheses doc (`HYPOTHESES.md`) | `$WORKDIR/plan/` | `../answer-research-question/hypothesis-generation/hypothesis-generation.md` |
| Distinguishability matrix (`distinguishability.json`) | `$WORKDIR/artifacts/develop_hypothesis/n*/` | `../answer-research-question/hypothesis-generation/hypothesis-generation.md` |
| Task package(s) | `causalab/tasks/…` or working-dir-local `code/tasks/<name>/` (per setup-task policy) | the setup-task guide (`../implementation/setup-task/setup-task.md`) |
| Prefix-gate interim reports (G1, G2) | `$WORKDIR/run/interim_reports/` | this pipeline |
| `RESEARCH_OBJECTIVE.md`, `PLAN.md` (with §D fixed-IIA wiring + §D.5 gates) | `$WORKDIR/plan/` | the plan phase |
| Task-setup figure (`task_setup.html`) | `$WORKDIR/plan/figures/` | this pipeline (`python -m causalab.runner.task_setup_figure`, Step 8) |
| `viewer_spec.yaml` (default §D selection) | `$WORKDIR/plan/viewer_spec.yaml` | the plan phase, then **this pipeline merges** the canonical fragment below (the fragment is effectively the viewer) |
| Runner config, run logs, tail interim reports | `$WORKDIR/run/` | the run phase |
| Raw artifacts (`subspace/fixed_k{k}/…` IIA `results.json`) | `$WORKDIR/artifacts/{task}/{model}/…` | the run phase |
| Final report (with per-stage outcomes) | `$WORKDIR/result/REPORT.md` | the interpret phase |
| Structured HTML artifact viewer | `$WORKDIR/result/artifact_viewer/` | the interpret phase (artifact-viewer render step) |

## Gate G1 — after `../explore-subspace/explore-subspace.md`

G1 is the first gate. Pass iff `refined_hypothesis.json` is present (not `more_info_needed.json`) **and** its verdict is a positive one. `verdict` ∈ {`confirmed`, `refined`, `disagreed`, `unresolved`}; only `confirmed`/`refined` pass — `disagreed` (derived hypothesis names a different axis) and `unresolved` (reconcile loop exhausted) do not.

```bash
RH="$WORKDIR/plan/refined_hypothesis.json"
MIN="$WORKDIR/plan/more_info_needed.json"
if [ -f "$RH" ] && printf '%s' "$(jq -r '.verdict' "$RH")" | grep -qE '^(confirmed|refined)$'; then
    echo "G1 PASS ($(jq -r '.verdict' "$RH"))"
else
    [ -f "$MIN" ] && jq -r '.actionable_requests[]?' "$MIN"   # print the asks on abort
    echo "G1 FAIL"
fi
```

On fail: write `run/interim_reports/characterize_subspace.md` (Outcome: aborted), print the actionable requests, halt.

## Gate G2 — after `../answer-research-question/hypothesis-generation/hypothesis-generation.md`

Read the matrix at `artifacts/develop_hypothesis/n*/distinguishability.json`. Pass iff **all** hold:

1. **Power vs null** — some focal `target` has a `datasets[*].per_target[target].vs_null` that is non-trivial (≳ 0.2; near 0 means the design can't move that target's output).
2. **A deconfounding design exists** — some `(target, alternative)` has a high `datasets[*].per_target[target].alternatives[alternative]` (≳ 0.7) where that alternative is **not** grouped with the target in `always_confounded`.
3. **Not inert / confounded-everywhere** — the focal targets are not all stuck in `always_confounded`, and none appears *only* paired with `null` there (that target is inert under the sampler — intervening never moves the output).

Surface the deciding numbers, then apply the rule:

```bash
DJ=$(ls "$WORKDIR"/artifacts/develop_hypothesis/n*/distinguishability.json | head -1)
jq '{
  targets,
  best_vs_null: ([.datasets[].per_target[].vs_null // 0] | max),
  best_alt_separation: ([.datasets[].per_target[].alternatives // {} | to_entries[]
                          | select(.key|IN("null","all")|not) | .value] | max),
  always_confounded,
  singletons
}' "$DJ"
```

On fail (nothing distinguishable / inert under the sampler): write `run/interim_reports/develop_hypothesis.md` (Outcome: aborted), halt. Localizing an indistinguishable hypothesis is meaningless.

## Viewer-merge fragment

This fragment **is** the canonical viewer (a run's the plan phase autonomous base selection is effectively empty, so the prepended fragment defines what the reader sees). Write it to `$WORKDIR/plan/viewer_spec_extra.yaml`, then merge with `cd ~/.silico/libraries/causalab-internal && python -m causalab.io.viewer_spec_merge --spec $WORKDIR/plan/viewer_spec.yaml --fragment $WORKDIR/plan/viewer_spec_extra.yaml --position prepend`. Globs resolve against `--root $WORKDIR` (so both `plan/figures/` and `artifacts/` resolve). Top-level globs are `**/`-prefixed to survive the `{task}/{model}/` middle path; **inside the per-task `repeat`**, globs resolve relative to each `artifacts/<task>/` dir (the `**/` survives the `<model>/` level). Unresolved figures are dropped, so a non-gradable task (no manifold/steering plots) just drops those slots and a non-task dir (e.g. `artifacts/develop_hypothesis/`) drops entirely.

Structure: global **Characterization** + global **Task setup**, then a per-task `repeat` block. The receptive-field decision map comes **last** within each task.

```yaml
# Global characterization (prefix-stage figures from ../explore-subspace/explore-subspace.md).
- heading: "Characterization evidence"
  items:
    - caption: "Subspace projection explorer (interactive)"
      candidates:
        - "**/characterize_subspace/**/projection_explorer.html"
        - "**/plan/figures/projection_explorer.html"
    - caption: "Webtext peak-token activation distribution"
      candidates:
        - "**/characterize_subspace/**/projection_distribution.html"
        - "**/plan/figures/projection_distribution.html"
    - caption: "Step-1 vs webtext overlay"
      candidates:
        - "**/characterize_subspace/**/step1_vs_webtext.html"
        - "**/plan/figures/step1_vs_webtext.html"

# Global task setup (the task(s) + worked prompt -> expected-answer examples).
- heading: "Task setup"
  items:
    - caption: "The task(s) and a few worked prompt -> expected-answer examples."
      candidates:
        - "plan/figures/task_setup.html"
        - "**/task_setup.html"

# Per-task evidence: one section per task dir under artifacts/.
- repeat:
    over: "artifacts/*/"
    label_from: {kind: dirname}
    heading: "Task: {label}"
    sections:
      - heading: "Baseline"
        items:
          - caption: "Confusion heatmap — gold class vs the model's output."
            candidates:
              - "**/baseline/**/confusion_heatmap.*"
          - caption: "Ground-truth structure (baseline dim-0)."
            candidates:
              - "**/baseline/**/ground_truth_dim0.*"
      - heading: "Localization (locate)"
        items:
          - caption: "Interchange heatmap — per-layer mediation at the readout position (where intervening flips the class toward the counterfactual)."
            candidates:
              - "**/locate/**/*heatmap*.png"
              - "**/locate/**/*heatmap*.*"
              - "**/locate/**/*.png"
      - heading: "Subspace geometry"
        items:
          - caption: "Subspace point cloud (3D) — the threaded fixed rotation."
            candidates:
              - "**/subspace/**/visualization/features_3d.html"
              - "**/subspace/**/*3d*.html"
          - caption: "Explained-variance spectrum / 2D projection (shown when the point cloud is degenerate, e.g. a binary target)."
            candidates:
              - "**/subspace/**/visualization/features_variance.png"
              - "**/subspace/**/visualization/features_2d.png"
              - "**/subspace/**/visualization/*.png"
      - heading: "Counterfactual geometry"
        items:
          - caption: "Activation manifold (spline) over the subspace as the target sweeps."
            candidates:
              - "**/activation_manifold/**/manifold_3d.html"
              - "**/activation_manifold/**/visualization/manifold_3d.html"
          - caption: "Hellinger output manifold (probability simplex, 3D)."
            candidates:
              - "**/output_manifold/**/hellinger_pca_3d.html"
              - "**/output_manifold/**/output_manifold_3d.html"
              - "**/output_manifold/**/hellinger_pca_2d.*"
      - heading: "Path steering — geometric vs linear faithfulness"
        items:
          - caption: "Dual-manifold embedding — subspace manifold vs output-behaviour manifold."
            candidates:
              - "**/path_steering/**/vis/dual_manifold_bars.html"
              - "**/path_steering/**/vis/dual_manifold.html"
          - row:
              - caption: "Isometry MDS — geometric paths."
                candidates:
                  - "**/path_steering/**/vis/isometry/geometric/isometry_mds.html"
              - caption: "Isometry MDS — linear paths."
                candidates:
                  - "**/path_steering/**/vis/isometry/linear/isometry_mds.html"
          - row:
              - caption: "Isometry / Shepard scatter — geometric paths."
                candidates:
                  - "**/path_steering/**/vis/isometry/geometric/isometry_scatter.*"
              - caption: "Isometry / Shepard scatter — linear paths."
                candidates:
                  - "**/path_steering/**/vis/isometry/linear/isometry_scatter.*"
      - heading: "Receptive-field decision map"
        items:
          - caption: "Argmax-class decision map over the top subspace PCs — interactive slice / class / path controls."
            candidates:
              - "**/path_steering/**/vis/receptive_field.html"
```

Notes:
- The **task-setup figure** (`plan/figures/task_setup.html`) is generated in Step 8 by `python -m causalab.runner.task_setup_figure`.
- **IIA evidence.** The IIA **score** (`subspace_score`, `full_cell_score`, `score_ratio`) is numeric in `subspace/fixed_k*/**/results.json` and reported by the interpret phase in `REPORT.md` — the shipped fixed analysis emits no interchange heatmap, so the viewer carries the threaded-subspace **geometry** instead.
- **Auto-discovery is method-agnostic.** `find_subspace_dirs` (`causalab/io/pipelines.py`) scans the whole `subspace/` dir, so `activation_manifold` / `path_steering` pick up `subspace/fixed_k*/` — no per-task refit.
- The **receptive-field map** is an opt-in `path_steering` viz; it only exists if Step 7 wired `path_steering.visualizations` to include `receptive_field` (otherwise the slot drops).
- The per-task `repeat` dedups on its `over` glob, so re-merging the fragment is idempotent (no duplicated repeat block).

## What this pipeline does NOT do

- Does not write `RESEARCH_OBJECTIVE.md` / `PLAN.md` (owned by the plan phase), runner configs (owned by the run phase), or `result/REPORT.md` + `result/artifact_viewer/` (owned by the interpret phase).
- Does not scaffold any IIA analysis — `method: fixed` is shipped in `subspace` (B2-1/#262); the plan **wires** it, it adds no analysis code.
- Does not introduce a second input type — the input is exactly a subspace bundle.
- Does not reimplement the artifact viewer — it ships from `causalab/io/artifact_viewer.py` via the interpret phase's artifact-viewer render step; this pipeline only merges `viewer_spec.yaml` (via the tested `causalab/io/viewer_spec_merge.py` helper).
- Does not reimplement the per-stage tail gating — it inherits it from the run phase.
- Does not add file-type dispatch to any downstream workflow.
- Does not prompt — autonomous-only, gate-free.

## Restrictions

- Confine this pipeline's writes to the working directory `$WORKDIR/` (plus the setup-task task package, per its own policy).
- Do not edit shipped `causalab/` at runtime from this pipeline; any executable helper it relies on ships under `causalab/` (shipped-helper convention) — here, `causalab/io/viewer_spec_merge.py`.
- Batch the setup-task step (and setup-methods / setup-analyses, via `../implementation/setup-methods/setup-methods.md` / `../implementation/setup-analyses/setup-analyses.md`, if a node turns out to be unshipped).
- Run against the causalab library the Lab maintains at `~/.silico/libraries/causalab-internal` (`cd` there for any causalab command); it is a managed clone that may be re-synced.
