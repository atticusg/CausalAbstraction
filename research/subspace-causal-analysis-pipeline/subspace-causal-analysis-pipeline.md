# Subspace causal-analysis pipeline

> **Stale — predates the protocol refactor.** This document was written against
> the Hydra runner (`scripts/run_exp.sh`), the `causalab/analyses/` tree,
> `methods/` as Python, and SLURM dispatch, all of which were retired in PR #20.
> The scientific guidance remains correct, but the commands, configuration
> formats, and code paths are no longer valid. Rewriting it is tracked as future
> work. For the current
> protocol, start at [`../answer-research-question/answer-research-question.md`](../answer-research-question/answer-research-question.md).

Name each stage as it starts, report each gate verdict in plain prose, and finish
with the report and viewer. Omit internal mechanics.

This document runs the full pipeline in a fixed order by chaining sibling workflows
and the plan, run, and interpret phases. It owns sequencing but does not reimplement
those stages. Do not pause for per-step researcher confirmation. The pipeline still
uses the fail-fast evidence gates below, including pre-plan gates G1 and G2.

This pipeline accepts **one input: a subspace bundle** — a rotation `.safetensors`,
a manifest `.json`, or a `{manifold_bundle, community_id}` pointer (any of
`../explore-subspace/explore-subspace.md`'s three forms). It tests whether the given
subspace is causal with interchange-IIA on the threaded `fixed` rotation. There is
no other input type or mode switch. To start from a **task or behavior** and discover
the mediating subspace, use
`../answer-research-question/answer-research-question.md`.

```
subspace bundle → pin provenance → explore-subspace → G1
  → develop-causal-model → G2   (seed: the characterized concept)
  → task setup (batched; ≥1 task must be path-steering-capable / graded)
  → [ setup-methods + setup-analyses — ONLY if the plan phase needs an unshipped node ]
  → the plan phase         (wires the full per-task chain: baseline → locate →
                              subspace[ method:fixed score:true ] →
                              activation_manifold → output_manifold →
                              path_steering[visualizations += receptive_field]; figure_format png)
  → task-setup figure + merge the canonical per-task viewer fragment
  → the run phase           (gated per-stage loop)
  → interpret phase         (REPORT.md + viewer)
```

## Required Reading

Before running this pipeline, read:
- `CONVENTIONS.md` (beside this document) — the lifecycle map, the G1/G2 gate-read contracts, the viewer-merge fragment, and restrictions.
- `causalab/analyses/characterize_subspace/README.md` — bundle ingest, the refined-hypothesis bundle, and the `more_info_needed.json` failure mode.
- `causalab/analyses/subspace/README.md` — the fixed-subspace threading contract (`method: fixed`, `score: true`, single `layers:[L]`, `k_features` = rotation `k`).

(Causalab code-path citations resolve under `~/.silico/libraries/causalab-internal/`.)

## Ownership of handoffs

This pipeline **controls sequencing**. After a sub-workflow
returns, ignore its handoff offer and continue with the next step here. This also
suppresses its mid-workflow confirmation steps; G1, G2, and the per-stage tail
gates replace them. Do not add file-type dispatch to downstream workflows.

## Steps

1. **Set up the working directory.** All of this pipeline's outputs land under a single working (output) directory. Keep the neutral sub-layout it references throughout: `plan/` (objective, plan, provenance, figures), `run/` (runner config, logs, interim reports), `artifacts/` (raw per-task outputs), and `result/` (final report + viewer). Below, all relative paths are relative to this working directory; set `WORKDIR` to its absolute path (the shell commands below use `$WORKDIR`).

2. **Guard against overwriting an existing plan (one plan per working dir).** If `plan/` already holds `RESEARCH_OBJECTIVE.md` or `PLAN.md`, stop — never overwrite an existing plan. Use a fresh working directory per plan.

3. **Pin subspace provenance.** The input is a rotation `.safetensors`, a manifest `.json`, or a `{manifold_bundle, community_id}` pointer. Record a provenance copy at `plan/subspace_manifest.json` (the rotation path/source, model, layer, `k`, and where it came from). Delegate parsing to `../explore-subspace/explore-subspace.md` Step 1; this step records provenance and the rotation identity needed by the later fixed-IIA wiring.

4. **Characterize the subspace, then gate G1.** Carry out `../explore-subspace/explore-subspace.md` on the bundle (ignore its end-of-skill handoff offer, per "Ownership of handoffs"). It writes `refined_hypothesis.json` on success or `more_info_needed.json` on a failed reproduction gate. **G1 passes iff** `plan/refined_hypothesis.json` exists and its `verdict` is `confirmed` or `refined` (G1 contract + jq in CONVENTIONS.md). On fail: write `run/interim_reports/characterize_subspace.md` (Outcome: aborted), print the `actionable_requests` from `more_info_needed.json` if it exists (a reproduction-gate failure; on a `disagreed`/`unresolved` verdict there is no such file — report the verdict instead), and **halt the chain**. Without a confirmed or refined hypothesis, there is no valid localization target. G1 is the first gate.

5. **Develop hypotheses, then gate G2.** Carry out `../answer-research-question/hypothesis-generation/hypothesis-generation.md` (ignore its handoff offer). It runs **before** task setup. Seed from the characterized concept (G1's `refined_hypothesis`): `../answer-research-question/hypothesis-generation/hypothesis-generation.md` authors the causal model **and** the distinguishing counterfactuals from that partial starting specification.

   It writes `plan/HYPOTHESES.md` and `artifacts/develop_hypothesis/n*/distinguishability.json`. **G2 passes iff** that distinguishability shows ≥1 target hypothesis with non-trivial power-vs-null **and** ≥1 deconfounding design, with the targets not confounded-everywhere / inert (G2 contract + jq in CONVENTIONS.md). On fail: write `run/interim_reports/develop_hypothesis.md` (Outcome: aborted), and **halt**. An indistinguishable hypothesis provides no valid localization target.

6. **Set up the task(s).** Follow the setup-task guide (`../implementation/setup-task/setup-task.md`) once, batched, with the task(s) `../answer-research-question/hypothesis-generation/hypothesis-generation.md` produced (the base causal-model task and, only if the IIA needs it, a latent-labeling MCQA task). Do **not** hardcode a task count or assume two conditions; take whatever develop-causal-model yielded. At least one task must support path steering and have a graded, non-degenerate output. The geometry, manifold, and path-steering analyses cannot use a categorical or binary target (for example, `could not convert string to float`). Do not add another pre-run check here. The run phase uses these files, and its §D.5 baseline check stops when the task is unusable.

7. **Plan the full per-task chain with the headline IIA wired.** Carry out the plan phase with an objective whose headline analysis is the **interchange-IIA of the subspace**, and whose §D wires the canonical **per-task** analysis chain:
   - **Gradable (graded/numeric-target) task:** `baseline → locate → subspace(…) → activation_manifold → output_manifold → path_steering`.
   - **Non-gradable (categorical/binary) task:** `baseline → locate → subspace(…)` only — the manifold/steering nodes would error.

   The `subspace` node is always `method: fixed`, `score: true`. Thread the **given** rotation from `plan/subspace_manifest.json` via `subspace.fixed.{artifact|source|feature_ids}`, a single `layers:[L]`, `k_features` = the rotation's `k`. Test whether the given subspace is causal.

   The produced rotation is **auto-discovered** by `activation_manifold` / `path_steering` with no per-task refit — discovery scans the whole `subspace/` dir (`find_subspace_dirs` in `causalab/io/pipelines.py`), so it picks up `subspace/fixed_k*/`. Require `§F visualization.figure_format: png` and **`path_steering.visualizations` to include `receptive_field`** (`[path_visualization, isometry_visualization, dual_manifold, receptive_field]`) so the path-steering decision map is actually produced for the viewer (keep `receptive_field.grid_res` modest). Do **not** scaffold a local mediation analysis (it is no longer needed). The setup-methods / setup-analyses guides (`../implementation/setup-methods/setup-methods.md` / `../implementation/setup-analyses/setup-analyses.md`) stay **conditional** — follow them (batched) only if the plan phase genuinely needs a node that is not shipped.

8. **Emit the task-setup figure, then merge the canonical viewer fragment.** First generate the task-setup figure (the task(s) + a few worked `prompt → expected answer` examples) — pass one spec per task the setup-task step created:
   ```bash
   cd ~/.silico/libraries/causalab-internal && python -m causalab.runner.task_setup_figure \
       --tasks '[{"name": "<task>", "target_variable": "<var>", "description": "<one-liner>"}]' \
       --n 3 --out "$WORKDIR/plan/figures/task_setup.html"
   ```
   Then, after the plan phase writes `plan/viewer_spec.yaml`, write the canonical fragment from CONVENTIONS.md §Viewer-merge fragment to `plan/viewer_spec_extra.yaml` — it includes the global *Characterization evidence* block (`../explore-subspace/explore-subspace.md` ran, so its figures exist) and the per-task repeat. Then merge it (idempotent — the per-task `repeat` dedups on its `over` glob; never clobbers the plan's selection):
   ```bash
   cd ~/.silico/libraries/causalab-internal && python -m causalab.io.viewer_spec_merge \
       --spec "$WORKDIR/plan/viewer_spec.yaml" \
       --fragment "$WORKDIR/plan/viewer_spec_extra.yaml" \
       --position prepend
   ```
   This feeds the artifact-viewer renderer run by the interpret phase's artifact-viewer render step (`--root $WORKDIR`), which resolves both `plan/figures/` and `artifacts/`. The per-task `repeat` renders one section per `artifacts/<task>/`, dropping figures a task didn't produce.

9. **Run the experiment, then interpret.** Carry out the run phase. It materializes the runner config(s), executes the §D chain **one gate-bounded stage at a time** with the capped iterate-until-signal loop, writes per-stage interim reports, and — including the subspace IIA node's §D.5 gate (the `fixed` headline analysis) and the baseline gate — fails fast on a dead stage. After it completes, carry out the **interpret phase**: write `result/REPORT.md` and render `result/artifact_viewer/index.html` from the merged spec. This pipeline does **not** prompt during this phase.

10. **Hand off.** Print the working-directory, plan, and artifacts paths plus both canonical deliverables — `result/REPORT.md` and `result/artifact_viewer/index.html`. Write no result file of your own; those two (owned by the interpret phase) are canonical. Note in one line whether every gate (G1, G2, and the tail gates) passed or where the chain aborted, and report the headline fixed-subspace IIA score from `REPORT.md`.
