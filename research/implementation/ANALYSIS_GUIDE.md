# Experiment and analysis catalog

CausaLab does not have a registry of Hydra analyses. Use a protocol document for
model execution and a workflow script step for deterministic processing.

## Shipped protocol documents

| Document | Use |
|---|---|
| `harvest.json` | Save activations from named sites and positions |
| `interchange.json` | Swap a counterfactual activation and score IIA |
| `path_patching.json` | Patch a sender-to-receiver path while freezing alternatives |
| `das.json` | Train and evaluate a distributed alignment search subspace |
| `dbm.json` | Train a differential binary mask |
| `hydra_effect.json` | Resample-ablate a component and measure downstream direct effects |
| `weekdays_locate_scan.json` | Sweep interchange over layers and positions |
| `weekdays_das_sweep.json` | Sweep DAS rank and seed at a selected site |
| `weekdays_das_apply.json` | Apply one fitted DAS artifact |

These files live under `causalab/configs/protocols/`. Treat them as examples:
replace the model, data, positions, sites, metrics, and outputs to match the
question.

## Reusable method

`causalab/configs/methods/interchange.json` is the current example of a reusable
method. It defines the intervention and scoring logic while leaving the model,
dataset, and layer open. The complete application is
`causalab/configs/runs/weekdays_8b_interchange.json`.

Use a method when the same experimental logic should transfer across tasks or
models. Use a flat protocol when reuse would add indirection without a real stable
interface.

## Shipped script steps

| Module | Use |
|---|---|
| `causalab.analysis.fit_pca` | Fit a centered PCA basis and write its spectrum |
| `causalab.analysis.harvest_difference` | Derive an intervention direction from two harvested tensors |
| `causalab.analysis.head_stats` | Summarize attention-head results |
| `causalab.analysis.paired_ttest` | Compare paired metric rows |
| `causalab.workflow.scripts.select` | Select values from a table for a later step |
| `causalab.io.plots.workflow_figures` | Render heatmaps or line plots and preserve plotted data |

A workflow may also name a repository-local script with
`{"path": "scripts/my_analysis.py"}`. Its content hash becomes part of the
workflow digest.

## Worked workflow

`causalab/configs/workflows/weekdays_8b.json` demonstrates the current full shape:

1. sweep layers and positions;
2. select the best location;
3. sweep DAS rank and seed at that location;
4. select the best fit;
5. apply the fitted artifact;
6. render the scan and rank curves.

Dependencies come from file and artifact references. The workflow does not contain
an authored sequence or scheduler configuration.

## Choosing whether to add Python

Use existing protocol vocabulary when the operation is a read, intervention,
metric, featurizer, sweep, or training objective. Add a workflow script when the
operation is deterministic processing over saved files. Add engine code only when
the experiment needs a new activation component, intervention mechanism, metric,
featurizer, or execution capability.

See [`setup-methods/setup-methods.md`](setup-methods/setup-methods.md) for method
documents and [`setup-analyses/setup-analyses.md`](setup-analyses/setup-analyses.md)
for workflow script steps.
