# Protocol examples — `conditions` vs `compositions` vs `intervened_models`

Six experiments, each written three ways: the readme's per-node `conditions`
(YAML), the named-table `compositions` (YAML), and the settled third format —
`intervened_models` in free JSON with a schema-aware parser (see
`../intervention_protocol_readme.md` §3.6–3.7,
`../intervention_protocol_simplification.md` sec. 2, the side-by-side verdict
in `../intervention_protocol_materialization.md`, and the final decisions +
sweep design in `../intervention_protocol_im.md`).

**Fairness protocol.** The pair members differ ONLY on the judged axis. All
orthogonal simplifications are applied to both variants equally:
`data:` (renamed from `counterfactual_dataset`), inline integer `pos`, no
`params` table (featurizer weights auto-declared), `objective` inside `train`,
`outputs` defaulting to metrics + unconsumed reads, no top-level `seeds`.

- `*_conditions.yaml` — one `interventionals:` table; each node carries
  `input:` and `conditions:`; `do:` defaults to Identity (a pure collect).
- `*_composition.yaml` — three tables: `compositions:` (name → input row +
  edits in force; the un-intervened `base` / `source` compositions exist
  implicitly), `reads:` (value producers; `in:` names the composition read
  under), `edits:` (inert effect definitions listed by compositions).

| # | Experiment | Files |
|---|---|---|
| 01 | activation harvesting | `01_harvest_{conditions,composition}.yaml`, `01_harvest_im.json` |
| 02 | interchange intervention | `02_interchange_{conditions,composition}.yaml`, `02_interchange_im.json` |
| 03 | path patching (IOI) | `03_path_patching_{conditions,composition}.yaml`, `03_path_patching_im.json` |
| 04 | DAS | `04_das_{conditions,composition}.yaml`, `04_das_im.json` |
| 05 | DBM | `05_dbm_{conditions,composition}.yaml`, `05_dbm_im.json` |
| 06 | Hydra effect, main experiment | `06_hydra_effect_{conditions,composition}.yaml`, `06_hydra_effect_im.json` |
| 07 | sweeping weekdays-8b pipeline (multi-file hydra composition) | `weekdays_sweep/` |
| 07/08 | the same pipeline as two self-contained IM documents with in-file sweep axes | `07_weekdays_locate_scan_im.json`, `08_weekdays_das_sweep_im.json` |
| 09 | apply a fitted DAS rotation (featurizer `file_path`, no train) | `09_das_apply_im.json` |

The `*_im.json` files are the settled format: `intervened_models` (the paper's
term for L_{b + I}) sit below `edits`; every declared edit must belong to at
least one intervened model; sweep axes are arrays on scalar-typed fields of
named table entries, expanded and parallelized by the parser (one harvest, many
fits) instead of by multirun. `weekdays_sweep/` is kept as the hydra-world
artifact that 07/08 replace.

The IM files also carry the output-contract decisions
(`../intervention_protocol_im.md` secs. 4-5): a mandatory non-empty `save` as
the last section, whose entries are explicit bindings
`{value, model, input, file_path}` cross-checked against the metric->read->IM
chain (reads, metrics, and trained featurizers saveable; every metric and
every trained featurizer must be saved - featurizer entries are
`{value, site, file_path}`); mandatory
`input` on every intervened model; one global namespace over all declared
names; every sweep axis as an explicit `{"sweep": ...}` wrapper; optional
featurizer `file_path` for loading fitted artifacts; `causal_model` in place
of `high_level` (`neural_model` is an accepted alias of `model`); reads bind
explicitly via `model` + `input` (the earlier `in:` field is retired); the
implicit-conventions audit (`../intervention_protocol_im.md` sec. 6) is
applied — all sites declared (incl. `lm_head`), `train.params` by featurizer
name, dataset refs as plain path/HF key with the content digest stamped at
load, singular `source` + brackets. The YAML pairs predate these decisions
and are frozen as comparison artifacts.

01–06 are deliberately concrete (no interpolations) so the syntax is judged
bare; `weekdays_sweep/` shows the same protocols as a hydra composition of
model/task/protocol groups with sweep axes.
