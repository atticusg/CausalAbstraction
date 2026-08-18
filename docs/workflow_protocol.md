# Workflow Protocol — specification v1

Self-contained specification of the second config type: a declarative
format for **chains of intervention-protocol executions plus their IO** —
what `causalab/analyses` used to be as Python. A workflow document
composes intervention protocols (`docs/intervention_protocol.md`, "the
IM spec" below); it never touches a neural network itself.

## 0. Principles

- **A campaign is a value, not a program.** One JSON document describes
  the whole pipeline: which intervention protocols run, how values flow
  between them, what is reduced, plotted, and kept. It can be hashed,
  diffed, shared, and re-run.
- **The runner owns execution.** The document says *what depends on
  what*; the runner derives *how* — order, parallelism, resume. There is
  no `sequence:`/`parallel:` construct: independent steps **are** the
  parallelism, and `explain` reports the derived schedule.
- **Everything declared must reach a sink.** A step nothing consumes and
  nothing saves is a load error, exactly as in the IM spec.
- **Closed vocabularies.** Step kinds, reductions, and plots are closed
  sets. Anything they cannot express is post-hoc analysis outside the
  record — the same trade the IM spec makes for metrics.
- **Format**: strict JSON (unknown keys = error); YAML accepted at the
  authoring surface. **v1 scope**: a finite acyclic step graph — no
  loops, no conditionals, no nested workflows, one workflow per run.

## 1. Document layout

Sections in this order (order enforced; `save` last):

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `description` | – | free text, the pipeline's intent |
| 3 | `steps` | ✓ | the named step table — the whole pipeline |
| 4 | `save` | ✓ | the complete output manifest — non-empty, last |

- **One namespace**: step names are unique, filesystem-safe
  (`[A-Za-z0-9_-]+`), and become the run tree's subdirectories.
- A workflow document is distinguished from an intervention protocol by
  its `steps` section; the CLI verbs dispatch on it (§9).

## 2. Section reference

### 2.1 `steps` — common fields

Every step is an object with a `type` from the closed set
`protocol · select · plot`, plus:

| field | meaning |
|---|---|
| `type` | ✓ — the step vocabulary below |
| `description` | – free text |
| `after` | – step names that must complete first, beyond the derived data dependencies (pure ordering; rare) |

Data dependencies are **derived, never authored**: a step depends on
every step whose outputs it references (§3). `after` adds ordering
without data flow.

### 2.2 `protocol` steps — run one intervention protocol

```json
"locate": {"type": "protocol", "document": "protocols/locate_scan.json"}
```

| field | meaning |
|---|---|
| `document` | ✓ — path to an intervention-protocol file, relative to the workflow file |
| `set` | – dotted-path overrides applied before loading, same syntax and semantics as the CLI `--set` (IM spec §9). Unlike the CLI form these are **part of the record**: they enter the canonical form and the digest |
| `max_points` | – override of the sweep point cap for this document (IM spec §5.14) |

- The document is loaded through the full IM pipeline (validate, expand,
  plan) with the workflow's resolution environment (§3); every load error
  of the inner document is a load error of the workflow.
- The step's outputs are exactly the inner document's `save` manifest,
  landing under `<run>/<step>/`.

### 2.3 `select` steps — reduce a result table to named values

The stage-1 → stage-2 seam: turn a swept metric table into the scalar(s)
the next protocol needs (the locate → DAS handoff), as data instead of a
notebook.

```json
"best": {
  "type": "select", "from": "locate", "table": "iia.parquet",
  "choose": "max", "value": "value",
  "emit": {"best_layer": "sites.target.layer", "best_pos": "positions.tap"}
}
```

| field | meaning |
|---|---|
| `from` | ✓ — the producing step |
| `table` | ✓ — a `.parquet` metric table among that step's outputs |
| `choose` | ✓ — `max` \| `min` |
| `value` | – the column ranked (default `value`) |
| `emit` | ✓ — `{artifact key: column}`; the emitted values come from the chosen row |

- Rows are first **grouped by the table's sweep-coordinate columns**
  (stamped by the backend, one per axis; derived, never authored) and
  aggregated by **mean** over examples — v1's one aggregation. `choose`
  then picks the best group; `emit` reads that group's coordinate (or
  value) columns.
- Output: `<run>/<step>/values.json` — an artifact **values table** in
  exactly the shape the IM spec's artifact-valued fields read
  (`{"artifact": "<step>", "key": "<emit key>"}`; the store resolves a
  step name to its `values.json`).

### 2.4 `plot` steps — closed figure vocabulary

```json
"scan": {"type": "plot", "plot": "heatmap", "from": "locate",
          "table": "iia.parquet", "x": "sites.target.layer",
          "y": "positions.tap", "value": "value", "file_path": "scan_iia.png"}
```

| field | meaning |
|---|---|
| `plot` | ✓ — `heatmap` \| `lines` |
| `from`, `table` | ✓ — as in `select` |
| `x` | ✓ — a coordinate column (the horizontal axis) |
| `y` | heatmap ✓ — the second coordinate column |
| `series` | lines – — one line per value of this coordinate column |
| `value` | – the plotted column (default `value`), mean-aggregated over examples |
| `file_path` | ✓ — `.png` or `.pdf`, within the step's output dir (relative, no parent escapes) |

- Two plots cover the v1 pipeline: scan grids (`heatmap`) and
  metric-vs-axis curves (`lines`). Every other figure is post-hoc
  analysis over the saved tables — deliberately outside the record.

### 2.5 `save`

Mandatory, non-empty, the last section — the complete manifest of what
leaves the workflow run, in the IM spec's binding-restated style:

```json
{"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"}
```

| field | meaning |
|---|---|
| `step` | ✓ — the producing step (cross-checked: `value` must be among its outputs) |
| `value` | ✓ — the output's name within the step: a protocol step's manifest path, a select step's `values.json`, a plot step's figure |
| `file_path` | ✓ — where it lands under the workflow's output root; unique per entry |

- Step outputs **not** saved remain in the run tree as intermediates
  (they are still on disk — the run tree is the working state, `save` is
  what the run *publishes*).

## 3. Cross-step wiring

One mechanism, inherited from the IM spec: **artifact references**.

- Inside a protocol step's document, an artifact ref whose first path
  segment names a step (`{"artifact": "best", "key": "best_layer"}`)
  resolves to that step's outputs; any other ref resolves against the
  external artifacts root. Step names shadow the external root — the
  canonical form stamps every resolved value, so the record shows which
  store answered.
- The same overlay applies to `file_path` loads (a featurizer bundle
  fitted by an earlier step: `"fit/rot.safetensors"`).
- These references **are** the derived dependency edges, together with
  `from` on select/plot steps and `after`. The step graph must be
  acyclic; its topological order is the schedule skeleton, and steps with
  no path between them may run in parallel — the runner's choice, never
  authored.

## 4. Execution semantics

- Steps run in a topological order of the derived graph; each step's
  outputs land under `<run>/<step>/`.
- A protocol step executes through the standard backend routing (IM spec
  §8) — capabilities derive from the union over the inner document's
  points.
- The runner writes a run manifest (`workflow.json`: the workflow digest,
  each step's inner digests and status) — the workflow-level analogue of
  point-protocol stamping.
- **Determinism**: with fixed datasets, artifacts, and model, a workflow
  run is a pure function of the document — same digests, same outputs.

## 5. Validation — load-error checklist

1. Strict keys everywhere; closed enums (`type`, `choose`, `plot`) reject
   with suggestions; derived fields may not be authored.
2. Section order per §1; `save` last, non-empty.
3. Step names unique and filesystem-safe; `save` file_paths unique,
   contained (relative, no parent escapes), and colliding with neither a
   step directory nor the reserved `workflow.json`.
4. Every reference resolves: `from`/`after` name declared steps; every
   `document` file exists and **loads as a valid intervention protocol**
   (with the step's `set` applied); a save entry's `step`/`value` name an
   actual output.
5. The derived step graph (artifact refs + `from` + `after`) is acyclic.
6. Sink rule: every step is consumed by a later step or by `save`.
7. `select`/`plot` column references (`x`, `y`, `series`, the ranked
   `value` column, `emit` values) must be sweep-coordinate columns (or
   `value`) of the referenced table's producing document — checkable at
   load from the inner document's axes; a plot must cover *every* axis of
   its producer (an uncovered axis would collapse into duplicate cells).
8. `select.table`/`plot.table` name `.parquet` outputs; plot `file_path`
   ends in `.png`/`.pdf`.
9. A protocol step's `set` paths must exist in the target document (an
   override that would create structure is a typo).
10. An artifact ref that names a step must name a `select` step (only
    they emit values tables), and a run-tree `file_path` load must name a
    file its step actually saves — both checkable at load from the emit
    table and the inner save manifests.

## 6. Derived — never authored

| property | derivation |
|---|---|
| step dependencies, schedule, parallelism | the reference graph (§3) |
| group-by columns of `select`/`plot` | the producing document's sweep axes |
| inner-document digests | the IM spec's canonicalization |
| the run manifest | stamped at execution |

## 7. Canonical form and digests

- The canonical form materializes every default (`value: "value"`, the
  mean aggregation), sorts `after` lists, and **stamps each protocol
  step with its document's digest**, computed with `set` applied: for a
  document with no in-run references this is the IM spec §7 campaign
  digest; a document that references step outputs (its values exist only
  at run time) stamps the digest of its overridden authored form, and
  the fully resolved per-point digests land in the run manifest. Either
  way the workflow digest changes exactly when the campaign it runs
  changes, without inlining documents.
- `digest = sha256(canonical bytes)`, same byte rules as the IM spec.

## 8. Runner contract

The runner is causalab-owned (there is no backend choice at the workflow
level — backends are chosen per protocol step):

| service | contract |
|---|---|
| schedule | topological order; independent steps may parallelize |
| stores | the run-tree/external artifact overlay (§3), one output root |
| select | group by coordinate columns, mean over examples, argmax/argmin, emit values table |
| plot | the closed figure vocabulary over aggregated tables |
| stamping | `workflow.json` manifest; inner runs stamp per the IM spec |
| resume | a step whose outputs exist with matching stamped digests may be skipped (`--resume`) |

## 9. CLI

The same four verbs (IM spec §9) accept workflow documents — dispatch on
the `steps` section:

| verb | effect |
|---|---|
| `run <wf> --out <dir>` | validate, schedule, execute steps, stamp the manifest |
| `validate <wf>` | the §5 checklist, including every inner document |
| `explain <wf>` | the derived schedule (levels of parallel steps), per-step inner digests/point counts, what `save` publishes |
| `digest <wf>` | the workflow digest |

## 10. Worked example — the weekdays-8b pipeline

The pipeline the analyses tree ran as hydra multirun (locate a layer ×
position cell, fit DAS rotations at it, apply the best fit on the test
split), as one workflow over the golden-corpus documents 07/08/09:

```json
{
  "version": "1",
  "description": "weekdays-8b: locate -> DAS k x seed fits at the best cell -> apply on test; scan heatmap + IIA-vs-k curves.",
  "steps": {
    "locate": {"type": "protocol", "document": "../methods/weekdays_locate_scan.json"},
    "best": {
      "type": "select", "from": "locate", "table": "iia.parquet",
      "choose": "max",
      "emit": {"best_layer": "sites.target.layer", "best_pos": "positions.tap"}
    },
    "fit": {"type": "protocol", "document": "../methods/weekdays_das_sweep.json",
             "set": {"positions.best": {"artifact": "best", "key": "best_pos"},
                     "sites.target.layer": {"artifact": "best", "key": "best_layer"}}},
    "apply": {"type": "protocol", "document": "../methods/weekdays_das_apply.json",
               "set": {"featurizers.rot.file_path": "fit/rot.safetensors"}},
    "scan_heatmap": {"type": "plot", "plot": "heatmap", "from": "locate",
                      "table": "iia.parquet", "x": "sites.target.layer",
                      "y": "positions.tap", "value": "value",
                      "file_path": "scan_iia.png"},
    "iia_by_k": {"type": "plot", "plot": "lines", "from": "fit",
                  "table": "iia.parquet", "x": "featurizers.rot.k",
                  "series": "train.seed", "value": "value",
                  "file_path": "iia_by_k.png"}
  },
  "save": [
    {"step": "best", "value": "values.json", "file_path": "best_cell.json"},
    {"step": "fit", "value": "iia.parquet", "file_path": "fit_iia.parquet"},
    {"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"},
    {"step": "scan_heatmap", "value": "scan_iia.png", "file_path": "scan_iia.png"},
    {"step": "iia_by_k", "value": "iia_by_k.png", "file_path": "iia_by_k.png"}
  ]
}
```

Derived schedule: `locate` → `best` → `fit` → `apply`, with
`scan_heatmap` free to run as soon as `locate` finishes and `iia_by_k`
after `fit` — two levels of parallelism nobody authored. The `set`
overrides on `fit` re-point the corpus document's artifact refs at the
in-run `best` step (the authored 08 document names a prior standalone
run's artifact path); `apply` loads the rotation the `fit` step saved,
with its ArtifactIdentity checked at load exactly as in a standalone run.

## 11. Open (for the gate review)

- **`select` aggregation**: v1 hard-codes mean-over-examples. Median /
  count-weighted variants would be new closed enum values, added when a
  pipeline needs them.
- **Cross-workflow threading** stays by external artifact reference (the
  same answer the IM spec §8 gives for cross-document threading).
- **Plot vocabulary**: deliberately two kinds. The old analyses' bespoke
  figures (manifolds, pullback visualizations) are post-hoc consumers of
  saved tables/tensors, not workflow steps — confirm this boundary.
- **Multi-tenant steps**: a step per backend (fit on Megatron, apply on
  serving) needs per-step backend pinning — deferred until a second
  backend exists; the field would be one optional `backend` per protocol
  step.
