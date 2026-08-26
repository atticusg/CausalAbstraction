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
- **Closed vocabularies.** Step kinds, reductions, plots, and the
  transform ops are closed sets. Anything they cannot express is post-hoc
  analysis outside the record — the same trade the IM spec makes for
  metrics. Closed is not frozen: the op registry grows by pull request
  (§11), never by document.
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
`protocol · transform · select · plot`, plus:

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
- **A `transform` producer is the exception: its table is ranked as
  written.** A transform step carries no sweep coordinates, and its op
  already decided what a row is, so there is nothing to group by and
  re-aggregating would collapse the very rows the document was validated
  against. The columns `value` and `emit` may name are exactly the ones
  the op declares (§2.4) — there is no implicit `value` column, so a
  select over a transform table names the column it ranks.
- Output: `<run>/<step>/values.json` — an artifact **values table** in
  exactly the shape the IM spec's artifact-valued fields read
  (`{"artifact": "<step>", "key": "<emit key>"}`; the store resolves a
  step name to its `values.json`).

### 2.4 `transform` steps — a registered, versioned, deterministic op

The other half of what an analysis does: turn a table or a tensor into
another one, deterministically, *inside* the record. Fitting a basis,
aggregating per-head statistics, running a paired t-test — none of these
touch a model, and before this step type each one was post-hoc analysis
over saved files, which is exactly what makes a pipeline inexpressible as
a document.

```json
"fit": {
  "type": "transform", "op": "fit_pca@1",
  "inputs": {"acts": {"step": "harvest", "value": "acts.safetensors"}},
  "params": {"k": 8},
  "outputs": {"weight": "basis.safetensors", "spectrum": "spectrum.parquet"}
}
```

| field | meaning |
|---|---|
| `op` | ✓ — `name@version` from the registry (`causalab/transform/`); an unknown name, or an unknown version of a known name, is a load error with suggestions |
| `inputs` | ✓ — `{slot: {step, value}}`, one entry per input slot the op declares; each **is** a derived dependency edge (§3). `slot`/`entry` additionally address one entry of a swept producer's bundle (IM spec §2.5) |
| `params` | – the op's declared parameters, checked against its schema at load; defaults are materialized into the canonical form |
| `outputs` | ✓ — `{slot: file_path}` under the step dir, one per output slot the op declares. A missing slot, an extra slot, or a path whose extension contradicts the slot's kind is a load error |

- **The op vocabulary is closed, and versioned.** `name@version` is the
  numerics contract: two runs of the same document must agree, so a
  behavioural change ships as `fit_pca@2` and documents written against
  `@1` keep digesting — and running — as written. The version is part of
  the canonical form; the op's *implementation* never is, the same rule
  that keeps backends out of protocol digests.
- **Determinism is the registry's admission criterion.** An op must be a
  pure function of its declared inputs and params. Anything stochastic
  takes an explicit `seed` param and must be bit-stable across devices;
  an op that cannot meet that does not belong in the registry, and that
  refusal is the point of a closed set. (`fit_pca@1` declares no seed
  because a full SVD has no randomness to pin — only a sign convention,
  which it fixes.)
- **An op declares the columns of every table it writes.** That is what a
  consuming `select`/`plot` step's column references are checked against
  at load (§5.7), since a transform step has no sweep axes to check
  against instead.
- **A transform step is not a new tensor channel.** Its tensor outputs are
  ordinary `.safetensors` bundles in the run tree, read back through the
  same `file_path` overlay and `entry` selector every other handoff uses
  (§3) — including by a later *protocol* step, which is how a fitted
  artifact re-enters a model-touching run.
- **Provenance.** A tensor a transform step writes is stamped with an
  ArtifactIdentity, so the §5.10 check a consuming document performs is a
  real one: the fields its tensor inputs agree on are **inherited** (a fit
  over activations from model X at site S is bound to X and S), the op may
  declare fields its params define (a basis's rank `k`), and the step
  stamps `produced_by` with the digest of its own canonical entry — its
  provenance unit, the analogue of a protocol point's digest. An op with
  no tensor input has nothing to inherit, so its tensor output cannot be
  consumed as a featurizer bundle; that is a limit, not an oversight.

### 2.5 `plot` steps — closed figure vocabulary

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

### 2.6 `save`

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
  fitted by an earlier step: `"fit/rot.safetensors"`). A step that swept
  writes one bundle holding one entry per point, so the loading document
  names which with `entry` (IM spec §2.5) — authored, or implied by its own
  sweep coordinates.
- A `transform` step's `inputs` are the third spelling of the same idea:
  each entry names a prior step's output directly, so the table of inputs
  **is** the step's dependency edges — nothing is discovered by walking a
  nested document, because a transform step has none.
- These references **are** the derived dependency edges, together with
  `from` on select/plot steps, `inputs` on transform steps, and `after`.
  The step graph must be acyclic; its topological order is the schedule skeleton, and steps with
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
- **What the runner knows about execution** (IM spec §8, execution scale):
  only the step dependency graph. It may run independent steps
  concurrently, but it owns no device, host, or job-system knowledge —
  those belong to the backends it is handed (device/dtype/point
  parallelism) and to site tooling outside the repo (job dispatch, which
  shards *document* runs via the CLI's `--points`; a workflow run is never
  sharded as a unit).
- **Determinism**: with fixed datasets, artifacts, and model, a workflow
  run is a pure function of the document — same digests, same outputs.

## 5. Validation — load-error checklist

1. Strict keys everywhere; closed enums (`type`, `choose`, `plot`, and a
   transform step's `op`) reject with suggestions; derived fields may not
   be authored. A transform step's `inputs`/`outputs` must name exactly
   the slots its op declares — no missing slot, no extra one — and its
   `params` must satisfy the op's parameter schema.
2. Section order per §1; `save` last, non-empty.
3. Step names unique and filesystem-safe; `save` file_paths unique,
   contained (relative, no parent escapes), and colliding with neither a
   step directory nor the reserved `workflow.json`.
4. Every reference resolves: `from`/`after` name declared steps (`from`
   names a `protocol` or a `transform` step — the two that produce
   tables); every `document` file exists and **loads as a valid
   intervention protocol** (with the step's `set` applied); a transform
   `inputs` entry names an output its producer really writes; a save
   entry's `step`/`value` name an actual output.
5. The derived step graph (artifact refs + `from` + `inputs` + `after`)
   is acyclic.
6. Sink rule: every step is consumed by a later step or by `save`.
7. `select`/`plot` column references (`x`, `y`, `series`, the ranked
   `value` column, `emit` values) must be sweep-coordinate columns (or
   `value`) of the referenced table's producing document — checkable at
   load from the inner document's axes; a plot must cover *every* axis of
   its producer (an uncovered axis would collapse into duplicate cells).
   Over a **transform** producer the same rule reads against the op's
   *declared* columns (§2.4) instead of sweep axes, with no implicit
   `value`; axis coverage is vacuous, since there are no axes.
8. `select.table`/`plot.table` name `.parquet` outputs; plot `file_path`
   ends in `.png`/`.pdf`; a transform `outputs` path matches its slot's
   kind (`.parquet` for a table, `.safetensors` for a tensor) and stays
   inside the step dir.
9. A protocol step's `set` paths must exist in the target document (an
   override that would create structure is a typo).
10. An artifact ref that names a step must name a `select` step (only
    they emit values tables) — a transform step's outputs are *files*,
    reached by the run-tree overlay, not a values table. A run-tree
    `file_path` load must name a file its step actually saves, whether
    that step is a `protocol` or a `transform` one — both checkable at
    load from the emit table, the inner save manifests, and a transform
    step's declared outputs. The load must also select an
    **entry** the producer will write, for every point of the consuming
    document: a producing document's entry names follow from its own
    expansion, which is deterministic at load (IM spec §3), so a mis-aimed
    tensor handoff fails before any step runs rather than after the
    producing step has spent its compute.

## 6. Derived — never authored

| property | derivation |
|---|---|
| step dependencies, schedule, parallelism | the reference graph (§3) |
| group-by columns of `select`/`plot` | the producing document's sweep axes — none for a transform producer, whose table is used as written (§2.3) |
| the columns of a transform step's table | its op's declared record (§2.4) |
| a transform step's digest, and the identity it stamps | its canonical entry, plus what its tensor inputs agree on (§2.4) |
| inner-document digests | the IM spec's canonicalization |
| the run manifest | stamped at execution |

## 7. Canonical form and digests

- The canonical form materializes every default (`value: "value"`, the
  mean aggregation, a transform op's parameter defaults), sorts `after`
  lists, records a transform step's `op` as `name@version`, and **stamps
  each protocol step with its document's digest**, computed with `set`
  applied: for a
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
    "locate": {"type": "protocol", "document": "../protocols/weekdays_locate_scan.json"},
    "best": {
      "type": "select", "from": "locate", "table": "iia.parquet",
      "choose": "max",
      "emit": {"best_layer": "sites.target.layer", "best_pos": "positions.tap"}
    },
    "fit": {"type": "protocol", "document": "../protocols/weekdays_das_sweep.json",
             "set": {"positions.best": {"artifact": "best", "key": "best_pos"},
                     "sites.target.layer": {"artifact": "best", "key": "best_layer"}}},
    "best_fit": {
      "type": "select", "from": "fit", "table": "iia.parquet",
      "choose": "max",
      "emit": {"best_k": "featurizers.rot.k", "best_seed": "train.seed"}
    },
    "apply": {"type": "protocol", "document": "../protocols/weekdays_das_apply.json",
               "set": {"featurizers.rot.file_path": "fit/rot.safetensors",
                       "featurizers.rot.k": {"artifact": "best_fit", "key": "best_k"},
                       "featurizers.rot.entry": {
                         "k": {"artifact": "best_fit", "key": "best_k"},
                         "seed": {"artifact": "best_fit", "key": "best_seed"}}}},
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
    {"step": "best_fit", "value": "values.json", "file_path": "best_fit.json"},
    {"step": "fit", "value": "iia.parquet", "file_path": "fit_iia.parquet"},
    {"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"},
    {"step": "scan_heatmap", "value": "scan_iia.png", "file_path": "scan_iia.png"},
    {"step": "iia_by_k", "value": "iia_by_k.png", "file_path": "iia_by_k.png"}
  ]
}
```

Derived schedule: `locate` → `best` → `fit` → `best_fit` → `apply`, with
`scan_heatmap` free to run as soon as `locate` finishes and `iia_by_k`
after `fit` — two levels of parallelism nobody authored. The `set`
overrides on `fit` re-point the corpus document's artifact refs at the
in-run `best` step (the authored 08 document names a prior standalone
run's artifact path). `fit` sweeps k × seed, so its bundle holds nine
rotations: `best_fit` names the winning cell the same way `best` named the
winning site, and `apply` selects that entry — one fit applied, provably the
one the numbers chose, with its ArtifactIdentity checked exactly as in a
standalone run.

## 11. Open (for the gate review)

- **`select` aggregation**: v1 hard-codes mean-over-examples. Median /
  count-weighted variants would be new closed enum values, added when a
  pipeline needs them.
- **Cross-workflow threading** stays by external artifact reference (the
  same answer the IM spec §8 gives for cross-document threading).
- **Plot vocabulary**: deliberately two kinds. The old analyses' bespoke
  figures (manifolds, pullback visualizations) are post-hoc consumers of
  saved tables/tensors, not workflow steps — confirm this boundary.
- **Growing the op registry.** A new op, or a new version of one, is a
  pull request against `causalab/transform/ops/`: a record (name, version,
  parameter schema, input and output slots, declared table columns), a
  body whose numerics are imported inside the function, and a unit test
  against a hand-computed oracle plus a determinism assertion. A document
  can never introduce one — that is what makes "the same document runs
  the same way" checkable. The seed set is `fit_pca@1`, `head_stats@1`
  and `paired_ttest@1`, chosen to exercise both directions of the IO
  contract and multi-input slots rather than to cover the analyses.
- **Ops with no consumer yet stay out.** The manifold family
  (`fit_spline`, `geodesic_path`, `hellinger_pca`, `path_scores`) arrives
  with the workflows that consume it, rather than pinning an API before
  its caller exists.
- **Multi-tenant steps**: a step per backend (fit on Megatron, apply on
  serving) needs per-step backend pinning — deferred until a second
  backend exists; the field would be one optional `backend` per protocol
  step.
