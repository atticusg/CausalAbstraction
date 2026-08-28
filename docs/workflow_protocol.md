# Workflow Protocol — specification v2

Self-contained specification of the second config type: a declarative format
for **chains of intervention-protocol executions plus the Python that processes
their outputs** — what `causalab/analyses` used to be. A workflow document
composes intervention protocols (`docs/intervention_protocol.md`, "the IM spec"
below); it never touches a neural network itself.

## 0. Principles

- **A campaign is a value.** One JSON document describes the whole pipeline:
  which intervention protocols run, how files flow between them, and what
  Python turns one into another. It can be hashed, diffed, shared and re-run —
  a workflow run is a pure function of the document **and the scripts it
  names**, both hashed into the digest (§7).
- **The runner owns execution.** The document says *what depends on what*; the
  runner derives *how* — order, parallelism, resume. There is no
  `sequence:`/`parallel:` construct: independent steps **are** the parallelism,
  and `explain` reports the derived schedule.
- **Two step types, one wiring mechanism.** `protocol` is declarative because
  that is where load-time bite lives; `script` is an escape hatch wide enough
  that the pipeline never has to leave the record. Everything they consume is
  spelled one way (§3).
- **Everything declared is published.** A step's declared outputs land in its
  own directory and stay there. There is no `save` section to keep in step with
  the steps, and no sink rule — a terminal plot or report step is legitimate.
- **Two record formats, ever.** JSON for anything structured and readable,
  safetensors for dense numerics. Visualization formats (`.png`, `.pdf`,
  `.html`) are legal outputs but carry no record — a figure is a rendering, not
  an artifact (§2.5).
- **Format**: strict JSON (unknown keys = error); YAML accepted at the authoring
  surface. **v2 scope**: a finite acyclic step graph — no loops, no
  conditionals, no nested workflows, one workflow per run.

## 1. Document layout

Sections in this order (order enforced):

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `description` | – | free text, the pipeline's intent |
| 3 | `output_dir` | ✓ | the workflow's own directory name — one path segment |
| 4 | `steps` | ✓ | the named step table — the whole pipeline |

- **One namespace**: step names are unique, filesystem-safe (`[A-Za-z0-9_-]+`),
  and become the run tree's subdirectories.
- A workflow document is distinguished from an intervention protocol by its
  `steps` section; the CLI verbs dispatch on it (§9).

### 1.1 `output_dir` and the run tree

`output_dir` is a single filesystem-safe path segment — not a nested path, not
absolute. The CLI supplies the root it sits under:

```
<out-root>/<output_dir>/<step>/<the step's declared outputs>
<out-root>/<output_dir>/<step>/_step.json      # the runner's per-step record
<out-root>/<output_dir>/workflow.json          # the run manifest
```

The split is deliberate. The *name* of a workflow's directory is a property of
the workflow; the *root* is a property of the site. So documents stay free of
absolute paths (IM spec §8) while a workflow still owns where its own outputs
gather. `output_dir` is therefore **excluded from the digest** — it names where,
not what (§7).

## 2. Section reference

### 2.1 `steps` — common fields

Every step is an object with a `type` from the closed set
`intervention_protocol · script`, plus:

| field | meaning |
|---|---|
| `type` | ✓ — the step vocabulary below |
| `description` | – free text |
| `after` | – step names that must complete first, beyond the derived data dependencies (pure ordering; rare) |

Data dependencies are **derived, never authored**: a step depends on every step
whose outputs it references (§3). `after` adds ordering without data flow.

### 2.2 `intervention_protocol` steps — run one intervention protocol

```json
"locate": {"type": "intervention_protocol", "document": "protocols/locate_scan.json"}
```

| field | meaning |
|---|---|
| `document` | ✓ — path to an intervention-protocol file, relative to the workflow file |
| `set` | – dotted-path overrides applied before loading, same syntax and semantics as the CLI `--set` (IM spec §9). Unlike the CLI form these are **part of the record**: they enter the canonical form and the digest |
| `max_points` | – override of the sweep point cap for this document (IM spec §5.14) |

- The document is loaded through the full IM pipeline (validate, expand, plan)
  with the workflow's resolution environment (§3); every load error of the inner
  document is a load error of the workflow.
- The step's outputs are exactly the inner document's `save` manifest, landing
  under `<step>/`. **A protocol document keeps its own `save` section** — that
  is what declares a protocol step's output names, and it is what §5 rule 4
  checks references against. It is a different section from the one v1 had at
  the workflow level, which is gone.

**Why `protocol` stays declarative.** A protocol step is not "a script that
happens to take a document". Keeping it a type preserves, all at load time:
inner-document validation (every IM spec §5 rule, before any step runs); sweep
expansion, so point counts and per-point digests reach `explain`; backend
capability routing over the union of the points; `--points` shard dispatch; and
the producer's **sweep axes**, which is what makes a downstream reduction
meaningful at all (§6).

### 2.3 `script` steps — inputs, one Python script, declared outputs

```json
"steer_direction": {
  "type": "script",
  "script": "scripts/harvest_difference.py",
  "inputs": {
    "acts_pos": {"step": "harvest_pos", "file": "acts.safetensors"},
    "acts_neg": {"step": "harvest_neg", "file": "acts.safetensors"},
    "normalize": true
  },
  "outputs": {"direction": "direction.safetensors", "stats": "stats.json"}
}
```

| field | meaning |
|---|---|
| `script` | ✓ — a **locator**: `{"module": "causalab.analysis.fit_pca"}` or `{"path": "scripts/probe.py"}` (§2.4) |
| `inputs` | ✓ — `{name: value}` in the §3 grammar; each reference **is** a derived dependency edge |
| `outputs` | ✓ — `{slot: file}` under the step dir, non-empty |
| `runtime` | – dependency isolation (§4.1) |
| `is_deterministic` | – default `true`; see §7 |

A script step is what makes the pipeline expressible without leaving the
record. It replaces v1's `transform` (a closed, versioned op registry),
`select` and `plot` step types: those were three vocabularies for "run some
Python over the previous step's output", and the registry's admission-by-pull-
request rule meant a one-off corpus-mean harvest could not be written at all.
The reductions v1 had as step types survive as **shipped scripts**, each filed
by subject rather than in one namespace: `causalab.workflow.scripts.select`,
`causalab.io.plots.workflow_figures`, and `causalab.analysis.*` (§2.4).

**Script steps are for deterministic Python analysis.** LLM and judge-style
work stays outside causalab — the protocol layer's determinism is what makes it
digestible, and judging lives in the research-protocol tooling that consumes
these outputs.

#### The output declaration

```json
"outputs": {
  "spectrum": {"file": "spectrum.json",
               "columns": {"component": "int64", "explained": "float64"}},
  "weight": "basis.safetensors"
}
```

The short form is a bare filename. `columns` is optional, allowed only on a
`.json` output, and is a promise about what the step publishes — verified **on
write** (§4). Because the digest covers outputs, a declared column set is part
of the record. Keeping the declaration in the *document* rather than inside the
file is what lets an empty table still be checked: there are no rows to infer
from, but there is always a declaration.

A `.json` output may instead declare **`keys`** — a flat *values object* rather
than a table, with one representative value per key:

```json
"outputs": {
  "values": {"file": "values.json",
             "keys": {"best_layer": 18, "best_pos": {"index": -1}}}
}
```

`columns` and `keys` are mutually exclusive: the first says "an array of row
objects", the second "one object mapping these names to values". `keys` is what
the `key` selector (§3) reads, and it is **load-bearing rather than
documentation**. A protocol step whose `set` pulls a value out of an earlier
step (`{"artifact": "best", "key": "best_layer"}`) cannot resolve it before the
run, so the loader substitutes the declared representative and validates the
inner document against *that* — which is why the representative is a value and
not a type: a position spec must type-check as a position spec, and `"int64"`
would not. The real value replaces it at run time and goes through the same
full load.

v1 derived these representatives from a `select` step's `emit` table plus the
producing document's sweep axes. With `select` a script, the document has to
say it, and saying it is cheap.

### 2.4 Addressing a script, and `file` vs `path`

`script` is a **locator**, the same shape an `inputs` reference uses (§3):

| form | resolves to |
|---|---|
| `{"module": "causalab.analysis.fit_pca"}` | an importable module, found via `importlib.util.find_spec` — which resolves a dotted name to a file **without executing it** |
| `{"path": "scripts/probe.py"}` | a file beside the workflow document, contained, no parent escapes |

The shipped scripts are filed **by subject**, not in one flat namespace:

| module | what it is |
|---|---|
| `causalab.analysis.fit_pca` · `harvest_difference` · `head_stats` · `paired_ttest` | numerical analysis — fits, statistics, and the operands an intervention consumes |
| `causalab.io.plots.workflow_figures` | rendering, beside the rest of `io/plots/` |
| `causalab.workflow.scripts.select` | the one script whose purpose *is* wiring steps together |

v1 spelled a shipped script `causalab:<name>`. That needed a registry — exactly
what this layer removes — and it hid *which* code ran behind a lookup. A module
path says it, and a script can then live where it belongs by subject instead of
where a resolver happens to search.

**`file` vs `path`.** Both words appear in a document and they are not
interchangeable:

- **`file`** is always *a name within some step's output directory*, declared by
  that step — `{"step": "harvest", "file": "acts.safetensors"}` on the way in,
  and a key of `outputs` on the way out. It is a filename, never a location;
  the runner owns where the step's directory lives. A `file` is checkable
  against a declaration.
- **`path`** is a location on disk that the workflow does not own — absolute, or
  relative to the repo root. A `path` is checkable only against the filesystem.

### 2.5 Formats: two that carry the record, three that visualize it

**Record formats — two, and nothing else.** **JSON** for everything structured
and readable (metric tables, values objects, manifests, stats), **safetensors**
for dense numerical weights and tensor artifacts. A metric table is a **native
JSON array of row objects** (`causalab/protocol/tables.py`):

```json
[
  {"example": 0, "sites.target.layer": 18, "value": 0.83},
  {"example": 1, "sites.target.layer": 18, "value": 0.91}
]
```

Labels repeat on every row. That is the deliberate trade — a file `jq` and a
human can both read, at the cost of size. There is no envelope and no embedded
column header: inventing a format inside a format would defeat the point of
having only two. One file per metric, so a document saving three metrics writes
three tables. Non-finite floats are written as `null`: bare `NaN`/`Infinity`
tokens are not JSON, and a metric that computed nothing is exactly the "no
value" a null means.

**Visualization formats — `.png`, `.pdf`, `.html`.** These are legal declared
outputs but carry **no record**: a figure is a *rendering* of an artifact rather
than an artifact itself. So a visualization output may declare no
`columns`/`keys`, and the runner neither checks its shape nor stamps it with an
ArtifactIdentity — existence is the whole contract.

- **`.png` is the default and is preferred over `.pdf`** unless a document asks
  for pdf explicitly. It is what a reviewer can open inline, in a PR, or in a
  notebook without a viewer. `.pdf` is for print or when a vector figure is
  genuinely needed; `.html` for interactive figures.
- The preference is implemented in one place —
  `causalab.io.plots.figure_format.normalize_figure_format(value, default="png")`
  — so every renderer inherits it rather than restating it.
- A step that renders should usually declare the **numbers as well**: the
  shipped `workflow_figures` script takes an optional `plotted` table output
  holding the exact rows it drew, which is what makes a figure checkable and
  lets a later step reference what it showed.

## 3. Cross-step wiring — the reference grammar

**A reference is a locator plus an optional selector.** One grammar, used by
every `inputs` entry.

| locator | resolves to |
|---|---|
| `{"path": P}` | a file on disk: **absolute** if `P` starts with `/`, otherwise **relative to the repo root** |
| `{"step": S, "file": F}` | the file `F` that step `S` declares, in the run tree |

| selector | requires | yields |
|---|---|---|
| *none* | — | the resolved absolute path — the script opens it itself |
| `"key": K` | the locator names a `.json` | the scalar at `K` inside it |
| `"entry": {…}` | the locator names a `.safetensors` | one tensor of a bundle, by coordinate match (IM spec §2.5) |

Anything carrying none of these keys is a **JSON literal**, passed through
unchanged. References are recognized **only at the top level of an `inputs`
entry**, so a nested object is always a literal: `{"cfg": {"step": 3}}` is
unambiguous and the loader never guesses.

```json
"inputs": {
  "layer_in_run":  {"step": "best", "file": "best_cell.json", "key": "best_layer"},
  "layer_on_disk": {"path": "/mnt/data/fits/best_cell.json",  "key": "best_layer"},
  "layer_in_repo": {"path": "configs/pinned_cell.json",       "key": "best_layer"},
  "k": 8
}
```

- **Why `path` carries a tag.** Nothing distinguishes a string that is a path
  from a string that is data: `"meta-llama/Llama-3.1-8B"` and
  `"scripts/prompt.txt"` are both strings. One tag settles it, and earns a
  load-time existence check a bare string could not justify.
- **Why `{"step": …}` cannot be a path.** The run tree's root is a CLI argument
  (§1.1), so `<out-root>/<output_dir>/<step>/…` is unknowable when the document
  is written. A cross-step reference names the step symbolically and lets the
  runner resolve it. That is the one irreducible indirection.
- **`key` and `entry` are one selector per format.** `key` is a name lookup in a
  JSON object and yields a scalar. `entry` is a coordinate match inside a
  safetensors bundle and yields the file plus the selected entry — the tensor
  stays in the file. `entry` cannot collapse into `key`: bundle keys are
  composite (`weight[k=8,seed=0]`) and matching is on **(name, value) pairs,
  never the rendered label**, because the label's field order follows the
  *producing* document's axis order, which the consumer cannot know
  (`causalab/protocol/bundles.py`).
- **No implicit filename.** `file` is always explicit. v1 could omit it because
  a `select` step always wrote `values.json`; with select a script that declares
  its own output names, there is no canonical filename left to default to.
- Inside a protocol step's *inner document*, the IM spec's own reference
  grammar applies unchanged (`{"artifact": …, "key": …}` and `file_path`
  against the artifacts root, with step names shadowing it). A `set` block on a
  protocol step authors those, so it is the one place both grammars meet.
- These references **are** the derived dependency edges, together with `after`.
  The step graph must be acyclic; its topological order is the schedule
  skeleton, and steps with no path between them may run in parallel — the
  runner's choice, never authored.

## 4. Execution semantics

- Steps run in a topological order of the derived graph; each step's outputs
  land under `<step>/`.
- A protocol step executes through the standard backend routing (IM spec §8) —
  capabilities derive from the union over the inner document's points.
- A script step is invoked **in-process by default**:

  ```python
  def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None: ...
  ```

  The script must create every declared output file. The runner then verifies
  each one exists (a missing one fails the step, named by slot), verifies
  declared `columns` against the JSON actually written, and **stamps
  `ArtifactIdentity`** on safetensors outputs — inherited from the step's tensor
  inputs, plus `produced_by` with the step's own digest. Identity stamping stays
  the runner's job so a script cannot forget it.
- The runner writes a per-step record (`<step>/_step.json`: the declared files,
  and for a protocol step its sweep axes and point digests) and a run manifest
  (`workflow.json`) — see §7.
- **What the runner knows about execution** (IM spec §8, execution scale): only
  the step dependency graph. It may run independent steps concurrently, but it
  owns no device, host, or job-system knowledge — those belong to the backends
  it is handed and to site tooling outside the repo (job dispatch, which shards
  *document* runs via the CLI's `--points`; a workflow run is never sharded as a
  unit).

### 4.1 `runtime` — dependency isolation

```json
"runtime": {"isolate": true, "deps": ["umap-learn>=0.5"]}
```

For a step whose dependency set differs from the runner's. An isolated step runs
in a subprocess with `deps` installed; `env` lists variable **names** to pass
through, never values — a secret never appears in a document, a canonical form
or a manifest.

`runtime` **is** part of the canonical form and the digest: which interpreter
and dependency set a step ran under changes what the step *is*, so `--resume`
must not skip across a change to it. That is distinct from IM spec §8's
"execution parameters never enter documents", which is about `--device` /
`--points` — the same computation on different hardware.

### 4.2 Script resolution and the torch-free guarantee

`validate` and `digest` **never import a user script**: importing one would pull
torch (or anything else) into a verb that must stay cheap and runnable on a
machine with no accelerator. So load-time checking of a script is deliberately
shallow — the file exists, `ast.parse` succeeds, a module-level `def main` is
present, and its bytes are hashed (hashing needs no import). Everything else
about the script is a run-time contract. This is a real reduction in load-time
bite compared with v1's op records, and it is the price of a vocabulary wide
enough to hold the work.

## 5. Validation — load-error checklist

**Scope: workflow documents.** The IM spec's own checklist for *intervention
protocol documents* is untouched; rules 8 and 9 below reach into a protocol
document, but they check the workflow's references to it — the document's own
validity stays with the IM loader, run in full.

1. Strict keys everywhere; closed `type` enum rejects with suggestions; derived
   fields may not be authored.
2. Section order per §1. `output_dir` is present and is a single filesystem-safe
   path segment — not nested, not absolute, no parent escapes.
3. Step names unique and filesystem-safe; no step may be named `workflow.json`,
   and no step directory may collide with the run manifest.
4. Every reference resolves: `step` names a declared step; `file` names a file
   that step really declares (a protocol step's inner `save` manifest, a script
   step's `outputs`); `after` names declared steps. A **repo-relative `path`
   must exist at load**; an **absolute `path` is not existence-checked**,
   because validation and execution routinely run on different hosts, so an
   absolute path naming another machine's data would fail a check it should
   pass — it becomes a run-time refusal, and `explain` lists them so the gap is
   visible before dispatch. A **selector must match its locator's format**:
   `key` only on a `.json`, `entry` only on a `.safetensors`, at most one
   selector per reference. A `key` selector on a locator naming an **in-run
   step's** output additionally requires that output to declare `keys`
   containing `K` — that declaration is what makes the reference checkable
   and is what a step-dependent inner document validates against (§2.3).
5. The derived step graph (`inputs` + inner-document artifact refs + `after`) is
   acyclic.
6. `script` names exactly one of `module` or `path`; it resolves — a dotted
   importable module, or a contained relative path — parses, and declares
   `main`. It is never imported (§4.2).
7. `outputs` is non-empty; each output's file is contained inside the step's own
   directory (relative, no parent escapes) and unique within the step; every
   output ends in `.json`, `.safetensors`, `.png`, `.pdf` or `.html` (§2.5);
   `columns` and `keys` are allowed only on a `.json` output, and are
   mutually exclusive.
8. A protocol step's `set` paths must exist in the target document (an override
   that would create structure is a typo), and the document must load as a valid
   intervention protocol with `set` applied.
9. A tensor `entry` selection is checkable at load when the producer is a
   `protocol` step — a producing document's entry names follow from its own
   expansion, which is deterministic at load (IM spec §3), so a mis-aimed tensor
   handoff fails before any step runs rather than after the producing step has
   spent its compute. Against a *script* producer it is a run-time check.
10. An isolated step declares its `deps`; `env` lists names only.
11. `is_deterministic` is a boolean if present.

**Two v1 rules are deliberately gone.** The sink rule ("every step is consumed
by a later step or by `save`") and every `save`-manifest rule die with the
`save` section. Consequence, accepted: nothing flags a step whose outputs
nobody reads. That was the rule's value, and losing it is the price of
"everything is published" — in v1 a terminal plot step needed a `save` entry to
be blessed.

## 6. Derived — never authored

| property | derivation |
|---|---|
| step dependencies, schedule, parallelism | the reference graph (§3) |
| a protocol step's sweep axes and point digests | the IM spec's expansion, republished in `_step.json` |
| the columns a script step's table actually has | verified against its declaration on write |
| a script step's digest, and the identity it stamps | its canonical entry, plus what its tensor inputs agree on |
| inner-document digests | the IM spec's canonicalization |
| the run manifest | stamped at execution |

**The axes still exist.** Because `protocol` stays declarative, the runner knows
each protocol step's sweep axes at load. It republishes them in the step's
`_step.json`, and the shipped `select`/`plot` scripts read them there. So
group-by-coordinates-then-mean survives as *behaviour*; it stops being derived
magic in the document model and becomes a documented thing a script does with
data the runner published. A user script gets the same record.

## 7. Canonical form, digests, and `--resume`

A script step's canonical entry is

```
{type, script, script_sha256, inputs, outputs, runtime, is_deterministic, after}
```

- **The script's content hash is in the digest.** Without it `--resume` is
  incorrect: a step whose script changed would be skipped as up to date.
  Hashing needs no import, so it costs the torch-free guarantee nothing.
- The canonical form materializes every default (`is_deterministic: true`, an
  output's long form), sorts `after` lists, and **stamps each protocol step with
  its document's digest**, computed with `set` applied: for a document with no
  in-run references this is the IM spec §7 campaign digest; a document that
  references step outputs (its values exist only at run time) stamps the digest
  of its overridden authored form, and the fully resolved per-point digests land
  in the run manifest. Either way the workflow digest changes exactly when the
  campaign it runs changes, without inlining documents.
- `output_dir` is **excluded**: it names where, not what.
- `digest = sha256(canonical bytes)`, same byte rules as the IM spec.

**`is_deterministic`** defaults `true` and is part of the digest. It buys two
things: `explain` reports "this workflow is not replayable" and names the steps
responsible, and `--resume` refuses to reuse a non-deterministic step's outputs
unless told to. A step that sets it false is asking for review.

**`workflow.json`** is the run-time record: per step the resolved input values,
the script path and hash, the `runtime` block, the step digest, and status.

## 8. Runner contract

The runner is causalab-owned (there is no backend choice at the workflow level —
backends are chosen per protocol step):

| service | contract |
|---|---|
| schedule | topological order; independent steps may parallelize |
| stores | the run-tree/external artifact overlay (§3), one output root |
| inputs | resolve the §3 grammar to paths and scalars |
| scripts | invoke `main(inputs, outputs)`, in-process or isolated |
| outputs | verify existence and declared columns; stamp identity on safetensors |
| stamping | `_step.json` per step; `workflow.json` for the run; inner runs stamp per the IM spec |
| resume | a step whose outputs exist with a matching stamped digest may be skipped (`--resume`), unless it is non-deterministic |

## 9. CLI

The same four verbs (IM spec §9) accept workflow documents — dispatch on the
`steps` section:

| verb | effect |
|---|---|
| `run <wf> --out <root>` | validate, schedule, execute steps, stamp the manifest |
| `validate <wf>` | the §5 checklist, including every inner document |
| `explain <wf>` | the derived schedule (levels of parallel steps), per-step inner digests/point counts, non-deterministic steps, unchecked absolute paths |
| `digest <wf>` | the workflow digest |

## 10. Worked example — the weekdays-8b pipeline

Locate a layer × position cell, fit DAS rotations at it, apply the best fit on
the test split, and plot — as one workflow over the golden-corpus documents
07/08/09.

```json
{
  "version": "1",
  "description": "weekdays-8b: locate -> DAS k x seed fits at the best cell -> apply on test; scan heatmap + IIA-vs-k curves.",
  "output_dir": "weekdays_8b",
  "steps": {
    "locate": {"type": "intervention_protocol", "document": "../protocols/weekdays_locate_scan.json"},
    "best": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_layer": "sites.target.layer", "best_pos": "positions.tap"}
      },
      "outputs": {"values": "values.json"}
    },
    "fit": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_sweep.json",
             "set": {"positions.best": {"artifact": "best", "key": "best_pos"},
                     "sites.target.layer": {"artifact": "best", "key": "best_layer"}}},
    "best_fit": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "fit", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_k": "featurizers.rot.k", "best_seed": "train.seed"}
      },
      "outputs": {"values": "values.json"}
    },
    "apply": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_apply.json",
               "set": {"featurizers.rot.file_path": "fit/rot.safetensors",
                       "featurizers.rot.k": {"artifact": "best_fit", "key": "best_k"},
                       "featurizers.rot.entry": {
                         "k": {"artifact": "best_fit", "key": "best_k"},
                         "seed": {"artifact": "best_fit", "key": "best_seed"}}}},
    "scan_heatmap": {
      "type": "script", "script": {"module": "causalab.io.plots.workflow_figures"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "plot": "heatmap", "x": "sites.target.layer", "y": "positions.tap"
      },
      "outputs": {"figure": "scan_iia.png"}
    }
  }
}
```

Derived schedule: `locate` → `best` → `fit` → `best_fit` → `apply`, with
`scan_heatmap` free to run as soon as `locate` finishes — parallelism nobody
authored. The `set` overrides on `fit` re-point the corpus document's artifact
refs at the in-run `best` step. `fit` sweeps k × seed, so its bundle holds nine
rotations: `best_fit` names the winning cell and `apply` selects that entry —
one fit applied, provably the one the numbers chose, with its ArtifactIdentity
checked exactly as in a standalone run.

Note the one wart the example makes visible: a `set` block authors *IM-spec*
references (`{"artifact": …}`), while the step's own `inputs` use the §3
grammar (`{"step": …}`). Both name the same referent. Aligning the two layers is
tracked as future work.

## 11. Open

- **`select` aggregation**: the shipped script hard-codes mean-over-examples.
  Median / count-weighted variants are script parameters, added when a pipeline
  needs them — no longer a spec-level enum.
- **Plot vocabulary**: the shipped `plot` script covers heatmap and lines.
  Anything else is now a user script rather than a spec change, which is the
  point of the step type.
- **Cross-workflow threading** stays by external path or artifact reference (the
  same answer the IM spec §8 gives for cross-document threading).
- **Two reference grammars** (§3): a `set` block is where the workflow grammar
  and the IM spec's meet. Options are to align the IM spec on the simpler form,
  or to let `set` accept the workflow grammar and translate.
- **Multi-tenant steps**: a step per backend (fit on Megatron, apply on serving)
  needs per-step backend pinning — deferred until a second backend exists; the
  field would be one optional `backend` per protocol step.
