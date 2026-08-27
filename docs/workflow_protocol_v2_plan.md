# Workflow protocol v2 — plan: one step shape, two step types

**Status: proposal.** Decisions taken in this document are marked **[decided]**;
everything else is a sub-decision this plan deliberately leaves open for the
implementation PR to settle in code review.

## 1. Why

`docs/workflow_protocol.md` v1 defines four closed step types — `protocol`,
`transform`, `select`, `plot` — over a closed, versioned op registry
(`causalab/transform/`). That is **~2,300 LOC of source, ~1,130 LOC of tests
and a 450-line spec**, with `causalab/protocol/workflow.py` alone at 1,332
lines.

Two problems, both structural rather than cosmetic.

**Three spellings for one idea.** Cross-step wiring happens as (a) artifact
refs buried inside a protocol step's *inner document*, discovered by walking
it; (b) `from` + `table` on a select/plot step; (c) `inputs` slots on a
transform step. §3 of the v1 spec already admits this — "the third spelling of
the same idea". Each spelling carries its own load-time checks, its own
canonicalization branch and its own runner branch.

**The vocabulary cannot hold the work we actually have.** PR #41's
`pipelines/README.md` indexes **nine** "Execution: stub" gaps. Read against the
v1 step types, they fall into two groups, and neither fits:

| stub | why no v1 step type expresses it |
|---|---|
| probing | the closed metric vocabulary is over `lm_head` reads, so a probe's own loss is inexpressible |
| logit lens | no primitive; the post-processing is ordinary Python |
| ablation / steering | the *mechanism* exists; the corpus-mean harvest and the harvest-and-difference that produce the direction do not |
| attribution | needs new protocol vocabulary, not a document |
| **feature labels** | the `sae` featurizer exists; **nothing on the labeling side** — labeling is an LLM call, and the handoff format is unspecified |
| behavioral analysis | "just run these prompts" batch greedy decode |
| hypothesis generation | `develop_hypothesis` was a harness around library calls |
| save results | persistence and promotion unsettled |

A `transform` op cannot close any of them: the registry's admission criterion
is **determinism** ("an op must be a pure function of its declared inputs and
params"), and an LLM call is not. A `protocol` step cannot close them either —
they do not touch a network through the intervention vocabulary. So today the
work happens *outside the record*, which is the precise failure the workflow
document was introduced to fix.

**Design target [decided]:** the contract must accommodate both the stub gaps
*and* agent-driven phases. An LLM is just another script.

## 2. The shape

Two step types **[decided]**: `protocol` (unchanged) and `script`.

```json
"label_features": {
  "type": "script",
  "script": "scripts/label_features.py",
  "inputs": {
    "acts":  {"step": "harvest", "file": "acts.safetensors"},
    "layer": {"step": "best", "key": "best_layer"},
    "n_labels": 32
  },
  "outputs": {"labels": "labels.parquet"}
}
```

Inputs → one script → declared outputs. Dependencies stay **derived**, now from
a single mechanism (`inputs`) instead of three. `save`, the run-tree/external
artifact overlay, the sink rule and the acyclicity requirement are unchanged.

### Why `protocol` stays special [decided]

A protocol step is not "a script that happens to take a document". Keeping it
declarative preserves, all at load time and all of which a generic script step
would forfeit:

- inner-document validation (every §5 rule of the IM spec, before any step runs);
- sweep expansion, so point counts and per-point digests are known to `explain`;
- backend capability routing (`choose_backend` over the union of the points);
- `--points START:STOP` shard dispatch;
- the producer's **sweep axes**, which is what makes a downstream reduction
  meaningful at all (see §4).

The cost is that the vocabulary is 2 rather than 1. That is the right trade:
`protocol` is where all the load-time bite lives, and the bite is the reason
the layer exists.

### The input value grammar

A `script` step's `inputs` map names to values:

| form | resolves to |
|---|---|
| a JSON literal (`32`, `"greedy"`, `[1,2]`, a nested object) | itself |
| `{"step": S, "file": P}` | an absolute `Path` to `<run>/S/P` |
| `{"step": S, "file": P, "entry": {…}}` | one entry of a swept `.safetensors` bundle (IM spec §2.5) |
| `{"step": S, "key": K}` | a scalar read from step `S`'s declared `.json` values table |
| `{"artifact": A, "key": K}` | the external artifacts root (IM spec, unchanged) |
| `{"file": P}` | a file in the external artifacts root |
| `{"document": P}` | a path relative to the workflow file (a protocol document, a prompt template) |

**References are recognized only at the top level of an `inputs` entry.** A
nested object is always a literal — no deep ref-walking, so
`{"cfg": {"step": 3}}` is unambiguously a literal and the loader never has to
guess. This is the one place v2 is *stricter* than v1, which walked nested
documents looking for refs.

### The output declaration

```json
"outputs": {
  "labels": {"file": "labels.parquet",
             "columns": {"feature": "int64", "label": "string"}},
  "basis": "basis.safetensors"
}
```

The short form is a bare path. `columns` is optional; declaring it is a promise
about the shape of what the step publishes, verified **on write** (§4). Because
the digest covers inputs and outputs, a declared column set is part of the
record.

## 3. Execution contract

**In-process by default, subprocess opt-in [decided].**

```python
def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None: ...
```

The script must create every declared output file. The runner then:

1. verifies each declared output exists (a missing one is a step failure, named
   by slot);
2. verifies declared `columns` against the parquet it actually wrote;
3. **stamps `ArtifactIdentity`** on `.safetensors` outputs — inherited from the
   step's tensor inputs, exactly as `_run_transform_step` does today, applied
   post-hoc by rewriting the file's metadata.

This departs from the current op contract (`(inputs, params) -> {slot: value}`,
runner owns paths and formats). The reason: a plot step, an LLM step and a
report step all want to write their own file, and paying for two contracts to
keep op unit tests filesystem-free is not worth it — `tmp_path` is one fixture.
Identity stamping stays centralized, which was the contract's real value.

Helpers move to `causalab/steps/io.py` (from `causalab/transform/io.py`): read
a parquet table, read a tensor bundle with `slot`/`entry` addressing, write
either. A script imports them; nothing forces it to.

### Isolation

An optional `runtime` block, for steps whose environment differs from the
runner's:

```json
"runtime": {"isolate": true, "deps": ["anthropic"],
            "env": ["ANTHROPIC_API_KEY"]}
```

Isolated steps run as `uv run --with <deps> python -m causalab.steps._shim`,
handed a resolved `inputs.json` and an `outputs.json`. `env` lists variable
**names** to pass through; values never appear in a document, a canonical form
or a manifest.

`runtime` **is** part of the canonical form and the digest — which interpreter
and which dependency set a step ran under changes what the step *is*. This is
distinct from the §8 "execution parameters never enter documents" rule, which
is about `--device` / `--dtype` / `--points`: those describe the same
computation on different hardware, whereas a different dep set is a different
computation. Worth confirming in review.

### Script resolution, and the torch-free guarantee

`script` is either a path relative to the workflow file (contained, no parent
escapes) or a shipped built-in named `causalab:<name>`.

**`validate` and `digest` must not import a user script** — that would import
torch (or `anthropic`, or anything) and break the guarantee pinned by
`tests/transform/test_load_is_torch_free.py`. So load-time checking of a script
is deliberately shallow: the file exists, `ast.parse` succeeds, and a
module-level `def main` is present. Everything else about the script is a
run-time contract. This is a real reduction in load-time bite, stated plainly
rather than papered over.

## 4. What moves from load time to run time

This is the substantive cost of the change, and the plan's main honesty
obligation.

| v1 checked at load | v2 |
|---|---|
| rule 7: `select`/`plot` column refs against the producer's sweep axes | the built-in `select`/`plot` scripts check their `group_by` against the producer's recorded axes, at their own step |
| rule 7 over a transform producer, against the op's declared columns | the op registry is gone; a declared `columns` block is verified on write |
| rule 1's op-record half (slots, param schema) | script signature is a run-time contract |
| rule 10's "an artifact ref naming a step must name a `select` step" | **strengthened**: `{"step": S, "key": K}` must name a step whose declared outputs include a `.json` — checkable for *any* step, because outputs are now declared |
| plot axis coverage (every axis of the producer covered) | the `plot` script's own refusal |

**The axes still exist.** Because `protocol` stays declarative, the runner
still computes `run_axes` at load. v2 has the runner write a per-step sidecar
(`<run>/<step>/_step.json`: declared files, and for a protocol step its sweep
axes and point digests), and the built-in `select`/`plot` scripts read it. So
group-by-coordinates-then-mean survives as *behaviour*; it stops being derived
magic in the document model and becomes a documented thing a script does with
data the runner published. A user script gets the same sidecar and can do the
same.

## 5. Reproducibility [decided]

**The digest covers inputs and outputs only.** A script step's canonical entry
is `{type, script, inputs, outputs, runtime, after}` — the script *path* names
which step you ran; its *contents* are not hashed into the identity.

Consequences, stated rather than glossed:

- v1 §0 ("a campaign is a value") and §4 ("a workflow run is a pure function of
  the document") are **no longer true** and must be rewritten. Two runs of the
  same document with an edited script digest identically.
- Determinism stops being an admission criterion, because there is no longer an
  admissions process. An LLM step is legal.

**Mitigation, and where reproducibility actually lives now:
`workflow.json`.** The manifest — which is stamped at execution and was always
the run-time record — gains per step: the resolved input values, the script
path **and its sha256 content hash**, the `runtime` block, and for
non-deterministic steps a pointer to the raw response artifact. So content
hashing still happens; it is *provenance in the manifest* rather than *identity
in the digest*. A reviewer asking "what produced this number" gets a complete
answer; a reviewer asking "do these two documents describe the same run" gets a
weaker answer than v1 gave. That is the trade taken.

## 6. Validation checklist (v2)

Replaces the v1 §5 list. Eleven rules, and every one of them is either
unchanged or *cheaper* than its v1 counterpart.

1. Strict keys; closed `type` ∈ {`protocol`, `script`}; derived fields not authored.
2. Section order per §1; `save` last, non-empty.
3. Step names unique and filesystem-safe; `save` file_paths unique, contained,
   colliding with neither a step directory nor `workflow.json`.
4. Every reference resolves: `step` names a declared step; `file` names a file
   that step really declares (a protocol step's inner `save` manifest, a script
   step's `outputs`); `key` names a step declaring a `.json` output; `after`
   names declared steps; a save entry's `step`/`value` name an actual output.
5. The derived graph (`inputs` + inner-document artifact refs + `after`) is acyclic.
6. Sink rule: every step is consumed by a later step or by `save`.
7. `script` resolves — a contained relative path or a `causalab:` built-in —
   parses, and declares `main`. Never imported.
8. Output paths are contained inside the step dir; a `columns` declaration
   requires `.parquet`; a `.safetensors` slot declares no columns.
9. A protocol step's `set` paths exist in the target document, and the document
   loads as a valid intervention protocol.
10. A tensor `entry` selection is checkable at load when the producer is a
    `protocol` step (its expansion is deterministic at load); against a script
    producer it is a run-time check.
11. An isolated step declares its `deps`; `env` lists names only.

## 7. Work plan

| phase | scope |
|---|---|
| 1 | **Spec v2** — rewrite `docs/workflow_protocol.md`: the two step types, the input grammar, the output declaration, the execution contract, the §6 checklist, the §5 reproducibility rewrite, and an explicit load-time → run-time ledger (§4 above) |
| 2 | **Document model** — `protocol/workflow.py`: two parse branches; graph from `inputs`; delete `check_columns`, `compute_representatives`, `_transform_table_columns`, `_check_transform_inputs`, the op-record re-raise path. Keep `_bundle_entries`/`_check_entry_selection` (rule 10) and path containment |
| 3 | **Runner** — one generic `_run_step`: resolve inputs, invoke (in-process or isolated), verify outputs, verify columns, stamp identity. Write the `_step.json` sidecar. `_run_protocol_step` survives nearly as-is |
| 4 | **Built-ins** — `causalab/steps/`: `select`, `plot` (heatmap + lines) as scripts reading the sidecar; the three transform ops (`fit_pca`, `head_stats`, `paired_ttest`) as plain scripts; `io.py` moved from `transform/`. Delete `causalab/transform/{schema,registry}.py` and `ops/`. Rewrite `configs/workflows/weekdays_8b.json` |
| 5 | **Tests** — rewrite `tests/protocol/test_workflow.py` (one test per v2 rule, asserted by rule number, as today); port `tests/transform/` (500 LOC: 3 op oracles survive as script tests, `test_registry.py` goes, `test_load_is_torch_free.py` is *retargeted* at the new rule 7 — validating a workflow whose script imports torch must still not import it); keep the end-to-end capstone in `tests/neural/pytorch_hooks/test_workflow_run.py` as the anchor — locate → select → DAS fit → apply → plots → publish must pass byte-identically. Re-pin `workflow_digests.json` |
| 6 | **Proof** — one LLM-generation step closing a real PR #41 stub. Feature labels is the right one: `sae` featurizer exists, the labeling side does not, and it exercises isolation (`deps: ["anthropic"]`, `env: ["ANTHROPIC_API_KEY"]`) plus the non-determinism path in the manifest |
| 7 | **Docs** — `docs/CODEBASE.md` §1/§4 (the `transform/` row goes, `steps/` arrives), `pipelines/README.md`'s stub table (which stubs this closes and which it does not) |

### Expected size

| file | v1 | v2 (est.) |
|---|---|---|
| `protocol/workflow.py` | 1,332 | ~800 |
| `workflow/runner.py` | 445 | ~380 |
| `transform/{schema,registry}.py` | 430 | 0 |
| `transform/ops/` (records) | 261 | ~200 as scripts |
| `docs/workflow_protocol.md` | 450 | ~300 |
| `tests/protocol/test_workflow.py` | 744 | ~550 |

Net **≈ −1,000 LOC of source**, step vocabulary 4 → 2, wiring mechanisms 3 → 1.

## 8. Landing

Recommended: **one PR stacked on `can/protocol-refactor`**, the same base #39
and #42/#44 use. #39 (`transform`) is already merged into that branch, so this
PR partly reverts it — worth saying so in the description rather than letting a
reviewer discover it. #42 and #44 touch the protocol layer, not the workflow
layer, so no conflict is expected.

PR #20's own description documents the v1 workflow layer at length and will go
stale the moment this lands. Either amend it with a forward pointer, or accept
the staleness explicitly.

### Open sub-decisions for review

1. Is `runtime` in the digest (§3) the right call, given §8's "execution
   parameters never enter documents"?
2. Do the ported ops keep their `@version` suffix as a *filename* convention
   (`fit_pca_v1.py`) now that nothing enforces it, or drop it?
3. Should a non-deterministic step be marked as such in the document (so
   `explain` can report "this workflow is not replayable"), or is the manifest's
   record enough?
4. Does `--resume` still mean anything? v1 skips a step whose outputs exist with
   matching stamped digests; with implementations out of the digest, a resumed
   step can skip after its script changed.
