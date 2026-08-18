# Intervention Protocol — materialization pass

> **Status: brainstorm, 2026-08-18.** Third document in the series, after
> `intervention_protocol_readme.md` (referenced `§n`) and
> `intervention_protocol_simplification.md` (referenced `sec. n`). Method: six
> experiments were written out twice under `protocol_examples/` — once with the
> readme's per-node `conditions`, once with an explicit `compositions:` table
> (the simplification doc's sec. 2 proposal, renamed) — plus a sweeping
> weekdays-8b pipeline as a multi-file hydra composition. The pairs differ
> *only* on the judged axis; all orthogonal simplifications are applied to
> both. This document is the fresh pass over the materialized YAML.
> **Revision note:** two results here were re-decided in
> `intervention_protocol_im.md`: the sec. 3 convergent form is set aside — the
> named-table variant wins on readability (user call), renamed
> `intervened_models` (the paper's own phrase, resolving sec. 4); and finding
> 3's hydra experiment-file sweep mechanism is superseded by in-document axes
> parsed by the backend (`protocol_examples/07/08_*_im.json`). The counts in
> sec. 1 and the remaining findings stand, except finding 6 (well-known
> implicit site names), reversed by the implicit-conventions audit
> (`intervention_protocol_im.md` sec. 6): every site is now declared.

## 1. The evidence

Non-comment, non-blank lines per variant:

| Experiment | `conditions` | `composition` | Δ |
|---|---|---|---|
| 01 activation harvesting | 15 | 15 | tie |
| 02 interchange | 15 | 18 | conditions −3 |
| 03 path patching | 23 | 26 | conditions −3 |
| 04 DAS | 27 | 30 | conditions −3 |
| 05 DBM | 28 | 31 | conditions −3 |
| 06 Hydra effect | 34 | 41 | conditions −7 |

**The conditions form wins or ties everywhere.** The composition table is never
shorter — not even on the Hydra effect, the many-condition-sets stress test it
was predicted to win. The reason is visible in the files: most condition sets
are *single-use* (each direct-effect injection conditions exactly one read), so
naming them adds a line per set plus two table headers, and buys nothing back.
The composition variant carries 6 top-level tables against 4 in every file.

Two things the composition variant does win, also visible in the files:

- **The forwards are legible.** `06_*_composition.yaml`'s `compositions:` block
  is the experiment's structure at a glance (1 ablation + 4 injections + 2
  implicit); in the conditions file the same fact is smeared across 16
  `conditions:` fields.
- **No `input:` noise.** The conditions file for the Hydra effect writes
  `input: base` fifteen times; path patching seven; the composition files,
  zero.

## 2. What the materialized files refute — on both sides

**Against the mandatory composition table** (simplification doc sec. 2, as
authored surface): refuted by the counts. A table that adds lines to all six
real experiments — including its own best case — is ceremony, not structure.
The user-side instinct ("conditions are fine") holds at the authoring surface.

**Against the conditions form as specified in the readme** — the examples
expose that two of its fields do no work:

1. **No edit ever carries `conditions`.** Across all six experiments, every
   node with `do:` has an empty condition set; conditions appear only on pure
   reads (the collect-under-intervention and the final logits). But edit-borne
   conditions are the *only* reason the transitive-closure canonicalization
   exists (§4.7: "conditions transitively closed and sorted"). The machinery is
   priced for a case no experiment used.
2. **Every edit's `input:` is redundant.** An edit executes only inside the
   forwards of reads that condition on it, and its input always matched theirs
   (it must — a base read cannot condition on a source edit). Fifteen
   `input: base` in one file, all derivable.

## 3. The convergent form

Deleting exactly the dead weight from the conditions form *reconstructs* the
composition semantics without the table. One node table; two node roles told
apart by their fields:

- **An edit has `do:` and nothing else contextual** — no `input`, no
  `conditions`. It is an inert effect definition (which is what the
  composition variant's `edits:` table already said).
- **A read has one context field, `in:`** — merging `input` + `conditions`:
  - `in: base` / `in: source[j]` — an un-intervened forward (default `base`);
  - `in: [edit, …]` — base plus these edits in force;
  - `in: {input: source[j], edits: [...]}` — the general form (rare: no
    example needed it);
  - `in: <name>` — a reference into an **optional** naming table for sets
    shared by several reads.

Interchange, convergent — 3 node lines, fewer tokens than either variant:

```yaml
interventionals:
  v_src:  {site: target,  pos: -1, in: source}
  patch:  {site: target,  pos: -1, do: {swap: v_src}}
  logits: {site: lm_head, pos: -1, in: [patch]}
```

Path patching, convergent — 9 node lines, no naming table needed:

```yaml
interventionals:
  v_sender:    {site: sender,   pos: -1, in: source}
  v_a10:       {site: a10,      pos: -1}                 # in: base default
  v_a11:       {site: a11,      pos: -1}
  swap_sender: {site: sender,   pos: -1, do: {swap: v_sender}}
  freeze_10:   {site: a10,      pos: -1, do: {swap: v_a10}}
  freeze_11:   {site: a11,      pos: -1, do: {swap: v_a11}}
  v_receiver:  {site: receiver, pos: -1, in: [swap_sender, freeze_10, freeze_11]}
  inject:      {site: receiver, pos: -1, do: {swap: v_receiver}}
  logits:      {site: lm_head,  pos: -1, in: [inject]}
```

Hydra effect, convergent: 16 node lines (as conditions) minus every
`input: base`; the three reads sharing the ablated context may either repeat
`in: [resample]` or name it once in the optional table.

What this buys, checked against both parents:

| Property | conditions (readme) | compositions (sec. 2) | convergent |
|---|---|---|---|
| shortest on all six examples | ✗ (input noise) | ✗ (table ceremony) | ✓ |
| transitive closure at load | required | — | **gone** (edits cannot nest context) |
| grouping/canonicalization | closure + sort + intern | authored table | content-intern `(input, sorted edit set)` — one dict |
| forwards visible in YAML | ✗ | ✓ | via optional naming + `explain` |
| §13 silent-second-forward footgun | present | impossible | present but lintable (two reads on one input whose edit sets differ by ⊂ — warn) |
| top-level tables | 4 | 6 | 4 (5 with optional naming) |
| operand/context namespaces checkable | ✗ (one namespace) | ✓ | ✓ per entry: operands must name `do:`-free entries; `in:`-lists must name `do:`-carrying entries |

Semantics is unchanged from the readme: pre-transform reads, visibility only
through `in:`, rule-3 classes per (site, pos-overlap, context), dead-edit and
dead-read checks (now symmetric). `num_forwards` ≥ number of distinct interned
`in:` values. The composition **concept** survives intact — an `in:` value *is*
a composition of edits — it is the mandatory *table* that dies. This supersedes
sec. 2 of the simplification doc.

## 4. Naming

The collision is no longer hypothetical: `weekdays_sweep/config.yaml` uses
"composition" in both senses in one header (hydra composes the YAMLs; the
protocol composes edits). In the convergent form the word almost vanishes from
the YAML surface — only the optional naming table needs a key — which makes
the collision cheap. Options for that table, in preference order:

1. `conditions: {ablated: [resample]}` — reuses the readme's word for exactly
   the readme's meaning, zero new vocabulary, and `in: ablated` reads well.
2. `compositions:` — the concept name; keep "a composition of edits" as the
   *prose* term regardless (it is the right description of what `in:` denotes).
3. `under:` instead of `in:` as the read field ("read logits under ablated") —
   equally good; pick one and never both.

## 5. Other findings from materializing

1. **`token_logit` is a missing metric kind.** The Hydra effect needs the logit
   of a per-example token column (`ml_token`); `dims` cannot express it (§3.4:
   dims are static) and §3.10 has no kind for it. One-line addition:
   `{kind: token_logit, of: <read>, token: <column>}`. Cross-read arithmetic
   (total effect = `te_clean − te_abl`) stayed **post hoc**, and that held the
   line well — it is off-model and cheap. Known cliff to record: the day a
   *training objective* needs a cross-read difference, the objective grammar
   needs a two-read term; until then, don't add it.
2. **The logit-lens recipe is O(k) forwards; the paper's method is O(1).**
   Measuring k direct effects via §3.9's inject-and-read costs 2k extra
   forwards (4 in the example files). The paper just multiplies collected
   contributions by the unembedding — off the forward path. That is B16
   (`ApplyModelFn`) resurfacing with a measured cost attached: a
   metric-side projection `{kind: unembed, of: <read>, token: <column>}` that
   the backend lowers to one (vocab-parallel) matmul would delete 4 of the 7
   forwards in `06_*`. It blurs the "metrics are pure arithmetic over node
   values" line (§3.10) because it touches a weight matrix — left **open**,
   but the Hydra effect is now the concrete argument for it.
3. **Sweep axes want a compose-time namespace.** `weekdays_sweep` routes every
   swept knob through `scan.*` (`scan.layer`, `scan.pos`), interpolated into
   the protocol — one stable override address per axis, strict parsing keeps
   the namespace out of the schema. The shipped runner states `layers: [28]`
   three times; the sweep tree states it zero times (stage 2 pulls it through
   `${artifact:...:best_layer}` — sec. 6.4 tier 1, now materialized in
   `RUN.md`).
   **And the axes themselves are YAML, not CLI.** A first draft of `RUN.md`
   passed the sweep ranges and artifact refs as multirun CLI arguments — which
   makes the command line out-of-band state and breaks "sharing YAMLs alone
   reproduces the experiment". Hydra already has the native fix (≥ 1.2):
   an *experiment file* (`exp_locate.yaml`, `exp_subspace.yaml`) that pins the
   config-group selection, the fixed values, `hydra.sweeper.params` for the
   axes, and `hydra.mode: MULTIRUN` — the launch command takes zero
   hyperparameters. CLI overrides stay possible for exploration, but an
   override that matters gets promoted into the file; the file is the record.
   Swept fields default to `???` in the experiment file so stripping the sweep
   block fails at compose time instead of running a silent default. The
   reproducibility chain is three YAML layers: experiment file → composed
   point protocol per cell → canonical stamped protocol + digest per run.
4. **Position indirection earned its keep under sweeps, not in standalones.**
   `scan.pos` sweeps a *name* into the task's `positions:` table; the
   standalone examples are fine with inline `-1`. Both sugar levels confirmed.
   Related: the swept files pin `logits` to `pos: last` while the tap position
   scans — per-node position independence is load-bearing; never factor `pos`
   up to the context/protocol level.
5. **Fields no example touched** — candidates to cut from the spec until
   demanded: `input:` on a composition (all six defaulted to base; the
   convergent general form `in: {input: …, edits: …}` covers the gap),
   edit-borne `conditions` (sec. 3 above), `dims` (zero uses across six
   protocols — featurizer shape did the selecting every time), `Pos`
   `relative_to`/`scope` (zero uses). None need deleting from the *design*;
   all should stay out of the tutorial surface.
6. **Singleton components are well-known site names.** A mechanical
   cross-reference check over all 19 files flagged every `site: lm_head` as
   undeclared — but the readme's own §6 example uses it undeclared too. Make it
   spec: components with no layer/head/expert axis (`lm_head`, `ln_final`,
   `embeddings`) are complete addresses and predeclared as site names, exactly
   as `base`/`source` are predeclared contexts. A declared `sites:` entry is
   only ever needed when an axis must be filled.
7. **The Hydra-effect grid confirms the generator tier.** 1 ablation × 2
   probes = 16 nodes; the paper's 32 × ~64 × 2 grid is unwriteable by hand and
   *should not* make the IR grow a loop — it is generated YAML plus a multirun
   axis over the ablation layer (simplification doc secs. 3.1, 6.4). The
   repetitive inject/read pattern in `06_*` is precisely what a 20-line
   generator emits.

## 6. Resulting changes to the earlier docs

- **Simplification doc sec. 2 is superseded** by sec. 3 here: keep its
  semantics, its differential test (unchanged — the convergent form still
  compiles to the same groups), and its `explain`-prints-the-forwards report;
  drop the mandatory `worlds`/`compositions`/`reads`/`edits` tables. The
  summary-table row 1 becomes: "edits lose `input`+`conditions`; reads gain
  `in:`; optional naming table; content-interned grouping".
- **Metric table** gains `token_logit`; `unembed` recorded as open (finding 2).
- **Schema after the pass** (simplification doc sec. 9) drops back to one node
  table: `version, model, high_level?, data, sites, positions?, featurizers?,
  params?, conditions?, interventionals, metrics?, outputs?, train?` — 13 keys,
  9 optional.
- The readme's §3.6/§3.7 stand, minus the closure rule (§4.7's "transitively
  closed" clause) once edits lose their `conditions` field.
