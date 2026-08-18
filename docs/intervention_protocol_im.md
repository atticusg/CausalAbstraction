# Intervention Protocol — the intervened-model pass

> **Status: brainstorm, 2026-08-18.** Fourth document in the series:
> `intervention_protocol_readme.md` (`§n`) → `intervention_protocol_simplification.md`
> (`simp. n`) → `intervention_protocol_materialization.md` (`mat. n`) → this.
> Materialized as the `*_im.json` files in `protocol_examples/` — a third
> implementation beside the `conditions` and `composition` YAML pairs.
>
> Decisions taken this pass (user calls, recorded as settled):
> 1. **The composition variant wins on readability**, overriding mat. 1's line
>    counts; mat. 3's convergent form is set aside. The reads/edits split stays.
> 2. **The table is renamed `intervened_models`** — the paper's own phrase for
>    ℒ_{b∪𝕀}, which finally closes the naming question (mat. 4): unlike
>    "world" and "composition", *intervened model* is causal-abstraction
>    vocabulary, and it does not collide with hydra's "composition".
>    Glossary row: `intervened_models.<name>` ↔ ℒ_{b∪𝕀} — theirs the word
>    *and* the object.
> 3. **Section order**: `reads` → `edits` → `intervened_models` (IMs below
>    edits). Rule: **every declared edit belongs to at least one
>    intervened_model** — a load-time error, not a warning (plus the reverse
>    checks: IM edit lists name declared edits; `in:` names an IM or an input
>    role; operands name reads).
> 4. **The format is free JSON with a schema-aware parser** that owns more of
>    the execution logic; hydra conventions are no longer followed.
> 5. **The output contract** (sec. 5): mandatory non-empty `save` as the last
>    section; only reads and metrics saveable; every metric must be saved;
>    `input` mandatory on every intervened_model; `high_level` renamed
>    `causal_model`, `neural_model` accepted as an alias of `model`.
> 6. **Explicit bindings and artifacts** (secs. 3–5, second revision): save
>    entries are objects `{value, model, input, file_path}` with the binding
>    cross-checked against the metric→read→IM chain; `original` names the
>    un-intervened model; one global namespace across all declared names; the
>    `{"sweep": ...}` wrapper is mandatory for every axis (list-on-scalar
>    retired); featurizers take an optional `file_path` to load a fitted
>    artifact instead of computing (`09_das_apply_im.json`).
> 7. **Trained featurizers must be saved** (sec. 5, third revision): entry
>    shape `{value, site, file_path}` — site restated and cross-checked
>    against usage; identity is stamped, never authored. `save` is now the
>    single complete manifest of everything that leaves a run.
> 8. **Reads bind explicitly** (sec. 4, third revision): `in:` is retired for
>    `model` + `input` on every read — `model` ∈ `original` | declared IMs,
>    `input` ∈ `base` | `source[j]`, cross-checked against the IM's declared
>    input. Reads, save entries, and IMs now all state their bindings in the
>    same explicit vocabulary.
> 9. **The implicit-conventions audit** (sec. 6): int position sugar kept and
>    documented; all sites declared (well-known implicit sites retired);
>    `train.params` names featurizers; dataset refs are a local path or HF
>    key with no digest (the resolved content digest is stamped at load);
>    singular `source` + `source[j]` ratified; column refs stay data-checked;
>    `neural_model` canonicalizes to `model`; IM edit lists are unordered
>    (canonical form sorts); the canonical-stamp principle is standing
>    policy.

## 1. The pivot: minimal config, smart parser

The config's job contracts to *the semantic description*; everything the parser
can derive or decide moves out of the file:

| Moves out of the config | Into |
|---|---|
| feature dims (`shape: [4096, k]` → author `k` only; `gate` authors nothing) | parser, from (model, site) metadata |
| hydra interpolation (`${facts.hidden}`, `${scan.pos}`) | unnecessary — derivation replaces it |
| config groups / defaults lists / `@package` headers | references: `model.key`, `dataset@digest`, `causal_model.key` resolve at load; one experiment = one self-contained file |
| the `${artifact:}` resolver syntax | an ordinary value form: `{"artifact": "<ref>", "key": "best_layer"}` |
| sweep expansion, forward fusion, fit parallelization | the parser/planner (sec. 2) |
| CLI overrides | optional `--set path=value`; exploration only, never the record |

The description-vs-program seam (simp. 0) is untouched: the file still never
says *how* — no "parallel", no forward counts, no device anything. The parser
derives the plan and `explain` reports it. What softens is "canonicalize
without touching anything" (simp. 10): deriving `d` needs the model's static
config (metadata, not weights). Accepted — the stamped canonical point
protocols are still fully concrete.

Costs owned: strict JSON has no comments (a `description` field per file
carries intent); no config-group reuse (the model/data header is ~5 lines per
file, and references were always the real sharing mechanism); hydra becomes an
*optional front-end* for teams that want it — since JSON is a YAML subset and
the object model is the format, a hydra pipeline can still compose and emit
these documents. `weekdays_sweep/` is kept as that world's artifact;
`07/08_*_im.json` are its replacement here.

## 2. Sweeps: the compute-sharing argument

The forcing example: DAS with k ∈ {8, 16, 32}. Under multirun (hydra or
otherwise) that is three processes, each loading the model and harvesting the
same activations — the sharing structure is invisible to any scheduler that
expands *before* the run, because it exists *inside* the value.

Declare the axes inside the document instead, and parallelization needs **no
new machinery**: the compiler already dedups shared sub-values by content (the
`(input, edits)` grouping that derives forwards). Expansion happens at load, so
the interning sees all points at once: the three rotations' source reads are
*the same read* → one harvest; the fits are independent params → parallel
optimizers over one activation stream. `08_weekdays_das_sweep_im.json` is 9
fits (k × seed) from one harvest; `07_weekdays_locate_scan_im.json` is a
64-cell scan whose source side collapses to one forward per row with 64 taps —
against 64 processes, 128 forward groups, and 64 model loads under multirun.

This **revises simp. 3.1**, and the reversal should be owned plainly: what that
section deleted was an execution construct (`sweep: [{for: …}]`, `ForEach` —
loops in the IR). What returns here is *plural values*: a scalar field carrying
a list denotes a set of points. "One protocol per run" becomes "one document
per run; a document denotes a set of point protocols." The provenance story
survives intact — expansion is deterministic, every expanded point materializes
as a concrete point protocol with its own digest (still the provenance unit),
and the document digest identifies the campaign.

The tier structure (simp. 6.4) collapses *into the parser*: the author declares
axes uniformly; the parser discovers what shares compute. Sweeping `model.key`
is legal — the parser finds zero sharing and schedules sequentially. The author
never chooses between "in-run sweep" and "multirun"; that distinction is now an
execution detail.

## 3. Declaration rules — the result of the discussion

1. **Every axis is an explicit `{"sweep": [...]}` wrapper — scalars included.**
   `"k": {"sweep": [8, 16, 32]}`, `"seed": {"sweep": [0, 1, 2]}`. *(Second
   revision: the first draft of this rule made a bare array on a scalar-typed
   field an axis, leaning on the parser's schema-awareness to disambiguate.
   Retired by decision: with the wrapper mandatory there is no ambiguity class
   at all — `anneal`'s `[start, end, frac]` needs no type lookup to read —
   axes are greppable by one keyword, and the parser no longer needs field
   types to find them. Cost: ~10 characters per axis.)*
2. **`{"sweep": {"range": [start, stop]}}`** (optional `"step"`) — the one
   convenience for long integer axes (`07_*`: 32 layers).
3. **List-typed fields sweep the same way** — `{"sweep": [[0,1], [0,1,2,3]]}`
   for a dims selection; rare by design, and now identical in form to every
   other axis.
4. **Axis identity = name identity.** An axis is declared on a *named table
   entry* (`sites.target.layer`, `positions.tap`, `featurizers.rot.k`), and
   every read/edit/metric that references the name moves together. This is the
   answer to the aliasing trap: writing the same list on two fields would be
   two axes (a wrong 4-way cross for the position scan); referencing one
   swept name is one axis, by construction. The name tables earn their keep a
   third time (after hydra overrides, simp. 5.4, and sweep addresses, mat. 5.3)
   — now as shared-axis identity. Inline scalars (`pos: -1`) stay for anything
   un-swept.
5. **Axes cross-multiply; nothing else.** Coordinates suffix derived names
   (`rot[k=8]`, `logits[target.layer=5]`) and key the result table. No zip, no
   conditionals, no expressions. A *dependent* axis — the Hydra effect's
   "probe layer > ablation layer" triangle — is out of scope: that tail stays a
   generator's job (simp. 3.1's principle), or runs rectangular and wastes the
   corner. The parser's `explain` prints the expanded point count, which is
   also the guard against accidental combinatorial explosion.
6. **Axes propagate through the reference graph.** Sweeping `target.layer`
   fans out every entity that transitively references `target`; entities off
   the axis stay singletons shared by all points — which *is* the
   parallelization. Contrast hydra: it needs dotted-path overrides because it
   edits a config tree it does not understand; the parser owns the graph, so
   the axis is declared where the value lives and nothing else is touched.

Worth noticing what rule 6 buys the Hydra effect: mat. 7 judged the paper's
full grid "unwriteable by hand". With axes, one probe site
`{"component": "attention_output", "layer": {"range": [14, 32]}}` fans the
entire read/inject/read pattern across 18 layers in a constant-size document —
the rectangular part of the grid became writable; only the triangular
dependency stays with generators.

## 4. The IM schema

Section order (presentational, enforced by the validator):
`version, description, model, causal_model?, data, positions?, sites,
featurizers?, reads, edits?, intervened_models?, metrics?, train?, save`.

- **Names follow the theory pair ℒ/ℋ**: `model` stays, with `neural_model`
  accepted as an alias (ℒ); `high_level` is renamed `causal_model` (ℋ) —
  matching the readme's own `CausalModelConfig` type (§3.3).
- `reads`: `{site, pos, model, input, featurizer?, dims?}` — *(third
  revision; the earlier single `in:` field conflated "which intervened model"
  with "which input row" by letting input roles double as model names.)*
  `model` is `original` or a declared intervened_model; `input` is `base` /
  `source[j]`. For a read in an intervened model, `input` is redundant with
  the IM's own declaration and is **cross-checked** — mismatch is a load
  error, same pattern as save entries: the file states every binding, the
  loader proves it. Reads never carry `do`.
- `edits`: `{site, pos, featurizer?, dims?, do}` — inert definitions; no
  `in`, no `input`, no `conditions` (edit-borne conditions stay dead, per
  mat. 2 — the closure never returns).
- `intervened_models`: `{input: base|source[j], edits: [...]}` — **below
  edits**, and **`input` is mandatory**. It used to default to `base`, which
  left the question "which input does this IM run on?" answerable only by
  knowing the default; IMs are few per document, so explicitness costs one
  token each and removes the last implicit binding in the file.
- **The membership rule**: every declared edit appears in ≥ 1
  intervened_model, every IM edit resolves, `in:` targets exist, operands name
  reads — all load-time errors (mechanically checked over all 8 files).
- **Cross-IM data flow has exactly one channel**: a read in IM_A may be the
  operand of an edit that IM_B puts in force — path patching *is* this
  (`v_receiver` read in `patched` → operand of `inject` → in force in
  `final`). There is deliberately no direct IM→IM wiring and no IM
  inheritance (`extends:`): none of the six experiments needed it, and it
  would blur the operand/context independence (§3.7) that makes path patching
  expressible. The IM graph (IM → edits → operand reads → IMs) must be
  acyclic; it is the schedule skeleton.
- **One global namespace.** All declared names — sites, positions,
  featurizers, reads, edits, intervened_models, metrics — share one namespace:
  a collision anywhere is a load error, as is using a reserved name (`base`,
  `source`, `source[j]`, `original`). Rationale:
  save entries, `explain` output, and result tables mix these names freely, so
  `de14c` (an IM) one underscore away from `de14_c` (a metric) is a hazard the
  schema should forbid rather than the reader disambiguate — enforcing the
  rule immediately drove 06's IMs to the clearer `with_inj14_clean` style.
- Position specs compact to their tag: `{"index": -1}`, `{"variable": "x"}`,
  `{"span": [a, b]}`.
- Featurizers author choices, not facts: `k`, `parametrization`; widths derive.
- **Featurizers take an optional `file_path`** that loads a fitted artifact
  instead of computing it — the fit→apply split as one field
  (`09_das_apply_im.json`). The artifact's `ArtifactIdentity` (model, site,
  k, parametrization, dtype — §8) is checked on load; a mismatch refuses. A
  featurizer with `file_path` may not appear in `train.params` (loading and
  fitting the same artifact is a contradiction; warm-starting a fit from a
  loaded init would be a distinct, explicit `init` form if ever wanted).

## 5. The output contract

The binding question — *which intervened or un-intervened model is a metric
applied to?* — is answered structurally by a chain: **a metric binds to
exactly one read (`of:`), and a read declares its model and input directly
(`model` / `input`, decision 8)**. Wanting
the same metric in two models means declaring two reads and two metrics —
`06_hydra_effect_im.json` does exactly this (`te_clean` on `logits_clean` in
the original model; `te_abl` on `logits_abl` in `ablated`).

*(Second revision.)* The chain alone proved too implicit at the boundary that
matters most — reading a `save` list, you could not see that path patching's
`logit_diff` belongs to the `final` model and not to `patched` without walking
metric → read → IM by hand. So the binding is now **restated at the save
site and cross-checked**: each save entry is an object

```json
{"value": "logit_diff", "model": "final", "input": "base",
 "file_path": "logit_diff.parquet"}
```

- `value` — the declared name of a read or metric (the only saveable kinds).
- `model` — `original` (the reserved name for the un-intervened model) or a
  declared intervened_model. Deliberately redundant with the chain: the
  loader resolves the chain and **errors on any mismatch**, so the field is
  drift-protected documentation, never a second source of truth.
- `input` — the input row (`base` / `source[j]`); redundant the same way,
  explicit for the same reason.
- `file_path` — where the value lands, relative to the run's output
  directory (never absolute; the config stays shareable). Tensors to
  `.safetensors`, per-example metric tables to `.parquet`. In a swept
  document the path is unchanged and the axis coordinates become columns
  (tables) or keyed entries (tensor files) under it.

What crosses the run boundary is governed by `save` — mandatory, non-empty,
and the **last section** of every config type:

1. **Three saveable kinds: reads, metrics, and trained featurizers.** Edits
   and intervened_models are not values; sites and positions are addresses.
   A featurizer entry has its own shape *(third revision — trained
   featurizers were initially exempted as an implicit training artifact;
   mandating them makes `save` the complete output manifest, no implicit
   channel left)*:

   ```json
   {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
   ```

   `value` names the featurizer — the saved unit is the whole bundle (all
   auto-declared slots) with its `ArtifactIdentity` stamped into the
   safetensors header. `site` is the featurizer's analogue of the read
   entry's `model`/`input`: the binding worth restating (reads disambiguate
   by intervened model; featurizers disambiguate by site), cross-checked
   against the sites its reads/edits actually use — mismatch is a load
   error. Deliberately absent: `model` and `input`/`trained_on` (document-
   global — one neural model, one train data spec per document),
   `k`/`parametrization`/`dtype` (already authored on the declaration), and
   the produced-by digest — all derived and stamped into the identity, which
   travels with the bytes, never with the entry.
2. **Every declared metric must appear in `save`** — a load error otherwise.
   No carve-out for metrics that serve `train.objective` / `eval` /
   `early_stop`: one rule, zero exceptions, and the loss trajectory is
   therefore always part of the results.
3. **Unsaved reads are legitimate intermediates** — operands and metric
   inputs (`v_src`, `v_receiver`, the Hydra-effect harvests) that never leave
   the device. `save` is what drives materialization: the backend keeps
   exactly `save` ∪ operand values, which is where `logits_to_keep` and the
   residency wins (B25) derive from. Saving full `logits` is a choice you
   make by writing it, not a default you forget to turn off.
4. **Everything declared must reach a sink** — the uniform load-error rule
   the membership rule started: an edit in no intervened_model, a metric not
   in `save`, a read that is neither saved nor a metric input nor an operand
   — all dead, all errors.
5. **Every trained featurizer must be saved** — the mirror of rule 2: a
   featurizer whose params appear in `train.params` without a save entry is a
   load error (training weights you then discard is almost always a mistake,
   and when intended should look intentional — it isn't expressible, by
   design). The converse holds too: an *untrained* featurizer in `save` is an
   error (identity has nothing to save; re-saving a `file_path`-loaded one is
   a pointless copy). `train.checkpoint` remains transient training state
   (resume); the save entry is the final artifact. This retires the earlier
   "two contracts, one home each" split: `save` is now the single, complete
   manifest of everything that leaves a run.

This supersedes the earlier `outputs`-with-defaults design (simp. 3.6): the
default was quiet convenience, `save` is an explicit contract — consistent
with mandatory IM `input` and the general no-silent-defaults trajectory of
this pass.

## 6. Convention decisions — the implicit-conventions audit

The examples were audited for every remaining implicit convention; each was
decided explicitly (fourth revision):

1. **Integer position sugar stays.** `"pos": -1` means `{"index": -1}`; a
   negative index counts from the end of the sequence; a non-negative index is
   rebased past any chat prefix (§3.4.1). Kept because it is intuitive —
   these three sentences are its complete documentation, and they live here
   and in the schema reference, not in folklore.
2. **`sites` is the complete tap inventory.** Every site a read or edit
   touches must be declared — `"lm_head": {"component": "lm_head"}` included.
   The well-known implicit site names are retired (this reverses mat. finding
   6); a site string that resolves to no declared entry is a load error.
3. **`train.params` names featurizers.** `["rot"]` trains every slot of
   `rot`; the dotted slot form (`"rot.weight"`) remains legal for the rare
   slot-level selection. Objective regularizers follow the same rule
   (`{"l1": "gate"}` = all of gate's params). `anneal` keeps dotted paths —
   its targets are slot hyperparameters, not param references.
4. **Dataset refs are a local path or an HF key — no digest.**
   `"weekdays/train"`, not `"weekdays/train@9ab2"`. Reproducibility does not
   regress: at load the parser resolves the ref and stamps the content digest
   into the canonical point protocol (decision 9 below), so the *record*
   stays pinned while the *authored file* stays a plain name. The cost owned:
   two runs of the same authored file at different times may see different
   data — the stamped protocols will say so.
5. **Singular `source` + brackets, ratified.** `data.source` holds the one
   counterfactual column in the k=1 case and reads say `input: "source"`;
   when `source` is an array, references index it as `source[j]`. No
   `sources` key.
6. **Column references stay data-checked.** `a: cf_answer`, `token: ml_token`
   resolve against the table at run time; a typo is caught by
   `validate --data`, not at load. Accepted as-is — a compose-time `columns:`
   declaration can be added later without breaking any file.
7. **`neural_model` is accepted at load and canonicalizes to `model`.** One
   field in the stamped form, two spellings at the authoring surface.
8. **IM edit lists are unordered.** Order in the file carries no meaning —
   per-address application order comes from the mechanism classes
   (simp. 3.10); the canonical form sorts the list.
9. **The canonical-stamp principle (standing policy).** The authored file may
   be minimal: absent `schedule` means constant LR, optimizer defaults apply,
   save dtypes follow the model, sweep coordinates get derived names
   (`rot[k=8]`). The stamped canonical point protocol materializes *every*
   default and every resolved reference — the authored document is for
   humans, the stamped one is the record, and nothing implicit survives into
   provenance.

## 7. Files

`protocol_examples/*_im.json`: 01 harvest · 02 interchange · 03 path patching
· 04 DAS (point) · 05 DBM · 06 Hydra effect — faithful translations of the
YAML pairs — plus the sweep demonstrations that replace the 10-file
`weekdays_sweep/` hydra tree with two self-contained documents:
07 locate scan (layer × position axes) · 08 DAS fit sweep (k × seed axes,
located cell via artifact-valued fields) · 09 DAS apply (featurizer
`file_path`, no train — the fit→apply split).

## 8. Open

- **Cross-document threading** stays by artifact reference (07 → 08); whether
  a "campaign" file listing several documents is ever needed, or whether tier-3
  programs/agents cover it (simp. 6.4), is deferred until it hurts.
- **Result schema for swept documents**: coordinates as columns is the sketch;
  the exact artifact layout (one bundle per point vs. one table) is a backend
  decision to prototype.
- **Axis caps**: `explain` prints the point count; whether to hard-cap
  cross-product size at load (refuse > N without an explicit flag) — lean yes,
  in keeping with the refusal culture.
- **Zip axes**: excluded now; revisit only with a concrete experiment that
  needs paired values on two names.
