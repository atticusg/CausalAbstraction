# Intervention Protocol — simplification pass

> **Status: brainstorm, 2026-08-18.** Companion to `intervention_protocol_readme.md`
> (same tree, neither implemented). References written `§n` point into that readme;
> internal cross-references are written `sec. n`.
> **Revision note:** sec. 2's proposal was tested by materializing six
> experiments both ways (`protocol_examples/`); the fresh pass
> (`intervention_protocol_materialization.md`) first replaced it with a
> convergent no-table form, and the final decision
> (`intervention_protocol_im.md`) restored the named table for readability
> under the paper's own term, `intervened_models`, with reads/edits split kept
> and edit-borne conditions dropped. Sec. 3.1's sweep-stays-out-of-the-value
> position is also revised there (in-document axes, parser-owned execution).
> The rest of this document stands. The abstractions there are taken
> as settled; this document attacks the *structure* — the YAML shape, the
> canonicalization machinery, the implementation details, and (sec. 6) the layer
> above the protocol where `causalab/analyses` chains methods today. Target:
> modular, composable, hydra-native, robust, and as simple as possible but no
> simpler.

## 0. The intent, restated

Everything in the readme is one move applied repeatedly: **make the experiment a
value, not a program.**

1. **Values can do what programs cannot.** A program (an nnsight trace, a `Plan`)
   cannot answer *which sites · needs gradient? · how many forwards · which model*
   without running (§1). A value can be hashed (provenance: config→run→artifact is
   one closed loop), diffed, composed (hydra), shipped (NDIF's whitelist, sbatch
   across the Mac↔cluster boundary), retargeted (Megatron has no HF modules to bind
   to), and statically checked before GPU time is spent.
2. **One closed vocabulary, pinned to the theory.** Each primitive is a causal-
   abstraction object (§11), so the expressive ceiling is a theorem, not an accident
   of what the first backend happened to support. The closed `do:` algebra is the
   lowerable subset; `PyTorchFn` is the escape hatch that must refuse loudly.
3. **Derive everything derivable** (`requires`, `num_forwards`, `digest`, §5) —
   anything authored twice can disagree, and a disagreement here is a wrong number,
   not an error.
4. **Put the backend seam above execution.** The protocol never contains anything
   only one backend could interpret: no tensors, no closures, no module references,
   no pre-resolved position ints. Resolution (`SiteResolver`, `PositionFrame`,
   forward grouping) is a backend service.
5. **YAML-only authoring makes hydra the composition algebra.** Methods become
   templates, experiments become config composition, and the merged document is the
   hashed unit — which deletes the entire Python-builder layer (§9).

The test this document applies: **complexity is justified only where it buys one of
those five.** Everywhere else it should go.

## 1. Where the remaining complexity lives

Reading §3–§13 with that test, the residual complexity clusters in two places:

- **Implicitness.** The forward-pass structure — the single most expensive derived
  fact — is *implicit* in per-node `conditions` sets. That implicitness is what
  forces the transitive-closure canonicalization (§3.6), the interning mitigation,
  the "crux" section (§3.7), rule 3's algebraic commutativity condition (§4.3,
  admittedly unimplemented), rule 5's dead-node warning, and the §13 footgun where
  omitting one name silently buys a second forward. Five mechanisms to manage one
  hidden variable.
- **Duplication.** `Param.shape` restates (and in the DAS example *contradicts the
  intuition of*) `Featurizer.shape` — `rot18` is authored `[4096, 8]` but
  `rot18.weight` is `[4096, 4096]` because cayley's skew matrix leaks into the
  authored surface (§7). `Param.trainable` duplicates `train.params`. `seeds`,
  `train.seed`, and `NoiseSpec.seed` are three seed homes. `objective` sits beside
  `train` but is meaningless without it. `outputs` restates "the metrics". The DAS
  example writes `dims: [0..7]` twice on a featurizer whose feature space *is*
  those 8 dims.

The same two clusters reappear one layer up, in how `causalab/analyses` chains
methods (execution order implicit in a defaults list; `layers: [28]` stated three
times per runner config) — sec. 6 treats that layer. Sec. 2–4 remove both clusters
from the protocol itself; sec. 5 adds the hydra conventions that keep the smaller
schema composable; sec. 7 the robustness additions; sec. 8 the floor — what must
not be simplified; sec. 9 the schema and both worked examples after the pass.

## 2. The one structural change: make worlds explicit

### 2.1 Proposal

The compiler's grouping key `(input, closed conditions)` (§3.7) *is* the paper's
ℒ_{b∪𝕀} — a model under a total setting of interventions. The readme keeps it
implicit and derives it; the paper writes it explicitly. Follow the paper: promote
it to a named table, and split the one node type along the two disjoint roles it
already has in practice (values consumed by name vs. effects listed in conditions —
rule 5 only makes sense because the roles never mix):

```yaml
worlds:   # name -> {input: base|source[j], edits: [EditName, ...]}
reads:    # name -> {site, pos, world, featurizer?, dims?}        # value producers
edits:    # name -> {site, pos, featurizer?, dims?, do, binds?}   # effect definitions
```

- A **world** is one forward: an input row plus the set of edits in force. The
  un-intervened worlds `base` and `source[j]` exist implicitly; you only author a
  world when it carries edits.
- A **read** produces a value: `featurize(read(site, pos) in world)[dims]` — the
  same pre-transform semantics as §3.6, with `world` naming what `conditions`
  encoded. Reads are what operands and metrics name.
- An **edit** is inert until a world lists it. It executes inside every world that
  lists it; its `f` (for `AddScaled`, `Lerp`, `Renormalize`) is the pre-value *in
  that world*, with the world's other edits governed by sec. 3.10's class order.
- **Operand/condition independence survives untouched** (§3.7's crux): an edit's
  operands name reads (values); a world's `edits:` list names effects. Consuming
  `v_receiver` still does not put you under the conditions it was harvested under —
  those live on `v_receiver.world`, which the consumer never inherits.
- **On the name.** `world` is *not* the paper's term. Geiger et al. define the
  object — the intervened model ℒ_{b∪𝕀}, whose notation names exactly this pair
  (`world.input` = b, `world.edits` = 𝕀) — and their word for a complete value
  assignment is a *(total) setting* (the readme uses it too, §3.6). "World" is
  borrowed from the counterfactual-semantics tradition the paper builds on
  (Lewis's possible worlds). It is still the right key: paper-native `setting`
  collides with config-speak in a YAML file; `run` collides with `causalab run`;
  and the operational candidate `forward_pass` is false in both directions —
  fusion puts several worlds into one literal forward (nnsight's fused
  multi-invoke trace), while staging, grad accumulation, and generation
  (prefill + N decode passes) spend many forwards on one world. That is why
  `num_forwards` is derived, ≥ |worlds| (§5), and must not be the table's name.
  The §11 glossary row is: `worlds.<name>` ↔ ℒ_{b∪𝕀} — ours the word, theirs
  the object.

### 2.2 What it deletes

| §3.6–§3.7 / §13 machinery | Under worlds |
|---|---|
| transitive closure of `conditions`, sorted, at load | gone — a world's edit list is already total |
| "set typed as tuple so it hashes" discussion | gone — same answer, but on an authored list |
| interning of condition sets (§13 mitigation 1) | gone — worlds are interned by construction |
| silent second forward when a name is omitted (§13) | impossible — forwards are the keys under `worlds:`; you cannot create one without writing it |
| named condition set as sugar (§13 mitigation 3) | *is* the design, not sugar |
| rule 5 asymmetric dead-node warning | symmetric and crisp: an edit no world lists is dead; a read nothing consumes is dead |
| `input` repeated on every node of one forward | stated once, on the world (`input: base` appears 7× in §6's example) |
| §3.7 needing to be "the crux" | the semantics is read off the page |

The node also shrinks: 7 fields (§3.6) → 5 per read and 5 per edit, and the two
types get disjoint reference namespaces — operands must name reads, worlds must
name edits, metrics must name reads — so every cross-reference is checkable with a
sharper error than "name resolves" (§4.1).

Derived facts get simpler, not weaker: `num_forwards` ≥ |worlds| (elision of
read-only depth unchanged); `paired_forward` ⇐ an edit in world *w* has an operand
read in a world with a different input; the **world DAG** (edits → operand reads →
worlds) is the schedule skeleton, and staging values across worlds as saved
constants falls out of its topological order. Acyclicity (§4.2) is now checked on
that small explicit DAG.

### 2.3 Equivalence and the differential test

Worlds are exactly the readme's canonical form with the closure pre-applied — the
translation both ways is mechanical (`(input, closed conditions)` ↔ world; node ↔
read or edit by `do == Identity`). That yields a free migration test: compile a
corpus both ways and assert identical `(input, conditions)` groups, forward counts,
and results. Worth pinning before the readme form is ever implemented, so the
simpler form can replace it without a semantics debate.

### 2.4 Honest costs

- One more table and two names (`reads`/`edits`) where §3.6 had one proud type.
  The counter: the readme already needs §3.7, rule 3, rule 5, and §13 to explain
  the one type. Concept count in the *explanation* is what matters, and it drops.
- World names enter the digest (renaming `patched` changes the hash). Same is
  already true of interventional names; names are semantic documentation and should
  hash. Non-issue, but worth stating.
- An edit in force in two worlds with *different operands* needs two edits. Also
  true today (operands are fixed per node). Non-regression.

## 3. Deletions and demotions

Each entry: what goes, why, and what replaces it.

### 3.1 `sweep` (and `ForEach`) — out of the protocol

A sweep is *n* protocols, not one; putting it inside the hashed value muddles what
the digest identifies (the artifact came from *a* layer, not from a scan). Hydra
already owns this: `causalab run -m protocol.yaml sites.target.layer='range(32)'`.
Each point gets its own digest — which is what provenance wants — and cluster
scheduling gets jobs, which is what SLURM wants. `runner/fanout.py` is the existing
in-house proof: sweeps as override cartesian products fanned into an array job, no
IR construct needed (sec. 6.4). The one thing lost is single-GPU amortization
(batching independent cells into the batch dim); recover it at the seam, not in the
IR: `run(protocols: list[Protocol])` where the backend fuses across protocols with
the *same mechanism* it already uses to fuse `(input, conditions)` groups within
one. Deletes `sweep`, deletes `ForEach`, deletes the §9 caveat about irregular
freeze sets needing either.

Irregular/variable-arity shapes (a k-node freeze set) are then *generated YAML* —
which is fine, and deserves a stated principle: **builders may exist as YAML
generators outside the boundary; nothing enters `run()` but the value.** That is
the honest version of "no Python builder" (§9) — the rule was never "no programs
write YAML" (hydra is one), it is "no second authoring surface feeds objects past
the validator." Sec. 6.4 leans on this principle hard.

### 3.2 Presets — delete the layer

§3.9's presets (`collect`, `replace`, `steer`, `interpolate`, `noise`) are Python
constructor residue. In a YAML-only world, `do: {swap: v}` *is* the preset —
one-to-one, as §3.9 itself observes ("the sign the algebra is the real primitive
set"). Method names live where §3.9 already puts the compositions: as templates
under `configs/protocol/`. One less layer, zero expressiveness lost.

### 3.3 `params` table — auto-declared, kept only for the exceptions

`Param.shape` is fully determined by `(Featurizer.kind, Featurizer.shape,
parametrization)` — the `[4096, 4096]` skew matrix in §7 is cayley's business, not
the author's, and authoring it is both duplication and a trap (change `k`, forget
the param row, get a shape error at best). So:

- Featurizer weights are **auto-declared**: each `kind` defines its slots
  (`subspace: weight`, `gate: theta`, `sae: enc, dec, b_enc, b_dec`, …), named
  `<featurizer>.<slot>`, shapes derived. The canonical form stamps them explicitly;
  the author never writes them.
- The `params:` table survives **optional**, for exactly two real cases: a *free*
  param that belongs to no featurizer (pullback's optimized written vector — an
  operand that is a `ParamName`), and a *constant* vector (a precomputed steering
  vector = `trainable: false` param in `weights.safetensors`). The second also
  closes an ambiguity: **constant vectors are untrainable Params**, never literals —
  YAML stays tensor-free with no judgement calls.
- `Param.trainable` is deleted: `train.params` is the single home for "what is
  optimized" (it already must exist for the optimizer). `sha256` and the fitted
  lifecycle are unchanged.

### 3.4 `seeds` — fold into the two real homes

Three seed homes (§3.3's `seeds:` top-level, `train.seed`, `NoiseSpec.seed`)
become two: noise seeds live in `NoiseSpec` (per-mechanism, formally an input
variable — §3.8, correct as is); everything else — init, data order — is
`train.seed`, because init only ever matters when something gets fitted. Top-level
`seeds` deleted.

### 3.5 `objective` — move inside `train`

`grad ⇐ params ≠ ∅ ∧ objective` (§5) is a two-clause rule because the two clauses
live in different places. An objective without training is meaningless; training
without an objective is ill-formed. One home: `train.objective`, and the derived
rule collapses to `grad ⇐ train ≠ null`. (Regularizers unchanged; `anneal`'s
dotted-path targets unchanged.) Fold `lr`, `schedule`, `clip` into the `optimizer`
dict while there — they are optimizer arguments everywhere else in the ecosystem —
and **move `resume` out of the protocol entirely**: it is a runtime affair like
backend kwargs (§3.1's own logic: backend-shaped things go to `run()`), and it must
not be in the hash. `TrainSpec` drops from 14 fields to 10 with no loss.

### 3.6 `outputs` — optional with a total default

Default: all metrics, plus any read consumed by nothing (an authored read you
didn't wire anywhere is more likely a wanted output than dead weight — and if it
is dead weight, the sec. 2.2 dead-read check names it). Explicit `outputs:` remains
for trimming. Deletes a required key from every simple protocol and kills the
silent "forgot to output the thing" failure.

### 3.7 `alignment` — a `binds:` field, not a container

`Alignment ⟨Π, π⟩` (§3.3) duplicates what the interventionals already encode: *the
(site, featurizer, dims) choice is the alignment* — that is the entire thesis of
DAS. The high-level variable names already exist in the protocol (Pos scopes:
`{variable: x}`). So: delete the top-level `alignment` container; add an optional
`binds: <hl_variable>` field on an edit. The alignment is then *derived* — the
collection of `(hl_variable ↔ site + featurizer + dims)` bindings — and sits next
to the thing it describes instead of in a parallel structure that can drift.
`high_level` stays: it is a name (provenance for what IIA compares against), not a
structure.

### 3.8 `weights.meta.json` — into the safetensors header

safetensors carries a JSON `__metadata__` map in its header. `ArtifactIdentity`
(§8) goes there; the bundle drops from three files to two
(`protocol.yaml` + `weights.safetensors`), and identity can never be separated
from the bytes it describes. Same refusal-on-mismatch behavior at `load`.

### 3.9 Derived fields never appear in authored YAML — even as comments

§6's example carries `requires: [paired_forward]   # derived` *inside the YAML*.
Comments rot into lies. Strict parsing (sec. 7.1) rejects derived keys in authored
documents; `causalab explain` (sec. 7.2) is where derived facts are displayed. The
readme's own examples should be corrected when it is next edited.

### 3.10 Rule 3 — syntactic mechanism classes, not algebraic commutativity

§4.3 requires overlapping writes to "commute (commutativity + left-annihilativity)"
and admits it is not checked — because it is not checkable in general without
algebraic reasoning over mechanisms. Replace it with a decidable rule. Tag each
`Mechanism` with a class:

| class | members | per (site, overlapping pos, world) |
|---|---|---|
| **absolute** | `Swap`, `Affine`, `Lerp`, `Clamp`, `Renormalize`, `PyTorchFn` | at most one |
| **additive** | `AddScaled`, `Gaussian` | any number |

and *define* the application order at one address: the absolute edit (if any)
first, then the additive edits summed. This is stronger than commutativity — it
gives well-defined set semantics even for non-commuting pairs (clamp-then-steer is
deterministic without clamp and steer commuting), it is checkable by pure syntax at
load, and it matches left-annihilativity in spirit (the absolute write annihilates,
additives stack on top). Two `Swap`s at one address become a load-time error — which
is correct, because that is a bug 100% of the time. Two `Lerp`s become "express it
as one `Affine`" — a rounding-error-sized loss for deleting an unimplementable
check. `dims`-disjoint writes remain freely composable (rule 4 unchanged).

## 4. Authoring-surface sugar — cheap, load-time, canonical form stays explicit

1. **Unify the `Pos` forms.** §3.4.1 carries two type tags (`index`, `span`) plus
   `variable` for what §3.4.1 itself says is one kind of thing ("single indices and
   windows are the same kind"). Authoring surface: `pos: -1` (int → index),
   `pos: [a, b]` (pair → span), `pos: {var: x}` (variable window), each optionally
   `scope:`/`relative_to:`. Canonical form keeps the explicit tagged records;
   nothing downstream changes.
2. **Inline `pos`.** A `positions:` table entry earns its keep only when a position
   is named once and used many times; `pos: -1` inline covers the (dominant) rest.
   §6's example loses its `positions:` table entirely; the table stays available
   for `{variable: x}` positions the *task* config group defines (sec. 5.2).
3. **Featurizer composition as a list.** `featurizer: [rot18, gate18]` instead of
   the `"rot18 >> gate18"` string micro-DSL (§7's DBM delta). No parser, trivially
   canonical, hashes as structure. `>>` survives in prose as the way to say it.
4. **`dims` defaults do the work.** For a `subspace` of shape `[4096, 8]`, the
   feature space *is* 8-dimensional — `dims: null` already means all of it (§3.6).
   §7's DAS example authors `dims: [0,…,7]` twice for nothing. DAS needs no `dims`
   at all; `dims` is for genuine sub-selection (SAE feature ids). Document this;
   delete it from the examples.
5. **Inline one-shot definitions — deferred.** Allowing a read to inline its site
   dict (auto-hoisted to a generated name at load) reads nicely but generated names
   pollute the override surface and the digest. Named tables are load-bearing for
   hydra (`sites.target.layer=24` needs a stable path). Revisit only if authoring
   friction is demonstrated; do not build it speculatively.

## 5. Hydra-native conventions — the composition contract

1. **One validator.** Hydra composes plain YAML documents; `Protocol.from_yaml`
   (well-formedness §4) is the *only* validator. No structured-config/ConfigStore
   schemas doing a second, weaker validation pass at compose time, no OmegaConf
   types leaking past the seam — `OmegaConf.to_container(cfg.protocol,
   resolve=True)` and hand it over. Interpolations therefore resolve *before*
   canonicalization; the hashed value is always post-resolution.
2. **The task group exports a declared interface.** Method templates reference
   dataset columns (`a: answer`, `target: label`) and prompt variables
   (`{var: x}`) that only the task defines. Make that a written contract: each
   `task/*.yaml` declares `columns: [answer, cf_answer, label]` and
   `variables: [x, …]` alongside `protocol.data`, and `validate` checks template
   references against it *at compose time* — the earliest point the information
   exists (from_yaml alone cannot see the table; `validate --data` re-checks
   against the real thing). Templates then compose against a contract, not luck.
3. **Model facts live beside the model, and `d` is authored nowhere.** The method
   template must not hardcode `4096` or `layer: 18`. `model/*.yaml` carries the
   handle (`protocol.model`) plus a `facts:` block (`hidden: 4096`, `layers: 32`)
   as a *sibling* of `protocol`, consumed by interpolation
   (`shape: [${facts.hidden}, 8]`, `layer: ${facts.mid_layer}`) and never entering
   the protocol schema (strict parsing keeps it out by construction). The `k` stays
   authored — it is a choice; `d` is a fact.
4. **Names are the override surface.** `sites.target.layer=24`,
   `featurizers.rot.shape.1=16`, `train.optimizer.lr=3e-4` — this is why the named
   tables survive every simplification above. Templates should use *role* names
   (`target`, `rot`, `patched`) rather than model-specific ones (`L18`), so
   overrides read as intent and templates port across models unchanged.
5. **Know the list-merge caveat.** OmegaConf overrides replace lists wholesale —
   `worlds.patched.edits` cannot be appended to from an override, only replaced.
   That is acceptable (a world's edit set is a semantic unit; partial merge of it
   would be a footgun), but it should be stated so nobody designs a config group
   around appending edits. Where composition *should* add edits (a template with an
   optional freeze set), model it as two named worlds or a generated protocol, not
   a list merge.

## 6. The campaign layer — how `analyses/` chains methods, and what it becomes

The protocol deliberately has **no `Pipeline`** (§3.1: "multi-step campaigns stay
hydra multirun; chains are recoverable post hoc because every artifact stamps the
producing digest"). `causalab/analyses` at `a50637c` is the reality check for that
sentence: 16 analysis packages, 9,197 lines of `main.py` alone, and real chains
that are *not* multirun-shaped — `subspace` needs `locate`'s **result** (the best
cell) before its own configuration is even known. Data-dependent control flow is
the campaign layer's defining feature, and it deserves an explicit design.

### 6.1 How chaining works today

- **A pipeline is a hydra defaults list, and its execution order is the list
  order** — "recovered at runtime via OmegaConf insertion order"
  (`runner/run_exp.py` docstring). Each analysis config mounts at `cfg.<name>` via
  `# @package <name>`; the runner walks the mounted slices in insertion order and
  dispatches `_name_` → `importlib.import_module(f"causalab.analyses.{name}.main")`.
- **Orchestration directives live inside config**: `_name_`, `_subdir`,
  `_output_dir` (`${experiment_root}/subspace/${.method}_k${.k_features}`), plus a
  `modes:` list some analyses interpret as an inner sweep, re-resolving `_subdir`
  per mode "so entries land in distinct dirs instead of overwriting each other".
- **Data flows through a conventional artifact tree.**
  `experiment_root: artifacts/${task.name}/${model.id}`; downstream steps
  auto-discover upstream outputs by scanning it — `io/pipelines.py:
  load_locate_result` walks `sorted(os.listdir(locate_root))`, takes the first
  directory with a `results.json` (falling back to `metadata.json`), and returns
  `{}` when nothing is found. `subspace.layers: null` means "auto-resolve from
  locate/".
- **Cross-step facts are hand-duplicated.**
  `runners/weekdays/weekdays_8b_pipeline.yaml` states `layers: [28]` three times
  (locate, subspace, activation_manifold), kept consistent by eye.
- **Sweeps are a separate orchestrator** — `runner/fanout.py` builds cartesian
  products of hydra overrides into a manifest, submits one slurm array job, each
  shard writes `<base>/shards/<id>`, then recombines.
- **Cross-analysis figures** run as registered `post:` handlers
  (`runner/post_steps.py`) that re-read prior steps' outputs.
- A `target_variables` loop mutates `cfg.task.target_variable` per iteration,
  with a `HANDLES_MULTI_VARIABLE` module flag for analyses that loop themselves.

### 6.2 The failure modes worth designing against

1. **Order-by-insertion is invisible.** The shipped weekdays pipeline has five of
   its seven steps commented out in the defaults list — the chain changes by
   deleting a `#`. Nothing states or checks step dependencies.
2. **Missing upstream is silent.** `load_locate_result` returning `{}` means
   downstream falls back to defaults — the §13 class of bug (a wrong number, not
   an error) at campaign scale.
3. **`sorted(listdir)[0]` is the resolution rule.** Run two locate methods and the
   alphabetically-first wins, silently.
4. **Facts triplicated across step configs**, or left `null` to trigger scanning —
   the two failure modes trade off against each other.
5. **Artifacts don't identify their producer.** `metadata.json` records knobs, not
   a config digest; reproduction is "the config snapshot near the directory".

### 6.3 The boundary the protocol makes possible

An analysis today braids three things: **(a)** model-touching loops — spec grids,
batch loops, plan runs, metric slicing; **(b)** off-model numerics — PCA/manifold
fitting, geometry, LLM judging, webtext probes; **(c)** artifact and figure IO.
The protocol absorbs (a) *whole*: build protocol(s) from a template, `run()`, read
`Result`. What remains of an analysis is a thin program around (b) and (c). That
is where most of the 9,197 lines go — not into YAML, but into deletion, because
the batching/spec/metric scaffolding is the backend's job now.

The analyses that look least YAML-able — `characterize_subspace` (judge, webtext
reproduction), `path_steering` (2,013 lines of manifold geometry) — are the
*confirmation*, not the counterexample: they are (b)-heavy programs that should
never have been asked to be declarative. The test of the boundary: **an analysis
touches the model only through protocols; everything else it does is ordinary
Python over Results and artifacts.**

### 6.4 Chaining in a fully YAML-driven codebase — three tiers

The campaign layer should not become one thing; it is three, and naming them
prevents the standard failure (a workflow-YAML dialect that grows conditionals,
loops, and templating until it is a program in the wrong syntax):

- **Tier 1 — data threading, no control flow: an `${artifact:...}` resolver.**
  A compose-time OmegaConf resolver that reads one typed value out of a prior
  step's artifact: `layer: ${artifact:locate/interchange:best_layer}`. It resolves
  before hashing (sec. 5.1), so the produced protocol is concrete, and the
  resolved ref — path *and* digest — is stamped into provenance. This deletes both
  halves of failure-mode 4: the runner config states `layers` zero times, and
  nothing scans directories. Missing artifact = compose-time error, not `{}`.
- **Tier 2 — static chains: an explicit `steps:` list.** The runner schema gets
  `steps: [{analysis: locate, …}, {analysis: subspace, needs: [locate]}, …]` —
  ordered, checkable, diffable — replacing insertion-order semantics and the
  `_name_`/`_subdir`/`_output_dir` directive vocabulary. Deliberately not
  Turing-complete: no conditionals, no loops, no templating. The moment either is
  wanted, that step is tier 3.
- **Tier 3 — data-dependent control flow is a program.** Argmax-then-fit,
  train-until, judge-and-iterate: Python — or an agent session, which is what
  causalab is built to be driven by. The sec. 3.1 generator principle is the
  governing rule: **programs may orchestrate, but every model-touching step they
  emit is a hashed protocol, and the `produced_by` digest chain across artifacts is
  the reproducibility record.** The pipeline *document* is optional; the
  provenance *chain* is not. Reproduction replays protocols by digest; it never
  needs to re-run the decision logic that chose them.

Supporting moves, one layer down:

- **Typed artifacts everywhere.** Extend §8's `ArtifactIdentity` from featurizer
  bundles to every inter-analysis handoff: each artifact carries a schema name +
  version (`locate.result@1`) and `produced_by: <digest>`. The `load_*` scan
  helpers collapse into one `Artifact.load(ref, expect="locate.result@1")` that
  refuses on missing or mismatched schema. This is the campaign-layer twin of the
  protocol's own refusal culture, and it is buildable *now*, before any protocol
  exists (sec. 10 amendment 4).
- **`fanout` survives as the sweep engine.** It already embodies sec. 3.1's
  decision (sweeps are override products above the value, not IR constructs); it
  gets simpler when shards are protocols — shard identity is the digest, and
  recombination keys on content instead of `shards/<id>` directory convention.
- **The multi-variable loop becomes a data axis.** `target_variable` selects
  dataset fields → it is a protocol difference → multirun/fanout it. Deletes the
  cfg-mutating loop and `HANDLES_MULTI_VARIABLE`.

### 6.5 What this deletes at the campaign layer

Insertion-order execution semantics; the `_name_`/`_subdir`/`_output_dir`
directive vocabulary and per-analysis `modes:` inner sweeps (fanout axes cover
them); the silent-`{}` discovery helpers and `sorted(listdir)` resolution;
`HANDLES_MULTI_VARIABLE`; the triple-stated cross-step facts; and — by explicit
refusal — any temptation toward a conditional/looping workflow-YAML dialect.

## 7. Robustness additions

1. **Strict parsing.** Unknown keys are errors, not ignored (`positon:` silently
   dropped is a wrong-numbers bug, the worst kind per §13). Closed enums reject
   with did-you-mean. Derived keys rejected in authored docs (sec. 3.9).
2. **`causalab explain protocol.yaml`** — the compile report as a first-class CLI
   verb beside `run`/`validate`/`digest` (§9): worlds → forward count and the world
   DAG, derived `requires`, auto-declared params with shapes, the digest, and what
   `outputs` defaulted to. §13's "print the derived groups" mitigation, promoted to
   the tool you run before burning cluster time.
3. **A pinned canonical corpus.** `tests/protocols/` holds every method template ×
   a reference model/task, each with its canonical form and digest pinned as golden
   files. Any change to canonicalization or schema shows up as a reviewable diff,
   and hash-breaking changes become deliberate (bump `version`, ship a loader
   migration) instead of accidental. This is the value-form's analogue of the §12
   round-trip test, extended over the whole method library.
4. **The worlds↔conditions differential test** (sec. 2.3) while both forms exist.
5. **Typed artifact loading with refusal** (sec. 6.4) — the campaign layer's
   version of items 1–2.
6. **Unchanged, and load-bearing:** load-time sugar expansion with an explicit
   canonical form, the refusal culture (`PyTorchFn` at construction, capability
   mismatches with generated messages), `ArtifactIdentity` refusing mismatched
   featurizer loads, datasets and models pinned by digest/revision in the ref
   (§13's reproducibility floor applies to `model.revision` exactly as to data —
   canonical form always stamps it explicitly).

## 8. What must NOT be simplified — the floor

Each of these was considered and rejected; recording why, so the next brainstorm
doesn't relitigate them.

- **Site × Pos × Featurizer stay three things.** They vary independently, and they
  are the hydra override surface. An "address blob" would couple what composition
  needs decoupled.
- **Metrics stay a closed vocabulary, not an expression DSL.** Lowering needs
  recognizable kinds — vocab-parallel CE must be *seen* as cross-entropy, and
  `logits_to_keep` must be derivable from `of:` targets (§3.10). §13's line holds:
  gather-then-reduce over node values and dataset columns, nothing else.
- **The `do:` algebra stays closed.** "Just name a torch fn" reopens B8 and forfeits
  NDIF and Megatron simultaneously. The sec. 3.10 class tags make the algebra
  *more* structured, not less closed.
- **Pre-transform read semantics stay** (§3.6) — post-transform makes
  read-before-write at the same address inexpressible. Worlds change the naming,
  not this.
- **`TrainSpec` stays declared**, however long it looks. The outer loop is exactly
  what a distributed optimizer must own (§3.1's rationale, megadas' 141KB
  re-implementation as the cautionary tale). Ten fields that replace a backend
  fork each are cheap.
- **Analyses stay programs.** The campaign layer's control flow is not a value and
  must not be YAML-encoded (sec. 6.4, tier 3). The declarative surface there is
  the artifacts, not the flow.
- **Canonicalization, `version`, digest, `ArtifactIdentity` stay.** They *are* the
  provenance purchase (sec. 0.1); every simplification above was chosen to make
  them smaller, never optional. One serializer though: canonical form is canonical
  YAML (JSON-compatible subset, sorted keys, canonical floats) and the digest is
  the sha256 of those bytes — not a separate `canonical_json` path (§8).
- **YAML-only authoring stays**, with the sec. 3.1 clarification: generators may
  write YAML; nothing but the value passes the validator.

## 9. The schema after the pass

Top-level: **16 keys → 13, of which 8 optional.**

```
version, model, high_level?, data,
sites, positions?, featurizers?, params?,
worlds?, reads, edits?,
metrics?, outputs?, train?
```

(`alignment`, `interventionals`, `objective`, `sweep`, `seeds` gone;
`counterfactual_dataset` renamed `data` — the base/sources keys already say
"counterfactual"; the name should say which slot it fills.)

The minimum protocol — collect activations — is five lines:

```yaml
version: "1"
model: {key: meta-llama/Llama-3.1-8B}
data:  {base: {dataset: "weekdays/train@9ab2", field: input}}
sites: {L18: {component: block_output, layer: 18}}
reads: {acts: {site: L18, pos: -1, world: base}}
```

### 9.1 Path patching, rewritten (vs. §6)

```yaml
version: "1"
model: {key: meta-llama/Llama-3.1-8B, revision: main}

data:
  base:   {dataset: "ioi/test@3f1c", field: input}
  source: {dataset: "ioi/test@3f1c", field: "counterfactual_inputs[0]"}

sites:
  sender:   {component: attention_value, layer: 9, head: 9}
  receiver: {component: block_input,     layer: 12}
  a10:      {component: attention_output, layer: 10}
  a11:      {component: attention_output, layer: 11}

worlds:                                   # base & source exist implicitly
  patched: {edits: [swap_sender, freeze_10, freeze_11]}   # input: base default
  final:   {edits: [inject]}

reads:
  v_sender:   {site: sender,   pos: -1, world: source}
  v_a10:      {site: a10,      pos: -1, world: base}
  v_a11:      {site: a11,      pos: -1, world: base}
  v_receiver: {site: receiver, pos: -1, world: patched}   # collect-under-intervention
  logits:     {site: lm_head,  pos: -1, world: final}

edits:
  swap_sender: {site: sender,   pos: -1, do: {swap: v_sender}}
  freeze_10:   {site: a10,      pos: -1, do: {swap: v_a10}}
  freeze_11:   {site: a11,      pos: -1, do: {swap: v_a11}}
  inject:      {site: receiver, pos: -1, do: {swap: v_receiver}}

metrics:
  logit_diff: {kind: logit_diff, of: logits, a: answer, b: cf_answer}
```

The four forwards of §6 are now *visible*: two implicit (`source`, `base`) and two
authored (`patched`, `final`). No `positions:` table, no `outputs:`, no `requires:`
comment, `input: base` written zero times instead of seven.

### 9.2 DAS, rewritten (vs. §7)

```yaml
version: "1"
model:      {key: meta-llama/Llama-3.1-8B, revision: main}
high_level: {key: "weekdays.causal_model"}

data:
  base:   {dataset: "weekdays/train@9ab2", field: input}
  source: {dataset: "weekdays/train@9ab2", field: "counterfactual_inputs[0]"}

sites:       {target: {component: block_output, layer: ${facts.mid_layer}}}
featurizers: {rot: {kind: subspace, shape: [${facts.hidden}, 8],
                    parametrization: cayley}}
# no params table: rot.weight auto-declared, shape derived from cayley
# no dims: the 8-dim feature space is the selection

worlds: {patched: {edits: [patch]}}

reads:
  v_src:  {site: target,  pos: -1, world: source, featurizer: rot}
  logits: {site: lm_head, pos: -1, world: patched}

edits:
  patch: {site: target, pos: -1, featurizer: rot, do: {swap: v_src},
          binds: answer_position}          # ← the alignment, derived not declared

metrics:
  iia: {kind: logit_diff,    of: logits, a: cf_answer, b: base_answer}
  ce:  {kind: cross_entropy, of: logits, target: label}

train:
  objective:  [[1.0, ce]]
  params:     [rot.weight]
  optimizer:  {name: adamw, lr: 1.0e-3, weight_decay: 0.0}
  steps:      {epochs: 10}
  batch:      {pairs: 16}
  precision:  {feature: fp32, loss: fp32, model: bf16}
  eval:       {every: {epochs: 1}, split: "weekdays/test@9ab2", metrics: [iia]}
  early_stop: {metric: iia, patience: 3, mode: max}
  checkpoint: {every: {epochs: 1}, scope: params}
  seed: 0
```

~30 authored lines against §7's ~45, with the `[4096, 4096]` param row, the double
`dims`, `seeds`, `outputs`, `requires`, and the top-level `objective` all gone —
and nothing a backend needs lost, because every deleted line was derivable. The
DBM delta shrinks the same way: add `gate` to `featurizers`, change one line to
`featurizer: [rot, gate]`, add the `l1` term and the `anneal`.

## 10. Considered and rejected — the radical options

For completeness, the bigger swings that fail the "no simpler" test:

- **A flat trace table** (one list of `(world, site, pos, do)` rows, spreadsheet
  style): deletes all names, and with them sharing, metrics addressing, and the
  hydra override surface. Names are the composition interface; keep them.
- **No `reads` section** (operands inline their `{site, pos, world}`): metrics and
  outputs need names for values anyway; inlining saves nothing and splits value
  identity across use sites.
- **Embedding in an existing IR** (torch.export, StableHLO-shaped things): those
  are program IRs — the entire point (sec. 0.1) is to stay a description. The
  protocol *lowers to* programs; it must not be one.
- **Metrics out of the protocol** (collect everything, score in pandas): forfeits
  `logits_to_keep`, vocab-parallel readout, and shared loss/metric vocabulary
  (§3.10) — the exact wins that motivated `Metric`.
- **Deriving featurizer `shape` from the model at load**: tempting (the protocol
  becomes size-agnostic) but canonicalization would need model metadata, breaking
  "hash without touching anything". Sec. 5.3's interpolation-from-facts gets the
  authoring win with a pure compose-time mechanism; canonical form stays concrete.
- **A workflow-YAML for campaigns** (steps with conditionals/loops/templating):
  the Argo failure mode — a program in the wrong syntax, unhashable in the ways
  that matter and unreadable in the ways that don't. Sec. 6.4's three tiers are
  the answer; tier 2 stays deliberately non-Turing-complete.

## 11. Impact on the build order (§12)

The §12 sequence survives with four amendments:

1. **Step 1 (round-trip + digest) targets the worlds form directly** — its
   canonicalization is a strict subset of the readme form's (no closure, no
   interning), so step 1 gets *cheaper*. Add the sec. 7.3 golden corpus here.
2. **Step 6 shrinks**: "conditions" work is replaced by the (simpler) worlds
   tables; the sec. 2.3 differential test replaces the correctness argument.
3. **Deleted from scope entirely**: presets (step 3 loses a deliverable it didn't
   need), `sweep`/`ForEach`, the `params` table for featurizer weights, top-level
   `objective`/`seeds`/`alignment`, `weights.meta.json`.
4. **One step added, and it can go first**: typed artifacts + the
   `${artifact:...}` resolver (sec. 6.4) touch no protocol machinery, fix the
   campaign layer's silent-`{}`/`sorted(listdir)` failures immediately, and lay
   the `produced_by` rail the digest will later ride. Like §12's step 1, it is
   useful alone.

The §12 empirical settlers stand unchanged and should still run first; add one:
draft `configs/` for the three worked methods (DAS, DBM, path patching) with a real
`model/` + `task/` split and confirm the sec. 5 conventions survive contact with
hydra's actual merge behavior — that is the cheapest way to falsify this document.

## 12. Summary — the pass in one table

| # | Change | Deletes | sec. |
|---|---|---|---|
| 1 | explicit `worlds` + `reads`/`edits` split | closure, interning, §13 footgun, rule 5 asymmetry, 7-field node | 2 |
| 2 | `sweep`/`ForEach` → hydra multirun/fanout + `run(list)` fusion | two IR constructs, digest ambiguity | 3.1 |
| 3 | presets deleted | one layer | 3.2 |
| 4 | params auto-declared from featurizer kinds | shape duplication, cayley leak, `trainable` dup | 3.3 |
| 5 | `seeds` → `train.seed` + `NoiseSpec` | one top-level key | 3.4 |
| 6 | `objective` (+`lr`/`clip`/`schedule`) into `train`; `resume` out to `run()` | two-clause grad rule, hash pollution | 3.5 |
| 7 | `outputs` optional with total default | one required key, a silent failure | 3.6 |
| 8 | `alignment` → per-edit `binds:` | a parallel structure that can drift | 3.7 |
| 9 | `ArtifactIdentity` into the safetensors header | one file | 3.8 |
| 10 | rule 3 → mechanism classes + defined order | an unimplementable check | 3.10 |
| 11 | Pos unification + inline ints + list composition + dims defaults | tags, tables, a string DSL, redundant dims | 4 |
| 12 | one validator, task interface, model facts, role names | double validation, hardcoded `d`/layers | 5 |
| 13 | campaign layer: typed artifacts + `${artifact:}` resolver + explicit `steps:` + tier-3-is-a-program | insertion-order chains, silent-`{}` discovery, triplicated facts, directive vocabulary | 6 |
| 14 | strict parsing, `explain`, golden corpus, differential test | silent wrongness | 7 |
