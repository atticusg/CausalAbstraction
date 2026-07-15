# Subject–Object Relations

A **factory** task over the 35 LRE (Linear Relational Embedding) relations from
the pair-disjoint DAS bundle at
`<lre-relations-source>/`. Each relation
maps a **subject** to an **object** (its answer) — e.g. `France → Paris`
(`country_capital_city`), `Lisa → woman` (`name_gender`), `STUDY → S`
(`word_first_letter`). Select a relation with `task.relation=<name>` (default
`word_first_letter`, a curation-green relation); see `data/manifest.json` for the
35 valid names.

The DAG mirrors [`identity_naming/`](../identity_naming/) with a subject→object
lookup:

```
subject ──> object ──> raw_output
    └──────> raw_input
```

`object` is a deterministic lookup `subject_to_object[subject]` (the relation's
law); `raw_input` fills the relation's template with the subject; `raw_output`
is `" " + object`. Phrasing templates are **not** causal variables — the causal
model renders from `templates[0]`, and `counterfactuals.py` provides template
variation by overriding `raw_input` per example.

## Data provenance (bundled, model-agnostic)

`data/build_relations.py` ingests each relation's **`dataset/filtered.jsonl`**
(the authoritative source — every row has `subject` / `object` / `template` /
`template_idx`) and writes a compact JSON per relation under `data/relations/`
plus `data/manifest.json` (~0.3 MB total, committed). It keeps only the relation
content — distinct subjects, the deterministic subject→object map, distinct
objects, deduped templates — and **drops every Llama-3.1-8B-specific token /
position field** (`prompt`, `prompt_token_ids`, `gold_first_id`, `subj_last*`,
`pred_first_id`, `candidate_first_ids`, `object_first_ids`, …): causalab
recomputes tokens and positions per tokenizer, so nothing model-specific is
carried. Runtime never touches `external artifact storage`.

Group / effective-dimension provenance (`group`, `C`, `eff_k`, `compact`) comes
from the bundle-level `effective_dimension.parquet` (answer-token site);
`category` from the per-relation meta when present. The 11 stripped-meta bias
relations (meta with no `templates` list / empty objects) are handled by reading
templates and objects from the JSONL, not the meta.

Source templates use a positional `{}` subject slot; the build script rewrites
it to the named `{subject}` placeholder that causalab's token-position parser
requires.

Regenerate with:

```bash
uv run python causalab/tasks/subject_object_relations/data/build_relations.py
```

The source has exactly 35 relation subdirs with a `dataset/filtered.jsonl` — the
in-scope set. (The epic's out-of-scope names — `addition` / `hours` / `months` /
`weekdays` / `entity_tracking` / `text_length` — do not live under
`lre_relations/`, so no exclusion is needed here.) One relation,
`person_plays_instrument`, has a single non-deterministic subject→object row in
the source; the build keeps the first (deterministic) mapping and records the
conflict count in the relation's `provenance`.

## Answer scoring (first-token / prefix-aware)

Objects may be multi-token ("Washington D.C."). The model declares
`output_tokens = build_output_tokens(distinct_objects)` with
`match_modes = {"object": "prefix"}`, and the loader derives a prefix checker
from that declaration (no `checker.py`). At `max_new_tokens=1` this credits a
single-token object exactly and a longer generation by prefix. **Consequence:**
relations whose objects are *not* single-token / first-token-distinct are graded
strictly (the model emits only the answer's first token) and score low — this is
the honest curation signal below, not a bug. Genuine first-token grading of
multi-token objects would require declaring first-token-only forms; deferred as
out of scope.

## Curation (Qwen3-4B-Instruct, chat-coherent)

`data/curation_sweep.py` measures per-relation base accuracy on the golden
fixture model (`chat-coherent` = `Qwen/Qwen3-4B-Instruct-2507`, chat template +
the golden answer directive) through the same building blocks the baseline
runner uses, plus single-token-decodability and first-token-distinctness of each
answer space. A relation is **green** when accuracy ≥ 0.9 **and** its answer
space is first-token-distinct. Accuracy reflects this *instruct* pipeline on the
completion-style LRE templates; most relations are flagged for genuine model
difficulty (e.g. `person_plays_*`) and/or the first-token grading of multi-token
objects (e.g. `country_capital_city`). Reproduce (GPU):

```bash
uv run python causalab/tasks/subject_object_relations/data/curation_sweep.py \
    --out /tmp/soc_curation.json --n 64 --seed 0
```

`C` / `eff_k` / `compact` are source DAS provenance (answer-token site); `acc` is
the measured base accuracy at seed 0, `n≈64` (post-dedup) examples per relation.

| relation | group | category | C | eff_k | compact | n_subj | n_obj | single-tok | 1st-tok-distinct | acc (Qwen3-4B) | status |
|---|---|---|--:|--:|:--:|--:|--:|--:|:--:|--:|:--:|
| `adj_antonym` | injective | linguistic | 70 | 16 | yes | 71 | 70 | 0.90 | yes | 0.719 | flagged |
| `adj_comparative` | injective | linguistic | 53 | 16 | yes | 54 | 54 | 0.70 | no | 0.023 | flagged |
| `adj_superlative` | injective | linguistic | 70 | 32 | yes | 71 | 71 | 0.45 | no | 0.053 | flagged |
| `characteristic_gender` | bias | - | 2 | 1 | yes | 26 | 2 | 1.00 | yes | 0.875 | flagged |
| `city_in_country` | injective | factual | 19 | 16 | no | 25 | 20 | 0.75 | no | 0.273 | flagged |
| `company_hq` | categorical | factual | 122 | 128 | no | 511 | 122 | 1.00 | yes | 0.234 | flagged |
| `country_capital_city` | injective | factual | 23 | 16 | no | 23 | 23 | 0.70 | yes | 0.000 | flagged |
| `country_language` | injective | factual | 9 | 8 | no | 19 | 9 | 0.89 | yes | 0.267 | flagged |
| `country_largest_city` | injective | factual | 23 | 16 | no | 23 | 23 | 0.87 | yes | 0.167 | flagged |
| `degree_gender` | bias | - | 2 | 1 | yes | 30 | 2 | 1.00 | yes | 0.960 | **green** |
| `food_from_country` | injective | factual | 23 | 16 | no | 26 | 23 | 0.96 | yes | 0.429 | flagged |
| `landmark_in_country` | categorical | factual | 85 | 64 | no | 718 | 86 | 1.00 | yes | 0.500 | flagged |
| `landmark_on_continent` | categorical | few_class_categorical | 4 | 16 | no | 196 | 4 | 1.00 | yes | 0.692 | flagged |
| `name_birthplace` | bias | - | 8 | 4 | yes | 28 | 8 | 1.00 | yes | 0.739 | flagged |
| `name_gender` | bias | - | 2 | 1 | yes | 18 | 2 | 1.00 | yes | 1.000 | **green** |
| `name_religion` | bias | - | 4 | 2 | yes | 15 | 4 | 0.75 | yes | 0.200 | flagged |
| `object_superclass` | categorical | commonsense | 9 | 4 | yes | 60 | 9 | 0.78 | yes | 0.478 | flagged |
| `occupation_age` | bias | - | 2 | 1 | yes | 22 | 2 | 1.00 | yes | 1.000 | **green** |
| `occupation_gender` | bias | - | 2 | 1 | yes | 19 | 2 | 1.00 | yes | 0.800 | flagged |
| `person_band_lead_singer` | injective | factual | 18 | 16 | no | 20 | 20 | 0.10 | no | 0.059 | flagged |
| `person_native_language` | bias | - | 24 | 8 | yes | 811 | 29 | 1.00 | yes | 0.492 | flagged |
| `person_occupation` | categorical | factual | 19 | 16 | no | 157 | 19 | 0.84 | yes | 0.121 | flagged |
| `person_plays_instrument` | categorical | factual | 5 | 4 | no | 371 | 5 | 1.00 | yes | 0.000 | flagged |
| `person_plays_position_in_sport` | categorical | factual | 12 | 4 | yes | 640 | 12 | 0.83 | yes | 0.000 | flagged |
| `person_plays_pro_sport` | categorical | factual | 5 | 4 | no | 289 | 5 | 1.00 | yes | 0.066 | flagged |
| `pokemon_evolutions` | injective | factual | 37 | 16 | yes | 40 | 40 | 0.03 | no | 0.000 | flagged |
| `product_by_company` | categorical | factual | 26 | 8 | yes | 390 | 26 | 1.00 | yes | 0.763 | flagged |
| `star_constellation` | categorical | factual | 22 | 8 | yes | 226 | 24 | 0.25 | no | 0.081 | flagged |
| `substance_phase` | categorical | few_class_categorical | 3 | 4 | no | 42 | 3 | 1.00 | yes | 0.962 | **green** |
| `task_done_by_person` | injective | commonsense | 24 | 16 | no | 24 | 24 | 0.62 | yes | 0.158 | flagged |
| `task_done_by_tool` | injective | commonsense | 33 | 16 | yes | 35 | 34 | 0.53 | no | 0.172 | flagged |
| `verb_past_tense` | injective | linguistic | 76 | 32 | yes | 76 | 76 | 0.99 | yes | 0.439 | flagged |
| `word_first_letter` | categorical | linguistic | 25 | 16 | no | 241 | 25 | 1.00 | yes | 1.000 | **green** |
| `word_last_letter` | categorical | linguistic | 18 | 8 | yes | 173 | 18 | 1.00 | yes | 0.623 | flagged |
| `word_sentiment` | categorical | few_class_categorical | 3 | 4 | no | 34 | 3 | 1.00 | yes | 1.000 | **green** |

**Green (6):** `degree_gender`, `name_gender`, `occupation_age`,
`substance_phase`, `word_first_letter`, `word_sentiment`.

### Test tiers wired to the curation

- **smoke** (`tests/end_to_end/configs/smoke/subject_object_relations.yaml`) —
  `name_gender` (small, green): runs the object-flip path on `tiny-random` (CPU),
  existence-only asserts.
- **golden** (`tests/end_to_end/configs/golden/subject_object_relations.yaml` +
  `tests/end_to_end/goldens/subject_object_relations.json`) — `word_first_letter`,
  the strongest green relation (25 first-token-distinct letters, 1.000 accuracy).

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` (the loader convention)
returns `n` **object-flip** pairs: a base subject and a counterfactual subject
whose *object differs*, so the answer flips (LRE interchange semantics).
Templates are cycled by index. `generate_resample_dataset` is an
independent-resample noise-floor reference. Both snapshot/restore the global RNG.
`COUNTERFACTUAL_GENERATORS` exposes zero-arg wrappers (`flip_object`,
`random_counterfactual`) over the default relation.

## Token Positions

`create_token_positions(pipeline, template=...)` returns `last_token` and
`subject` (the last token of the `{subject}` span), built declaratively via
`build_token_positions` — same pattern as `identity_naming`.

## Files

| File | Role |
|---|---|
| `config.py` | `SubjectObjectRelationsConfig` (loads the bundled JSON) + `load_manifest` / `relation_names` |
| `causal_models.py` | `create_causal_model` + the `GET_*` accessors used by `tasks/loader.py` |
| `counterfactuals.py` | object-flip `generate_dataset`, `generate_resample_dataset`, `COUNTERFACTUAL_GENERATORS` |
| `token_positions.py` | `create_token_positions` |
| `data/build_relations.py` | ingest the bundle → committed model-agnostic JSON |
| `data/curation_sweep.py` | per-relation base accuracy on Qwen3-4B (curation table) |
| `data/relations/*.json`, `data/manifest.json` | committed relation content |
| `summary.ipynb` | CPU-only task walkthrough (no model) |

## Known gaps

- **No task `numerical_unit` pin.** Like the other factory tasks
  (`graph_walk`, `identity_naming`, `natural_domains_arithmetic`), this task has
  no LM-free numerical pin — `tests/_helpers/task_pins.py::walk_task_samples`
  raises `NotImplementedError` for `FACTORY_TASKS` until a serialisable
  `task_cfg` shim lands. A follow-up issue tracks factory-task pin support.
