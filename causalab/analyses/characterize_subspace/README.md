# characterize_subspace

Independent reproduction + characterization of a phase-1-supplied subspace,
ending in a refined hypothesis bundle for downstream consumption.

## What it does

1. Loads a rotation matrix from a `.safetensors` artifact (adaptive shape
   inspection — see `loading.py`).
2. Projects a phase-1 dataset through the subspace, runs the adaptive
   metric gate (`reproduction.py`). Failure → writes
   `more_info_needed.json` and stops.
3. Streams broad-corpus webtext (default: FineWeb-Edu) and projects it.
   Each document is reduced to its **peak-norm token** — the single (non-BOS)
   token whose k-dim subspace projection has the largest Euclidean norm
   (`‖proj‖₂` over *all* subspace dimensions, not just the dim-0 coordinate) —
   rather than a per-document mean, which washes out the sparse token a
   concept-selective subspace actually fires on. The norm measures how strongly
   a token fires in the subspace in any direction, so a concept living off the
   leading axis is still captured; the BOS attention sink is excluded.
   Documents are binned by that norm; the kept extrema are the
   **strongest-activating** (top-k) and **weakest-activating / most generic**
   (bottom-k) documents (`webtext.py`). Because the score is a magnitude,
   **direction is discarded** — the judge characterises *what makes tokens fire
   strongly in this subspace*, and a bipolar subspace's opposite poles can both
   appear among the strongest. The text shown for each document is a
   ±`context_window` token window around its peak token, with the peak wrapped
   `<<…>>`.
   (Because the model is decoder-only, the peak token's activation only
   reflects tokens at or before it; the right half of the window is shown
   for human readability and did not influence the projection.)
4. Calls an LLM judge to **derive** a hypothesis from the webtext +
   phase-1 evidence — without seeing the user-supplied significance
   description (`judge.derive_hypothesis`).
5. Calls the LLM a second time to **reconcile** the derived hypothesis
   against the provided significance description
   (`judge.reconcile_hypotheses`). Up to `max_reconciliation_iterations`
   framings are tried; the first `confirmed` short-circuits.
6. Writes the bundle (`bundle.py`).

## Judge-independence invariant

Three layers prevent the derivation call from seeing the provided
significance description:

| Layer | Where |
|---|---|
| Type — derive does not accept `Significance` | `judge.derive_hypothesis` signature |
| Schema — `Step1Summary` carries no significance fields | `schemas.Step1Summary` |
| Runtime — substring guard on the rendered derive prompt | `causalab.methods.llm_judge.assert_no_forbidden_substrings` |

The runtime guard takes a caller-supplied list of forbidden substrings —
typically `Significance.non_empty_values()` — and raises
`ForbiddenSubstringError` on a hit. Smoke tested in
`tests/test_characterize_subspace_judge.py`.

## Output bundle layout

```
${out_dir}/
├── metadata.json
├── refined_hypothesis.json       # consumed downstream
├── evidence.safetensors          # projections (webtext + step1)
├── evidence.meta.json            # spans + provenance
├── reconciliation_trace.json     # per-iteration framing + verdict
├── figures/
│   ├── projection_distribution.html   # static: webtext max-token distribution
│   ├── step1_vs_webtext.html          # static: step-1 vs webtext overlay
│   └── projection_explorer.html       # interactive: histogram ↔ context windows ↔ 3D PCA
└── more_info_needed.json         # written iff the gate fails (mutually exclusive
                                  # with refined_hypothesis.json)
```

`projection_explorer.html` is a single linked app over the per-document
peak-norm representation the judge sees: a clickable histogram of each doc's
peak-token subspace-activation norm (`‖peak_kdim‖₂`) where clicking a bin both
lists *all* of that bin's marked context windows in a side panel **and**
renders a 3D PCA scatter of the *same* documents (coloured by that same norm,
with a colorbar; hover shows the line-wrapped window and norm). The bin range,
panel value, scatter colour, and hover magnitude are one quantity, so the list
and scatter always show the same set. The per-document data embedded into it is
capped at `max_docs_embedded` (decile-stratified sample, logged when applied);
`evidence.safetensors` keeps the full `(N, k)` set.

## Terminal states

| Verdict | Meaning | Downstream action |
|---|---|---|
| `confirmed` | Derived hypothesis agrees with the provided one. | Downstream uses the refined hypothesis. |
| `refined` | Derived hypothesis sharpens or broadens the provided one along the same axis. | Downstream uses the refined hypothesis. |
| `disagreed` | Derived hypothesis names a different axis. | User decides whether to plan experiments anyway. |
| `unresolved` | Reconcile loop exhausted without a clean answer. | User reviews evidence, may rerun with different framings or webtext sample. |
| `insufficient_handoff` | Reproduction gate failed before the judge ran. | User addresses items in `more_info_needed.json` (see "Failure mode" below). |

## Failure mode — `more_info_needed.json`

Written when the reproduction gate fails (e.g. phase-1 projections show
no variance, no separation along the subspace's leading direction, or
the topology metric falls below threshold). Schema:

```json
{
  "status": "insufficient_handoff",
  "failed_metrics": [{"name", "value", "threshold", "axis", "notes"}],
  "missing_inputs": [{"field", "why"}],
  "actionable_requests": [str, ...],
  "diagnostic": str,
  "skipped_metrics": [str, ...],
  "step1_summary": {...}
}
```

When this file is present, `refined_hypothesis.json` is absent; the
downstream consumer branches on file existence.

## Configuration

See `causalab/configs/analysis/characterize_subspace.yaml`. Required
fields (no defaults): `subspace.model`, `subspace.layer`,
`subspace.step1_dataset`, plus **exactly one** of `subspace.artifact` or
`subspace.source` (see below). Webtext, metrics, and judge sections ship
sensible defaults.

## Supplying the subspace

Two ways, mutually exclusive:

1. `subspace.artifact` — a ready `.safetensors` rotation matrix
   (`loading.load_subspace` inspects key + orientation).
2. `subspace.source` — an SAE feature cluster
   `{sae_checkpoint, clusters_path, cluster_id, orthonormalize}`. The
   analysis builds the rotation under its output dir from the cluster's
   decoder directions (`methods.sae.decoder_subspace`, orthonormal basis by
   default) before loading it. Build provenance (feature ids, shapes) is
   recorded in `metadata.json`.

The producer is also a standalone CLI:

```bash
uv run python -m causalab.analyses.characterize_subspace.subspace_builder \
    --sae-checkpoint /path/to/sae.pt \
    --clusters /path/to/clustered_sae_latent_semantic_labels.json \
    --cluster-id 950 \
    --out /path/to/subspace.safetensors
```

## Supplying the subspace from a manifold-SAE bundle

A manifold-SAE-autointerp run is a *bundle* of per-community records, not a
single `.safetensors`. To characterize one community, first run the
`manifold_bundle_ingest` analysis — it turns `(bundle_dir, community_id)` into a
complete input set (rotation safetensors + `step1_dataset.json` + a manifest)
with no hand-editing — then point this analysis's `subspace.artifact`,
`step1_dataset`, and `significance` at the emitted manifest. See
`causalab/analyses/manifold_bundle_ingest/README.md` (issue #265), which maps a
`{"manifold_bundle": ..., "community_id": ...}` pointer to a complete input set.

## Auth

The judge calls go through OpenRouter by default. Set
`OPENROUTER_API_KEY` in the environment. For native OpenAI, set
`judge.provider: openai` in the config and `OPENAI_API_KEY` in the
environment.

The runner auto-loads a `.env` at the repo root on entry
(`causalab.runner.env.load_project_dotenv`), so the canonical place for the key
is `.env` — copy `.env.example`, fill it in, and it is picked up automatically
on every run (including `--slurm` jobs, which read the same shared-FS `.env`).
No manual `export`, `source`, or `sed` parsing is needed. An explicitly
exported key still takes precedence over the file. The credential check is
fail-fast: a missing key errors in seconds, before the model + webtext work.

## Entry points

- Direct invocation via the Hydra runner: same path as other shipped
  analyses (`scripts/run_exp.sh <runner-config-name>`).

## Limitations (first cut)

- Only `site: residual` is implemented. `attn-out` / `mlp-out` raise
  `NotImplementedError`; they need forward hooks.
- `step1_dataset` must be a JSON path. Task-name resolution is TODO.
- Figure-based reproduction metrics (Procrustes, KDE JS divergence)
  are not yet implemented; the side-car schema for figure points is
  still open. Tracked in `more_info_needed.skipped_metrics` on output.
