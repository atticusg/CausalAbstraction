# Build and run experiments

Run commands from the CausaLab repository root.

## 1. Choose a starting document

Use the closest shipped example:

- `causalab/configs/protocols/` contains complete protocol documents.
- `causalab/configs/methods/` contains reusable method halves.
- `causalab/configs/runs/` contains complete documents written as an application
  plus a method.
- `causalab/configs/workflows/` contains pipelines of protocol and script steps.

Copy a document into the research worktree when an experiment needs durable
changes. Use `--set path=value` only for quick exploration; a meaningful run
should be represented by a reviewable document.

## 2. Build the dataset table

Protocol documents read serialized JSON tables. For a shipped task:

```bash
uv run python scripts/build_task_dataset.py \
  --task natural_domains_arithmetic \
  --set domain_type=weekdays \
  --n 300 --seed 0 \
  --target-variable result \
  --out data/weekdays/train.json
```

The command also writes `train.manifest.json` with build provenance. Re-run the
same command with `--check` to confirm that the table bytes remain deterministic.

## 3. Validate before loading a model

```bash
uv run causalab validate experiment.json --data-root data --data
uv run causalab explain experiment.json --data-root data
uv run causalab digest experiment.json --data-root data
```

`validate --data` checks referenced columns. `explain` reports sweep axes, point
count, required engine capabilities, forward groups, products, and digests.
`digest` prints the campaign identity.

## 4. Run

```bash
uv run causalab run experiment.json \
  --data-root data \
  --artifacts-root runs \
  --out runs \
  --device cuda \
  --dtype bf16
```

The same command accepts a workflow document. A workflow creates
`runs/<output_dir>/`, one directory per step, and `workflow.json`. A protocol run
writes `protocol.json` before execution so a failed run still records what it
attempted.

## 5. Run a sweep in parts

First inspect the deterministic point order with `explain`. Then distribute
non-overlapping half-open ranges:

```bash
uv run causalab run experiment.json ... --points 0:16
uv run causalab run experiment.json ... --points 16:32
```

Point ranges do not change campaign or point digests. Scheduling these commands is
an infrastructure concern outside CausaLab.

## 6. Resume a workflow

```bash
uv run causalab run workflow.json \
  --data-root data --artifacts-root runs --out runs \
  --device cuda --resume
```

The runner reuses a completed step only when its digest and declared outputs
match. Editing a protocol, script, input, or dependency invalidates the affected
step. A step declared nondeterministic is not reused unless
`--reuse-nondeterministic` is supplied.

## 7. Inspect results

- Read `protocol.json` or `workflow.json` first.
- Read each workflow step's `_step.json` before consuming its files.
- JSON metric files are arrays of row objects and include sweep coordinates.
- Safetensors bundles may contain several coordinate-keyed entries.
- Use the identity metadata when passing a fitted tensor into another protocol.

Do not infer that a merged document has run. The output tree and its records are
the evidence that execution occurred.
