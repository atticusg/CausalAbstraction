# Running the weekdays pipeline — YAML-first

**Every hyperparameter lives in YAML, sweep axes included.** The experiment
files (`exp_locate.yaml`, `exp_subspace.yaml`) carry the config-group
selection, the fixed values, and the sweep axes (`hydra.sweeper.params`), and
set `hydra.mode: MULTIRUN` so even `-m` is not needed on the CLI. Sharing this
directory is the complete experiment spec; the commands below take no
hyperparameters.

## Stage 1 — locate: layer × position scan (64 cells, 64 digests)

```sh
causalab run --config-dir . --config-name exp_locate
```

On the cluster the same product goes through `runner/fanout.py` as one slurm
array job; locally, `run([protocols])` may fuse cells that differ only
off-model. Collect the scan into a `locate.result@1` artifact
({best_layer, best_pos, grid}).

## Stage 2 — subspace at the best cell: k × seed sweep (12 runs)

```sh
causalab run --config-dir . --config-name exp_subspace
```

`exp_subspace.yaml` pulls the located cell through the `${artifact:...}`
resolver (materialization doc; simplification doc sec. 6.4) — the layer is
stated in NO config file and on NO command line, unlike the shipped
`runners/weekdays/weekdays_8b_pipeline.yaml`, which states `layers: [28]`
three times. A missing stage-1 artifact is a compose-time error, not a silent
default.

## Method axis — DBM alongside DAS

Add it to the experiment file, like any other axis:

```yaml
hydra:
  sweeper:
    params:
      protocol: das_composition,dbm_composition
```

(a `dbm_*.yaml` twin of `das_*.yaml` — same shape as `05_dbm_*.yaml` one
directory up.)

## CLI overrides — possible, not default

Any value can still be overridden ad hoc; overrides compose on top of the
experiment file and are for exploration, never the record:

```sh
causalab run --config-dir . --config-name exp_locate 'scan.layer=range(8,24)'
```

An exploratory override that turns out to matter gets promoted into the
experiment file (or a new one) — the file is the source of truth. `config.yaml`
stays as the interactive single-point entry for poking at one cell.

## Reproducibility chain

Three layers, all YAML: the **experiment file** (selection + fixed values +
sweep axes) → the **composed point protocol** per cell (concrete after
interpolation) → the **canonical stamped protocol + digest** each run writes
next to its results. Sharing layer 1 reproduces the campaign; layer 3 makes
every individual artifact independently replayable even if the experiment file
is lost.

## Downstream pipeline stages

`activation_manifold`, `output_manifold`, `path_steering`, `pullback` from the
shipped pipeline are tier-3 analyses (simplification doc sec. 6): programs that
consume these stages' artifacts and this scan's harvests. They are launched per
stage, not composed into the protocol.
