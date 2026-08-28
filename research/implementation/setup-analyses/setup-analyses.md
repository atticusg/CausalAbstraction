# Add deterministic analysis to a workflow

Model execution belongs in an intervention protocol. Deterministic processing of
saved outputs belongs in a workflow `script` step. The retired Python analysis and
Hydra configuration interface should not be used.

Read the script-step section of
[`../../../docs/workflow_protocol.md`](../../../docs/workflow_protocol.md) before
adding code.

## 1. Decide whether Python is necessary

Do not add a script for behavior already expressed by protocol vocabulary:
activation reads, interventions, metrics, featurizers, training, and sweeps belong
in the protocol document.

Use a script for deterministic calculations over files, such as a fit, statistical
test, selection, table transformation, or figure. LLM and human judgments stay
outside CausaLab's deterministic workflow record.

## 2. Choose where the script lives

- Put reusable numerical code in the relevant package, usually
  `causalab/analysis/`.
- Put reusable plotting code under `causalab/io/plots/`.
- Put workflow wiring helpers under `causalab/workflow/scripts/`.
- Put research-specific code beside its workflow and reference it with a relative
  `path` locator.

Loaders locate and hash scripts without importing them. A script must define:

```python
def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    ...
```

It must create every declared output.

## 3. Define inputs and outputs

Workflow inputs may be literals, paths, or references to an earlier step's files.
Those references create dependency edges. JSON outputs are either tables or values
objects; dense tensors use safetensors.

Declare table columns or representative keys when a downstream step needs to
validate the shape before the producer runs. Use `causalab.io.step_io` helpers to
read and write supported formats.

The runner stamps safetensors outputs with inherited identity and the producing
step digest. Scripts should declare only identity fields that only they know, such
as a fitted rank.

## 4. Add the workflow step

Copy [`templates/workflow.json`](templates/workflow.json) and
[`templates/summarize.py`](templates/summarize.py). A module locator looks like
`{"module": "causalab.analysis.fit_pca"}`. A repository-local script locator
looks like `{"path": "scripts/summarize.py"}` and is relative to the workflow
document.

Use `runtime.isolate` only when the script genuinely needs dependencies that
differ from the runner. Dependency specifications and passed environment variable
names become part of the workflow digest.

## 5. Test

Test the function directly with temporary input and output paths. Then validate and
run the smallest workflow that crosses both boundaries you use:

```bash
uv run causalab validate path/to/workflow.json --data-root data --data
uv run causalab explain path/to/workflow.json --data-root data
uv run causalab run path/to/workflow.json --data-root data --out /tmp/runs
```

Run the existing script-step integration test when the change affects shared
workflow behavior:

```bash
uv run pytest -q tests/neural/engines/pytorch_hooks/test_script_step_run.py
```

## 6. Review the record

Confirm that the step directory contains every declared file and `_step.json`, and
that `workflow.json` records the script path, content hash, digest, and completion
status. When a later protocol loads a tensor output, verify its artifact identity
as part of the test.
