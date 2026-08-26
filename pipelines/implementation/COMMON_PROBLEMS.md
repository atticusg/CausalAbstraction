# Common Problems and Solutions

This document describes common problems encountered during experiments and how to resolve them.

## Import Errors

### TokenPosition import fails
- **Error**: `ModuleNotFoundError: No module named 'causalab.neural.token_position'`
- **Cause**: TokenPosition is in `token_position_builder.py`, not `token_position.py`
- **Fix**: Use `from causalab.neural.token_position_builder import TokenPosition`

### pipeline.tokenize() doesn't exist
- **Error**: `AttributeError: 'LMPipeline' object has no attribute 'tokenize'`
- **Cause**: LMPipeline doesn't have a `tokenize()` method
- **Fix**: Use `pipeline.tokenizer.encode(text, add_special_tokens=False)`

## Intervention Errors

### Variable-length token indices
- **Error**: `ValueError: expected sequence of length X at dim 1 (got Y)`
- **Cause**: Token positions return different numbers of indices across examples
- **Fix**: Ask the user what to do


## Unexpected Results

### Interventions produce garbage outputs (GPT-2 only)
- **Symptom**: Intervention outputs are nonsense like ' the beach.' instead of expected names
- **Symptom**: Patching final layer at final position gives 0% accuracy (should be ~100%)
- **Cause**: Missing `position_ids` for left-padded inputs
- **Explanation**: GPT-2 does not compute position IDs from the attention mask, so with left padding positions are incorrectly assigned as 0,1,2,3,4,5,6... instead of 0,0,0,0,17,18,19... Most newer models (Llama, Pythia, etc.) handle this automatically. HuggingFace's `generate()` also handles this, but pyvene's forward pass does not.
- **Fix**: Enable position_ids when creating the pipeline (GPT-2 only):
  ```python
  pipeline = LMPipeline(
      model_name,
      ...,
      position_ids=True,  # Required for GPT-2 with left-padded inputs
  )
  ```
- **Note**: Only needed for GPT-2. Other models compute correct position IDs from the attention mask automatically.

### Near-zero intervention accuracy
- **Symptom**: Residual stream patching shows ~0% accuracy
- **Possible causes**:
  1. Missing `position_ids=True` if using GPT-2 (see above)
  2. Wrong token position targeted
  3. There is a problem with the metric:
    * What we measure
    * How many tokens are being generated


## Task Setup Issues

### Space token prediction issue
- **Symptom**: Model predicts " answer" instead of "answer"
- **Cause**: Template doesn't have trailing space
- **Fix**: Add trailing space to template, remove leading space from expected output

### Token-level mismatch in raw_output (breaks token-level experiments)
- **Symptom**: Experiments that operate at the token level (loss functions, metrics, etc.) show ~20-30% accuracy, while string-level scoring looks normal
- **Root cause**: `raw_output` has a leading space (e.g. `" 13"`) but the template already ends with a trailing space, so the model generates `"13"` (no leading space). The two are different tokens.
- **Why checker doesn't catch it**: The checker uses `.strip()` to normalize whitespace, so `" 13"` and `"13"` compare as equal at the string level. But token-level operations see them as completely different tokens. Residual stream scoring uses the checker too, so it also looks fine.
- **The rule**: If the template ends with a space → `raw_output` must NOT have a leading space. If the template has no trailing space → `raw_output` SHOULD have a leading space (since the model will predict the space as part of its output).
- **How to detect**: First-line, model-free — from the causalab library checkout run `cd ~/.silico/libraries/causalab-internal && PYTHONPATH="$WORKDIR/code:$PYTHONPATH" uv run python -m causalab.tasks.preflight --task {task_name} --model {model_name}` (the shipped tokenizer pre-flight; no weights, sub-second; the `PYTHONPATH` prefix is only needed for a session-local task under `$WORKDIR/code`). It flags an orphan trailing-whitespace token directly and exits non-zero. For the model-confirmed version, run the task's `tests/test_with_model.py` (or the template at `setup-task/templates/tests/test_with_model.py`), which includes a token alignment check (Test 3).
- **How to fix**: Drop the trailing space from `prompt_suffix` / the template so the prompt ends on a non-whitespace character (the common case), or remove the leading space from the `raw_output` compute function (e.g. change `" " + str(sum)` to `str(sum)`).


## Workflow / Dispatch Issues

### Chained dispatch bypassed pre-flight
- **Symptom**: A downstream analysis (e.g. `locate`) ran on top of an upstream stage (e.g. `baseline`) that failed its gate — producing artifacts that are meaningless and waste GPU time.
- **Root cause**: The runner config chained the whole DAG (`defaults: [baseline, locate, …]`) and was dispatched in one call. Hydra ran every step back-to-back; the pre-flight gate was never checked *between* stages, so a 0.0-accuracy baseline didn't stop `locate`.
- **The rule** (see `running-experiments.md` § "The gate concept"): dispatch **one gate-bounded stage at a time**. Run the upstream node alone, read its gate artifact, compare to the gate's sign-of-life floor, and only then dispatch the downstream node — which auto-discovers the upstream artifacts from the shared `--experiment-root`. **Never** issue a single runner call whose `defaults:` list spans a gate boundary.
- **How to detect**: before a full run, do a quick 16-example debug pass (`task.n_train=16`) — it surfaces a dead baseline in ~2 min instead of after the full chain finishes. Inspect `baseline/accuracy.json` *before* queuing anything downstream.
- **Transcript** (@goodatticus, the session that motivated this rule):

  > I dispatched baseline → locate as a single chain in one runner call, so locate ran on top of a baseline showing 0.0 accuracy. The plan's §E gate explicitly says "baseline pre-flight fails → stop the chain", but I didn't check it between analyses — Hydra ran them both back-to-back. That wasted ~15 minutes of H200 time on meaningless intervention data. The right move was to run baseline alone first, gate on accuracy, then submit locate as a second invocation — or do a quick 16-example debug pass to surface the 0.0 accuracy in ~2 min.


