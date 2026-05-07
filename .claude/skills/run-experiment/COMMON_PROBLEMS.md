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
- **How to detect**: Run the task's `tests/test_with_model.py` (or the template at `.claude/skills/setup-task/templates/tests/test_with_model.py`), which includes a token alignment check (Test 3). The `baseline` analysis also exposes the mismatch via `counterfactual_sanity.json` once it runs.
- **How to fix**: Remove the leading space from the `raw_output` compute function (e.g. change `" " + str(sum)` to `str(sum)`).


