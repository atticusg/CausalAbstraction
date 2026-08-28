# Define a reusable method

A method is the transferable half of an intervention protocol. It defines the
experimental logic while leaving the model, dataset, and task-specific site
addresses open. Methods are JSON documents under `causalab/configs/methods/`, not
Python modules.

Read the method and application section of
[`../../../docs/intervention_protocol.md`](../../../docs/intervention_protocol.md)
before authoring one.

## 1. Decide whether a method is useful

Create a method only when the same intervention and scoring logic should transfer
across several applications. Keep a one-off experiment as a flat protocol.

Write down:

- the causal question;
- required input roles, usually `base` and optionally `counterfactual`;
- named sites and which address fields an application must supply;
- reads and interventions;
- metrics and outputs;
- optional featurizers and training behavior.

Do not put a model or dataset in the method. Do not fix a layer, position, or
component unless that value is intrinsic to the method.

## 2. Start from the template

Copy [`templates/method.json`](templates/method.json) or the shipped
`causalab/configs/methods/interchange.json`. Keep the required section order. A
standalone method has `"type": "method"`.

The method must contain `reads` and `save`. A site may contain only the fields the
method fixes. Missing address fields become the method's signature.

## 3. Inspect the signature

```bash
uv run causalab explain path/to/method.json
uv run causalab validate path/to/method.json
uv run causalab digest path/to/method.json
```

`explain` prints every model, data role, and site field that an application must
supply. Refine the method until that signature expresses the intended interface.

## 4. Build a complete run

Copy [`templates/run.json`](templates/run.json). Its `application` supplies the
model, data, and open site fields. Its `method` may be the relative path to the
method file.

An application may complete a method, but it may not override it. The composed
document must digest identically to the same experiment written as one flat
protocol.

## 5. Validate and test

```bash
uv run causalab validate path/to/run.json --data-root data --data
uv run causalab explain path/to/run.json --data-root data
uv run pytest -q tests/protocol/test_method.py
```

Add a focused test when the new method relies on behavior not already covered by
the protocol corpus. If the method requires new protocol vocabulary, implement and
test that vocabulary first.

## 6. Ship

Commit the reusable method, at least one complete application, and any dataset or
artifact fixtures needed to validate it. Describe which fields are intentionally
fixed and which remain open.
