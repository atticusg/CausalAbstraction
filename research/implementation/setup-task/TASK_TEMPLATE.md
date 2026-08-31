# Task specification

## Research purpose

- Behavior to study:
- Why causal abstraction is useful:
- Alternatives the task should distinguish:

## Inputs and outputs

- Input variables and allowed values:
- Prompt form:
- Output variable and allowed values:
- Surface forms for every output value:
- Exact or prefix matching:

## Causal model

For every variable, state its parents, mechanism, value domain, and whether it is a
candidate intervention target. Explain how each hypothesized intermediate affects
`raw_output`.

## Counterfactual datasets

For every generator, state:

- what changes and what remains fixed;
- which hypotheses it should distinguish;
- whether it is wide, narrow, or changes one input token;
- intended train or evaluation role;
- balancing and validity constraints.

## Configuration

- Singleton or factory task:
- Factory fields and defaults:
- Normal target variable:
- Named generator used for the broad random sample:

## Serialized table

- Dataset reference:
- Required columns beyond the standard serializer output:
- Build command and seed:
- Determinism check:

## Verification

- Mechanism totality tests:
- Output-form coverage:
- Generator validity and determinism:
- Distinguishability expectations:
- Protocol used for an end-to-end smoke test:
