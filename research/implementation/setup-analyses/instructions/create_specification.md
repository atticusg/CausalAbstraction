# Write a workflow script specification

Fill [`../SET_UP_ANALYSIS_TEMPLATE.md`](../SET_UP_ANALYSIS_TEMPLATE.md) before
writing code. Define the file interface first. Every input must be representable in
the workflow reference grammar, and every output must be JSON, safetensors, or an
explicit visualization format.

Explain why the operation is not already a protocol read, metric, intervention,
featurizer, training objective, or sweep. Record every source of nondeterminism.
Prefer a deterministic implementation; otherwise set `is_deterministic` to false
in the workflow and explain the consequence for resume.
