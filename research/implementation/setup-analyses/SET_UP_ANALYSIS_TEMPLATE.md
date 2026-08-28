# Workflow script specification

## Purpose

What deterministic calculation does this script perform, and why is protocol
vocabulary insufficient?

## Inputs

For each input, state whether it is a literal, JSON table, values object, tensor
bundle, or filesystem path. Name any required table columns or tensor slots.

## Outputs

For each output, state its filename, format, columns or keys, and whether a later
protocol consumes it.

## Determinism and dependencies

- Sources of randomness and their declared seeds:
- External dependencies:
- Whether isolation is required:
- Environment variable names that must be passed:

## Provenance

- Identity inherited from tensor inputs:
- Identity fields only the script can declare:
- Downstream identity checks:

## Tests

- Direct unit test:
- Minimal workflow test:
- Expected failure cases:
