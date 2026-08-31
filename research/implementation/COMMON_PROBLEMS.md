# Common problems

## A dataset reference does not resolve

`data.*.dataset` is relative to `--data-root` and omits the `.json` suffix. Build
the table first and confirm that the resulting path matches the reference. Use
`causalab validate ... --data` to check both the file and referenced columns.

## A metric names a missing answer column

Expected answers and equivalent surface forms are task data. Add them while
serializing the task table; do not compute them in the protocol. Inspect one JSON
row and compare its columns with the metric's `expected`, `a`, `b`, or `token`
field.

## A token position resolves incorrectly

Use a variable or column position when the location changes by row. Fixed indexes
are rebased after a chat prefix and negative indexes count from the end. Run
`validate --data` so column positions are checked. An absent or repeated prompt
variable is an authoring error, not a position to guess around.

## A sweep creates too many points

Every explicit axis participates in a cross product. Put the sweep on a named site
or position that all dependent reads and writes reference. If two fields must move
together, generate the desired documents or points instead of declaring two
independent axes. Inspect the point count with `explain` before running.

## A method will not compose with an application

An application completes a method; it never overrides it. Run `causalab explain`
on the method alone to see its open signature. Supply exactly those model, data,
and site fields in the application. Move a disputed value into the application if
it is not truly part of the reusable method.

## The engine refuses a capability

`explain` lists requirements derived from the document. Select an installed engine
that advertises all of them or implement the missing capability. Do not silently
drop a read, write, training section, or precision requirement to make the document
run.

## A fitted artifact is rejected

The consuming model, site, dtype, featurizer parameters, and selected bundle entry
must agree with the safetensors identity. Read the producer's `_step.json`, the
bundle metadata, and the consuming declaration. A mismatch normally means the
wrong point or artifact was selected.

## A script step did not produce its outputs

The script must create every path in its `outputs` mapping. JSON tables must match
any declared columns, values objects must match declared keys, and tensor outputs
must use safetensors. Run the script's unit test directly, then inspect the step's
error and `_step.json`.

## Resume reruns a step

Resume is digest-based. Changes to a protocol, script contents, inputs, overrides,
or upstream products correctly invalidate reuse. Compare the old and new step
records instead of forcing reuse. Nondeterministic steps require the explicit
`--reuse-nondeterministic` option.

## Results exist but provenance is unclear

Treat files without `protocol.json`, `workflow.json`, or `_step.json` as
unverified. The run records and content digests are part of the result, not optional
logging.
