# Workflow script template

`experiment.json` measures an expected-answer logit. `workflow.json` runs that
protocol and then invokes the repository-local `summarize.py` script, which writes
a one-row summary table.

Copy both files into a research directory, update the protocol path and expected
columns, then run `causalab validate` and `causalab explain` before execution.
