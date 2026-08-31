# Task package layout

The authoritative layout and export contract is
[`../../../../causalab/tasks/README.md`](../../../../causalab/tasks/README.md).

Create `causalab/tasks/<name>/` and begin from the closest shipped package. The
core is `causal_models.py`, `counterfactuals.py`, and `__init__.py`. Add a config
dataclass for a factory task, token-position compatibility helpers when callers
need them, and a custom checker only when `output_tokens` cannot express grading.

Task packages do not contain protocol documents, run configuration, task YAML,
model execution, or scheduler settings. Build their examples into JSON tables with
`scripts/build_task_dataset.py`; protocol documents reference those bytes through
their data root.
