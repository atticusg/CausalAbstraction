"""Hydra analysis: exploratory evidence about a model's algorithm on a task.

The entry point is :func:`causalab.analyses.exploration.main.main` (``main(cfg)``),
dispatched by the runner when a config carries ``_name_: exploration``. It reads
``cfg.exploration.*`` (a ``mode`` plus that mode's inputs) and ``cfg.model``; it
is **task-less** (its inputs are raw, hand-authored prompts/manifests, not a
runner-generated task dataset). The four modes live in sibling modules, each
exposing ``run(pipeline, acfg, out_dir)``:

* :mod:`causalab.analyses.exploration.probe_prompts` — ``mode: probe``.
* :mod:`causalab.analyses.exploration.logit_lens_inputs` — ``mode: logit_lens``.
* :mod:`causalab.analyses.exploration.pair_trace` — ``mode: pair``.
* :mod:`causalab.analyses.exploration.pca_critical_tokens` — ``mode: pca``.
"""
