"""Hydra analysis: certify how counterfactual datasets distinguish hypotheses.

The entry point is :func:`causalab.analyses.develop_hypothesis.main.main`
(``main(cfg)``), dispatched by the runner when a config carries
``_name_: develop_hypothesis``. It reads ``cfg.develop_hypothesis.*``, is
task-less and model-less (CPU, causal-model level), and delegates the heavy
lifting to ``causalab.causal.causal_utils.distinguishability_report``.
"""
