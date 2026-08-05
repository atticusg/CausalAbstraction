"""Path patching — isolate the *direct* sender→receiver edge effect.

Path patching perturbs a sender component from a counterfactual input while
freezing every other path to its clean-base value, so only the direct
sender→receiver residual channel carries the perturbation. It is the
direct-effect specialization of causalab's interchange intervention; see
``causalab/analyses/path_patching/README.md`` for the analysis that drives it.

Receivers:

* ``output`` (default) — the **direct effect on the logits** (the IOI Fig. 3
  head sweep), read from a single intervened forward.
* internal receivers — ``head_value_input`` (a head's value vector),
  ``mlp_input``, ``residual`` — measured by the two-pass collect/inject runner.

The restorer set — which component families are frozen between sender and
receiver — is the estimand definition, selectable via ``restore``.

Public surface:

* :class:`ReceiverSpec` / :data:`OUTPUT` / :func:`build_receiver_unit` — the receiver.
* :func:`build_restorer_set` / :func:`build_path_patching_target` — assemble the
  two-group sender/restorer :class:`InterchangeTarget`.
* :func:`run_path_patching` / :func:`run_path_patching_scan` — the runner
  (one-pass for ``output``, two-pass for internal receivers).
* :func:`run_path_patching_set` / :func:`run_path_patching_set_scan` — patch one
  sender into a *set* of internal receivers simultaneously (IOI Fig. 4 / Fig. 5).
* :func:`make_logit_diff_metric` — the direct-effect logit-difference metric.
"""

from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_path_patching_target,
    build_receiver_unit,
    build_restorer_set,
    deepest_receiver,
    group_path_patching_target,
    sender_reaches_any,
    sender_reaches_receiver,
)
from causalab.methods.path_patching.run import (
    run_path_patching,
    run_path_patching_scan,
    run_path_patching_set,
    run_path_patching_set_scan,
)

# Re-exported from the general metric module (it is not path-patching-specific);
# kept on the package surface so ``from causalab.methods.path_patching import
# make_logit_diff_metric`` continues to resolve.
from causalab.methods.metric import make_logit_diff_metric

__all__ = [
    "OUTPUT",
    "ReceiverSpec",
    "build_path_patching_target",
    "build_receiver_unit",
    "build_restorer_set",
    "deepest_receiver",
    "group_path_patching_target",
    "sender_reaches_any",
    "sender_reaches_receiver",
    "run_path_patching",
    "run_path_patching_scan",
    "run_path_patching_set",
    "run_path_patching_set_scan",
    "make_logit_diff_metric",
]
