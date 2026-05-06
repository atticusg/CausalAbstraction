"""{{METHOD_NAME}}: {{ONE_LINE_PURPOSE}}

{{LONGER_PURPOSE_PARAGRAPH_FROM_SPEC}}

This is a *method* (interpretability primitive) — see `ARCHITECTURE.md` §3.
Layering rules respected by this module:
  - imports only from causalab/{neural,methods,io,causal,tasks}/ and third-party libs
  - no imports from causalab/runner/ or causalab/analyses/
  - no hyperparameter defaults (the consuming analysis's Hydra config supplies them)
  - no disk I/O (the consuming analysis decides where results land)
"""

from __future__ import annotations

# {{IMPORTS_FROM_SPEC_SECTION_3}}
# Example shape — replace with the actual imports listed in set_up_method.md §3.
# import torch
# from causalab.neural.pipeline import LMPipeline


def {{METHOD_NAME}}(
    # {{POSITIONAL_INPUTS_FROM_SPEC_SECTION_2}}
    *,
    # {{HYPERPARAMETERS_FROM_SPEC_SECTION_4_NO_DEFAULTS}}
):  # type: ignore[no-untyped-def]
    """{{ONE_LINE_PURPOSE}}

    Parameters
    ----------
    {{PARAM_DOCS_FROM_SPEC_SECTIONS_2_AND_4}}

    Returns
    -------
    {{RETURN_STRUCTURE_FROM_SPEC_SECTION_2}}
    """
    raise NotImplementedError(
        "{{METHOD_NAME}} not yet implemented. "
        "See set_up_method.md alongside this file for the spec."
    )
