"""Shape and dtype test for {{METHOD_NAME}}.

Lives at ${SESSION_DIR}/code/methods/{{METHOD_NAME}}/tests/test_{{METHOD_NAME}}.py.

The first version of this test asserts that calling the method raises
``NotImplementedError`` so the file passes immediately after scaffolding. When
implementation begins (Step 4 of /setup-methods), flip the body to a real shape
and dtype check on randomly-generated inputs that match the signature in
set_up_method.md §2, then iterate until green.
"""

from __future__ import annotations

import pytest

from methods.{{METHOD_NAME}} import {{METHOD_NAME}}


def test_{{METHOD_NAME}}_shape_and_dtype():
    """Smoke test — flip from `pytest.raises(NotImplementedError)` to a real
    shape check once the body is implemented.
    """
    # TODO: replace this with a real call once {{METHOD_NAME}} is implemented.
    # Example shape (adapt to set_up_method.md §2 for this specific method):
    #
    #     activations = torch.randn(4, 12, 768)
    #     target_token_ids = torch.randint(0, 50000, (4,))
    #     out = {{METHOD_NAME}}(
    #         activations=activations,
    #         target_token_ids=target_token_ids,
    #         layer=8,
    #         head=3,
    #     )
    #     assert out["logit_diff"].shape == (4,)
    #     assert out["logit_diff"].dtype == torch.float32
    with pytest.raises(NotImplementedError):
        {{METHOD_NAME}}()  # type: ignore[call-arg]
