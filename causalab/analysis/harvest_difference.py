"""``causalab.analysis.harvest_difference`` — a steering direction from two harvests.

```json
"direction": {
  "type": "script", "script": {"module": "causalab.analysis.harvest_difference"},
  "inputs": {"positive": {"step": "harvest_pos", "file": "acts.safetensors"},
             "negative": {"step": "harvest_neg", "file": "acts.safetensors"},
             "normalize": true},
  "outputs": {"weight": "direction.safetensors",
              "stats": {"file": "stats.json",
                        "columns": {"dim": "int64", "value": "float64"}}}
}
```

**This is the step type earning its keep.** "Harvest activations on two
contrasting corpora and subtract the means" is the direction half of every
steering experiment, and before this it was expressible nowhere: it touches no
network through the intervention vocabulary, so it is not a protocol document;
and it is a one-off reduction, so the old registry — which admitted ops only by
pull request — could not hold it either. The science was already in the research
protocol; only the invocation was missing.

The output is a ``(d,)`` direction in a ``weight``-slotted bundle, which a
downstream protocol step loads as a ``params`` constant and applies with an
additive ``do``. Its ArtifactIdentity is **inherited from both harvests** by the
runner, keeping only the fields they agree on — so a direction built from model
X at site S is bound to X and S, and the two harvests disagreeing about the site
drops the site rather than letting one of them speak for the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, write_table, write_tensor

__all__ = ["main"]


def _rows(tensor: Any, slot: str) -> Any:
    """Flatten leading dimensions: ``[examples, positions, d]`` and ``[rows, d]``
    both mean rows of ``d``. A ``(d,)`` mean-reduced harvest is one row."""
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim < 2:
        raise StepError(
            f"harvest_difference: {slot!r} has shape {tuple(tensor.shape)}; "
            "expected rows of a hidden dimension"
        )
    return tensor.reshape(-1, tensor.shape[-1])


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    from causalab.io.step_io import read_tensor

    def resolve(slot: str) -> Any:
        value = inputs[slot]
        if isinstance(value, (str, Path)):
            return read_tensor(Path(value), what=f"harvest_difference: {slot!r}")
        return value

    positive = _rows(resolve("positive"), "positive")
    negative = _rows(resolve("negative"), "negative")
    if positive.shape[-1] != negative.shape[-1]:
        raise StepError(
            f"harvest_difference: the two harvests have different widths "
            f"({positive.shape[-1]} vs {negative.shape[-1]}) — a difference "
            "needs one hidden dimension"
        )
    # float64 for the accumulation: a bf16 mean over thousands of rows loses
    # the low bits the difference is made of (the same reason save-time
    # `reduce` accumulates in fp32).
    delta = positive.to(torch.float64).mean(dim=0) - negative.to(torch.float64).mean(
        dim=0
    )
    norm = float(torch.linalg.vector_norm(delta))
    if bool(inputs.get("normalize", False)):
        if norm == 0.0:
            raise StepError(
                "harvest_difference: the two harvests have identical means, so "
                "there is no direction to normalize — a zero direction would "
                "steer nothing while looking like it steered"
            )
        delta = delta / norm

    write_tensor(outputs["weight"], delta.to(torch.float32), slot="weight")
    if "stats" in outputs:
        write_table(
            Path(outputs["stats"]),
            [{"dim": i, "value": float(v)} for i, v in enumerate(delta)],
        )
