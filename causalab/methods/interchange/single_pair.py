"""Single-pair residual stream trace via interchange interventions."""

import logging
from typing import Dict, Any, List

from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.activations.targets import build_residual_stream_targets

logger = logging.getLogger(__name__)


def _trace_from_prompt(text: str) -> CausalTrace:
    """Build a ``raw_input``-only :class:`CausalTrace` from a bare prompt."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _flatten_indices(idxs: Any) -> List[int]:
    """Flatten a (possibly nested) index return into a flat ``list[int]``."""
    if isinstance(idxs, int):
        return [idxs]
    flat: List[int] = []
    for x in idxs:
        if isinstance(x, (list, tuple)):
            flat.extend(_flatten_indices(x))
        else:
            flat.append(int(x))
    return flat


def _positions_in_bounds(
    pipeline: LMPipeline,
    prompt: str,
    counterfactual_prompt: str,
    token_positions: List[TokenPosition],
) -> tuple[List[TokenPosition], List[str], int, int]:
    """Keep only positions the base *and* counterfactual can both supply.

    The trace patches ``"sources->base"`` — the counterfactual (source) residual
    at position ``p`` is gathered and written into the base at ``p``.
    ``get_list_of_each_token`` builds positions whose index ``p`` is fixed to the
    *base* tokenization and ignores its input, so when the counterfactual
    tokenizes shorter than the base, ``p`` exceeds the counterfactual's length
    and the source-side gather indexes out of bounds — a CUDA scatter/gather
    assertion that poisons the context (#176). Such positions have no source
    token to patch, so they are not traceable: drop them.

    Both prompts are loaded as single-example batches (no padding), so the
    ``attention_mask`` sum is the exact in-bounds count for the unpadded-frame
    indices interventions use. Resolving each position with the same
    ``is_original`` flags the interchange uses keeps paired/combined positions
    correct while leaving fixed-index positions untouched.

    Returns ``(kept_positions, dropped_ids, base_len, counterfactual_len)``.
    """
    base_trace = _trace_from_prompt(prompt)
    counterfactual_trace = _trace_from_prompt(counterfactual_prompt)
    base_len = int(pipeline.load([base_trace])["attention_mask"][0].sum())
    counterfactual_len = int(
        pipeline.load([counterfactual_trace])["attention_mask"][0].sum()
    )

    kept: List[TokenPosition] = []
    dropped_ids: List[str] = []
    for tp in token_positions:
        base_idxs = _flatten_indices(tp.index(base_trace, is_original=True))
        counterfactual_idxs = _flatten_indices(
            tp.index(counterfactual_trace, is_original=False)
        )
        if all(0 <= i < base_len for i in base_idxs) and all(
            0 <= i < counterfactual_len for i in counterfactual_idxs
        ):
            kept.append(tp)
        else:
            dropped_ids.append(tp.id)
    return kept, dropped_ids, base_len, counterfactual_len


def run_single_pair_trace(
    pipeline: LMPipeline,
    prompt: str,
    counterfactual_prompt: str,
    token_positions: List[TokenPosition],
    layers: List[int] | None = None,
    verbose: bool = True,
    source_pipeline: LMPipeline | None = None,
) -> Dict[str, Any]:
    """Trace information flow through residual stream interventions.

    Runs interchange interventions at each (layer, token_position) combination
    using a single prompt/counterfactual pair.  Returns raw intervention outputs
    (no scoring or metric).

    Args:
        pipeline: Target LMPipeline where interventions are applied.
        prompt: Original prompt text.
        counterfactual_prompt: Counterfactual prompt text.
        token_positions: List of TokenPosition objects to analyze.
        layers: Specific layers to analyze (default: all layers including -1).
        verbose: Whether to print progress information.
        source_pipeline: If provided, collect activations from this pipeline
            instead of the target pipeline.  Enables cross-model patching.

    Returns:
        Dictionary containing:
            - intervention_results: raw intervention outputs keyed by (layer, pos_id)
            - metadata: experiment configuration
            - token_positions: the positions actually traced, after dropping any
              the counterfactual can't supply (#176). Use these — not the input
              list — to label axes so they match the traced cells.
    """
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    counterfactual_dataset: list[CounterfactualExample] = [  # type: ignore[assignment]
        {
            "input": {"raw_input": prompt},
            "counterfactual_inputs": [{"raw_input": counterfactual_prompt}],
        }
    ]

    num_layers = pipeline.model.config.num_hidden_layers
    if layers is None:
        layers = [-1] + list(range(num_layers))

    # Drop any position the counterfactual (source) can't supply — patching at a
    # nonexistent source token gathers out of bounds and triggers a CUDA
    # scatter/gather assertion that poisons the context (#176).
    token_positions, dropped_ids, base_len, counterfactual_len = _positions_in_bounds(
        pipeline, prompt, counterfactual_prompt, token_positions
    )
    if dropped_ids:
        logger.warning(
            "single_pair_trace: skipping %d/%d token position(s) outside the base "
            "(%d tokens) or counterfactual (%d tokens) — no source token to patch "
            "there (#176): %s",
            len(dropped_ids),
            len(dropped_ids) + len(token_positions),
            base_len,
            counterfactual_len,
            dropped_ids,
        )

    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    intervention_results = {}
    for (layer, pos_id), target in tqdm(
        targets.items(), desc="Running interventions", disable=not verbose
    ):
        outputs = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=counterfactual_dataset,
            interchange_target=target,
            batch_size=1,
            output_scores=False,
            source_pipeline=source_pipeline,
        )
        intervention_results[(layer, pos_id)] = outputs

    model_name = getattr(pipeline, "model_or_name", None)
    token_position_ids = [pos.id for pos in token_positions]
    metadata = {
        "model": model_name,
        "prompt": prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "num_layers": num_layers,
        "layers_used": layers,
        "num_token_positions": len(token_positions),
        "token_position_ids": token_position_ids,
        "total_interventions": len(layers) * len(token_positions),
    }

    return {
        "intervention_results": intervention_results,
        "metadata": metadata,
        # The effective positions after dropping any the counterfactual can't
        # supply (#176). Consumers must use these — not the input list — to label
        # axes so the heatmap matches the traced cells.
        "token_positions": token_positions,
    }
