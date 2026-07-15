"""Curation sweep for ``subject_object_relations``: per-relation base accuracy.

Loads the golden fixture model ONCE — ``chat-coherent`` = Qwen/Qwen3-4B-Instruct-2507
with the chat template + the golden answer directive — and measures 64-example base
accuracy for every bundled relation via the *same* building blocks the baseline
runner uses (``load_pipeline`` → ``generate_datasets`` → ``compute_base_accuracy``),
so the numbers track what the golden tier would see. Also records, per relation, the
single-token-decodability and first-token-distinctness of the answer space.

The printed table is transcribed into ``README.md``'s curation section; relations
clearing the gate (accuracy ≥ threshold AND first-token-distinct) are eligible for
the smoke / golden tiers.

GPU. Run via SLURM (a few minutes for 35 relations at ``max_new_tokens=1``)::

    uv run python causalab/tasks/subject_object_relations/data/curation_sweep.py \
        --out /tmp/soc_curation.json --n 64 --seed 0
"""

from __future__ import annotations

import argparse
import json

from causalab.io.pipelines import load_pipeline
from causalab.methods.metric import compute_base_accuracy
from causalab.runner.helpers import generate_datasets
from causalab.tasks.loader import load_task
from causalab.tasks.subject_object_relations.config import (
    SubjectObjectRelationsConfig,
    relation_names,
)

# Matches tests/end_to_end/configs/model/chat-coherent.yaml exactly.
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ANSWER_DIRECTIVE = (
    "Reply with only the final answer word and nothing else. "
    "Do not restate the question."
)


def _token_stats(pipeline, objects: list[str]) -> tuple[float, bool]:
    """Single-token fraction and first-token distinctness of the answer space."""
    tok = pipeline.tokenizer
    first_ids: list[int | None] = []
    n_single = 0
    for obj in objects:
        ids = tok.encode(" " + obj, add_special_tokens=False)
        if len(ids) == 1:
            n_single += 1
        first_ids.append(ids[0] if ids else None)
    single_frac = n_single / len(objects) if objects else 0.0
    first_distinct = len(set(first_ids)) == len(first_ids)
    return single_frac, first_distinct


def _load_task(relation: str):
    task = load_task(
        "subject_object_relations",
        task_cfg=SubjectObjectRelationsConfig(relation=relation),
    )
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/soc_curation.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    rels = relation_names()
    pipeline = load_pipeline(
        MODEL,
        task=_load_task(rels[0]),
        max_new_tokens=1,
        device="auto",
        dtype="bfloat16",
        use_chat_template=True,
        chat_answer_directive=ANSWER_DIRECTIVE,
    )

    results: dict[str, dict] = {}
    header = (
        f"{'relation':35s} {'group':11s} {'nobj':>4s} {'stok':>5s} "
        f"{'ftd':>5s} {'n':>4s} {'acc':>6s} {'pacc':>6s} {'verdict':>8s}"
    )
    print(header)
    for rel in rels:
        task = _load_task(rel)
        model = task.causal_model
        objects = model.values["object"]
        single_frac, first_distinct = _token_stats(pipeline, objects)

        n = min(args.n, model.n_unique_inputs)
        train, _ = generate_datasets(
            task,
            n_train=n,
            n_test=0,
            seed=args.seed,
            enumerate_all=False,
            resample_variable="all",
        )
        acc = compute_base_accuracy(
            train,
            pipeline,
            task.checker,
            batch_size=args.batch_size,
            answer_fn=task.score_answer,
        )
        accuracy = float(acc["accuracy"])
        pacc = acc.get("prob_accuracy")
        verdict = (
            "green" if (accuracy >= args.threshold and first_distinct) else "flagged"
        )
        results[rel] = {
            "group": task.causal_model._in_config.group,  # type: ignore[attr-defined]
            "n_objects": len(objects),
            "single_token_frac": round(single_frac, 3),
            "first_token_distinct": first_distinct,
            "n_eval": len(train),
            "accuracy": round(accuracy, 4),
            "prob_accuracy": None if pacc is None else round(float(pacc), 4),
            "verdict": verdict,
        }
        pacc_s = "  n/a " if pacc is None else f"{float(pacc):6.3f}"
        print(
            f"{rel:35s} {str(results[rel]['group']):11s} {len(objects):4d} "
            f"{single_frac:5.2f} {str(first_distinct):>5s} {len(train):4d} "
            f"{accuracy:6.3f} {pacc_s} {verdict:>8s}"
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    green = sorted(r for r, v in results.items() if v["verdict"] == "green")
    print(
        f"\n{len(green)}/{len(rels)} relations green "
        f"(accuracy >= {args.threshold} AND first-token-distinct)."
    )
    print("green:", ", ".join(green) if green else "(none)")
    # Best green candidate for the golden tier: highest accuracy, then most objects.
    if green:
        best = max(
            green,
            key=lambda r: (results[r]["accuracy"], results[r]["n_objects"]),
        )
        print(f"golden candidate: {best} (acc={results[best]['accuracy']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
