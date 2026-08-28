"""Emit the weekdays tables. Fully enumerated, so there is no seed to record:
7 base days x the counterfactual offsets assigned to each split."""

import json
import sys
import pathlib

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATE = "If today is {day}, tomorrow is"


def rows(offsets):
    out = []
    for k in offsets:
        for i, day in enumerate(DAYS):
            cf_day = DAYS[(i + k) % 7]
            out.append(
                {
                    "input": TEMPLATE.format(day=day),
                    "counterfactual_inputs": [TEMPLATE.format(day=cf_day)],
                    "answer": " " + DAYS[(i + 1) % 7],
                    "base_answer": " " + DAYS[(i + 1) % 7],
                    "cf_answer": " " + DAYS[(i + k + 1) % 7],
                    "label": " " + DAYS[(i + k + 1) % 7],
                    # Per-role prompt variables. A bare `subject` column would
                    # be looked up against BOTH roles' texts and the
                    # counterfactual names a different day, so a
                    # {"variable": "subject"} position would refuse on it. The
                    # `<field>_variables` sibling is the per-role spelling:
                    # `input_variables` for the base column, and a list for the
                    # list-valued counterfactual column, indexed the same way
                    # the field is.
                    "input_variables": {"subject": day},
                    "counterfactual_inputs_variables": [{"subject": cf_day}],
                }
            )
    return out


root = pathlib.Path(sys.argv[1])
for name, offsets in (("train", (1, 3)), ("test", (2,))):
    (root / f"{name}.json").write_text(json.dumps(rows(offsets), indent=2) + "\n")
    print(name, len(rows(offsets)), "rows")
