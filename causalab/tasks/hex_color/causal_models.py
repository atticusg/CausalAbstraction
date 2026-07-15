"""Causal model for the ``hex_color`` perceptual colour-classification task.

The task casts colour naming as a tiny causal DAG over a single stimulus::

    hex → color → raw_output
    hex → raw_input

``hex`` is the input: a hue-jittered ``#RRGGBB`` code drawn from the bundled 600
stimuli (100 per colour × 6). ``color`` is the perceptual label the stimulus
maps to (one of the six colour words), looked up from the bundled data.
``raw_input`` is the filled prompt (the six choices inlined MCQA-style);
``raw_output`` is ``" " + color`` — the word the model is expected to emit.

Design notes:

* **Singleton task.** The stimulus set is fixed (bundled ``data/hex_color.json``),
  so the model is a module-level ``CAUSAL_MODEL`` constant — no factory config.
* **Periodic hue embedding.** ``color`` carries a 1-D embedding (its hue-centre
  in degrees) with a 360° period, mirroring ``natural_domains_arithmetic``'s
  cyclic ``result``. Manifold/geometry analyses read these; the baseline does
  not require them.
* **Scoring is the colour word.** Unlike MCQA (which scores an option *letter*
  by default and needs a ``score_by: value`` mode), the answer here *is* the
  colour word, so the default path already scores "the value". ``output_tokens``
  on ``color`` drives both the probability path and the *derived* string checker
  (``causalab.tasks.loader._resolve_checker`` → ``derive_checker``): all six
  colours are single-token, so plain exact match suffices and the task ships no
  bespoke ``checker.py``.
* **``indigo`` dropped (7 → 6).** The source dataset had seven classes; ``indigo``
  (hue 258°, wedged between blue 235° and purple 285°) is excluded because the
  golden fixture (Qwen3-4B-Instruct) labels indigo swatches "purple"
  ~0.999-confident, capping 7-colour accuracy at ~0.80 (< the 0.9 golden floor).
  Dropping it both makes the task viable on the fixture *and* removes the only
  multi-token colour (``indigo`` → ``["ind", "igo"]``), which is why no bespoke
  checker is needed. Decided during epic #522 orchestration.

Data provenance: the stimuli come from Llama-3.1-8B DAS work
(``<hex-color-das-source>/data.json``),
but only the model-agnostic stimulus content is bundled — no tokenizer/position
fields, and the 100 ``indigo`` rows are excluded at build time. Colours/
hue-centres/template come from that dataset's ``config.json``.
"""

from __future__ import annotations

import json

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from .config import (
    COLORS,
    DATA_PATH,
    HUE_CENTERS_DEG,
    HUE_PERIOD,
    PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Bundled stimulus data
# ---------------------------------------------------------------------------


def _load_stimuli() -> list[dict]:
    """Load the bundled 600-stimulus set (never reads ``external artifact storage`` at runtime)."""
    with open(DATA_PATH, "r") as f:
        return json.load(f)


STIMULI: list[dict] = _load_stimuli()

# Ordered hex list (the ``hex`` input variable's value domain) and the
# hex → colour-label lookup that drives the ``color`` mechanism.
HEXES: list[str] = [row["hex"] for row in STIMULI]
HEX_TO_LABEL: dict[str, str] = {row["hex"]: row["label"] for row in STIMULI}

# Convenience index used by the counterfactual generators: colour → its hexes.
HEXES_BY_COLOR: dict[str, list[str]] = {c: [] for c in COLORS}
for _row in STIMULI:
    HEXES_BY_COLOR[_row["label"]].append(_row["hex"])


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


def _compute_color(t: CausalTrace) -> str:
    """The perceptual colour label of the stimulus hex."""
    return HEX_TO_LABEL[t["hex"]]


def _fill_template(t: CausalTrace) -> str:
    """Render ``raw_input`` — the prompt with the stimulus hex substituted."""
    return PROMPT_TEMPLATE.replace("{hex}", t["hex"])


def _compute_raw_output(t: CausalTrace) -> str:
    """Expected next-token output: a leading space then the colour word."""
    return " " + t["color"]


# ---------------------------------------------------------------------------
# Causal model
# ---------------------------------------------------------------------------


def _build_causal_model() -> CausalModel:
    mechanisms: dict[str, Mechanism] = {
        "hex": input_var(HEXES),
        "color": Mechanism(parents=["hex"], compute=_compute_color),
        "raw_input": Mechanism(parents=["hex"], compute=_fill_template),
        "raw_output": Mechanism(parents=["color"], compute=_compute_raw_output),
    }
    values: dict[str, list | None] = {
        "hex": HEXES,
        "color": COLORS,
        "raw_input": None,
        "raw_output": None,
    }
    return CausalModel(
        mechanisms,
        values,
        id="hex_color",
        # The answer is the colour word (``raw_output = " " + color``). Declaring
        # the mechanical ``[" red", "red"]`` forms drives both the probability
        # path (score-token resolution / per-class distributions) and the derived
        # exact-match grader. All six colours are single-token, so no bespoke
        # checker.py is needed.
        output_tokens={"color": build_output_tokens(COLORS)},
        # Periodic hue embedding: each colour sits at its hue-centre on a circle
        # that wraps at 360°.
        embeddings={"color": lambda c: [HUE_CENTERS_DEG[c]]},
        periods={"color": HUE_PERIOD},
    )


CAUSAL_MODEL = _build_causal_model()


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

TARGET_VARIABLE = "color"
TEMPLATE = PROMPT_TEMPLATE
