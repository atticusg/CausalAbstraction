"""Configuration for the identity_naming factory task.

Maps entities to their canonical names/numbers. Each domain provides:
- A list of entities and their corresponding result values
- Multiple phrasing templates for prompt variation (NOT causal variables)
- An embedding function for entities and results

First domain: pitch_midi (note name -> MIDI number).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class IdentityNamingConfig:
    """Configuration for an identity-naming task.

    Attributes:
        domain_type: One of the keys in DOMAIN_PRESETS (e.g. "pitch_midi").
        entities: The input domain values.
        entity_to_result: Mapping from entity to its canonical result string.
        templates: List of prompt templates with {entity} placeholder.
        output_prefix: String prepended to result in raw_output.
        entity_embedding: Custom embedding function for entities.
        result_embedding: Custom embedding function for results.
        seed: Random seed.
    """

    domain_type: str
    entities: list[str] = field(default_factory=list)
    entity_to_result: dict[str, str] = field(default_factory=dict)
    templates: list[str] = field(default_factory=list)
    output_prefix: str = " "
    entity_embedding: Callable[[str], list[float]] | None = None
    result_embedding: Callable[[str], list[float]] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        valid = set(DOMAIN_PRESETS.keys())
        if self.domain_type not in valid:
            raise ValueError(
                f"domain_type must be one of {sorted(valid)}, got '{self.domain_type}'"
            )
        if not self.entities:
            preset = DOMAIN_PRESETS[self.domain_type]
            for k, v in preset.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Pitch MIDI helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_NAME_TO_OFFSET = {n: i for i, n in enumerate(_NOTE_NAMES)}


def _note_to_midi(note: str) -> int:
    """Convert note name like 'C#4' to MIDI number (C#4 = 61)."""
    if len(note) >= 2 and note[1] == "#":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return (octave + 1) * 12 + _NOTE_NAME_TO_OFFSET[name]


def _build_note_range(start_midi: int, end_midi: int) -> list[str]:
    """Build list of note names from MIDI numbers."""
    notes = []
    for m in range(start_midi, end_midi + 1):
        name = _NOTE_NAMES[m % 12]
        octave = m // 12 - 1
        notes.append(f"{name}{octave}")
    return notes


# Range: C2 (MIDI 36) to C6 (MIDI 84) — 49 notes, common piano range
_PITCH_MIDI_START = 36
_PITCH_MIDI_END = 84
_PITCH_MIDI_NOTES = _build_note_range(_PITCH_MIDI_START, _PITCH_MIDI_END)
_PITCH_MIDI_MAP = {note: str(_note_to_midi(note)) for note in _PITCH_MIDI_NOTES}
_PITCH_MIDI_RESULTS = sorted(set(_PITCH_MIDI_MAP.values()), key=int)


def _pitch_midi_entity_embed(v: str) -> list[float]:
    return [float(_note_to_midi(v))]


def _pitch_midi_result_embed(v: str) -> list[float]:
    return [float(int(v))]


_PITCH_MIDI_TEMPLATES = [
    "The MIDI number for {entity} is ",
]


# ---------------------------------------------------------------------------
# Domain presets
# ---------------------------------------------------------------------------

DOMAIN_PRESETS: dict[str, dict] = {
    "pitch_midi": dict(
        entities=_PITCH_MIDI_NOTES,
        entity_to_result=_PITCH_MIDI_MAP,
        templates=_PITCH_MIDI_TEMPLATES,
        output_prefix="",
        entity_embedding=_pitch_midi_entity_embed,
        result_embedding=_pitch_midi_result_embed,
    ),
}
