"""Structured load errors for the intervention-protocol layer.

Every rejection the loader can produce is one of two exception classes, each
carrying a machine-readable code so tests pin *which* rule fired, not just
that something raised:

* :class:`ParseError` — the document is not a well-formed protocol object:
  bad JSON/YAML, a non-object where a table is required, a wrong value type,
  an unknown key (code ``P<n>``).
* :class:`ValidationError` — the document parses but violates one of the
  load-error rules of the spec's validation checklist
  (``docs/intervention_protocol.md`` §5); ``rule`` is the checklist item
  number (code ``V1`` … ``V<CHECKLIST_RULES>``).

Both derive from :class:`ProtocolError` so callers that only care about
"this document was refused" catch one type. ``path`` is the JSON-pointer-ish
location of the offending value (``sites.target.layer``), kept human-readable
rather than RFC 6901 — it is for error messages and test assertions, not for
re-indexing the document.
"""

from __future__ import annotations

import difflib
from typing import Iterable, Sequence

__all__ = [
    "CHECKLIST_RULES",
    "ParseError",
    "ProtocolError",
    "ValidationError",
    "suggest",
]

#: How many rules the §5 load-error checklist has. Named so the range guard
#: below and the spec move together: adding a rule means editing §5 and this
#: number, and nothing else.
CHECKLIST_RULES: int = 18


class ProtocolError(ValueError):
    """Base class for every intervention-protocol load rejection.

    Subclasses set :attr:`code`; the message always leads with it so a bare
    ``str(err)`` in a log or CI transcript identifies the rule without the
    exception type.
    """

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        where = f" at {path}" if path else ""
        super().__init__(f"[{code}]{where} {message}")


class ParseError(ProtocolError):
    """The document is not a well-formed protocol object (strict parse).

    Codes:

    * ``P1`` — not valid JSON/YAML, or the top level is not an object
    * ``P2`` — a section or field has the wrong type / shape
    * ``P3`` — unknown key (strict keys; suggestions offered)
    * ``P4`` — a closed enum received an unknown value (suggestions offered)
    * ``P5`` — a derived field was authored (spec §6)
    """

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        if code not in {"P1", "P2", "P3", "P4", "P5"}:
            raise AssertionError(f"unknown ParseError code {code!r}")
        super().__init__(code, message, path=path)


class ValidationError(ProtocolError):
    """A well-formed document violates checklist rule ``rule`` (spec §5).

    ``rule`` is the 1-based item number of the load-error checklist in
    ``docs/intervention_protocol.md`` §5 — the tests' contract is one failing
    document per rule, asserted by this number.
    """

    def __init__(self, rule: int, message: str, *, path: str | None = None) -> None:
        if not 1 <= rule <= CHECKLIST_RULES:
            raise AssertionError(f"checklist rule out of range: {rule}")
        self.rule = rule
        super().__init__(f"V{rule}", message, path=path)


def suggest(unknown: str, known: Iterable[str]) -> str:
    """A ``did you mean …?`` suffix for unknown-key/enum rejections.

    Returns an empty string when nothing is close enough — the caller can
    always append the result unconditionally.
    """
    candidates: Sequence[str] = difflib.get_close_matches(
        str(unknown), [str(k) for k in known], n=3
    )
    if not candidates:
        return ""
    quoted = ", ".join(repr(c) for c in candidates)
    return f" — did you mean {quoted}?"
