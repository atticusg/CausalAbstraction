"""Unified arithmetic model with a SHARED INTERNAL CALCULATOR.

One causal model, `domain` as an input, covering standard addition (integer, age)
and the natural domains (weekdays, months, hours, alphabet). The hypothesis: a
single computational module produces `raw_sum` -- the pre-reduction integer sum --
and that variable is the SAME, on a live path to the output, for every domain.

Variable decomposition (this is the point -- the shipped task hides all of this
inside one `result` mechanism):

  entity, number, domain  (inputs)
    entity_index = encode(entity, domain)     # Monday->0 ; "five"->5 ; "C"->2
    number_value = encode(number, domain)     # "three"->3
    raw_sum      = entity_index + number_value # <- SHARED CALCULATOR (domain-free)
    reduced      = raw_sum % modulus(domain)   # or raw_sum if the domain is linear
    result       = decode(reduced, domain)     # int -> "Thursday" / "8" / "F"
    raw_input, raw_output

Because `raw_sum` is one variable in one model, run_interchange(base, {"raw_sum":
cf}) patches it ACROSS domains: take a weekday base, an integer counterfactual,
and transplant the integer sum -- base reduces/decodes it under ITS OWN domain.
"""

from __future__ import annotations

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_HOURS = [str(h) for h in range(1, 25)]
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
]
_W2I = {w: i + 1 for i, w in enumerate(_WORDS)}  # "one"->1 .. "fifteen"->15


class Domain:
    def __init__(self, entities, numbers, enc, num2int, modulus, decode, template):
        self.entities = entities
        self.numbers = numbers
        self.enc = enc  # entity token -> int
        self.num2int = num2int  # number token -> int
        self.modulus = modulus  # int or None (linear)
        self.decode = decode  # int -> output token (total: defined for any int)
        self.template = template


DOMAINS: dict[str, Domain] = {
    "weekdays": Domain(
        _DAYS,
        _WORDS[:7],
        {d: i for i, d in enumerate(_DAYS)},
        _W2I,
        7,
        lambda r: _DAYS[r % 7],
        "Q: What day is {number} days after {entity}?\nA:",
    ),
    "months": Domain(
        _MONTHS,
        _WORDS[:7],
        {m: i for i, m in enumerate(_MONTHS)},
        _W2I,
        12,
        lambda r: _MONTHS[r % 12],
        "Q: What month is {number} months after {entity}?\nA:",
    ),
    "hours": Domain(
        _HOURS,
        _WORDS[:12],
        {h: i for i, h in enumerate(_HOURS)},
        _W2I,
        24,
        lambda r: _HOURS[r % 24],
        "Q: What hour is {number} hours after {entity}?\nA:",
    ),
    "alphabet": Domain(
        _LETTERS[:25],
        _WORDS[:4],
        {c: i for i, c in enumerate(_LETTERS)},
        _W2I,
        None,
        lambda r: _LETTERS[r % 26],
        "Starting at {entity}, increment by {number}. Result:",
    ),
    "integer": Domain(
        _WORDS,
        _WORDS[:9],
        dict(_W2I),
        _W2I,
        None,
        lambda r: str(r),
        "Q: What is {number} added to {entity}?\nA:",
    ),
    "age": Domain(
        [str(i) for i in range(1, 31)],
        [str(i) for i in range(1, 10)],
        {str(i): i for i in range(1, 31)},
        {str(i): i for i in range(1, 10)},
        None,
        lambda r: str(r),
        "Alice is {entity}; Bob is {number} older. Bob is",
    ),
}

_ALL_ENT = sorted({e for d in DOMAINS.values() for e in d.entities})
_ALL_NUM = sorted({n for d in DOMAINS.values() for n in d.numbers})

# entity and domain are COUPLED inputs (a token is only valid inside its domain).
# An interchange that patches the entity token alone into a base of another domain
# creates an off-distribution (entity, domain) pair, so the per-domain encoder
# would KeyError. We make encoding total with a global fallback so every
# intervention is well-defined; the off-distribution semantics this introduces are
# exactly why the surface-token hypotheses read oddly across domains.
_GLOBAL_ENC: dict[str, int] = {}
for _d in DOMAINS.values():
    for _e, _i in _d.enc.items():
        _GLOBAL_ENC.setdefault(_e, _i)
_GLOBAL_NUM: dict[str, int] = {}
for _d in DOMAINS.values():
    for _n, _i in _d.num2int.items():
        _GLOBAL_NUM.setdefault(_n, _i)


def _enc_entity(t):
    d = DOMAINS[t["domain"]]
    return d.enc.get(t["entity"], _GLOBAL_ENC.get(t["entity"], 0))


def _enc_number(t):
    d = DOMAINS[t["domain"]]
    return d.num2int.get(t["number"], _GLOBAL_NUM.get(t["number"], 0))


def _reduce(t):
    m = DOMAINS[t["domain"]].modulus
    return t["raw_sum"] % m if m is not None else t["raw_sum"]


def _decode(t):
    return DOMAINS[t["domain"]].decode(t["reduced"])


_mechanisms = {
    "entity": input_var(_ALL_ENT),
    "number": input_var(_ALL_NUM),
    "domain": input_var(list(DOMAINS)),
    "entity_index": Mechanism(parents=["entity", "domain"], compute=_enc_entity),
    "number_value": Mechanism(parents=["number", "domain"], compute=_enc_number),
    "raw_sum": Mechanism(
        parents=["entity_index", "number_value"],
        compute=lambda t: t["entity_index"] + t["number_value"],
    ),
    "reduced": Mechanism(parents=["raw_sum", "domain"], compute=_reduce),
    "result": Mechanism(parents=["reduced", "domain"], compute=_decode),
    "raw_input": Mechanism(
        parents=["entity", "number", "domain"],
        compute=lambda t: DOMAINS[t["domain"]].template.format(
            entity=t["entity"], number=t["number"]
        ),
    ),
    "raw_output": Mechanism(
        parents=["result"], compute=lambda t: " " + str(t["result"])
    ),
}

_values = {
    "entity": _ALL_ENT,
    "number": _ALL_NUM,
    "domain": list(DOMAINS),
    "entity_index": None,
    "number_value": None,
    "raw_sum": None,
    "reduced": None,
    "result": None,
    "raw_input": None,
    "raw_output": None,
}

unified = CausalModel(_mechanisms, _values, id="unified_arithmetic")

MODELS = {"unified": unified}
DEFAULT_MODEL = "unified"

HYPOTHESES = {
    "raw_sum": ("unified", ["raw_sum"]),  # the shared calculator
    "operands": ("unified", ["entity_index", "number_value"]),  # full upstream slice
    "entity_index": ("unified", ["entity_index"]),  # partial: just the entity operand
    "number_value": ("unified", ["number_value"]),  # partial: just the number operand
    "reduced": ("unified", ["reduced"]),  # post-modulus (domain-entangled)
    "result": ("unified", ["result"]),  # decoded token
    "surface_entity": ("unified", ["entity"]),
    "domain": ("unified", ["domain"]),
    "all": ("unified", ["raw_output"]),
}
TARGETS = ["raw_sum"]
