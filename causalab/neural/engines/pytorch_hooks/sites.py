"""Moved to :mod:`causalab.neural.shared.sites` — the component → module map
and the write-policy tables are shared across engines (plan §2.3); only the
eager attention-pattern write machinery (attention_probs.py) is this
engine's. This re-export keeps the old import path for one deprecation beat.
"""

from causalab.neural.shared.sites import *  # noqa: F401,F403
