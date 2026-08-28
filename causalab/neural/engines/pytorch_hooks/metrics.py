"""Moved to :mod:`causalab.neural.shared.metrics` — engine-neutral (plan §2.4).

This re-export keeps the old import path alive for one deprecation beat;
new code imports the shared home.
"""

from causalab.neural.shared.metrics import *  # noqa: F401,F403
