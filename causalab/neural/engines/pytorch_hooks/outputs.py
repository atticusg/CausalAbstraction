"""Moved to :mod:`causalab.neural.shared.outputs` — engine-neutral (plan §2.4).

This re-export keeps the old import path alive for one deprecation beat;
new code imports the shared home.
"""

from causalab.neural.shared.outputs import *  # noqa: F401,F403
