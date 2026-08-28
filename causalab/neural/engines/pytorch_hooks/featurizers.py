"""Moved to :mod:`causalab.neural.shared.featurizers` — applying a featurizer
is engine-neutral tensor math (plan §2.4); only the train loop (train.py) is
engine work. This re-export keeps the old import path for one deprecation
beat.
"""

from causalab.neural.shared.featurizers import *  # noqa: F401,F403
