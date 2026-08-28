"""What every execution engine uses — and none may fork.

The §8 services that are protocol work, not hook work: tokenization and the
batch frame (:mod:`.encoding`), contract layouts (:mod:`.layout`), payload
math (:mod:`.mechanisms`), metric lowering (:mod:`.metrics`), artifact
writing and stamping (:mod:`.outputs`), bundle loading, role resolution and
identity records (:mod:`.services`), and the per-layer hybrid stream table
(:mod:`.streams`). An engine owns *loading, site resolution and execution*;
everything here is the shared remainder, single-homed so two engines can
never disagree about it (plan §2.4).
"""
