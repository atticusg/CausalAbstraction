"""Neural-network execution for intervention protocols.

After the protocol refactor this package holds exactly two things: the
reference engine over native pytorch hooks (:mod:`.pytorch_hooks`, the
only subpackage — spec §8 of ``docs/intervention_protocol.md``) and the
backbone-agnostic token-position utilities (:mod:`.token_positions`) the
task packages' position vocabularies are written against.
"""
