# causalab.neural

The execution side of the intervention-protocol stack:

- `pytorch_hooks/` — the reference backend (spec §8 of
  `docs/intervention_protocol.md`): SiteResolver over raw module hooks,
  position resolution against the padded batch frame, the closed `do`
  mechanism set, featurizers with the error-term contract, metric
  lowering, the train loop, and ArtifactIdentity stamping.
- `token_positions.py` — char→token position utilities (offset-mapping
  based, chat-prefix aware). Backbone-agnostic; the task packages'
  `token_positions.py` modules build on it. The protocol-native position
  service lives in `pytorch_hooks/encoding.py`; this module remains the
  home of the legacy declarative vocabulary the tasks encode.

Everything else that used to live here (the Plan IR and its scheduler,
the nnsight pipeline, spec persistence) was replaced by the protocol
layer (`causalab/protocol`) plus the backend above — see the PR that
introduced `docs/intervention_protocol.md`.
