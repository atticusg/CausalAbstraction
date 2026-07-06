"""Analytic path patching (Hanna et al. 2023 semantics) with architecture guards.

Quick start::

    from causalab.methods.path_patching import (
        PatchEngine, PathSpec, build_patch_cache, resolve_descriptor
    )

    desc = resolve_descriptor(pipeline.model)
    clean = build_patch_cache(pipeline, desc, clean_texts, {"end": -1})
    cf = build_patch_cache(pipeline, desc, cf_texts, {"end": -1})
    engine = PatchEngine(desc, clean, cf)      # construction runs the guards

    # sender a9.h1's direct edge to the logits:
    logits = engine.patched_logits(("head", 9, 1), PathSpec.cascade())
    # its paths through MLPs 8-11 (closed cascade, direct edge off-path):
    logits = engine.patched_logits(
        ("head", 9, 1),
        PathSpec.cascade([8, 9, 10, 11], direct_to_logits=False),
    )

See ``docs`` for the freeze-recipe semantics, the edge-set specification,
and a worked example.
"""

from .cache import PatchCache, build_patch_cache, padded_position
from .descriptor import (
    SUPPORTED_MODEL_TYPES,
    ArchitectureDescriptor,
    resolve_descriptor,
)
from .edges import PathSpec
from .engine import PatchEngine, Sender
from .guards import GuardError, default_tolerances, run_construction_guards
from .kv import (
    AttnDetailCache,
    KVEdge,
    KVHead,
    KVPatchEngine,
    SlidingWindowError,
    build_attn_detail_cache,
)
from .provenance import (
    CapturePoint,
    UnsupportedArchitectureError,
    capture_provenance,
    check_capability,
    coverage_table,
    pyvene_pin,
)
from .reference import reference_patched_logits

__all__ = [
    "ArchitectureDescriptor",
    "AttnDetailCache",
    "CapturePoint",
    "GuardError",
    "KVEdge",
    "KVHead",
    "KVPatchEngine",
    "SlidingWindowError",
    "UnsupportedArchitectureError",
    "capture_provenance",
    "check_capability",
    "coverage_table",
    "pyvene_pin",
    "PatchCache",
    "PatchEngine",
    "PathSpec",
    "SUPPORTED_MODEL_TYPES",
    "Sender",
    "build_attn_detail_cache",
    "build_patch_cache",
    "default_tolerances",
    "padded_position",
    "reference_patched_logits",
    "resolve_descriptor",
    "run_construction_guards",
]
