"""The backend-free intervention-protocol layer.

``docs/intervention_protocol.md`` is the normative spec; this package is
its loader, validator, canonicalizer, sweep expander, planner, backend
contract, and CLI. Nothing here imports torch or an execution engine — a
document is a value, and this layer owns everything decidable from the
value plus a resolution environment.
"""

from causalab.protocol.backend import (
    Backend,
    ExecutionRequest,
    RunResult,
    choose_backend,
    requires,
)
from causalab.protocol.canonical import canonical_bytes, canonicalize, digest
from causalab.protocol.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.loader import LoadedProtocol, load
from causalab.protocol.plan import PointPlan, plan_point
from causalab.protocol.resolve import (
    ArtifactStore,
    DatasetResolver,
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
)
from causalab.protocol.schema import Document, load_raw, parse_document
from causalab.protocol.sweep import Expansion, expand, find_axes
from causalab.protocol.validate import validate_document

__all__ = [
    "ArtifactStore",
    "Backend",
    "DatasetResolver",
    "Document",
    "ExecutionRequest",
    "Expansion",
    "FileArtifacts",
    "FileDatasets",
    "LoadedProtocol",
    "ParseError",
    "PointPlan",
    "ProtocolError",
    "ResolutionEnv",
    "RunResult",
    "ValidationError",
    "canonical_bytes",
    "canonicalize",
    "choose_backend",
    "digest",
    "expand",
    "find_axes",
    "load",
    "load_raw",
    "parse_document",
    "plan_point",
    "requires",
    "validate_document",
]
