"""Spline-based manifold with TPS and 1D cubic spline backends."""

from causalab.methods.spline.cubic import CubicSpline1D, NaturalCubicSpline1D
from causalab.methods.spline.manifold import SplineManifold
from causalab.methods.spline.tps import ThinPlateSpline, thin_plate_kernel

__all__ = [
    "SplineManifold",
    "ThinPlateSpline",
    "CubicSpline1D",
    "NaturalCubicSpline1D",
    "thin_plate_kernel",
]
