"""
RBF with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

Python implementation of the RBF-based implicit fitting algorithm from:

    Li, Q. (2004).
    "Implicit fitting using radial basis functions with ellipsoidal
    constraint."
    Computer Graphics Forum, 23(1), 89–96. Wiley/Blackwell.
    https://doi.org/10.1111/j.1467-8659.2004.00756.x

The package provides the :func:`fit_ellipsoid` function, which implements
the RBF implicit fitting with ellipsoidal constraint algorithm from Li (2004)
to fit an ellipsoid to scattered 3-D point data.  A synthetic data generator,
multi-format data loader, and residual utilities are also included.
"""

from .ellipsoid_fit import (
    fit_ellipsoid,
    algebraic_distance,
    residuals_rms,
)
from .data_generator import generate_ellipsoid_points
from .loaders import (
    load_point_cloud,
    load_csv,
    load_xyz,
    load_obj,
    load_ply,
    load_matlab,
    load_npy,
    load_npz,
    FORMAT_LOADERS,
)
__all__ = [
    # Algebraic ellipsoid fitting (Li, CGF 2004)
    "fit_ellipsoid",
    "algebraic_distance",
    "residuals_rms",
    # Synthetic data generator
    "generate_ellipsoid_points",
    # Multi-format data loader
    "load_point_cloud",
    "load_csv",
    "load_xyz",
    "load_obj",
    "load_ply",
    "load_matlab",
    "load_npy",
    "load_npz",
    "FORMAT_LOADERS",
]

__version__ = "1.0.0"
__author__ = "QL-UoHull"
__license__ = "MIT"
