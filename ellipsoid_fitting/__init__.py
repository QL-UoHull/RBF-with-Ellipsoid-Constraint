"""
RBR with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

Python implementation of the algorithm described in:

    Li, Q. (2004).
    "Implicit fitting using radial basis functions with ellipsoidal
    constraint."
    Computer Graphics Forum, 23(1), 89–96. Wiley/Blackwell.
    https://doi.org/10.1111/j.1467-8659.2004.00756.x

The package provides the :func:`fit_ellipsoid` function, which uses a
Radial Basis (function) Representation (RBR) together with the
ellipsoid-specific constraint to fit an ellipsoid to scattered 3-D point
data.  A synthetic data generator and residual utilities are also included.
"""

from .ellipsoid_fit import (
    fit_ellipsoid,
    algebraic_distance,
    residuals_rms,
)
from .data_generator import generate_ellipsoid_points

__all__ = [
    "fit_ellipsoid",
    "algebraic_distance",
    "residuals_rms",
    "generate_ellipsoid_points",
]

__version__ = "1.0.0"
__author__ = "QL-UoHull"
__license__ = "MIT"
