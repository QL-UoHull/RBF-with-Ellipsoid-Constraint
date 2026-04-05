"""
Ellipsoid Fitting via Least Squares with Ellipsoid-Specific Constraints
(Li & Griffiths, 2004)

Python implementation of the algorithm described in:

    Li, Q. and Griffiths, J. G. (2004).
    "Least squares ellipsoid specific fitting."
    Proceedings of the Geometric Modeling and Processing, 2004.
    IEEE, pp. 335-340.
    https://doi.org/10.1109/GMAP.2004.1290055
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
