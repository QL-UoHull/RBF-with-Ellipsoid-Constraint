"""
RBF with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

Python implementation of the algorithm described in:

    Li, Q. and Griffiths, J. G. (2004).
    "Radial basis functions for surface reconstruction from unorganised
    point clouds with applications to bone reconstruction."
    Computer Graphics Forum, 23(1), 67–78. Wiley-Blackwell.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x

The package provides :func:`fit_rbf_ellipsoid_linear` and
:func:`evaluate_model_linear` for RBF implicit surface fitting with an
ellipsoid constraint.  A synthetic data generator and multi-format data
loader are also included.
"""

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
from .rbf_ellipsoid import (
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
)

__all__ = [
    # RBF with Ellipsoid Constraint (Li & Griffiths, CGF 2004)
    "fit_rbf_ellipsoid_linear",
    "evaluate_model_linear",
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

__version__ = "2.0.0"
__author__ = "QL-UoHull"
__license__ = "MIT"
