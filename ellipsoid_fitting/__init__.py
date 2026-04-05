"""
Ellipsoid Fitting – two complementary algorithms:

1. **Li–Griffiths (2004)** algebraic least-squares fitting
   (``fit_ellipsoid``).

2. **RBF with Ellipsoid Constraint (Li & Griffiths, CGF 2004)** – implicit
   surface reconstruction using a linear radial basis function kernel
   (``fit_rbf_ellipsoid_linear`` / ``evaluate_model_linear``).

A unified multi-format **data loader** (``load_point_cloud``) supports CSV,
OBJ, PLY (ASCII and binary), XYZ/TXT/PTS, MATLAB `.m` scripts, and NumPy
NPY/NPZ archives.

References
----------
Li, Q. and Griffiths, J. G. (2004).
    *Least squares ellipsoid specific fitting.*
    Proceedings of the Geometric Modeling and Processing, 2004.
    IEEE, pp. 335–340.
    https://doi.org/10.1109/GMAP.2004.1290055

Li, Q. and Griffiths, J. G. (2004).
    *Radial basis functions for surface reconstruction from unorganised
    point clouds with applications to bone reconstruction.*
    Computer Graphics Forum, 23(1), 67–78.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x
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
from .rbf_ellipsoid import (
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
)

__all__ = [
    # Algebraic ellipsoid fitting (Li & Griffiths, GMAP 2004)
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
    # RBF with Ellipsoid Constraint (Li & Griffiths, CGF 2004)
    "fit_rbf_ellipsoid_linear",
    "evaluate_model_linear",
]

__version__ = "2.0.0"
__author__ = "QL-UoHull"
__license__ = "MIT"
