"""
Ellipsoid fitting via least squares with ellipsoid-specific constraints.

This module provides two public fitting functions:

1. ``fit_ellipsoid`` — **pure algebraic variant** (Li & Griffiths, GMAP 2004).
   This is a direct implementation of the constrained least-squares algebraic
   fitting described in the paper.  It does **not** estimate surface normals,
   does **not** generate off-surface point layers, and does **not** compute
   radial basis function (RBF) weights.  The output is a set of geometric
   parameters (centre, radii, axes) derived from the 10 algebraic polynomial
   coefficients.

2. ``fit_ellipsoid_no_normals`` — **algebraic + RBF variant (no normal
   estimation, no off-surface layers)**.  This extends the algebraic fit by
   also recovering a set of RBF weights, using only the original on-surface
   points as the training set (target values d = 0).  No normal vectors are
   estimated and no additional offset point layers are generated; this is an
   implementation choice distinct from variants that augment the training set
   with off-surface points.

Both variants fit a general algebraic ellipsoid of the form

    F(x, y, z) = Ax² + By² + Cz² + 2Dyz + 2Exz + 2Fxy
                 + 2Gx + 2Hy + 2Iz + J = 0

to a set of 3-D data points using a constrained least-squares approach.
The constraint matrix (with parameter k = 4) guarantees that only
ellipsoidal solutions are admitted, distinguishing the method from
unconstrained quadric fitting.

Reference
---------
Li, Q. and Griffiths, J. G. (2004).
    "Least squares ellipsoid specific fitting."
    Proceedings of the Geometric Modeling and Processing, 2004.
    IEEE, pp. 335-340.
    https://doi.org/10.1109/GMAP.2004.1290055
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eig, inv
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _design_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Build the (N × 10) algebraic design matrix.

    Each row corresponds to one data point and contains the monomials

        [x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z, 1]

    Parameters
    ----------
    x, y, z : 1-D array-like of length N
        Cartesian coordinates of the data points.

    Returns
    -------
    D : ndarray, shape (N, 10)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    return np.column_stack([
        x * x, y * y, z * z,
        2 * y * z, 2 * x * z, 2 * x * y,
        2 * x, 2 * y, 2 * z,
        np.ones_like(x),
    ])


def _constraint_matrix(k: float = 4.0) -> np.ndarray:
    """Build the (10 × 10) ellipsoid-specific constraint matrix.

    Following equation (7) of Li & Griffiths (2004) the constraint
    matrix is

        ⎡  0   k/2 k/2  0   0   0  0  0  0  0 ⎤
        ⎢ k/2   0  k/2  0   0   0  0  0  0  0 ⎥
        ⎢ k/2  k/2  0   0   0   0  0  0  0  0 ⎥
        ⎢  0    0   0  -k   0   0  0  0  0  0 ⎥
        ⎢  0    0   0   0  -k   0  0  0  0  0 ⎥
        ⎢  0    0   0   0   0  -k  0  0  0  0 ⎥
        ⎣  0    0   0   0   0   0  0  0  0  0 ⎦  (rows 6-9 are zero)

    With the default k = 4, this is the standard Li-Griffiths constraint.

    Parameters
    ----------
    k : float, optional
        Constraint parameter (default 4).

    Returns
    -------
    C : ndarray, shape (10, 10)
    """
    C = np.zeros((10, 10))
    C[0, 1] = C[1, 0] = k / 2
    C[0, 2] = C[2, 0] = k / 2
    C[1, 2] = C[2, 1] = k / 2
    C[3, 3] = -k
    C[4, 4] = -k
    C[5, 5] = -k
    return C


def _algebraic_to_geometric(v: np.ndarray) -> dict:
    """Convert algebraic coefficients to geometric (centre, axes, radii).

    Parameters
    ----------
    v : 1-D ndarray of length 10
        Algebraic coefficients [A, B, C, D, E, F, G, H, I, J].

    Returns
    -------
    dict with keys:
        ``centre``      – ndarray (3,) : centre of the ellipsoid
        ``radii``       – ndarray (3,) : semi-axis lengths (sorted descending)
        ``axes``        – ndarray (3, 3) : corresponding unit-vector columns
        ``M``           – ndarray (4, 4) : homogeneous matrix of the quadric
        ``coefficients``– ndarray (10,)  : original algebraic coefficients
    """
    A, B, C, D, E, F, G, H, I, J = v

    # 3 × 3 matrix of second-order terms
    M3 = np.array([
        [A, F, E],
        [F, B, D],
        [E, D, C],
    ])

    # 4 × 4 homogeneous matrix
    M4 = np.array([
        [A,  F,  E,  G],
        [F,  B,  D,  H],
        [E,  D,  C,  I],
        [G,  H,  I,  J],
    ])

    # Centre: centre = -M3⁻¹ * [G, H, I]
    centre = -inv(M3) @ np.array([G, H, I])

    # Eigendecomposition of M3 to obtain axes and radii
    # Radii from: r_i = sqrt(-det(M4) / (lambda_i * det(M3)))
    eigenvalues, eigenvectors = eig(M3)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    det_M4 = np.linalg.det(M4)
    det_M3 = np.linalg.det(M3)

    # Guard against near-singular matrices
    if abs(det_M3) < 1e-12:
        raise ValueError(
            "Degenerate solution: the 3×3 second-order matrix is singular. "
            "The fitted surface may not be a valid ellipsoid."
        )

    radii_sq = -det_M4 / (eigenvalues * det_M3)

    if np.any(radii_sq <= 0):
        raise ValueError(
            "Non-ellipsoidal solution: one or more squared semi-axes are "
            "non-positive. Check that the data lie on (or near) an ellipsoid."
        )

    radii = np.sqrt(radii_sq)

    # Sort by descending radius
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    eigenvectors = eigenvectors[:, order]

    return {
        "centre": centre,
        "radii": radii,
        "axes": eigenvectors,
        "M": M4,
        "coefficients": v,
    }


def _rbf_matrix(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Compute the linear RBF kernel matrix between two point sets.

    The linear kernel is φ(r) = r (Euclidean distance).

    Parameters
    ----------
    pts_a : ndarray, shape (M, 3)
        Query points.
    pts_b : ndarray, shape (N, 3)
        Centre points.

    Returns
    -------
    Phi : ndarray, shape (M, N)
        Phi[i, j] = ||pts_a[i] - pts_b[j]||
    """
    return cdist(pts_a, pts_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_ellipsoid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float = 4.0,
) -> dict:
    """Fit an ellipsoid to 3-D point data — **pure algebraic variant**.

    This is a direct implementation of the constrained least-squares algebraic
    fitting described in Li & Griffiths (2004, GMAP).  It is the **pure
    algebraic variant**: it does **not** estimate surface normals, does **not**
    generate off-surface point layers, and does **not** compute radial basis
    function (RBF) weights.  If you also need RBF weights (with the same
    on-surface-only training set), use :func:`fit_ellipsoid_no_normals`.

    The method minimises the sum of squared algebraic distances subject to
    the constraint that the fitted quadric is an ellipsoid.  It solves the
    generalised eigenvalue problem

        S v = λ C v,

    where **S = Dᵀ D** is the scatter matrix and **C** is the
    ellipsoid-specific constraint matrix, and selects the eigenvector
    corresponding to the largest positive eigenvalue.

    Parameters
    ----------
    x, y, z : array-like of shape (N,)
        Cartesian coordinates of the 3-D data points.  At least 10 points
        are required (one per degree of freedom of the general quadric).
    k : float, optional
        Constraint parameter (default 4).  Li & Griffiths (2004) prove that
        any *k* in the open interval (0, 4] yields ellipsoid-specific
        constraints; k = 4 is the recommended value.

    Returns
    -------
    result : dict
        A dictionary containing:

        ``centre``      : ndarray (3,)
            Coordinates of the ellipsoid centre.
        ``radii``       : ndarray (3,)
            Semi-axis lengths, sorted in descending order.
        ``axes``        : ndarray (3, 3)
            Unit vectors of the semi-axes (columns), in the same order as
            *radii*.
        ``M``           : ndarray (4, 4)
            Homogeneous 4 × 4 matrix representation of the quadric.
        ``coefficients``: ndarray (10,)
            Raw algebraic coefficients [A, B, C, D, E, F, G, H, I, J].

    Raises
    ------
    ValueError
        If fewer than 10 data points are supplied, or if the fitted surface
        is not a valid ellipsoid.

    Examples
    --------
    >>> import numpy as np
    >>> from ellipsoid_fitting import fit_ellipsoid, generate_ellipsoid_points
    >>> pts = generate_ellipsoid_points(centre=(1, 2, 3), radii=(5, 3, 2),
    ...                                  n_points=200, noise_std=0.05)
    >>> result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
    >>> print("Centre:", result["centre"])
    >>> print("Radii :", result["radii"])
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    n = len(x)
    if n != len(y) or n != len(z):
        raise ValueError("x, y and z must have the same length.")
    if n < 10:
        raise ValueError(
            f"At least 10 data points are required; got {n}."
        )

    D = _design_matrix(x, y, z)
    S = D.T @ D                   # 10 × 10 scatter matrix
    C = _constraint_matrix(k)     # 10 × 10 constraint matrix

    # Solve the generalised eigenvalue problem S v = λ C v.
    # We reformulate as the standard eigenvalue problem on the 6 × 6
    # upper-left sub-problem (the lower-right 4 × 4 block of C is zero).
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]
    C11 = C[:6, :6]    # only non-zero block

    # Solve the 6 × 6 reduced eigensystem (eq. 11 of Li & Griffiths, 2004)
    try:
        S22_inv = inv(S22)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The scatter matrix S22 is singular. "
            "Consider adding more (or more diverse) data points."
        ) from exc

    M = inv(C11) @ (S11 - S12 @ S22_inv @ S21)
    eigenvalues, eigenvectors = eig(M)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Select the eigenvector corresponding to the largest eigenvalue.
    # A small tolerance handles perfectly noise-free data where the optimal
    # eigenvalue may be numerically indistinguishable from zero.
    _tol = 1e-6 * np.max(np.abs(eigenvalues))
    positive_mask = eigenvalues > -_tol
    if not np.any(positive_mask):
        raise ValueError(
            "No positive eigenvalue found.  The data may not lie on an "
            "ellipsoid, or may be too noisy / degenerate."
        )
    best_idx = np.argmax(eigenvalues)
    u1 = eigenvectors[:, best_idx]

    # Recover the full 10-element coefficient vector
    u2 = -S22_inv @ S21 @ u1
    v = np.concatenate([u1, u2])

    # Normalise so that the quadric has a consistent scale
    v /= np.linalg.norm(v)

    return _algebraic_to_geometric(v)


def fit_ellipsoid_no_normals(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float = 4.0,
) -> dict:
    """Fit an ellipsoid to 3-D point data — **algebraic + RBF variant, no normal estimation**.

    This function extends :func:`fit_ellipsoid` by also computing a set of
    radial basis function (RBF) weights in addition to the algebraic polynomial
    coefficients.  It is an implementation choice to use **only the original
    on-surface points** as the training set (target values d = 0), without
    estimating surface normals and without generating off-surface point layers
    (no ``pts_plus`` / ``pts_minus`` construction).

    This is the **on-surface-only, no-normal-estimation** variant:

    * Training set: T = P (the N input points only).
    * Target vector: d = 0 (on-surface constraint only).
    * Scatter matrix: S = Dᵀ D computed from the N-row design matrix.
    * The same constrained eigenvalue problem as :func:`fit_ellipsoid` is
      solved to obtain polynomial coefficients **β**.
    * RBF weights **w** are then recovered by solving the linear system
      Φ w ≈ −D β, where Φ is the N × N linear RBF kernel matrix (φ(r) = r)
      evaluated between all pairs of input points.

    Parameters
    ----------
    x, y, z : array-like of shape (N,)
        Cartesian coordinates of the 3-D data points.  At least 10 points
        are required.
    k : float, optional
        Constraint parameter (default 4).  Li & Griffiths (2004) prove that
        any *k* in the open interval (0, 4] yields ellipsoid-specific
        constraints; k = 4 is the recommended value.

    Returns
    -------
    result : dict
        A dictionary containing:

        ``centre``      : ndarray (3,)
            Coordinates of the ellipsoid centre.
        ``radii``       : ndarray (3,)
            Semi-axis lengths, sorted in descending order.
        ``axes``        : ndarray (3, 3)
            Unit vectors of the semi-axes (columns), in the same order as
            *radii*.
        ``M``           : ndarray (4, 4)
            Homogeneous 4 × 4 matrix representation of the quadric.
        ``coefficients``: ndarray (10,)
            Raw algebraic coefficients [A, B, C, D, E, F, G, H, I, J].
        ``rbf_weights`` : ndarray (N,)
            RBF weight vector **w** (one weight per input point), recovered
            by solving Φ w ≈ −D β with a linear RBF kernel φ(r) = r.

    Raises
    ------
    ValueError
        If fewer than 10 data points are supplied, if the arrays have
        mismatched lengths, or if the fitted surface is not a valid ellipsoid.

    Examples
    --------
    >>> import numpy as np
    >>> from ellipsoid_fitting import fit_ellipsoid_no_normals, generate_ellipsoid_points
    >>> pts = generate_ellipsoid_points(centre=(1, 2, 3), radii=(5, 3, 2),
    ...                                  n_points=200, noise_std=0.05)
    >>> result = fit_ellipsoid_no_normals(pts[:, 0], pts[:, 1], pts[:, 2])
    >>> print("Centre     :", result["centre"])
    >>> print("Radii      :", result["radii"])
    >>> print("RBF weights:", result["rbf_weights"].shape)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    n = len(x)
    if n != len(y) or n != len(z):
        raise ValueError("x, y and z must have the same length.")
    if n < 10:
        raise ValueError(
            f"At least 10 data points are required; got {n}."
        )

    pts = np.column_stack([x, y, z])   # (N, 3)

    D = _design_matrix(x, y, z)
    S = D.T @ D                   # 10 × 10 scatter matrix
    C = _constraint_matrix(k)     # 10 × 10 constraint matrix

    # Solve the same constrained eigenvalue problem as fit_ellipsoid.
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]
    C11 = C[:6, :6]

    try:
        S22_inv = inv(S22)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The scatter matrix S22 is singular. "
            "Consider adding more (or more diverse) data points."
        ) from exc

    M = inv(C11) @ (S11 - S12 @ S22_inv @ S21)
    eigenvalues, eigenvectors = eig(M)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    _tol = 1e-6 * np.max(np.abs(eigenvalues))
    positive_mask = eigenvalues > -_tol
    if not np.any(positive_mask):
        raise ValueError(
            "No positive eigenvalue found.  The data may not lie on an "
            "ellipsoid, or may be too noisy / degenerate."
        )
    best_idx = np.argmax(eigenvalues)
    u1 = eigenvectors[:, best_idx]

    u2 = -S22_inv @ S21 @ u1
    v = np.concatenate([u1, u2])
    v /= np.linalg.norm(v)

    # Recover RBF weights w by solving: Phi * w ≈ d - D * v
    # Training target d = 0 (on-surface only), so: Phi * w ≈ -D * v
    Phi = _rbf_matrix(pts, pts)               # (N, N) linear RBF matrix
    rhs = -(D @ v)                            # (N,) right-hand side
    rbf_weights, _, _, _ = np.linalg.lstsq(Phi, rhs, rcond=None)

    result = _algebraic_to_geometric(v)
    result["rbf_weights"] = rbf_weights
    return result


def algebraic_distance(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate the algebraic distance F(x, y, z) for each data point.

    Parameters
    ----------
    x, y, z : array-like of shape (N,)
        Cartesian coordinates.
    coefficients : array-like of shape (10,)
        Algebraic coefficients [A, B, C, D, E, F, G, H, I, J].

    Returns
    -------
    d : ndarray of shape (N,)
        Algebraic distance values (ideally zero for exact ellipsoid points).
    """
    D = _design_matrix(x, y, z)
    return D @ np.asarray(coefficients, dtype=float)


def residuals_rms(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    result: dict,
) -> float:
    """Root-mean-square algebraic residual of a fit.

    Parameters
    ----------
    x, y, z : array-like of shape (N,)
        Cartesian coordinates.
    result : dict
        Return value of :func:`fit_ellipsoid`.

    Returns
    -------
    rms : float
        RMS algebraic residual (lower is better).
    """
    d = algebraic_distance(x, y, z, result["coefficients"])
    return float(np.sqrt(np.mean(d ** 2)))
