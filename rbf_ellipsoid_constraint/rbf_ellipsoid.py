"""Radial Basis Function (RBF) surface fitting with ellipsoid constraint.

Python implementation of the algorithm described in:

    Li, Q. and Griffiths, J. G. (2004).
    "Radial basis functions for surface reconstruction from unorganised
    point clouds with applications to bone reconstruction."
    *Computer Graphics Forum*, 23(1), 67–78.
    Wiley-Blackwell.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x

The algorithm fits an implicit surface F(x, y, z) = 0 to a set of 3-D
surface points using a **linear** radial basis function kernel
(φ(r) = r) together with a second-order polynomial basis.

An ellipsoid constraint is imposed via a generalised eigenvalue problem,
ensuring that the recovered implicit surface is topologically an ellipsoid.

Algorithm overview
------------------
Given N surface points **p₁, …, pN**:

1. Normalise the data (zero centroid, unit bounding radius).
2. Build the N × N RBF kernel matrix **A** where A_ij = φ(‖pᵢ − pⱼ‖),
   with an optional smoothing diagonal regulariser.
3. Build the N × 10 polynomial basis matrix **B** with monomials
   ``[1, x, y, z, x², y², z², xy, xz, yz]``.
4. Solve **A X = B** to obtain the 10-column matrix **X**.
5. Form **D = Bᵀ X** (10 × 10).
6. Build the 10 × 10 ellipsoid constraint matrix **C** (non-zero only in
   the six second-order coefficient positions).
7. Solve the generalised eigenvalue problem **D β = λ C β**; select the
   eigenvector **β** corresponding to the smallest positive eigenvalue.
8. Recover the RBF weights **α = −X β**.

Evaluation
----------
For any query point **q**:

    F(q) = Σ_i αᵢ φ(‖q − pᵢ‖) + β₀ + β₁ x + … + β₉ yz

Points where F(q) ≈ 0 lie on the reconstructed surface.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig, solve, pinv
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_rbf_ellipsoid_linear(
    points: np.ndarray,
    smooth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Fit an implicit ellipsoidal surface using a linear RBF kernel.

    The input *points* are assumed to lie on (or near) an ellipsoidal
    surface.  The function normalises the data internally so that the
    result is scale-invariant.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        3-D surface points.  At least 10 points are required.
    smooth : float, optional
        Non-negative regularisation parameter added to the diagonal of the
        RBF matrix (default 0.0 – no smoothing).  Increase this value if
        the system is ill-conditioned or the data are noisy.

    Returns
    -------
    alpha : ndarray of shape (N,)
        RBF weights.
    beta : ndarray of shape (10,)
        Polynomial coefficients ``[β₀, β₁, …, β₉]`` corresponding to the
        monomials ``[1, x, y, z, x², y², z², xy, xz, yz]``.
    centroid : ndarray of shape (3,)
        Centroid used for normalisation (in original coordinates).
    scale : float
        Radius used for normalisation (in original coordinates).

    Returns ``None`` if no valid (positive) eigenvalue is found.

    Raises
    ------
    ValueError
        If fewer than 10 points are supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from rbf_ellipsoid_constraint import generate_ellipsoid_points
    >>> from rbf_ellipsoid_constraint.rbf_ellipsoid import fit_rbf_ellipsoid_linear
    >>> pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=200,
    ...                                  noise_std=0.02, seed=0)
    >>> result = fit_rbf_ellipsoid_linear(pts)
    >>> result is not None
    True
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a 2-D array of shape (N, 3).")
    N = len(points)
    if N < 10:
        raise ValueError(
            f"At least 10 points are required; got {N}."
        )

    # --- 1. Normalise ---
    centroid = np.mean(points, axis=0)
    scale = float(np.max(np.linalg.norm(points - centroid, axis=1)))
    if scale == 0.0:
        raise ValueError("All points are identical – cannot fit a surface.")
    pts = (points - centroid) / scale

    # --- 2. RBF kernel matrix (linear kernel: φ(r) = r) ---
    dists = cdist(pts, pts)
    A = dists + np.eye(N) * smooth

    # --- 3. Polynomial basis matrix (N × 10) ---
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    B = np.column_stack([
        np.ones(N),   # β₀  (constant)
        x,            # β₁
        y,            # β₂
        z,            # β₃
        x * x,        # β₄  (x²)
        y * y,        # β₅  (y²)
        z * z,        # β₆  (z²)
        x * y,        # β₇  (xy)
        x * z,        # β₈  (xz)
        y * z,        # β₉  (yz)
    ])

    # --- 4. Solve A X = B ---
    try:
        X = solve(A, B, assume_a="sym")
    except Exception:
        X = pinv(A) @ B

    # --- 5. D = Bᵀ X ---
    D = B.T @ X

    # --- 6. Ellipsoid constraint matrix C (10 × 10) ---
    # Non-zero only in the second-order monomial block (indices 4–9).
    # Diagonal entries for x², y², z² are 1; for xy, xz, yz are 0.5
    # (so that the constraint penalises non-ellipsoidal coefficient ratios).
    C = np.zeros((10, 10))
    C[4, 4] = 1.0   # x²
    C[5, 5] = 1.0   # y²
    C[6, 6] = 1.0   # z²
    C[7, 7] = 0.5   # xy
    C[8, 8] = 0.5   # xz
    C[9, 9] = 0.5   # yz

    # --- 7. Generalised eigenvalue problem D β = λ C β ---
    eigvals, eigvecs = eig(D, C)

    candidates = [
        (eigvals[i].real, eigvecs[:, i].real)
        for i in range(len(eigvals))
        if abs(eigvals[i].imag) < 1e-8 and eigvals[i].real > 1e-10
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0])
    best_beta = candidates[0][1]

    # --- 8. RBF weights ---
    alpha = -(X @ best_beta)

    return alpha, best_beta, centroid, scale


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_linear(
    eval_pts: np.ndarray,
    norm_pts: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Evaluate the fitted implicit surface F at arbitrary query points.

    Both *eval_pts* and *norm_pts* are expected to be in **normalised**
    coordinates (i.e. subtract the fitted centroid and divide by the fitted
    scale before calling this function).

    Parameters
    ----------
    eval_pts : ndarray of shape (M, 3)
        Query points in normalised coordinates.
    norm_pts : ndarray of shape (N, 3)
        Training points in normalised coordinates (the RBF centres).
    alpha : ndarray of shape (N,)
        RBF weights returned by :func:`fit_rbf_ellipsoid_linear`.
    beta : ndarray of shape (10,)
        Polynomial coefficients returned by :func:`fit_rbf_ellipsoid_linear`.
    chunk_size : int, optional
        Number of query points evaluated per batch to control peak memory
        usage (default 5000).

    Returns
    -------
    values : ndarray of shape (M,)
        Implicit surface values F(q) for each query point.  Points where
        F(q) ≈ 0 lie on the reconstructed surface.

    Examples
    --------
    >>> import numpy as np
    >>> from rbf_ellipsoid_constraint import generate_ellipsoid_points
    >>> from rbf_ellipsoid_constraint.rbf_ellipsoid import (
    ...     fit_rbf_ellipsoid_linear, evaluate_model_linear)
    >>> pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=200, seed=0)
    >>> result = fit_rbf_ellipsoid_linear(pts)
    >>> alpha, beta, cent, scale = result
    >>> norm = (pts - cent) / scale
    >>> vals = evaluate_model_linear(norm, norm, alpha, beta)
    >>> float(np.mean(np.abs(vals))) < 0.5
    True
    """
    eval_pts = np.asarray(eval_pts, dtype=float)
    norm_pts = np.asarray(norm_pts, dtype=float)

    results: list[np.ndarray] = []
    for start in range(0, len(eval_pts), chunk_size):
        batch = eval_pts[start: start + chunk_size]
        bx, by, bz = batch[:, 0], batch[:, 1], batch[:, 2]

        # RBF part: φ(r) = r  (linear kernel)
        d = cdist(batch, norm_pts)
        rbf_part = d @ alpha

        # Polynomial part
        poly_part = (
            beta[0]
            + beta[1] * bx
            + beta[2] * by
            + beta[3] * bz
            + beta[4] * bx ** 2
            + beta[5] * by ** 2
            + beta[6] * bz ** 2
            + beta[7] * bx * by
            + beta[8] * bx * bz
            + beta[9] * by * bz
        )
        results.append(rbf_part + poly_part)

    return np.concatenate(results)
