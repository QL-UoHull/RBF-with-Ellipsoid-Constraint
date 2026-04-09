"""
Implicit fitting using radial basis functions with ellipsoidal constraint.

Python implementation of the algorithm described in:

    Li, Q. (2004).
    "Implicit fitting using radial basis functions with ellipsoidal constraint."
    Computer Graphics Forum, 23(1), 89–96.
    Wiley/Blackwell.
    https://doi.org/10.1111/j.1467-8659.2004.00756.x

The method represents the implicit surface as

    f(**x**) = Σ_i w_i φ(‖**x** − **c**_i‖) + β^T **p**(**x**) = 0

where φ(r) = r is the biharmonic kernel, **c**_i are the N surface-point
centres, and **p**(**x**) = [x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z, 1]^T
is the 10-term degree-2 polynomial basis.

Off-surface training points **P**± = **P** ± ε **n̂** are generated at
displacement ε along the locally estimated surface normals, forming the
augmented training set T = [**P**; **P**+; **P**−] with corresponding
implicit-function targets d = [0; +ε; −ε].  The polynomial part β is
found by minimising the sum of squared algebraic distances over all 3N
training points subject to the Li–Griffiths ellipsoidal constraint:

    min_β  β^T **S** β    s.t.  β^T **C** β > 0

where **S** = **D**_T^T **D**_T is the augmented scatter matrix, **D**_T is
the (3N × 10) polynomial design matrix evaluated at T, and **C** is the
(10 × 10) ellipsoid-specific constraint matrix with parameter k = 4.

The polynomial coefficients β are selected as the eigenvector corresponding
to the largest positive eigenvalue of the reduced 6 × 6 problem

    **M** u = ν u,   **M** = **C**₁₁⁻¹ (**S**₁₁ − **S**₁₂ **S**₂₂⁻¹ **S**₂₁)

where **S** and **C** are partitioned into their 6 × 6 (upper) and 4 × 4
(lower) blocks.  Given β, the biharmonic RBF weights are recovered from
the interpolation system Φ **w** ≈ d − **D**_T β.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import eig, inv
from scipy.spatial import cKDTree


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

    The 6 × 6 upper-left block is defined by the ellipsoidal constraint:

        C[0,1] = C[1,0] = C[0,2] = C[2,0] = C[1,2] = C[2,1] = k/2
        C[3,3] = C[4,4] = C[5,5] = -k

    For the default k = 4 the constraint β^T C β > 0 is necessary and
    sufficient for the algebraic quadric to be an ellipsoid.

    Parameters
    ----------
    k : float, optional
        Constraint parameter, k ∈ (0, 4]. Default 4.

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

def _estimate_normals(pts: np.ndarray, k_neighbours: int = 15) -> np.ndarray:
    """Estimate outward unit normals at each surface point via local PCA.

    For each point the *k* nearest neighbours are gathered and the smallest
    eigenvector of the local covariance matrix is taken as the normal
    direction.  Orientations are made consistent by ensuring each normal
    points away from the cloud centroid.

    Parameters
    ----------
    pts : ndarray, shape (N, 3)
        Surface point cloud.
    k_neighbours : int, optional
        Neighbourhood size for local PCA.  Default 15.

    Returns
    -------
    normals : ndarray, shape (N, 3)
        Unit-length outward normal vectors.
    """ 
    n = len(pts)
    k_neighbours = min(k_neighbours, n)
    tree = cKDTree(pts)
    normals = np.empty_like(pts)
    for i in range(n):
        _, idx = tree.query(pts[i], k=k_neighbours)
        nbh = pts[idx] - pts[i]
        _, vecs = np.linalg.eigh(nbh.T @ nbh)
        normals[i] = vecs[:, 0]

    # Consistent outward orientation
    centroid = pts.mean(axis=0)
    outward = pts - centroid
    flip = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def _rbf_matrix(
    query_pts: np.ndarray,
    center_pts: np.ndarray,
) -> np.ndarray:
    """Build the RBF evaluation matrix using the biharmonic kernel φ(r) = r.

    Parameters
    ----------
    query_pts : ndarray, shape (M, 3)
        Points at which the RBF sum is evaluated (rows of the output).
    center_pts : ndarray, shape (N, 3)
        RBF centre locations (columns of the output).

    Returns
    -------
    Phi : ndarray, shape (M, N)
        Phi[i, j] = ‖query_pts[i] − center_pts[j]‖
    """ 
    diff = query_pts[:, np.newaxis, :] - center_pts[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

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

    M3 = np.array([
        [A, F, E],
        [F, B, D],
        [E, D, C],
    ])

    M4 = np.array([
        [A,  F,  E,  G],
        [F,  B,  D,  H],
        [E,  D,  C,  I],
        [G,  H,  I,  J],
    ])

    centre = -inv(M3) @ np.array([G, H, I])

    eigenvalues, eigenvectors = eig(M3)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    det_M4 = np.linalg.det(M4)
    det_M3 = np.linalg.det(M3)

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

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_ellipsoid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float = 4.0,
    epsilon: float | None = None,
    k_neighbours: int = 15,
) -> dict:
    """Fit an ellipsoid to 3-D point data using RBF implicit fitting.

    Implements the **Radial Basis (function) Representation (RBR) with
    Ellipsoidal Constraint** method described in:

        Li, Q. (2004).
        "Implicit fitting using radial basis functions with ellipsoidal
        constraint."
        *Computer Graphics Forum*, 23(1), 89–96. Wiley/Blackwell.
        https://doi.org/10.1111/j.1467-8659.2004.00756.x

    The algorithm:

    1. Estimates surface normals **n̂**_i via local PCA on k nearest
       neighbours.
    2. Generates off-surface points **P**± = **P** ± ε **n̂** at distance ε.
    3. Builds the augmented training set T = [**P**; **P**+; **P**−] and
       the (3N × 10) polynomial design matrix **D**_T.
    4. Solves the ellipsoid-constrained scatter problem on the augmented
       scatter matrix **S** = **D**_T^T **D**_T:

           min_β  β^T **S** β    s.t.  β^T **C** β > 0

       using the 6 × 6 generalised eigenvalue reduction of Li (2004).
    5. Recovers the biharmonic RBF weights from the residual
       Φ **w** ≈ d − **D**_T β.

    Parameters
    ----------
    x, y, z : array-like of shape (N,)
        Cartesian coordinates of the 3-D data points.  At least 10 points
        are required.
    k : float, optional
        Constraint parameter for the ellipsoidal constraint matrix **C**.
        Must satisfy k ∈ (0, 4].  Default 4.
    epsilon : float or None, optional
        Off-surface displacement distance used to generate the training
        set.  If *None* (default) it is set to 5 % of the mean coordinate
        standard deviation.
    k_neighbours : int, optional
        Number of nearest neighbours used for local-PCA normal estimation.
        Default 15.

    Returns
    -------
    result : dict
        A dictionary containing:

        ``centre``      : ndarray (3,)
            Coordinates of the ellipsoid centre.
        ``radii``       : ndarray (3,)
            Semi-axis lengths, sorted in descending order.
        ``axes``        : ndarray (3, 3)
            Unit vectors of the semi-axes (columns), same order as *radii*.
        ``M``           : ndarray (4, 4)
            Homogeneous 4 × 4 matrix representation of the quadric.
        ``coefficients``: ndarray (10,)
            Algebraic coefficients [A, B, C, D, E, F, G, H, I, J].
        ``rbf_weights`` : ndarray (N,)
            Biharmonic RBF weights w_i for the N surface-point centres.

    Raises
    ------
    ValueError
        If fewer than 10 data points are supplied, array lengths differ,
        or the fitted surface is not a valid ellipsoid.

    Examples
    --------
    >>> import numpy as np
    >>> from rbf_implicit_fitting import fit_ellipsoid, generate_ellipsoid_points
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

    pts = np.column_stack([x, y, z])

    # ------------------------------------------------------------------
    # Step 1 – estimate surface normals via local PCA
    # ------------------------------------------------------------------
    normals = _estimate_normals(pts, k_neighbours=k_neighbours)

    # ------------------------------------------------------------------
    # Step 2 – determine off-surface displacement ε
    # ------------------------------------------------------------------
    if epsilon is None:
        epsilon = float(np.mean(np.std(pts, axis=0))) * 0.05
        epsilon = max(epsilon, 1e-4)

    # ------------------------------------------------------------------
    # Step 3 – build training set T = [P; P+; P−]
    #          targets d = [0_N; +ε_N; −ε_N]
    # ------------------------------------------------------------------
    pts_plus  = pts + epsilon * normals
    pts_minus = pts - epsilon * normals
    T = np.vstack([pts, pts_plus, pts_minus])   # (3N, 3)
    d = np.concatenate([
        np.zeros(n),
        np.full(n,  epsilon),
        np.full(n, -epsilon),
    ])                                           # (3N,)

    # ------------------------------------------------------------------
    # Step 4 – augmented scatter matrix S = D_T^T D_T  (10 × 10)
    # ------------------------------------------------------------------
    D_T = _design_matrix(T[:, 0], T[:, 1], T[:, 2])   # (3N, 10)
    S = D_T.T @ D_T

    # ------------------------------------------------------------------
    # Step 5 – solve the ellipsoid-constrained scatter problem
    #
    #   Partition S and C into 6×6 (quadratic) and 4×4 (linear+const)
    #   blocks, then solve the reduced 6×6 eigenvalue problem:
    #       M u = ν u,  M = C₁₁⁻¹ (S₁₁ − S₁₂ S₂₂⁻¹ S₂₁)
    # ------------------------------------------------------------------
    C = _constraint_matrix(k)
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]
    C11 = C[:6, :6]

    try:
        S22_inv = inv(S22)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The S₂₂ block of the scatter matrix is singular. "
            "Consider adding more (or more diverse) data points."
        ) from exc

    M_mat = inv(C11) @ (S11 - S12 @ S22_inv @ S21)
    eigenvalues, eigenvectors = eig(M_mat)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    _tol = 1e-6 * max(np.max(np.abs(eigenvalues)), 1.0)
    if not np.any(eigenvalues > -_tol):
        raise ValueError(
            "No positive eigenvalue found. The data may not lie on an "
            "ellipsoid, or may be too noisy / degenerate."
        )
    best_idx = np.argmax(eigenvalues)
    u1 = eigenvectors[:, best_idx]
    u2 = -S22_inv @ S21 @ u1
    v = np.concatenate([u1, u2])
    v /= np.linalg.norm(v)

    # ------------------------------------------------------------------
    # Step 6 – recover biharmonic RBF weights for the N surface centres
    #
    #   Solve Φ w ≈ d − D_T β  in the least-squares sense, where
    #   Φ[i,j] = ‖T_i − pts_j‖  (biharmonic kernel, 3N centres = pts).
    # ------------------------------------------------------------------
    Phi = _rbf_matrix(T, pts)                     # (3N, N)
    residual = d - D_T @ v                         # (3N,)
    rbf_weights, *_ = np.linalg.lstsq(Phi, residual, rcond=None)  # (N,)

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

    Evaluates the polynomial part of the implicit function:
    F(**x**) = β^T **p**(**x**).  Ideally zero for points on the fitted
    ellipsoid surface.

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
