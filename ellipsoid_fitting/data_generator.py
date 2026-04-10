"""
Synthetic data generator for RBF implicit surface fitting experiments.

Provides generators for ellipsoid and several non-ellipsoidal 3-D point
clouds (torus, superquadric, bumpy sphere, saddle patch) as well as a
convenience dispatcher ``generate_synthetic_points``.
"""

from __future__ import annotations

import numpy as np


def generate_ellipsoid_points(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radii: tuple[float, float, float] = (3.0, 2.0, 1.0),
    rotation: np.ndarray | None = None,
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate 3-D points sampled uniformly on an ellipsoid surface.

    Points are first generated on a unit sphere using the Fibonacci
    sphere algorithm (quasi-uniform coverage), then scaled to the
    requested semi-axis lengths, optionally rotated, shifted to the
    given centre, and perturbed with Gaussian noise.

    Parameters
    ----------
    centre : tuple of float, optional
        (cx, cy, cz) – ellipsoid centre (default (0, 0, 0)).
    radii : tuple of float, optional
        (a, b, c) – semi-axis lengths along x, y, z *before* rotation
        (default (3, 2, 1)).
    rotation : ndarray of shape (3, 3) or None, optional
        Orthogonal rotation matrix applied after scaling.
        When *None* (default) the ellipsoid is axis-aligned.
    n_points : int, optional
        Number of surface points to generate (default 300).
    noise_std : float, optional
        Standard deviation of isotropic Gaussian noise added to each
        point (default 0.05).  Set to 0 for noise-free data.
    seed : int or None, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    pts : ndarray of shape (n_points, 3)
        Columns are x, y, z coordinates.

    Examples
    --------
    >>> pts = generate_ellipsoid_points(centre=(1, 2, 3), radii=(5, 3, 2),
    ...                                  n_points=200, noise_std=0.0)
    >>> pts.shape
    (200, 3)
    """
    rng = np.random.default_rng(seed)

    # Fibonacci sphere for quasi-uniform coverage
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n_points)
    theta = 2 * np.pi * i / golden          # azimuthal angle
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)  # polar angle

    xs = np.sin(phi) * np.cos(theta)
    ys = np.sin(phi) * np.sin(theta)
    zs = np.cos(phi)

    pts = np.column_stack([xs, ys, zs])

    # Scale to ellipsoid
    pts *= np.array(radii, dtype=float)

    # Optional rotation
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a (3, 3) array.")
        pts = pts @ rotation.T

    # Shift to centre
    pts += np.array(centre, dtype=float)

    # Add noise
    if noise_std > 0:
        pts += rng.normal(0.0, noise_std, size=pts.shape)

    return pts


def generate_torus_points(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    major_radius: float = 3.0,
    minor_radius: float = 1.0,
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate 3-D points sampled on a torus surface.

    The torus lies in the xy-plane centred at `centre`.  Its surface is
    parametrised by two angles (u, v):

        x = (R + r·cos v)·cos u
        y = (R + r·cos v)·sin u
        z = r·sin v

    where R = `major_radius` and r = `minor_radius`.

    Parameters
    ----------
    centre : 3-tuple of float
        Centre of the torus (default (0, 0, 0)).
    major_radius : float
        Distance from the centre of the tube to the centre of the torus
        (default 3.0).  Must be > minor_radius.
    minor_radius : float
        Radius of the tube (default 1.0).
    n_points : int
        Number of surface points (default 300).
    noise_std : float
        Isotropic Gaussian noise added to each point (default 0.05).
    seed : int or None
        Random seed (default 42).

    Returns
    -------
    ndarray of shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)
    R, r = float(major_radius), float(minor_radius)

    # Draw u uniformly; use rejection sampling for v so the density on
    # the torus surface is uniform (acceptance probability ∝ R + r·cos v).
    pts_list = []
    max_weight = R + r
    while len(pts_list) < n_points:
        batch = n_points * 4
        u = rng.uniform(0, 2 * np.pi, batch)
        v = rng.uniform(0, 2 * np.pi, batch)
        accept_prob = (R + r * np.cos(v)) / max_weight
        mask = rng.uniform(0, 1, batch) < accept_prob
        u, v = u[mask], v[mask]
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        pts_list.append(np.column_stack([x, y, z]))

    pts = np.vstack(pts_list)[:n_points]
    pts += np.array(centre, dtype=float)

    if noise_std > 0:
        pts += rng.normal(0.0, noise_std, size=pts.shape)

    return pts


def generate_superquadric_points(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scales: tuple[float, float, float] = (3.0, 2.0, 1.5),
    exponents: tuple[float, float] = (0.5, 0.5),
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate 3-D points sampled on a superquadric surface.

    A superquadric is parameterised by two shape exponents (ε₁, ε₂):

        x = a · sign(cos η)·|cos η|^ε₁ · sign(cos ω)·|cos ω|^ε₂
        y = b · sign(cos η)·|cos η|^ε₁ · sign(sin ω)·|sin ω|^ε₂
        z = c · sign(sin η)·|sin η|^ε₁

    where η ∈ [−π/2, π/2] and ω ∈ [−π, π].

    Setting ε₁ = ε₂ = 1 gives an ellipsoid; values < 1 give pinched
    ("peanut") shapes; values > 1 give cuboidal shapes.

    Parameters
    ----------
    centre : 3-tuple of float
    scales : 3-tuple of float
        Semi-axis lengths (a, b, c).
    exponents : 2-tuple of float
        Shape exponents (ε₁, ε₂) — both > 0.  Default (0.5, 0.5) gives a
        pinched/peanut shape.
    n_points : int
    noise_std : float
    seed : int or None

    Returns
    -------
    ndarray of shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)
    a, b, c = float(scales[0]), float(scales[1]), float(scales[2])
    e1, e2 = float(exponents[0]), float(exponents[1])

    eta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)
    omega = rng.uniform(-np.pi, np.pi, n_points)

    def _signed_pow(base: np.ndarray, exp: float) -> np.ndarray:
        return np.sign(base) * (np.abs(base) ** exp)

    cos_eta = np.cos(eta)
    sin_eta = np.sin(eta)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    x = a * _signed_pow(cos_eta, e1) * _signed_pow(cos_omega, e2)
    y = b * _signed_pow(cos_eta, e1) * _signed_pow(sin_omega, e2)
    z = c * _signed_pow(sin_eta, e1)

    pts = np.column_stack([x, y, z])
    pts += np.array(centre, dtype=float)

    if noise_std > 0:
        pts += rng.normal(0.0, noise_std, size=pts.shape)

    return pts


def generate_bumpy_sphere_points(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_radius: float = 3.0,
    bump_amplitude: float = 0.4,
    n_bumps: int = 6,
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate a point cloud on a bumpy-sphere surface.

    A unit sphere is uniformly sampled (Fibonacci lattice), then each
    point's radius is modulated by a sum of `n_bumps` sinusoidal lobes
    with random frequencies and phases, creating a smooth closed surface
    that is topologically a sphere but clearly not an ellipsoid.

    Parameters
    ----------
    centre : 3-tuple of float
    base_radius : float
        Mean radius of the surface (default 3.0).
    bump_amplitude : float
        Peak-to-peak amplitude of the surface bumps as a fraction of
        `base_radius` (default 0.4 → ±40 % modulation).
    n_bumps : int
        Number of sinusoidal bump components (default 6).
    n_points : int
    noise_std : float
    seed : int or None

    Returns
    -------
    ndarray of shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)

    # Fibonacci sphere for quasi-uniform unit directions
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n_points)
    theta = 2 * np.pi * i / golden           # azimuthal angle
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)  # polar angle

    # Random frequencies and phases for the bump components
    A_k = rng.uniform(0.5, 1.0, n_bumps)
    A_k /= A_k.sum()                         # normalise amplitudes
    f_k = rng.integers(1, 4, size=n_bumps).astype(float)
    g_k = rng.integers(1, 4, size=n_bumps).astype(float)
    p_k = rng.uniform(0, 2 * np.pi, n_bumps)
    q_k = rng.uniform(0, 2 * np.pi, n_bumps)

    bump = np.zeros(n_points)
    for k in range(n_bumps):
        bump += A_k[k] * np.sin(f_k[k] * theta + p_k[k]) * np.cos(g_k[k] * phi + q_k[k])

    r = base_radius * (1.0 + bump_amplitude * bump)

    xs = r * np.sin(phi) * np.cos(theta)
    ys = r * np.sin(phi) * np.sin(theta)
    zs = r * np.cos(phi)

    pts = np.column_stack([xs, ys, zs])
    pts += np.array(centre, dtype=float)

    if noise_std > 0:
        pts += rng.normal(0.0, noise_std, size=pts.shape)

    return pts


def generate_saddle_points(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_x: float = 3.0,
    scale_y: float = 3.0,
    curvature: float = 1.0,
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate a point cloud on a saddle (hyperbolic paraboloid) surface.

    The surface is defined by z = curvature*(x²/scale_x² − y²/scale_y²)
    for (x, y) uniformly distributed in [−scale_x, scale_x] × [−scale_y, scale_y].

    Parameters
    ----------
    centre : 3-tuple of float
    scale_x, scale_y : float
        Half-width of the sampled domain in x and y (default 3.0).
    curvature : float
        Controls how steeply the surface curves (default 1.0).
    n_points : int
    noise_std : float
    seed : int or None

    Returns
    -------
    ndarray of shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-scale_x, scale_x, n_points)
    y = rng.uniform(-scale_y, scale_y, n_points)
    z = curvature * (x ** 2 / scale_x ** 2 - y ** 2 / scale_y ** 2)

    pts = np.column_stack([x, y, z])
    pts += np.array(centre, dtype=float)

    if noise_std > 0:
        pts += rng.normal(0.0, noise_std, size=pts.shape)

    return pts


def generate_synthetic_points(
    shape: str = "bumpy_sphere",
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int | None = 42,
    **kwargs,
) -> np.ndarray:
    """Generate synthetic point clouds of various non-ellipsoidal shapes.

    A convenience dispatcher.  Use `shape` to select the surface type;
    remaining keyword arguments are forwarded to the underlying generator.

    Parameters
    ----------
    shape : str
        One of ``"ellipsoid"``, ``"torus"``, ``"superquadric"``,
        ``"bumpy_sphere"``, ``"saddle"``.
    n_points : int
    noise_std : float
    seed : int or None
    **kwargs
        Passed verbatim to the selected generator.

    Returns
    -------
    ndarray of shape (n_points, 3)

    Raises
    ------
    ValueError
        If `shape` is not one of the supported names.
    """
    _generators = {
        "ellipsoid": generate_ellipsoid_points,
        "torus": generate_torus_points,
        "superquadric": generate_superquadric_points,
        "bumpy_sphere": generate_bumpy_sphere_points,
        "saddle": generate_saddle_points,
    }
    if shape not in _generators:
        raise ValueError(
            f"Unknown shape {shape!r}. "
            f"Choose one of: {sorted(_generators)}"
        )
    return _generators[shape](n_points=n_points, noise_std=noise_std, seed=seed, **kwargs)
