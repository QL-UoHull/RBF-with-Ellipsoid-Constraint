"""
Synthetic data generators for RBF implicit surface fitting experiments.

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

    The torus lies in the xy-plane centred at `centre`. Parametrised by:
        x = (R + r·cos v)·cos u
        y = (R + r·cos v)·sin u
        z = r·sin v
    where R = major_radius, r = minor_radius.

    Uses rejection-corrected random sampling: u uniform in [0,2π];
    v accepted with probability proportional to (R + r·cos v) / (R + r).

    Returns ndarray of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    R, r = major_radius, minor_radius

    pts = []
    while len(pts) < n_points:
        # Over-sample to account for rejection
        batch = max(n_points * 4, 1000)
        u = rng.uniform(0, 2 * np.pi, batch)
        v = rng.uniform(0, 2 * np.pi, batch)
        # Acceptance probability proportional to (R + r*cos(v)) / (R + r)
        prob = (R + r * np.cos(v)) / (R + r)
        accept = rng.uniform(0, 1, batch) < prob
        u = u[accept]
        v = v[accept]
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        pts.extend(zip(x, y, z))

    pts = np.array(pts[:n_points], dtype=float)
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
    """Generate 3-D points on a superquadric surface.

    Parametrised by two shape exponents (ε₁, ε₂):
        x = a · sign(cos η)·|cos η|^ε₁ · sign(cos ω)·|cos ω|^ε₂
        y = b · sign(cos η)·|cos η|^ε₁ · sign(sin ω)·|sin ω|^ε₂
        z = c · sign(sin η)·|sin η|^ε₁
    where η ∈ [−π/2, π/2] and ω ∈ [−π, π].
    Default exponents (0.5, 0.5) give a pinched/peanut shape.

    Returns ndarray of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    a, b, c = scales
    e1, e2 = exponents

    eta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)
    omega = rng.uniform(-np.pi, np.pi, n_points)

    def _signed_pow(base, exp):
        return np.sign(base) * np.abs(base) ** exp

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

    Uses a Fibonacci lattice for quasi-uniform directions, then modulates
    each point's radius by a sum of n_bumps sinusoidal lobes with random
    frequencies and phases. The result is a smooth closed surface
    topologically equivalent to a sphere but clearly not an ellipsoid.

    Returns ndarray of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)

    # Fibonacci lattice for quasi-uniform sphere coverage
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n_points)
    theta = 2 * np.pi * i / golden
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)

    # Unit directions
    dirs = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])

    # Random bump frequencies and phases
    freq_u = rng.integers(1, 4, size=n_bumps).astype(float)
    freq_v = rng.integers(1, 4, size=n_bumps).astype(float)
    phase = rng.uniform(0, 2 * np.pi, size=n_bumps)

    # Modulate radius
    radii = base_radius * np.ones(n_points)
    for k in range(n_bumps):
        radii += bump_amplitude * np.sin(freq_u[k] * theta + phase[k]) * np.cos(freq_v[k] * phi)

    pts = dirs * radii[:, np.newaxis]
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

    z = curvature * (x²/scale_x² − y²/scale_y²)
    for (x, y) uniformly distributed in [−scale_x, scale_x] × [−scale_y, scale_y].

    Returns ndarray of shape (n_points, 3).
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
    """Convenience dispatcher for all synthetic point cloud generators.

    shape: one of "ellipsoid", "torus", "superquadric", "bumpy_sphere", "saddle".
    Raises ValueError for unknown shape names.
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
            f"Unknown shape '{shape}'. "
            f"Choose one of: {', '.join(sorted(_generators))}."
        )
    return _generators[shape](n_points=n_points, noise_std=noise_std, seed=seed, **kwargs)
