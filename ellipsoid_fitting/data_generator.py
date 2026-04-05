"""
Synthetic data generator for ellipsoid fitting experiments.

Generates 3-D point sets that lie on (or near) the surface of an
axis-aligned or arbitrarily oriented ellipsoid.
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
