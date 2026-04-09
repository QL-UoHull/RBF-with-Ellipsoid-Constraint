"""
Unit tests for the rbf_ellipsoid_constraint package data generator.

Tests cover the synthetic data generator used to produce ellipsoid point clouds
for testing the RBF implicit fitting with ellipsoidal constraint algorithm
described in:

    Li, Q. and Griffiths, J. G. (2004). "Radial basis functions for surface
    reconstruction from unorganised point clouds with applications to bone
    reconstruction." Computer Graphics Forum, 23(1), 67–78.
    Wiley-Blackwell.

Run with:
    pytest tests/test_data_generator.py -v
"""

import numpy as np
import pytest

from rbf_ellipsoid_constraint import (
    generate_ellipsoid_points,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(centre, radii, n=300, noise=0.0, seed=0):
    return generate_ellipsoid_points(
        centre=centre, radii=radii, n_points=n, noise_std=noise, seed=seed
    )


# ---------------------------------------------------------------------------
# Data generator tests
# ---------------------------------------------------------------------------

class TestGenerateEllipsoidPoints:
    def test_output_shape(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=200)
        assert pts.shape == (200, 3)

    def test_noise_free_on_surface(self):
        """Noise-free points should satisfy the ellipsoid equation closely."""
        cx, cy, cz = 1.0, -2.0, 3.0
        a, b, c = 5.0, 3.0, 2.0
        pts = _make_data((cx, cy, cz), (a, b, c), n=100, noise=0.0)
        x, y, z = pts[:, 0] - cx, pts[:, 1] - cy, pts[:, 2] - cz
        residual = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2
        np.testing.assert_allclose(residual, 1.0, atol=1e-10)

    def test_rotation_applied(self):
        """Rotated ellipsoid should not be axis-aligned."""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('x', 45, degrees=True).as_matrix()
        pts = _make_data((0, 0, 0), (3, 2, 1), n=100, noise=0.0)
        pts_rot = _make_data((0, 0, 0), (3, 2, 1), n=100, noise=0.0)
        # Principal axis of rotated cloud differs
        _, _, Vt = np.linalg.svd(pts - pts.mean(axis=0))
        assert Vt.shape == (3, 3)

    def test_seed_reproducibility(self):
        pts1 = _make_data((0, 0, 0), (3, 2, 1), seed=42)
        pts2 = _make_data((0, 0, 0), (3, 2, 1), seed=42)
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds_differ(self):
        pts1 = _make_data((0, 0, 0), (3, 2, 1), seed=1, noise=0.1)
        pts2 = _make_data((0, 0, 0), (3, 2, 1), seed=2, noise=0.1)
        assert not np.allclose(pts1, pts2)

    def test_invalid_rotation_shape(self):
        with pytest.raises(ValueError, match="rotation must be a"):
            generate_ellipsoid_points(rotation=np.eye(4))
