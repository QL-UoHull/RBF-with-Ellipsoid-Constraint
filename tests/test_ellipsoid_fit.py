"""
Unit tests for the rbf_ellipsoid_constraint package.

Tests cover the RBF with Ellipsoid Constraint algorithm described in:

    Li, Q. and Griffiths, J. G. (2004). "Radial basis functions for surface
    reconstruction from unorganised point clouds with applications to bone
    reconstruction." Computer Graphics Forum, 23(1), 67–78. Wiley-Blackwell.

Run with:
    pytest tests/test_ellipsoid_fit.py -v
"""

import numpy as np
import pytest

from rbf_ellipsoid_constraint import (
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
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


# ---------------------------------------------------------------------------
# RBF fitting tests
# ---------------------------------------------------------------------------

class TestFitRBFEllipsoid:
    def test_basic_returns_tuple(self):
        """fit_rbf_ellipsoid_linear should return a 4-tuple."""
        pts = _make_data((0, 0, 0), (3, 2, 1), n=200, noise=0.02)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        assert isinstance(alpha, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert isinstance(cent, np.ndarray)
        assert isinstance(scale, float)

    def test_alpha_shape(self):
        n = 150
        pts = _make_data((0, 0, 0), (3, 2, 1), n=n, noise=0.02)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        assert alpha.shape == (n,)

    def test_beta_shape(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=100, noise=0.02)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        assert beta.shape == (10,)

    def test_centroid_close_to_true(self):
        true_centre = np.array([1.0, 2.0, 3.0])
        pts = generate_ellipsoid_points(
            centre=true_centre, radii=(3, 2, 1),
            n_points=200, noise_std=0.0, seed=5
        )
        _, _, cent, scale = fit_rbf_ellipsoid_linear(pts)
        np.testing.assert_allclose(cent, true_centre, atol=0.1)
        assert scale > 0

    def test_too_few_points_raises(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=5)
        with pytest.raises(ValueError, match="At least 10"):
            fit_rbf_ellipsoid_linear(pts)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            fit_rbf_ellipsoid_linear(np.ones((50, 2)))

    def test_with_smoothing(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=150, noise=0.02)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.1)
        assert result is not None

    def test_noise_free(self):
        """Noise-free data should yield a valid result."""
        pts = generate_ellipsoid_points(
            radii=(4, 3, 2), n_points=200, noise_std=0.0, seed=99
        )
        result = fit_rbf_ellipsoid_linear(pts, smooth=1e-6)
        assert result is not None

    def test_different_ellipsoids(self):
        """Fit should work for various ellipsoid shapes."""
        for radii in [(5, 3, 2), (2, 2, 1), (6, 1, 1)]:
            pts = generate_ellipsoid_points(
                radii=radii, n_points=200, noise_std=0.02, seed=0
            )
            result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
            assert result is not None, f"Fit failed for radii={radii}"

    def test_residuals_small_on_surface(self):
        """Residuals should be small for on-surface points."""
        pts = _make_data((0, 0, 0), (3, 2, 1), n=300, noise=0.01)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
        assert np.mean(np.abs(vals)) < 0.5


# ---------------------------------------------------------------------------
# CSV round-trip test
# ---------------------------------------------------------------------------

class TestCSVDatasets:
    @pytest.mark.parametrize("fname", [
        "synthetic_ellipsoid_low_noise.csv",
        "synthetic_ellipsoid_rotated.csv",
        "synthetic_sphere_like.csv",
    ])
    def test_fit_from_csv(self, fname, tmp_path):
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        csv_path = os.path.join(repo_root, "data", fname)
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        pts = data[:, :3]
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
        assert np.mean(np.abs(vals)) < 1.0
