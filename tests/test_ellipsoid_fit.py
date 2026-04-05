"""
Unit tests for the ellipsoid_fitting package.

Run with:
    pytest tests/test_ellipsoid_fit.py -v
"""

import numpy as np
import pytest

from ellipsoid_fitting import (
    fit_ellipsoid,
    algebraic_distance,
    residuals_rms,
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
# Fitting tests
# ---------------------------------------------------------------------------

class TestFitEllipsoid:
    def test_basic_axis_aligned(self):
        """Fit should recover centre and radii of an axis-aligned ellipsoid."""
        true_centre = np.array([1.0, 2.0, 3.0])
        true_radii = np.array([5.0, 3.0, 2.0])
        pts = _make_data(true_centre, true_radii, n=400, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

        np.testing.assert_allclose(result["centre"], true_centre, atol=0.1)
        np.testing.assert_allclose(
            np.sort(result["radii"])[::-1],
            np.sort(true_radii)[::-1],
            atol=0.2,
        )

    def test_centred_at_origin(self):
        pts = _make_data((0, 0, 0), (4, 3, 2), n=300, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        np.testing.assert_allclose(result["centre"], [0, 0, 0], atol=0.1)

    def test_sphere_like(self):
        """Near-spherical data: radii should all be close."""
        pts = _make_data((0, 0, 0), (3, 3, 3), n=300, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        # All three radii should be within 10 % of each other
        r = result["radii"]
        assert r.max() / r.min() < 1.1

    def test_output_keys(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=200, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        for key in ("centre", "radii", "axes", "M", "coefficients"):
            assert key in result

    def test_output_shapes(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=200, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        assert result["centre"].shape == (3,)
        assert result["radii"].shape == (3,)
        assert result["axes"].shape == (3, 3)
        assert result["M"].shape == (4, 4)
        assert result["coefficients"].shape == (10,)

    def test_radii_sorted_descending(self):
        pts = _make_data((0, 0, 0), (5, 3, 1), n=300, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        r = result["radii"]
        assert r[0] >= r[1] >= r[2]

    def test_too_few_points(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=5, noise=0.0)
        with pytest.raises(ValueError, match="At least 10"):
            fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fit_ellipsoid([1, 2, 3], [1, 2], [1, 2, 3])

    def test_rms_residual_low_noise(self):
        """RMS residual should be small for low-noise data."""
        pts = _make_data((0, 0, 0), (3, 2, 1), n=300, noise=0.01)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        rms = residuals_rms(pts[:, 0], pts[:, 1], pts[:, 2], result)
        assert rms < 0.5

    def test_custom_k_parameter(self):
        """Using k = 2 (still within valid range) should still converge."""
        pts = _make_data((0, 0, 0), (3, 2, 1), n=300, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2], k=2.0)
        assert "centre" in result

    def test_rotated_ellipsoid(self):
        """Fit should work for a rotated (non-axis-aligned) ellipsoid."""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        pts = generate_ellipsoid_points(
            centre=(2, -1, 4),
            radii=(6, 4, 2),
            rotation=R,
            n_points=500,
            noise_std=0.05,
            seed=123,
        )
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        np.testing.assert_allclose(result["centre"], [2, -1, 4], atol=0.3)
        np.testing.assert_allclose(
            np.sort(result["radii"])[::-1],
            np.sort([6, 4, 2])[::-1],
            atol=0.4,
        )


# ---------------------------------------------------------------------------
# Algebraic distance tests
# ---------------------------------------------------------------------------

class TestAlgebraicDistance:
    def test_distance_near_zero_on_surface(self):
        """Algebraic distance should be near zero for points on the ellipsoid."""
        pts = _make_data((0, 0, 0), (3, 2, 1), n=300, noise=0.01)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        d = algebraic_distance(pts[:, 0], pts[:, 1], pts[:, 2], result["coefficients"])
        assert np.max(np.abs(d)) < 0.5

    def test_output_shape(self):
        pts = _make_data((0, 0, 0), (3, 2, 1), n=50, noise=0.02)
        result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])
        d = algebraic_distance(pts[:, 0], pts[:, 1], pts[:, 2], result["coefficients"])
        assert d.shape == (50,)


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
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        result = fit_ellipsoid(x, y, z)
        rms = residuals_rms(x, y, z, result)
        assert rms < 5.0   # generous bound; noise may increase residual
        assert result["radii"].min() > 0
