"""
Tests for the multi-format data loader and RBF ellipsoid fitting.

Run with:
    pytest tests/test_loaders_and_rbf.py -v
"""

from __future__ import annotations

import os
import struct
import numpy as np
import pytest

from rbf_ellipsoid_constraint import (
    load_point_cloud,
    load_csv,
    load_xyz,
    load_obj,
    load_ply,
    load_matlab,
    load_npy,
    load_npz,
    generate_ellipsoid_points,
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _pts(n=100, seed=0):
    """Small synthetic ellipsoid point cloud for use in tests."""
    return generate_ellipsoid_points(
        centre=(0, 0, 0),
        radii=(3, 2, 1),
        n_points=n,
        noise_std=0.02,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_returns_array(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid_low_noise.csv")
        pts = load_csv(path)
        assert isinstance(pts, np.ndarray)
        assert pts.shape[1] == 3

    def test_matches_np_loadtxt(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid_low_noise.csv")
        pts = load_csv(path)
        expected = np.loadtxt(path, delimiter=",", skiprows=1)[:, :3]
        np.testing.assert_array_equal(pts, expected)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_point_cloud("/nonexistent/path/file.csv")

    def test_too_few_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("x,y\n1,2\n3,4\n")
        with pytest.raises(ValueError, match="at least 3 columns"):
            load_csv(str(p))


# ---------------------------------------------------------------------------
# load_xyz
# ---------------------------------------------------------------------------

class TestLoadXYZ:
    def test_roundtrip(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.xyz")
        np.savetxt(path, orig, fmt="%.8f")
        loaded = load_xyz(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_via_dispatcher_xyz(self, tmp_path):
        orig = _pts(30)
        path = str(tmp_path / "pts.xyz")
        np.savetxt(path, orig, fmt="%.8f")
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_via_dispatcher_txt(self, tmp_path):
        orig = _pts(30)
        path = str(tmp_path / "pts.txt")
        np.savetxt(path, orig, fmt="%.8f")
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_via_dispatcher_pts(self, tmp_path):
        orig = _pts(30)
        path = str(tmp_path / "pts.pts")
        np.savetxt(path, orig, fmt="%.8f")
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-6)


# ---------------------------------------------------------------------------
# load_obj
# ---------------------------------------------------------------------------

class TestLoadOBJ:
    def _write_obj(self, path, pts):
        with open(path, "w") as fh:
            fh.write("# test OBJ\n")
            for x, y, z in pts:
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            # Add a face line that should be ignored
            fh.write("f 1 2 3\n")

    def test_roundtrip(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.obj")
        self._write_obj(path, orig)
        loaded = load_obj(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_shape(self, tmp_path):
        orig = _pts(80)
        path = str(tmp_path / "pts.obj")
        self._write_obj(path, orig)
        loaded = load_obj(path)
        assert loaded.shape == (80, 3)

    def test_empty_obj_raises(self, tmp_path):
        path = str(tmp_path / "empty.obj")
        with open(path, "w") as fh:
            fh.write("# no vertices here\nf 1 2 3\n")
        with pytest.raises(ValueError, match="No vertex"):
            load_obj(path)

    def test_via_dispatcher(self, tmp_path):
        orig = _pts(40)
        path = str(tmp_path / "pts.obj")
        self._write_obj(path, orig)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_existing_sample_file(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid.obj")
        if not os.path.exists(path):
            pytest.skip("Sample OBJ file not found")
        pts = load_obj(path)
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# load_ply (ASCII)
# ---------------------------------------------------------------------------

class TestLoadPLYAscii:
    def _write_ply(self, path, pts):
        n = len(pts)
        with open(path, "w") as fh:
            fh.write("ply\n")
            fh.write("format ascii 1.0\n")
            fh.write(f"element vertex {n}\n")
            fh.write("property float x\n")
            fh.write("property float y\n")
            fh.write("property float z\n")
            fh.write("end_header\n")
            for x, y, z in pts:
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def test_roundtrip(self, tmp_path):
        orig = _pts(60)
        path = str(tmp_path / "pts.ply")
        self._write_ply(path, orig)
        loaded = load_ply(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_shape(self, tmp_path):
        orig = _pts(70)
        path = str(tmp_path / "pts.ply")
        self._write_ply(path, orig)
        loaded = load_ply(path)
        assert loaded.shape == (70, 3)

    def test_via_dispatcher(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.ply")
        self._write_ply(path, orig)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_existing_sample_file(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid.ply")
        if not os.path.exists(path):
            pytest.skip("Sample PLY file not found")
        pts = load_ply(path)
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0

    def test_not_ply_raises(self, tmp_path):
        path = str(tmp_path / "bad.ply")
        with open(path, "wb") as fh:
            fh.write(b"notaplyfile\n")
        with pytest.raises(ValueError, match="Not a valid PLY"):
            load_ply(path)

    def test_missing_xyz_properties_raises(self, tmp_path):
        path = str(tmp_path / "bad.ply")
        with open(path, "w") as fh:
            fh.write("ply\n")
            fh.write("format ascii 1.0\n")
            fh.write("element vertex 2\n")
            fh.write("property float x\n")
            fh.write("property float y\n")
            # no z property
            fh.write("end_header\n")
            fh.write("1.0 2.0\n")
            fh.write("3.0 4.0\n")
        with pytest.raises(ValueError, match="x, y, z"):
            load_ply(path)


# ---------------------------------------------------------------------------
# load_ply (binary little-endian)
# ---------------------------------------------------------------------------

class TestLoadPLYBinary:
    def _write_binary_ply(self, path, pts):
        n = len(pts)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        with open(path, "wb") as fh:
            fh.write(header)
            for x, y, z in pts:
                fh.write(struct.pack("<fff", float(x), float(y), float(z)))

    def test_roundtrip(self, tmp_path):
        orig = _pts(60)
        path = str(tmp_path / "pts_bin.ply")
        self._write_binary_ply(path, orig)
        loaded = load_ply(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_existing_binary_sample(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid_binary.ply")
        if not os.path.exists(path):
            pytest.skip("Binary PLY sample not found")
        pts = load_ply(path)
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# load_matlab
# ---------------------------------------------------------------------------

class TestLoadMatlab:
    def _write_m(self, path, pts):
        with open(path, "w") as fh:
            fh.write("% MATLAB point cloud\n")
            fh.write("data = [\n")
            for x, y, z in pts:
                fh.write(f"  {x:.6f}, {y:.6f}, {z:.6f};\n")
            fh.write("];\n")

    def test_roundtrip(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.m")
        self._write_m(path, orig)
        loaded = load_matlab(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_shape(self, tmp_path):
        orig = _pts(40)
        path = str(tmp_path / "pts.m")
        self._write_m(path, orig)
        loaded = load_matlab(path)
        assert loaded.shape == (40, 3)

    def test_no_pattern_raises(self, tmp_path):
        path = str(tmp_path / "bad.m")
        with open(path, "w") as fh:
            fh.write("% no data variable here\n")
        with pytest.raises(ValueError, match="data = "):
            load_matlab(path)

    def test_via_dispatcher(self, tmp_path):
        orig = _pts(30)
        path = str(tmp_path / "pts.m")
        self._write_m(path, orig)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_existing_sample_file(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid.m")
        if not os.path.exists(path):
            pytest.skip("Sample .m file not found")
        pts = load_matlab(path)
        assert pts.shape[1] == 3
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# load_npy / load_npz
# ---------------------------------------------------------------------------

class TestLoadNumpy:
    def test_npy_roundtrip(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.npy")
        np.save(path, orig)
        loaded = load_npy(path)
        np.testing.assert_array_equal(loaded, orig)

    def test_npz_roundtrip(self, tmp_path):
        orig = _pts(50)
        path = str(tmp_path / "pts.npz")
        np.savez(path, data=orig)
        loaded = load_npz(str(tmp_path / "pts.npz"))
        np.testing.assert_array_equal(loaded, orig)

    def test_npz_missing_key_raises(self, tmp_path):
        orig = _pts(20)
        path = str(tmp_path / "bad.npz")
        np.savez(path, points=orig)
        with pytest.raises(ValueError, match="'data' key"):
            load_npz(str(tmp_path / "bad.npz"))

    def test_npy_wrong_shape_raises(self, tmp_path):
        path = str(tmp_path / "bad.npy")
        np.save(path, np.ones((50,)))  # 1-D
        with pytest.raises(ValueError, match="2-D"):
            load_npy(path)

    def test_via_dispatcher_npy(self, tmp_path):
        orig = _pts(30)
        path = str(tmp_path / "pts.npy")
        np.save(path, orig)
        loaded = load_point_cloud(path)
        np.testing.assert_array_equal(loaded, orig)


# ---------------------------------------------------------------------------
# load_point_cloud dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_unknown_extension_raises(self, tmp_path):
        path = str(tmp_path / "pts.stl")
        open(path, "w").close()
        with pytest.raises(ValueError, match="Unrecognised file extension"):
            load_point_cloud(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_point_cloud("/tmp/does_not_exist_12345.csv")

    def test_csv_via_dispatcher(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid_low_noise.csv")
        pts = load_point_cloud(path)
        assert pts.shape == (300, 3)


# ---------------------------------------------------------------------------
# fit_rbf_ellipsoid_linear
# ---------------------------------------------------------------------------

class TestFitRBFEllipsoidLinear:
    def test_returns_four_tuple(self):
        pts = _pts(150)
        result = fit_rbf_ellipsoid_linear(pts)
        assert result is not None
        alpha, beta, cent, scale = result
        assert isinstance(alpha, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert isinstance(cent, np.ndarray)
        assert isinstance(scale, float)

    def test_alpha_shape(self):
        pts = _pts(120)
        alpha, beta, cent, scale = fit_rbf_ellipsoid_linear(pts)
        assert alpha.shape == (120,)

    def test_beta_shape(self):
        pts = _pts(100)
        alpha, beta, cent, scale = fit_rbf_ellipsoid_linear(pts)
        assert beta.shape == (10,)

    def test_centroid_and_scale(self):
        true_centre = np.array([1.0, 2.0, 3.0])
        pts = generate_ellipsoid_points(
            centre=true_centre, radii=(3, 2, 1),
            n_points=200, noise_std=0.0, seed=5
        )
        _, _, cent, scale = fit_rbf_ellipsoid_linear(pts)
        np.testing.assert_allclose(cent, true_centre, atol=0.1)
        assert scale > 0

    def test_too_few_points_raises(self):
        pts = _pts(5)
        with pytest.raises(ValueError, match="At least 10"):
            fit_rbf_ellipsoid_linear(pts)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            fit_rbf_ellipsoid_linear(np.ones((50, 2)))

    def test_with_smoothing(self):
        pts = _pts(150)
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


# ---------------------------------------------------------------------------
# evaluate_model_linear
# ---------------------------------------------------------------------------

class TestEvaluateModelLinear:
    def test_output_shape(self):
        pts = _pts(150)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
        assert vals.shape == (150,)

    def test_residuals_small_on_surface(self):
        """Residuals should be small for points lying on the ellipsoid."""
        pts = generate_ellipsoid_points(
            radii=(3, 2, 1), n_points=300, noise_std=0.01, seed=42
        )
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
        assert np.mean(np.abs(vals)) < 0.5

    def test_chunking_consistent(self):
        """Results should be identical regardless of chunk_size."""
        pts = _pts(200)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        v1 = evaluate_model_linear(norm_pts, norm_pts, alpha, beta, chunk_size=50)
        v2 = evaluate_model_linear(norm_pts, norm_pts, alpha, beta, chunk_size=200)
        np.testing.assert_allclose(v1, v2, atol=1e-10)


# ---------------------------------------------------------------------------
# Integration: load file -> RBF fit
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.parametrize("fname", [
        "synthetic_ellipsoid.obj",
        "synthetic_ellipsoid.ply",
        "synthetic_ellipsoid_binary.ply",
        "synthetic_ellipsoid.xyz",
        "synthetic_ellipsoid.m",
    ])
    def test_load_and_fit_rbf(self, fname):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            pytest.skip(f"Sample file not found: {fname}")
        pts = load_point_cloud(path)
        assert pts.shape[1] == 3
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None

    def test_load_csv_and_fit_rbf(self):
        path = os.path.join(DATA_DIR, "synthetic_ellipsoid_low_noise.csv")
        pts = load_point_cloud(path)
        result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
        assert result is not None
        alpha, beta, cent, scale = result
        norm_pts = (pts - cent) / scale
        vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
        assert np.mean(np.abs(vals)) < 1.0
