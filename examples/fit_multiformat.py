"""
Multi-format example: load a 3-D point cloud from various file formats and
fit an ellipsoid using the RBF with Ellipsoid Constraint algorithm.

Supported input formats demonstrated here:
  .csv  .obj  .ply  .xyz  .txt  .pts  .m  .npy  .npz

Run with:
    python examples/fit_multiformat.py
    python examples/fit_multiformat.py data/synthetic_ellipsoid.obj
    python examples/fit_multiformat.py data/synthetic_ellipsoid.ply
    python examples/fit_multiformat.py data/synthetic_ellipsoid.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend so the script runs without a display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rbf_ellipsoid_constraint import (
    load_point_cloud,
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
    FORMAT_LOADERS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_sep(title=""):
    width = 60
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"\n{'─' * (pad // 2)} {title} {'─' * (pad - pad // 2)}")
    else:
        print("─" * width)


def demo_rbf(pts: np.ndarray, label: str, smooth: float = 0.05):
    """Fit using the RBF-with-Ellipsoid-Constraint (Li & Griffiths, CGF 2004)."""
    result = fit_rbf_ellipsoid_linear(pts, smooth=smooth)
    if result is None:
        print("  [RBF] No valid eigenvalue found – fit failed.")
        return None
    alpha, beta, cent, scale = result
    norm_pts = (pts - cent) / scale
    vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
    mean_err = float(np.mean(np.abs(vals)))
    max_err = float(np.max(np.abs(vals)))
    print(f"  [RBF] Centroid : {cent.round(4)}")
    print(f"  [RBF] Scale    : {scale:.4f}")
    print(f"  [RBF] Mean |F| : {mean_err:.6f}")
    print(f"  [RBF] Max  |F| : {max_err:.6f}")
    return alpha, beta, cent, scale


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_and_save(pts, rbf_result, label, out_path):
    """Save a scatter + RBF isosurface plot."""
    try:
        from skimage import measure
        has_skimage = True
    except ImportError:
        has_skimage = False

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=4, alpha=0.4, c="k", label="Points",
    )
    if rbf_result is not None and has_skimage:
        alpha_rbf, beta_rbf, cent, scale = rbf_result
        norm_pts = (pts - cent) / scale
        res = 40
        limit = 1.3
        gx = np.linspace(-limit, limit, res)
        GX, GY, GZ = np.meshgrid(gx, gx, gx, indexing="ij")
        grid_pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        vol = evaluate_model_linear(grid_pts, norm_pts, alpha_rbf, beta_rbf)
        vol = vol.reshape(res, res, res)
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, 0.0)
            step = 2 * limit / (res - 1)
            verts_w = verts * step - limit
            verts_w = verts_w * scale + cent
            ax.plot_trisurf(
                verts_w[:, 0], verts_w[:, 1], verts_w[:, 2],
                triangles=faces, color="lime", alpha=0.35,
                edgecolor="none", shade=True,
            )
        except Exception:
            pass
    elif rbf_result is not None:
        ax.set_title("Install scikit-image for RBF isosurface")
    ax.set_title(f"RBF with Ellipsoid Constraint\n({label})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_on_file(filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    label = os.path.basename(filepath)

    _print_sep(label)
    print(f"  Format : {ext!r}")

    try:
        pts = load_point_cloud(filepath)
    except Exception as exc:
        print(f"  ERROR loading file: {exc}")
        return

    print(f"  Points : {len(pts)}")

    rbf = None
    try:
        rbf = demo_rbf(pts, label)
    except Exception as exc:
        print(f"  [RBF] FAILED: {exc}")

    out_png = os.path.join(
        os.path.dirname(__file__),
        f"fit_{os.path.splitext(label)[0]}.png",
    )
    try:
        visualise_and_save(pts, rbf, label, out_png)
    except Exception as exc:
        print(f"  [Plot] FAILED: {exc}")


def main():
    if len(sys.argv) > 1:
        # Run on files provided via command-line arguments
        for path in sys.argv[1:]:
            run_on_file(path)
    else:
        # Default: run on all files in data/ that have a supported extension
        supported_ext = set(FORMAT_LOADERS.keys())
        data_files = sorted(
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if os.path.splitext(f)[1].lower() in supported_ext
        )
        if not data_files:
            print("No data files found in data/.  Run the data generator first.")
            return
        _print_sep("Multi-format RBF with Ellipsoid Constraint demo")
        print(f"  Found {len(data_files)} data file(s).")
        for path in data_files:
            run_on_file(path)
        _print_sep()
        print("Done.")


if __name__ == "__main__":
    main()
