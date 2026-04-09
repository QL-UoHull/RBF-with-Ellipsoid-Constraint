"""
Multi-format example: load a 3-D point cloud from various file formats and
fit an ellipsoid using the algebraic ellipsoid fitting algorithm in the package.

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

from rbf_implicit_fitting import (
    load_point_cloud,
    fit_ellipsoid,
    residuals_rms,
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


def demo_algebraic(pts: np.ndarray, label: str) -> None:
    """Fit using the RBF implicit fitting with ellipsoidal constraint (Li, CGF 2004)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    result = fit_ellipsoid(x, y, z)
    rms = residuals_rms(x, y, z, result)
    print(f"  [Algebraic] Centre : {result['centre'].round(4)}")
    print(f"  [Algebraic] Radii  : {result['radii'].round(4)}")
    print(f"  [Algebraic] RMS    : {rms:.6f}")
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_and_save(pts, algebraic_result, label, out_path):
    """Save a scatter + surface plot."""
    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(111, projection="3d")
    ax1.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=4, alpha=0.6, c="steelblue", label="Points",
    )
    if algebraic_result is not None:
        cx, cy, cz = algebraic_result["centre"]
        r1, r2, r3 = algebraic_result["radii"]
        axes = algebraic_result["axes"]
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        xs = r1 * np.outer(np.cos(u), np.sin(v))
        ys = r2 * np.outer(np.sin(u), np.sin(v))
        zs = r3 * np.outer(np.ones_like(u), np.cos(v))
        shape = xs.shape
        pm = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]) @ axes.T
        ax1.plot_surface(
            pm[:, 0].reshape(shape) + cx,
            pm[:, 1].reshape(shape) + cy,
            pm[:, 2].reshape(shape) + cz,
            alpha=0.15, color="orange",
        )
    ax1.set_title(f"Algebraic fit (Li, CGF 2004)\n({label})")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend(fontsize=7)

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

    alg = None
    try:
        alg = demo_algebraic(pts, label)
    except Exception as exc:
        print(f"  [Algebraic] FAILED: {exc}")

    out_png = os.path.join(
        os.path.dirname(__file__),
        f"fit_{os.path.splitext(label)[0]}.png",
    )
    try:
        visualise_and_save(pts, alg, label, out_png)
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
        _print_sep("Multi-format ellipsoid fitting demo")
        print(f"  Found {len(data_files)} data file(s).")
        for path in data_files:
            run_on_file(path)
        _print_sep()
        print("Done.")


if __name__ == "__main__":
    main()
