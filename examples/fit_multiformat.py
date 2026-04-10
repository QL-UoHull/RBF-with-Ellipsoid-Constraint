"""
Multi-format example: load a 3-D point cloud from various file formats and
fit an ellipsoid using the RBF ellipsoid fitting algorithm in the package.

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


def demo_rbf(pts: np.ndarray, label: str) -> None:
    """Fit using the RBF implicit fitting with ellipsoidal constraint (Li & Griffiths, CGF 2004)."""
    result = fit_rbf_ellipsoid_linear(pts)
    if result is None:
        print(f"  [RBF] Fitting failed — no valid eigenvalue found.")
        return None
    alpha, beta, centroid, scale = result
    norm_pts = (pts - centroid) / scale
    vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
    rms = float(np.sqrt(np.mean(vals ** 2)))
    print(f"  [RBF] Centroid : {centroid.round(4)}")
    print(f"  [RBF] RMS      : {rms:.6f}")
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_and_save(pts, rbf_result, label, out_path):
    """Save a scatter plot."""
    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(111, projection="3d")
    ax1.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=4, alpha=0.6, c="steelblue", label="Points",
    )
    ax1.set_title(f"RBF fit (Li & Griffiths, CGF 2004)\n({label})")
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
        alg = demo_rbf(pts, label)
    except Exception as exc:
        print(f"  [RBF] FAILED: {exc}")

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
