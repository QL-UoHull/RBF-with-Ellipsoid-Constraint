"""
CSV dataset example: load a CSV file and fit an RBF implicit surface to a 3-D point cloud.

Reference:
    Li, Q. and Griffiths, J. G. (2004). "Radial basis functions for surface
    reconstruction from unorganised point clouds with applications to bone
    reconstruction." Computer Graphics Forum, 23(1), 67-78. Wiley-Blackwell.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x

Run with:
    python examples/fit_from_csv.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from rbf_ellipsoid_constraint import fit_rbf_ellipsoid_linear, evaluate_model_linear

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fit_from_csv(csv_path: str) -> None:
    print(f"\nFitting ellipsoid to: {os.path.basename(csv_path)}")
    print("-" * 60)

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pts = data[:, :3]
    print(f"  Points loaded  : {len(pts)}")

    result = fit_rbf_ellipsoid_linear(pts)
    if result is None:
        print("  Fitting failed — no valid eigenvalue found.")
        return

    alpha, beta, centroid, scale = result
    norm_pts = (pts - centroid) / scale
    vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
    rms = float(np.sqrt(np.mean(vals ** 2)))

    print(f"  Centroid       : {centroid.round(4)}")
    print(f"  RMS residual   : {rms:.6f}")
    print(f"  RBF weights    : {alpha.shape[0]} values")


if __name__ == "__main__":
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            fit_from_csv(os.path.join(DATA_DIR, fname))
