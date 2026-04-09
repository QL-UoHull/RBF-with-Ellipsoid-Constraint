"""
CSV dataset example: load a CSV file and fit an ellipsoid using RBF with
ellipsoid constraint.

Reference:
    Li, Q. and Griffiths, J. G. (2004). "Radial basis functions for surface
    reconstruction from unorganised point clouds with applications to bone
    reconstruction." Computer Graphics Forum, 23(1), 67–78.
    Wiley-Blackwell. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x

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

    result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
    if result is None:
        print("  Fit failed: no valid eigenvalue found.")
        return

    alpha, beta, cent, scale = result
    norm_pts = (pts - cent) / scale
    vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)

    print(f"  Centroid       : {cent.round(4)}")
    print(f"  Scale          : {scale:.4f}")
    print(f"  Mean |F|       : {float(np.mean(np.abs(vals))):.6f}")
    print(f"  Max  |F|       : {float(np.max(np.abs(vals))):.6f}")


if __name__ == "__main__":
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            fit_from_csv(os.path.join(DATA_DIR, fname))
