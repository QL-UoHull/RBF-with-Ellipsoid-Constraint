"""
CSV dataset example: load a CSV file and fit an ellipsoid.

Run with:
    python examples/fit_from_csv.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ellipsoid_fitting import fit_ellipsoid, residuals_rms

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fit_from_csv(csv_path: str) -> None:
    print(f"\nFitting ellipsoid to: {os.path.basename(csv_path)}")
    print("-" * 55)

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    print(f"  Points loaded : {len(x)}")

    result = fit_ellipsoid(x, y, z)
    rms = residuals_rms(x, y, z, result)

    print(f"  Centre        : {result['centre'].round(4)}")
    print(f"  Radii         : {result['radii'].round(4)}")
    print(f"  RMS residual  : {rms:.6f}")


if __name__ == "__main__":
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            fit_from_csv(os.path.join(DATA_DIR, fname))
