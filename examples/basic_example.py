"""
Basic usage example: fit an ellipsoid to a synthetic noisy point cloud
using RBF implicit fitting with ellipsoidal constraint.

Reference:
    Li, Q. and Griffiths, J. G. (2004). "Radial basis functions for surface
    reconstruction from unorganised point clouds with applications to bone
    reconstruction." Computer Graphics Forum, 23(1), 67–78.
    Wiley-Blackwell. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x

Run with:
    python examples/basic_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rbf_ellipsoid_constraint import (
    fit_rbf_ellipsoid_linear,
    evaluate_model_linear,
    generate_ellipsoid_points,
)

# -------------------------------------------------------------------
# 1.  Generate a synthetic noisy point cloud on a known ellipsoid
# -------------------------------------------------------------------
TRUE_CENTRE = np.array([1.0, 2.0, 3.0])
TRUE_RADII  = np.array([5.0, 3.0, 2.0])

pts = generate_ellipsoid_points(
    centre=TRUE_CENTRE,
    radii=TRUE_RADII,
    n_points=300,
    noise_std=0.05,
    seed=42,
)

# -------------------------------------------------------------------
# 2.  Fit using RBF with Ellipsoid Constraint
# -------------------------------------------------------------------
result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
if result is None:
    print("Fit failed: no valid eigenvalue found.")
    sys.exit(1)

alpha, beta, cent, scale = result
norm_pts = (pts - cent) / scale
residuals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)

print("=" * 60)
print("RBF with Ellipsoid Constraint  (Li & Griffiths, CGF 2004)")
print("=" * 60)
print(f"\nTrue  centre : {TRUE_CENTRE}")
print(f"Fitted centroid: {cent.round(4)}")
print(f"\nScale    : {scale:.4f}")
print(f"Mean |F| : {np.mean(np.abs(residuals)):.6f}  (≈0 on surface)")
print(f"Max  |F| : {np.max(np.abs(residuals)):.6f}")
print("=" * 60)

# -------------------------------------------------------------------
# 3.  Visualise
# -------------------------------------------------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.5, label="Noisy data")
ax.set_title("RBF with Ellipsoid Constraint (Li & Griffiths, CGF 2004)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "examples", "basic_example_plot.png")
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved to {out_path}")
plt.show()
