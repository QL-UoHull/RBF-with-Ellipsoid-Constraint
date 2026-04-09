"""
Basic usage example: fit an ellipsoid to a synthetic noisy point cloud
using RBF implicit fitting with ellipsoidal constraint.

Reference:
    Li, Q. (2004). "Implicit fitting using radial basis functions with
    ellipsoidal constraint." Computer Graphics Forum, 23(1), 89-96.
    Wiley/Blackwell. https://doi.org/10.1111/j.1467-8659.2004.00756.x

Run with:
    python examples/basic_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ellipsoid_fitting import fit_ellipsoid, generate_ellipsoid_points, residuals_rms

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
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

# -------------------------------------------------------------------
# 2.  Fit the ellipsoid using RBF with Ellipsoid Constraint
# -------------------------------------------------------------------
result = fit_ellipsoid(x, y, z)

print("=" * 60)
print("RBF with Ellipsoid Constraint  (Li, CGF 2004)")
print("=" * 60)
print(f"\nTrue  centre : {TRUE_CENTRE}")
print(f"Fitted centre: {result['centre'].round(4)}")
print(f"\nTrue  radii  : {np.sort(TRUE_RADII)[::-1]}")
print(f"Fitted radii : {result['radii'].round(4)}")
print(f"\nRBF weights  : {result['rbf_weights'].shape[0]} values, "
      f"‖w‖ = {np.linalg.norm(result['rbf_weights']):.4f}")

rms = residuals_rms(x, y, z, result)
print(f"\nRMS algebraic residual: {rms:.6f}")
print("=" * 60)

# -------------------------------------------------------------------
# 3.  Visualise
# -------------------------------------------------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, s=4, alpha=0.5, label="Noisy data")

# Draw the fitted ellipsoid as a mesh
cx, cy, cz = result["centre"]
r1, r2, r3 = result["radii"]
axes = result["axes"]

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = r1 * np.outer(np.cos(u), np.sin(v))
ys = r2 * np.outer(np.sin(u), np.sin(v))
zs = r3 * np.outer(np.ones_like(u), np.cos(v))

# Apply axes rotation and translate to centre
shape = xs.shape
pts_mesh = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]) @ axes.T
xs2 = pts_mesh[:, 0].reshape(shape) + cx
ys2 = pts_mesh[:, 1].reshape(shape) + cy
zs2 = pts_mesh[:, 2].reshape(shape) + cz

ax.plot_surface(xs2, ys2, zs2, alpha=0.15, color="orange")
ax.set_title("RBF with Ellipsoid Constraint (Li, CGF 2004)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "examples", "basic_example_plot.png")
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved to {out_path}")
plt.show()
