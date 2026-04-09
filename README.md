# RBF with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

[![Tests](https://img.shields.io/badge/tests-77%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **implicit fitting using radial basis functions
(RBFs) with ellipsoidal constraint** algorithm described in:

> **Reference paper:**  
> Li, Q, et al. (2004). *Implicit fitting using radial basis functions with
> ellipsoidal constraint.*  
> *Computer Graphics Forum*, 23(1), 89–96. Wiley/Blackwell.  
> DOI: [10.1111/j.1467-8659.2004.00756.x](https://doi.org/10.1111/j.1467-8659.2004.00756.x)


## Algorithm

Fits an implicit surface `F(x,y,z) = 0` using a **linear RBF kernel**
`φ(r) = r` together with a second-order polynomial basis.  An ellipsoid
constraint is imposed via a generalised eigenvalue problem, ensuring that the
reconstructed surface is topologically an ellipsoid.

The repository name **RBF** stands for **Radial Basis Functions**,
the term used by the author for the RBF-based implicit fitting approach combined
with an ellipsoid-specific constraint.  The method fits an implicit ellipsoidal
surface to scattered 3-D point data.

### Key idea

The implicit surface is expressed as a combination of biharmonic radial basis
functions (RBFs) and a degree-2 polynomial:

```
f(x) = Σ_i w_i φ(‖x − c_i‖) + β^T p(x) = 0
```

where φ(r) = r is the biharmonic kernel, **c**_i are the N surface-point
centres, and **p**(x) = [x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z, 1]^T is
the 10-term degree-2 polynomial basis.

### Algorithm

1. **Normal estimation.** For each input point the surface normal **n̂**_i
   is estimated via local PCA on k nearest neighbours.

2. **Off-surface training set.** Two additional layers of points are
   generated: **P**+ = **P** + ε**n̂** (outside the surface, target f = +ε)
   and **P**− = **P** − ε**n̂** (inside the surface, target f = −ε).
   Together with the original N surface points (target f = 0), this gives a
   3N-row augmented training set T.

3. **Augmented scatter matrix.** The (3N × 10) polynomial design matrix
   **D**_T is evaluated at T and the scatter matrix **S** = **D**_T^T **D**_T
   is formed.

4. **Ellipsoid-constrained fitting.** The polynomial coefficients β are
   found by solving:

   ```
   min_β  β^T S β    subject to  β^T C β > 0
   ```

   using the block-decomposition/eigenvector approach from Li (2004), where
   **C** is the (10 × 10) ellipsoid-specific constraint matrix with parameter
   k = 4.

5. **RBF weight recovery.** Given β, the biharmonic RBF weights are
   recovered from the residual Φ **w** ≈ **d** − **D**_T β.

6. **Geometric parameters.** The algebraic coefficients β =
   [A, B, C, D, E, F, G, H, I, J] are converted to the ellipsoid centre,
   semi-axis lengths, and axis directions.

---

## Repository structure

```
RBF-with-Ellipsoid-Constraint/
├── ellipsoid_fitting/
│   ├── __init__.py
│   ├── ellipsoid_fit.py       # Core RBF fitting algorithm (Li, CGF 2004)
│   ├── rbf_ellipsoid.py       # Alternative lower-level RBF interface
│   ├── loaders.py             # Multi-format point-cloud loader
│   └── data_generator.py      # Synthetic data generator
├── data/
│   ├── synthetic_ellipsoid_low_noise.csv
│   ├── synthetic_ellipsoid_rotated.csv
│   ├── synthetic_sphere_like.csv
│   ├── synthetic_ellipsoid.obj
│   ├── synthetic_ellipsoid.ply
│   ├── synthetic_ellipsoid_binary.ply
│   ├── synthetic_ellipsoid.xyz
│   ├── synthetic_ellipsoid.m
│   ├── femur.m
│   ├── head.m
│   └── Tibia.csv
├── examples/
│   ├── basic_example.py       # RBF fit on a synthetic point cloud
│   ├── fit_from_csv.py        # Load CSV datasets and fit
│   └── fit_multiformat.py     # Load any supported format and fit
├── notebooks/
│   └── ellipsoid_fitting_demo.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_ellipsoid_fit.py       # Tests for fit_ellipsoid (22 tests)
│   └── test_loaders_and_rbf.py     # Tests for loaders + RBF (52 tests)
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/QL-UoHull/RBF-with-Ellipsoid-Constraint.git
cd RBF-with-Ellipsoid-Constraint

# Install dependencies
pip install -r requirements.txt

# (Optional) install the package in editable mode
pip install -e .
```

Dependencies: `numpy >= 1.22`, `scipy >= 1.8`, `matplotlib >= 3.5`.  
Optional (for isosurface visualisation): `scikit-image`.

---

## Quick start


```python
from ellipsoid_fitting import fit_ellipsoid, generate_ellipsoid_points
import numpy as np

pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=300, noise_std=0.05)
result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

print("Centre:", result["centre"])
print("Radii: ", result["radii"])
```

### Loading different file formats

```python
from ellipsoid_fitting import load_point_cloud

# Auto-detect format from file extension
pts = load_point_cloud("data/synthetic_ellipsoid.obj")   # Wavefront OBJ
pts = load_point_cloud("data/synthetic_ellipsoid.ply")   # PLY (ASCII or binary)
pts = load_point_cloud("data/synthetic_ellipsoid.xyz")   # whitespace-separated
pts = load_point_cloud("data/synthetic_ellipsoid.m")     # MATLAB .m script
pts = load_point_cloud("data/synthetic_ellipsoid_low_noise.csv")  # CSV
```

---

## Supported file formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.csv` | Comma-separated values | Header row required; first 3 columns are x, y, z |
| `.txt` | Whitespace-delimited | No header; first 3 columns are x, y, z |
| `.xyz` | XYZ point cloud | Same as `.txt` |
| `.pts` | Point cloud | Same as `.txt` |
| `.obj` | Wavefront OBJ | Only vertex lines (`v x y z`) are parsed |
| `.ply` | Stanford PLY | ASCII and binary (little-endian / big-endian) |
| `.m` | MATLAB script | Parses `data = [...];` matrix literal |
| `.npy` | NumPy binary | Array must be 2-D with ≥ 3 columns |
| `.npz` | NumPy compressed | Array stored under key `"data"` |

---

## API Reference

### Primary interface

#### `fit_ellipsoid(x, y, z, k=4, epsilon=0.01, k_neighbours=6) → dict`

Fit an implicit ellipsoidal surface to 3-D point data using the RBF
implicit fitting with ellipsoidal constraint algorithm (Li, CGF 2004).

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like (N,) | x-coordinates of surface points |
| `y` | array-like (N,) | y-coordinates of surface points |
| `z` | array-like (N,) | z-coordinates of surface points |
| `k` | int | Ellipsoid constraint parameter (default 4) |
| `epsilon` | float | Off-surface displacement for normal estimation (default 0.01) |
| `k_neighbours` | int | Number of nearest neighbours for normal estimation (default 6) |

Returns a `dict` with keys `centre`, `radii`, `axes`, `algebraic`,
`rbf_weights`, and `poly_coeffs`.

### Lower-level / alternative RBF interface

#### `fit_rbf_ellipsoid_linear(points, smooth=0.0) → (alpha, beta, centroid, scale) | None`

Fit an implicit ellipsoidal surface using a linear RBF kernel
(Li, CGF 2004).

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | ndarray (N, 3) | 3-D surface points |
| `smooth` | float | Diagonal regulariser; increase for noisy data (default 0) |

Returns a 4-tuple `(alpha, beta, centroid, scale)` where `alpha` are the
RBF weights (shape `(N,)`), `beta` the polynomial coefficients (shape
`(10,)`), and `centroid` / `scale` are normalisation parameters.  Returns
`None` if no valid eigenvalue is found.

#### `evaluate_model_linear(eval_pts, norm_pts, alpha, beta, chunk_size=5000) → ndarray (M,)`

Evaluate the implicit surface `F(q)` at arbitrary query points (both arrays
must be in normalised coordinates).  Points where `F(q) ≈ 0` lie on the
reconstructed surface.

---

Four datasets are included in `data/` (header row: `x,y,z`):

| File | Points | Description |
|------|--------|-------------|
| `synthetic_ellipsoid_low_noise.csv` | 300 | Axis-aligned ellipsoid, centre (1,2,3), radii (5,3,2), σ=0.05 |
| `synthetic_ellipsoid_rotated.csv` | 500 | Arbitrarily rotated ellipsoid, radii (6,4,2.5), σ=0.15 |
| `synthetic_sphere_like.csv` | 200 | Near-spherical ellipsoid (radii all ≈ 4), centre (5,−3,1), σ=0.10 |
| `Tibia.csv` | — | Real-world bone surface scan |
---

### Synthetic data generator

#### `generate_ellipsoid_points(centre, radii, rotation, n_points, noise_std, seed) → ndarray (N, 3)`

Generate 3-D points sampled uniformly on an ellipsoid surface.


---

## Citation

If you use this code in academic work, please cite the relevant paper:

```bibtex
@article{li2004implicit,
  title     = {Implicit fitting using radial basis functions with
               ellipsoidal constraint},
  author    = {Li, Qingde},
  journal   = {Computer Graphics Forum},
  volume    = {23},
  number    = {1},
  pages     = {89--96},
  year      = {2004},
  publisher = {Wiley/Blackwell},
  doi       = {10.1111/j.1467-8659.2004.00756.x}
}
```

A `CITATION.cff` file is also provided for automated citation tools
(e.g. GitHub's *Cite this repository* button).

---

## Licence

This project is released under the [MIT Licence](LICENSE).
