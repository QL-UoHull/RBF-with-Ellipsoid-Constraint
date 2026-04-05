# RBF with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **implicit fitting using radial basis functions
(RBFs) with ellipsoidal constraint** algorithm described in:

> **Reference paper:**  
> Li, Q. (2004). *Implicit fitting using radial basis functions with
> ellipsoidal constraint.*  
> *Computer Graphics Forum*, 23(1), 89–96. Wiley/Blackwell.  
> DOI: [10.1111/j.1467-8659.2004.00756.x](https://doi.org/10.1111/j.1467-8659.2004.00756.x)
# RBF with Ellipsoid Constraint

[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of two complementary **ellipsoid surface-fitting**
algorithms and a unified **multi-format data loader** supporting CSV, OBJ,
PLY, XYZ, MATLAB `.m`, and NumPy formats.

---

## Algorithms

### 1 · Algebraic Least-Squares Fitting (Li & Griffiths, GMAP 2004)

> Li, Q. and Griffiths, J. G. (2004).  
> *Least squares ellipsoid specific fitting.*  
> Proceedings of the Geometric Modeling and Processing, 2004. IEEE, pp. 335–340.  
> DOI: [10.1109/GMAP.2004.1290055](https://doi.org/10.1109/GMAP.2004.1290055)

The repository name **RBR** stands for **Radial Basis (function) Representation**,
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
Fits a general algebraic ellipsoid `F(x,y,z) = 0` to 3-D point data using a
constrained least-squares approach.  The constraint matrix (with parameter
`k = 4`) guarantees that only ellipsoidal solutions are admitted.

### 2 · RBF with Ellipsoid Constraint (Li & Griffiths, CGF 2004)

> Li, Q. and Griffiths, J. G. (2004).  
> *Radial basis functions for surface reconstruction from unorganised point
> clouds with applications to bone reconstruction.*  
> Computer Graphics Forum, 23(1), 67–78. Wiley-Blackwell.  
> DOI: [10.1111/j.1467-8659.2004.00005.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x)

Fits an implicit surface `F(x,y,z) = 0` using a **linear RBF kernel**
`φ(r) = r` together with a second-order polynomial basis.  An ellipsoid
constraint is imposed via a generalised eigenvalue problem, ensuring that the
reconstructed surface is topologically an ellipsoid.

---

## Repository structure

```
├── ellipsoid_fitting/       # Core Python package
│   ├── __init__.py          # Public API
│   ├── ellipsoid_fit.py     # RBR fitting algorithm (main implementation)
│   └── data_generator.py   # Synthetic data generator
├── data/                    # Reproducible CSV datasets
│   ├── synthetic_ellipsoid_low_noise.csv
│   ├── synthetic_ellipsoid_rotated.csv
│   ├── synthetic_sphere_like.csv
│   └── Tibia.csv
├── examples/                # Runnable example scripts
│   ├── basic_example.py     # Fit and visualise a synthetic cloud
│   └── fit_from_csv.py      # Load CSV datasets and fit
├── notebooks/               # Jupyter notebook workflow
│   └── ellipsoid_fitting_demo.ipynb
├── tests/                   # pytest test suite (25 tests)
│   └── test_ellipsoid_fit.py
├── CITATION.cff             # Machine-readable citation metadata
├── LICENSE                  # MIT licence
├── pyproject.toml           # Package metadata and build config
└── requirements.txt         # Runtime dependencies
├── ellipsoid_fitting/        # Core Python package
│   ├── __init__.py           # Public API
│   ├── ellipsoid_fit.py      # Algebraic Li–Griffiths fitting (GMAP 2004)
│   ├── rbf_ellipsoid.py      # RBF with ellipsoid constraint (CGF 2004)
│   ├── loaders.py            # Multi-format point-cloud data loader
│   └── data_generator.py    # Synthetic data generator
├── data/                     # Reproducible point-cloud datasets
│   ├── synthetic_ellipsoid_low_noise.csv
│   ├── synthetic_ellipsoid_rotated.csv
│   ├── synthetic_sphere_like.csv
│   ├── synthetic_ellipsoid.obj          ← Wavefront OBJ
│   ├── synthetic_ellipsoid.ply          ← PLY ASCII
│   ├── synthetic_ellipsoid_binary.ply   ← PLY binary (little-endian)
│   ├── synthetic_ellipsoid.xyz          ← space-separated XYZ
│   ├── synthetic_ellipsoid.m            ← MATLAB-style script
│   └── Tibia.csv                        ← real bone surface scan
├── examples/                 # Runnable example scripts
│   ├── basic_example.py      # Algebraic fit on a synthetic cloud
│   ├── fit_from_csv.py       # Load CSV datasets and fit
│   └── fit_multiformat.py    # Load any supported format + both fits
├── notebooks/                # Jupyter notebook workflow
│   └── ellipsoid_fitting_demo.ipynb
├── tests/                    # pytest test suite
│   ├── test_ellipsoid_fit.py # Algebraic fitting tests (22 tests)
│   └── test_loaders_and_rbf.py  # Loader + RBF tests (52 tests)
├── CITATION.cff              # Machine-readable citation metadata
├── LICENSE                   # MIT licence
├── pyproject.toml            # Package metadata and build config
└── requirements.txt          # Runtime dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/QL-UoHull/RBR-with-Ellipsoid-Constraint.git
cd RBR-with-Ellipsoid-Constraint

# Install dependencies
pip install -r requirements.txt

# (Optional) install the package in editable mode
pip install -e .
```

Dependencies: `numpy >= 1.22`, `scipy >= 1.8`, `matplotlib >= 3.5`.  
Optional (for isosurface visualisation): `scikit-image`.

---

## Quick start

### Algebraic fitting

```python
import numpy as np
from ellipsoid_fitting import fit_ellipsoid, generate_ellipsoid_points

# Generate a synthetic noisy point cloud
pts = generate_ellipsoid_points(
    centre=(1.0, 2.0, 3.0),
    radii=(5.0, 3.0, 2.0),
    n_points=300,
    noise_std=0.05,
)

# Fit the ellipsoid using RBR with ellipsoidal constraint
result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

print("Centre:", result["centre"])      # → [1.0, 2.0, 3.0]
print("Radii :", result["radii"])       # → [5.0, 3.0, 2.0] (descending)
print("Axes  :", result["axes"])        # 3×3 orthonormal matrix
print("RBF w :", result["rbf_weights"].shape)  # (300,)
pts = generate_ellipsoid_points(centre=(1, 2, 3), radii=(5, 3, 2),
                                 n_points=300, noise_std=0.05)
result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

print("Centre:", result["centre"])   # → [1.0, 2.0, 3.0]
print("Radii :", result["radii"])    # → [5.0, 3.0, 2.0]  (sorted descending)
```

### RBF fitting

```python
from ellipsoid_fitting import (
    fit_rbf_ellipsoid_linear, evaluate_model_linear, generate_ellipsoid_points
)
import numpy as np

pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=300, noise_std=0.05)
alpha, beta, cent, scale = fit_rbf_ellipsoid_linear(pts, smooth=0.05)

norm_pts = (pts - cent) / scale
residuals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
print(f"Mean |F|: {np.mean(np.abs(residuals)):.4f}")  # close to 0 on-surface
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

## Running tests

```bash
pytest tests/ -v
```

All 25 tests should pass.
All 74 tests should pass.

---

## Running examples

```bash
# Algebraic fit on a synthetic noisy point cloud (with 3-D visualisation)
python examples/basic_example.py

# Fit all CSV datasets and print results
python examples/fit_from_csv.py

# Multi-format demo: load each supported file type, run both algorithms
python examples/fit_multiformat.py

# Pass a specific file to the multi-format demo
python examples/fit_multiformat.py data/synthetic_ellipsoid.obj
python examples/fit_multiformat.py data/synthetic_ellipsoid.ply
python examples/fit_multiformat.py data/Tibia.csv
```

---

## Jupyter notebook

An end-to-end reproducible workflow is provided in
`notebooks/ellipsoid_fitting_demo.ipynb`.  Launch with:

```bash
jupyter notebook notebooks/ellipsoid_fitting_demo.ipynb
```

---

## API reference

### `fit_ellipsoid(x, y, z, k=4.0, epsilon=None, k_neighbours=15) → dict`

Fit an ellipsoid to 3-D point data using RBF implicit fitting.
### Data loader

#### `load_point_cloud(filename) → ndarray (N, 3)`

Auto-detect file format from extension and return an (N, 3) NumPy array.

Individual loaders (`load_csv`, `load_obj`, `load_ply`, `load_xyz`,
`load_matlab`, `load_npy`, `load_npz`) are also exported and can be called
directly.

---

### Algebraic fitting

#### `fit_ellipsoid(x, y, z, k=4.0) → dict`

Fit an ellipsoid to 3-D point data (Li & Griffiths, GMAP 2004).

| Parameter | Type | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | array-like (N,) | Cartesian coordinates (≥ 10 points) |
| `k` | float | Constraint parameter, k ∈ (0, 4]; default 4.0 |
| `epsilon` | float or None | Off-surface displacement; auto-set if None |
| `k_neighbours` | int | Neighbours for local-PCA normal estimation; default 15 |

**Returns** a `dict` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `centre` | `(3,)` | Ellipsoid centre |
| `radii` | `(3,)` | Semi-axis lengths (descending) |
| `axes` | `(3, 3)` | Unit-vector columns (axes of the ellipsoid) |
| `M` | `(4, 4)` | Homogeneous quadric matrix |
| `coefficients` | `(10,)` | Algebraic coefficients `[A,B,C,D,E,F,G,H,I,J]` |
| `rbf_weights` | `(N,)` | Biharmonic RBF weights for the N surface-point centres |

### `generate_ellipsoid_points(centre, radii, rotation, n_points, noise_std, seed)`

Generate synthetic 3-D surface points on an (optionally rotated) ellipsoid.

### `algebraic_distance(x, y, z, coefficients) → ndarray`
#### `algebraic_distance(x, y, z, coefficients) → ndarray (N,)`

Evaluate the polynomial part F(x,y,z) = β^T p(x) for each data point.

#### `residuals_rms(x, y, z, result) → float`

Root-mean-square algebraic residual.

---

### RBF fitting

#### `fit_rbf_ellipsoid_linear(points, smooth=0.0) → (alpha, beta, centroid, scale) | None`

Fit an implicit ellipsoidal surface using a linear RBF kernel
(Li & Griffiths, CGF 2004).

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

## Datasets

| File | Points | Description |
|------|--------|-------------|
| `synthetic_ellipsoid_low_noise.csv` | 300 | Axis-aligned, centre (1,2,3), radii (5,3,2), σ=0.05 |
| `synthetic_ellipsoid_rotated.csv` | 500 | Arbitrarily rotated, radii (6,4,2.5), σ=0.15 |
| `synthetic_sphere_like.csv` | 200 | Near-spherical, radii ≈ 4, centre (5,−3,1), σ=0.10 |
| `synthetic_ellipsoid.obj` | 200 | Same cloud as above in Wavefront OBJ format |
| `synthetic_ellipsoid.ply` | 200 | Same cloud in PLY ASCII format |
| `synthetic_ellipsoid_binary.ply` | 200 | Same cloud in PLY binary (little-endian) |
| `synthetic_ellipsoid.xyz` | 200 | Same cloud in plain XYZ format |
| `synthetic_ellipsoid.m` | 200 | Same cloud as MATLAB `data = [...];` script |
| `Tibia.csv` | 1 484 | Real tibia bone surface scan |

---

## Citation

If you use this code in academic work, please cite the relevant paper(s):

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

@article{li2004rbf,
  title     = {Radial basis functions for surface reconstruction from
               unorganised point clouds with applications to bone
               reconstruction},
  author    = {Li, Qingde and Griffiths, John G.},
  journal   = {Computer Graphics Forum},
  volume    = {23},
  number    = {1},
  pages     = {67--78},
  year      = {2004},
  publisher = {Wiley-Blackwell},
  doi       = {10.1111/j.1467-8659.2004.00005.x}
}
```

A `CITATION.cff` file is also provided for automated citation tools
(e.g. GitHub's *Cite this repository* button).

---

## Licence

This project is released under the [MIT Licence](LICENSE).
