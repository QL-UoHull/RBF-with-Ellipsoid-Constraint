# RBR with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **implicit fitting using radial basis functions
with ellipsoidal constraint** algorithm described in:

> **Reference paper:**  
> Li, Q. (2004). *Implicit fitting using radial basis functions with
> ellipsoidal constraint.*  
> *Computer Graphics Forum*, 23(1), 89–96. Wiley/Blackwell.  
> DOI: [10.1111/j.1467-8659.2004.00756.x](https://doi.org/10.1111/j.1467-8659.2004.00756.x)

---

## Overview

The repository implements the **Radial Basis (function) Representation (RBR)
with Ellipsoid Constraint** method for fitting an implicit ellipsoidal surface
to scattered 3-D point data.

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

---

## Quick start

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
```

### Fitting from a CSV file

```python
import numpy as np
from ellipsoid_fitting import fit_ellipsoid

data = np.loadtxt("data/synthetic_ellipsoid_low_noise.csv", delimiter=",", skiprows=1)
result = fit_ellipsoid(data[:, 0], data[:, 1], data[:, 2])
print(result)
```

---

## Running tests

```bash
pytest tests/ -v
```

All 25 tests should pass.

---

## Running examples

```bash
# Visualise a fit on a synthetic noisy point cloud
python examples/basic_example.py

# Fit all CSV datasets and print results
python examples/fit_from_csv.py
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

Evaluate the polynomial part F(x,y,z) = β^T p(x) for each data point.

### `residuals_rms(x, y, z, result) → float`

Root-mean-square algebraic residual.

---

## Datasets

Four datasets are included in `data/` (header row: `x,y,z`):

| File | Points | Description |
|------|--------|-------------|
| `synthetic_ellipsoid_low_noise.csv` | 300 | Axis-aligned ellipsoid, centre (1,2,3), radii (5,3,2), σ=0.05 |
| `synthetic_ellipsoid_rotated.csv` | 500 | Arbitrarily rotated ellipsoid, radii (6,4,2.5), σ=0.15 |
| `synthetic_sphere_like.csv` | 200 | Near-spherical ellipsoid (radii all ≈ 4), centre (5,−3,1), σ=0.10 |
| `Tibia.csv` | — | Real-world bone surface scan |

---

## Citation

If you use this code in academic work, please cite the original paper:

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
