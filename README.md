# Ellipsoid Fitting via Least Squares with Ellipsoid-Specific Constraints

[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **Li–Griffiths (2004)** least-squares
ellipsoid-specific fitting algorithm.

> **Reference paper:**  
> Li, Q. and Griffiths, J. G. (2004). *Least squares ellipsoid specific fitting.*  
> *Proceedings of the Geometric Modeling and Processing, 2004.* IEEE, pp. 335–340.  
> DOI: [10.1109/GMAP.2004.1290055](https://doi.org/10.1109/GMAP.2004.1290055) · [ResearchGate](https://www.researchgate.net/publication/4070857_Least_squares_ellipsoid_specific_fitting)

---

## Overview

Fitting a general quadric surface to noisy 3-D point data is a classic
least-squares problem; however, unconstrained fitting may produce
hyperboloids, paraboloids, or other non-ellipsoidal quadrics.  Li & Griffiths
(2004) introduced an **ellipsoid-specific constraint matrix** that forces the
solution to be a valid ellipsoid.

The algorithm:

1. Represents every data point as a row of the 10-column **design matrix**  
   `d = [x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z, 1]`
2. Forms the **scatter matrix** `S = DᵀD`
3. Applies the **ellipsoid-specific constraint matrix** `C` (with parameter
   `k = 4`) to the 6 × 6 upper-left sub-problem
4. Selects the eigenvector corresponding to the **largest positive eigenvalue**
   of the reduced generalised eigenvalue problem

The algebraic coefficients `[A, B, C, D, E, F, G, H, I, J]` are then
converted to the geometric parameters (centre, semi-axis lengths, axis
directions).

---

## Repository structure

```
├── ellipsoid_fitting/       # Core Python package
│   ├── __init__.py          # Public API
│   ├── ellipsoid_fit.py     # Li–Griffiths fitting algorithm
│   └── data_generator.py   # Synthetic data generator
├── data/                    # Reproducible CSV datasets
│   ├── synthetic_ellipsoid_low_noise.csv
│   ├── synthetic_ellipsoid_rotated.csv
│   └── synthetic_sphere_like.csv
├── examples/                # Runnable example scripts
│   ├── basic_example.py     # Fit and visualise a synthetic cloud
│   └── fit_from_csv.py      # Load CSV datasets and fit
├── notebooks/               # Jupyter notebook workflow
│   └── ellipsoid_fitting_demo.ipynb
├── tests/                   # pytest test suite
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
git clone https://github.com/QL-UoHull/Ellipsoid-Fitting-via-Least-Squares-with-Ellipsoid-Specific-Constraints-Li-Griffiths-2004-.git
cd Ellipsoid-Fitting-via-Least-Squares-with-Ellipsoid-Specific-Constraints-Li-Griffiths-2004-

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

# Fit the ellipsoid
result = fit_ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2])

print("Centre:", result["centre"])   # → [1.0, 2.0, 3.0]
print("Radii :", result["radii"])    # → [5.0, 3.0, 2.0]  (sorted descending)
print("Axes  :", result["axes"])     # 3×3 orthonormal matrix
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

All 22 tests should pass.

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

### `fit_ellipsoid(x, y, z, k=4.0) → dict`

Fit an ellipsoid to 3-D point data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | array-like (N,) | Cartesian coordinates |
| `k` | float | Constraint parameter; must be in `(0, 4]`; default `4.0` |

**Returns** a `dict` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `centre` | `(3,)` | Ellipsoid centre |
| `radii` | `(3,)` | Semi-axis lengths (descending) |
| `axes` | `(3, 3)` | Unit-vector columns (axes of the ellipsoid) |
| `M` | `(4, 4)` | Homogeneous quadric matrix |
| `coefficients` | `(10,)` | Raw algebraic coefficients `[A,B,C,D,E,F,G,H,I,J]` |

### `generate_ellipsoid_points(centre, radii, rotation, n_points, noise_std, seed)`

Generate synthetic 3-D surface points.

### `algebraic_distance(x, y, z, coefficients) → ndarray`

Evaluate `F(x,y,z)` for each data point (ideally 0 on the ellipsoid).

### `residuals_rms(x, y, z, result) → float`

Root-mean-square algebraic residual.

---

## Datasets

Three CSV datasets are included in `data/` (header row: `x,y,z`):

| File | Points | Description |
|------|--------|-------------|
| `synthetic_ellipsoid_low_noise.csv` | 300 | Axis-aligned ellipsoid, centre (1,2,3), radii (5,3,2), σ=0.05 |
| `synthetic_ellipsoid_rotated.csv` | 500 | Arbitrarily rotated ellipsoid, radii (6,4,2.5), σ=0.15 |
| `synthetic_sphere_like.csv` | 200 | Near-spherical ellipsoid (radii all ≈ 4), centre (5,−3,1), σ=0.10 |

---

## Citation

If you use this code in academic work, please cite the original paper:

```bibtex
@inproceedings{li2004ellipsoid,
  title     = {Least squares ellipsoid specific fitting},
  author    = {Li, Qingde and Griffiths, John G.},
  booktitle = {Proceedings of the Geometric Modeling and Processing, 2004},
  pages     = {335--340},
  year      = {2004},
  publisher = {IEEE},
  doi       = {10.1109/GMAP.2004.1290055}
}
```

A `CITATION.cff` file is also provided for automated citation tools
(e.g. GitHub's *Cite this repository* button).

---

## Licence

This project is released under the [MIT Licence](LICENSE).
