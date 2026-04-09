# RBF with Ellipsoid Constraint

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **implicit fitting using radial basis functions
(RBFs) with ellipsoidal constraint** algorithm described in:

> **Reference paper:**  
> Li, Q. and Griffiths, J. G. (2004). *Radial basis functions for surface
> reconstruction from unorganised point clouds with applications to bone
> reconstruction.*  
> *Computer Graphics Forum*, 23(1), 67–78. Wiley-Blackwell.  
> DOI: [10.1111/j.1467-8659.2004.00005.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x)

---

## Algorithm

Fits an implicit surface `F(x,y,z) = 0` using a **linear RBF kernel**
`φ(r) = r` together with a second-order polynomial basis.  An ellipsoid
constraint is imposed via a generalised eigenvalue problem, ensuring that the
reconstructed surface is topologically an ellipsoid.

### Key idea

The implicit surface is expressed as a combination of linear radial basis
functions and a degree-2 polynomial:

```
F(q) = Σ_i α_i φ(‖q − p_i‖) + β₀ + β₁ x + … + β₉ yz = 0
```

where φ(r) = r is the linear kernel, **p**_i are the N surface-point centres,
and **β** is the 10-term degree-2 polynomial coefficient vector.

### Steps

1. **Normalise.** Data are centred and scaled to unit bounding radius.

2. **RBF kernel matrix.** Build the N × N matrix **A** where A_ij = ‖p_i − p_j‖,
   with an optional smoothing diagonal regulariser.

3. **Polynomial basis matrix.** Build the N × 10 matrix **B** with monomials
   `[1, x, y, z, x², y², z², xy, xz, yz]`.

4. **Solve A X = B** to obtain the 10-column matrix **X**.

5. **Form D = Bᵀ X** (10 × 10).

6. **Ellipsoid constraint matrix C** (non-zero only in the six second-order
   coefficient positions).

7. **Generalised eigenvalue problem D β = λ C β**; select the eigenvector
   **β** corresponding to the smallest positive eigenvalue.

8. **Recover RBF weights α = −X β**.

---

## Repository structure

```
├── rbf_ellipsoid_constraint/    # Core Python package
│   ├── __init__.py              # Public API
│   ├── rbf_ellipsoid.py         # RBF with ellipsoid constraint (main implementation)
│   ├── loaders.py               # Multi-format point-cloud data loader
│   └── data_generator.py        # Synthetic data generator
├── data/                        # Reproducible point-cloud datasets
│   ├── synthetic_ellipsoid_low_noise.csv
│   ├── synthetic_ellipsoid_rotated.csv
│   ├── synthetic_sphere_like.csv
│   ├── synthetic_ellipsoid.obj          ← Wavefront OBJ
│   ├── synthetic_ellipsoid.ply          ← PLY ASCII
│   ├── synthetic_ellipsoid_binary.ply   ← PLY binary (little-endian)
│   ├── synthetic_ellipsoid.xyz          ← space-separated XYZ
│   ├── synthetic_ellipsoid.m            ← MATLAB-style script
│   └── Tibia.csv                        ← real bone surface scan
├── examples/                    # Runnable example scripts
│   ├── basic_example.py         # Fit and visualise a synthetic cloud
│   ├── fit_from_csv.py          # Load CSV datasets and fit
│   └── fit_multiformat.py       # Load any supported format and fit
├── notebooks/                   # Jupyter notebook workflow
│   └── rbf_ellipsoid_demo.ipynb
├── tests/                       # pytest test suite
│   ├── test_ellipsoid_fit.py    # Data generator + RBF fitting tests
│   └── test_loaders_and_rbf.py  # Loader + RBF tests
├── CITATION.cff                 # Machine-readable citation metadata
├── LICENSE                      # MIT licence
├── pyproject.toml               # Package metadata and build config
└── requirements.txt             # Runtime dependencies
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

### RBF fitting

```python
from rbf_ellipsoid_constraint import (
    fit_rbf_ellipsoid_linear, evaluate_model_linear, generate_ellipsoid_points
)
import numpy as np

pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=300, noise_std=0.05)
result = fit_rbf_ellipsoid_linear(pts, smooth=0.05)
if result is not None:
    alpha, beta, cent, scale = result
    norm_pts = (pts - cent) / scale
    residuals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
    print(f"Centroid : {cent.round(4)}")
    print(f"Mean |F| : {np.mean(np.abs(residuals)):.4f}")  # close to 0 on-surface
```

### Loading different file formats

```python
from rbf_ellipsoid_constraint import load_point_cloud

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

---

## Running examples

```bash
# RBF fit on a synthetic noisy point cloud (with 3-D visualisation)
python examples/basic_example.py

# Fit all CSV datasets and print results
python examples/fit_from_csv.py

# Multi-format demo: load each supported file type, run RBF fitting
python examples/fit_multiformat.py

# Pass a specific file to the multi-format demo
python examples/fit_multiformat.py data/synthetic_ellipsoid.obj
python examples/fit_multiformat.py data/synthetic_ellipsoid.ply
python examples/fit_multiformat.py data/Tibia.csv
```

---

## Jupyter notebook

An end-to-end reproducible workflow is provided in
`notebooks/rbf_ellipsoid_demo.ipynb`.  Launch with:

```bash
jupyter notebook notebooks/rbf_ellipsoid_demo.ipynb
```

---

## API reference

### `fit_rbf_ellipsoid_linear(points, smooth=0.0) → (alpha, beta, centroid, scale) | None`

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

### `evaluate_model_linear(eval_pts, norm_pts, alpha, beta, chunk_size=5000) → ndarray (M,)`

Evaluate the implicit surface `F(q)` at arbitrary query points (both arrays
must be in normalised coordinates).  Points where `F(q) ≈ 0` lie on the
reconstructed surface.

### Data loader

#### `load_point_cloud(filename) → ndarray (N, 3)`

Auto-detect file format from extension and return an (N, 3) NumPy array.

Individual loaders (`load_csv`, `load_obj`, `load_ply`, `load_xyz`,
`load_matlab`, `load_npy`, `load_npz`) are also exported and can be called
directly.

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

If you use this code in academic work, please cite:

```bibtex
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
