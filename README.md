# RBF with Ellipsoid Constraint — Implicit Fitting Using Radial Basis Functions

[![Tests](https://img.shields.io/badge/tests-71%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the **implicit fitting using radial basis functions
(RBFs) with ellipsoidal constraint** algorithm described in:

> **Reference paper:**  
> Li, Q., et al. (2004). *Implicit fitting using radial basis functions with ellipsoid constraint.*  
> *Computer Graphics Forum*, 23(1), 67–78. Wiley-Blackwell.  
> https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00005.x


## Algorithm

### RBF with Ellipsoid Constraint (Li &amp; et al., CGF 2004)

Fits an implicit surface `F(x,y,z) = 0` using a **linear RBF kernel**
`φ(r) = r` together with a second-order polynomial basis.  An ellipsoid
constraint is imposed via a generalised eigenvalue problem, ensuring that the
reconstructed surface is topologically an ellipsoid.

The repository name **RBF** stands for **Radial Basis Functions**,
the term used by the authors for the RBF-based implicit fitting approach combined
with an ellipsoid-specific constraint.  The method fits an implicit ellipsoidal
surface to scattered 3-D point data.

#### Key idea

The implicit surface is expressed as a combination of linear radial basis
functions (RBFs) and a degree-2 polynomial:

```
F(x) = Σ_i αᵢ φ(‖x − pᵢ‖) + β₀ + β₁x + β₂y + β₃z + β₄x² + β₅y² + β₆z² + β₇xy + β₈xz + β₉yz = 0
```

where φ(r) = r is the linear kernel, **p**_i are the N surface-point
centres, and [β₀ … β₉] is the 10-term degree-2 polynomial basis.

#### Algorithm

1. **Normalise** the input data (zero centroid, unit bounding radius).
2. **Build the RBF kernel matrix** **A** where A_ij = φ(‖pᵢ − pⱼ‖).
3. **Build the polynomial basis matrix** **B** (N × 10).
4. **Solve A X = B** to obtain X.
5. **Form D = Bᵀ X** (10 × 10).
6. **Build the ellipsoid constraint matrix C** (10 × 10).
7. **Solve the generalised eigenvalue problem D β = λ C β**; select the
   eigenvector β with smallest positive eigenvalue.
8. **Recover RBF weights** α = −X β.

---

## Repository structure

```
RBF-with-Ellipsoid-Constraint/
├── rbf_ellipsoid_constraint/
│   ├── __init__.py
│   ├── rbf_ellipsoid.py       # Core RBF fitting algorithm (Li, et al., CGF 2004)
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
│   ├── test_data_generator.py      # Tests for data generator
│   └── test_loaders_and_rbf.py     # Tests for loaders
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
from rbf_ellipsoid_constraint import fit_rbf_ellipsoid_linear, evaluate_model_linear, generate_ellipsoid_points
import numpy as np

pts = generate_ellipsoid_points(radii=(3, 2, 1), n_points=300, noise_std=0.05)
result = fit_rbf_ellipsoid_linear(pts)

if result is not None:
    alpha, beta, centroid, scale = result
    norm_pts = (pts - centroid) / scale
    vals = evaluate_model_linear(norm_pts, norm_pts, alpha, beta)
    print("RMS residual:", np.sqrt(np.mean(vals**2)))
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

## API Reference

### RBF with Ellipsoid Constraint (Li &amp; et al., CGF 2004)

#### `fit_rbf_ellipsoid_linear(points, smooth=0.0) → tuple | None`

Fit an implicit ellipsoidal surface to 3-D point data using a linear RBF
kernel with ellipsoid constraint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | ndarray (N, 3) | 3-D surface points. At least 10 required. |
| `smooth` | float | Regularisation parameter added to the RBF diagonal (default 0.0) |

Returns `(alpha, beta, centroid, scale)` or `None` if no valid eigenvalue is found.

#### `evaluate_model_linear(eval_pts, norm_pts, alpha, beta, chunk_size=5000) → ndarray`

Evaluate the fitted implicit surface F at query points (in normalised coordinates).

---

### Synthetic data generator

#### `generate_ellipsoid_points(centre, radii, rotation, n_points, noise_std, seed) → ndarray (N, 3)`

Generate 3-D points sampled uniformly on an ellipsoid surface.


---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Citation

If you use this code in academic work, please cite the original paper:

```bibtex
@article{li2004rbf,
  title     = {Implicit fitting using radial basis functions with ellipsoid constraint},
  author    = {Qingde Li, et al.},
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
