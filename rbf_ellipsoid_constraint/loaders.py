"""Multi-format 3-D point-cloud data loader.

Supported formats
-----------------
.csv   Comma-separated values (header row; first three columns are x, y, z).
.txt   Whitespace-delimited columns (no header; first three columns are x, y, z).
.xyz   Same as .txt.
.pts   Same as .txt.
.obj   Wavefront OBJ – vertex lines (``v x y z``) are extracted.
.ply   Stanford PLY – ASCII and binary (little-endian / big-endian).
.m     MATLAB script – parses a ``data = [...];`` matrix literal.
.npy   NumPy binary array (shape must be (N, ≥3)).
.npz   NumPy compressed archive (array stored under the key ``"data"``).
"""

from __future__ import annotations

import os
import re
import struct
import numpy as np


# ---------------------------------------------------------------------------
# Individual format loaders
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> np.ndarray:
    """Load x, y, z from a comma-separated file with a header row.

    Parameters
    ----------
    filename : str
        Path to the CSV file.  The first row is treated as a header and
        skipped.  The first three numeric columns are used as x, y, z.

    Returns
    -------
    pts : ndarray of shape (N, 3)
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            f"CSV file must have at least 3 columns; got {data.shape[1]}."
        )
    return data[:, :3].astype(float)


def load_xyz(filename: str) -> np.ndarray:
    """Load x, y, z from a whitespace-delimited file (no header).

    Parameters
    ----------
    filename : str
        Path to the file.  Each row contains at least three numbers.

    Returns
    -------
    pts : ndarray of shape (N, 3)
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            f"XYZ file must have at least 3 columns; got {data.shape[1]}."
        )
    return data[:, :3].astype(float)


def load_obj(filename: str) -> np.ndarray:
    """Load vertex positions from a Wavefront OBJ file.

    Only vertex lines beginning with ``v `` are parsed; face, normal,
    texture-coordinate, and material lines are ignored.

    Parameters
    ----------
    filename : str
        Path to the OBJ file.

    Returns
    -------
    pts : ndarray of shape (N, 3)

    Raises
    ------
    ValueError
        If no vertex entries are found in the file.
    """
    vertices: list[list[float]] = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
    if not vertices:
        raise ValueError(
            f"No vertex ('v') entries found in OBJ file: {filename!r}"
        )
    return np.array(vertices, dtype=float)


# ---------------------------------------------------------------------------
# PLY loader (ASCII and binary)
# ---------------------------------------------------------------------------

#: Mapping from PLY scalar type names to (struct format char, byte size)
_PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char":    ("b", 1), "uchar":   ("B", 1),
    "short":   ("h", 2), "ushort":  ("H", 2),
    "int":     ("i", 4), "uint":    ("I", 4),
    "float":   ("f", 4), "float32": ("f", 4),
    "double":  ("d", 8), "float64": ("d", 8),
    "int8":    ("b", 1), "uint8":   ("B", 1),
    "int16":   ("h", 2), "uint16":  ("H", 2),
    "int32":   ("i", 4), "uint32":  ("I", 4),
    "int64":   ("q", 8), "uint64":  ("Q", 8),
}


def load_ply(filename: str) -> np.ndarray:
    """Load the vertex x, y, z coordinates from a Stanford PLY file.

    Handles both **ASCII** and **binary** (little-endian or big-endian) PLY
    files.  Only the first element named ``vertex`` is read; face, edge, and
    other element data are skipped.

    Parameters
    ----------
    filename : str
        Path to the PLY file.

    Returns
    -------
    pts : ndarray of shape (N, 3)

    Raises
    ------
    ValueError
        If the file is not a valid PLY file, if it has no vertex element, or
        if the vertex element lacks ``x``, ``y``, ``z`` properties.
    """
    with open(filename, "rb") as fh:
        # ---- Parse header ----
        header_lines: list[str] = []
        found_end_header = False
        for _ in range(1000):          # hard limit guards against malformed files
            raw = fh.readline()
            if not raw:                # EOF before end_header
                break
            line = raw.decode("ascii", errors="replace").rstrip("\r\n")
            header_lines.append(line)
            if line.strip() == "end_header":
                found_end_header = True
                break

        if not header_lines or header_lines[0].strip() != "ply":
            raise ValueError(f"Not a valid PLY file: {filename!r}")
        if not found_end_header:
            raise ValueError(
                f"PLY file is missing 'end_header': {filename!r}"
            )

        # Determine binary/ascii format
        ply_format = "ascii"
        for ln in header_lines:
            if ln.startswith("format"):
                ply_format = ln.split()[1]
                break

        # Parse element / property declarations
        n_vertex = 0
        x_idx = y_idx = z_idx = -1
        prop_info: list[tuple[str, str | None, int | None]] = []
        # Each entry: (prop_name, struct_fmt_char | None, byte_size | None)
        # None values indicate a list property (unsupported for fixed dtype)
        in_vertex_element = False

        for ln in header_lines:
            tokens = ln.split()
            if not tokens:
                continue
            if tokens[0] == "element":
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    n_vertex = int(tokens[2])
                    prop_info = []
                    x_idx = y_idx = z_idx = -1
            elif tokens[0] == "property" and in_vertex_element:
                if tokens[1] == "list":
                    prop_info.append(("_list", None, None))
                    continue
                pname = tokens[2]
                fmt_char, bsize = _PLY_SCALAR_TYPES.get(tokens[1], ("f", 4))
                idx = len(prop_info)
                prop_info.append((pname, fmt_char, bsize))
                if pname == "x":
                    x_idx = idx
                elif pname == "y":
                    y_idx = idx
                elif pname == "z":
                    z_idx = idx

        if n_vertex == 0:
            raise ValueError(
                f"PLY file has no vertex elements: {filename!r}"
            )
        if x_idx < 0 or y_idx < 0 or z_idx < 0:
            raise ValueError(
                f"PLY vertex element must have x, y, z properties: {filename!r}"
            )

        # ---- Read vertex data ----
        if ply_format == "ascii":
            pts = _ply_read_ascii(fh, n_vertex, x_idx, y_idx, z_idx)
        else:
            endian = "<" if ply_format == "binary_little_endian" else ">"
            pts = _ply_read_binary(
                fh, n_vertex, x_idx, y_idx, z_idx, prop_info, endian
            )

    return pts


def _ply_read_ascii(
    fh,
    n_vertex: int,
    xi: int,
    yi: int,
    zi: int,
) -> np.ndarray:
    pts = np.empty((n_vertex, 3), dtype=float)
    for i in range(n_vertex):
        row = fh.readline().decode("ascii", errors="replace").split()
        pts[i, 0] = float(row[xi])
        pts[i, 1] = float(row[yi])
        pts[i, 2] = float(row[zi])
    return pts


def _ply_read_binary(
    fh,
    n_vertex: int,
    xi: int,
    yi: int,
    zi: int,
    prop_info: list,
    endian: str,
) -> np.ndarray:
    """Read fixed-width binary vertex records."""
    # Validate: list properties are not supported in binary mode
    for pname, fmt_char, _ in prop_info:
        if fmt_char is None:
            raise ValueError(
                "Binary PLY files with list properties in the vertex element "
                "are not supported by this loader.  "
                "Convert to ASCII PLY first."
            )

    # Build numpy structured dtype from property list
    dtype_fields = [
        (pname, endian + fmt_char)
        for pname, fmt_char, _ in prop_info
    ]
    dt = np.dtype(dtype_fields)
    raw = fh.read(n_vertex * dt.itemsize)
    arr = np.frombuffer(raw, dtype=dt)

    x_name = prop_info[xi][0]
    y_name = prop_info[yi][0]
    z_name = prop_info[zi][0]
    pts = np.column_stack(
        [arr[x_name], arr[y_name], arr[z_name]]
    ).astype(float)
    return pts


# ---------------------------------------------------------------------------
# MATLAB .m loader
# ---------------------------------------------------------------------------

def load_matlab(filename: str) -> np.ndarray:
    """Load a point matrix from a MATLAB-style (``.m``) script file.

    Looks for the pattern::

        data = [...];

    in the file and parses the enclosed numbers as an (N × 3) array.
    Row separators can be either semicolons or newlines; column separators
    can be commas or spaces.

    Parameters
    ----------
    filename : str
        Path to the ``.m`` file.

    Returns
    -------
    pts : ndarray of shape (N, 3)

    Raises
    ------
    ValueError
        If the expected pattern is not found, or if the number of parsed
        values is not divisible by 3.
    """
    with open(filename, "r") as fh:
        content = fh.read()

    match = re.search(r"data\s*=\s*\[(.*?)\];", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not find 'data = [...];' pattern in MATLAB file: {filename!r}"
        )

    data_str = match.group(1).replace(",", " ").replace(";", " ")
    try:
        numbers = [float(x) for x in data_str.split()]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric data inside 'data = [...];' in {filename!r}"
        ) from exc

    if len(numbers) % 3 != 0:
        raise ValueError(
            f"Number of values ({len(numbers)}) is not divisible by 3 "
            f"in MATLAB file {filename!r}."
        )
    return np.array(numbers, dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------------
# NumPy binary loaders
# ---------------------------------------------------------------------------

def load_npy(filename: str) -> np.ndarray:
    """Load a point cloud from a NumPy ``.npy`` binary file.

    Parameters
    ----------
    filename : str
        Path to the ``.npy`` file.  The array must be 2-D with at least
        three columns.

    Returns
    -------
    pts : ndarray of shape (N, 3)
    """
    data = np.load(filename)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"NumPy array in {filename!r} must be 2-D with ≥3 columns; "
            f"got shape {data.shape}."
        )
    return data[:, :3].astype(float)


def load_npz(filename: str) -> np.ndarray:
    """Load a point cloud from a NumPy ``.npz`` compressed archive.

    The archive must contain an array stored under the key ``"data"``,
    which must be 2-D with at least three columns.

    Parameters
    ----------
    filename : str
        Path to the ``.npz`` file.

    Returns
    -------
    pts : ndarray of shape (N, 3)
    """
    archive = np.load(filename)
    if "data" not in archive:
        raise ValueError(
            f"Expected a 'data' key in NPZ archive {filename!r}; "
            f"available keys: {list(archive.keys())}."
        )
    data = archive["data"]
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Array 'data' in {filename!r} must be 2-D with ≥3 columns; "
            f"got shape {data.shape}."
        )
    return data[:, :3].astype(float)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

#: Mapping from lowercase file extension to the corresponding loader function.
FORMAT_LOADERS: dict = {
    ".csv": load_csv,
    ".txt": load_xyz,
    ".xyz": load_xyz,
    ".pts": load_xyz,
    ".obj": load_obj,
    ".ply": load_ply,
    ".m":   load_matlab,
    ".npy": load_npy,
    ".npz": load_npz,
}


def load_point_cloud(filename: str) -> np.ndarray:
    """Load a 3-D point cloud from a file, auto-detecting the format.

    The file format is inferred from the file extension (case-insensitive).

    Supported extensions and their expected structure:

    =========  ==============================================================
    ``.csv``   Comma-separated; first row is a header; first 3 columns x,y,z
    ``.txt``   Whitespace-delimited; no header; first 3 columns are x, y, z
    ``.xyz``   Same as ``.txt``
    ``.pts``   Same as ``.txt``
    ``.obj``   Wavefront OBJ; only vertex lines (``v x y z``) are read
    ``.ply``   Stanford PLY; ASCII and binary (little/big endian)
    ``.m``     MATLAB script; parses ``data = [...];`` matrix literal
    ``.npy``   NumPy binary array of shape (N, ≥3)
    ``.npz``   NumPy compressed archive with an array stored as ``"data"``
    =========  ==============================================================

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    pts : ndarray of shape (N, 3)
        Point-cloud coordinates, one point per row.

    Raises
    ------
    FileNotFoundError
        If *filename* does not exist.
    ValueError
        If the file extension is not recognised, or the file cannot be parsed.

    Examples
    --------
    >>> pts = load_point_cloud("data/synthetic_ellipsoid_low_noise.csv")
    >>> pts.shape
    (300, 3)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename!r}")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in FORMAT_LOADERS:
        raise ValueError(
            f"Unrecognised file extension {ext!r}.  "
            f"Supported formats: {sorted(FORMAT_LOADERS.keys())}."
        )

    return FORMAT_LOADERS[ext](filename)
