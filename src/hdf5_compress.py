"""
HDF5 Compression Utilities for PINN/CFD Flow Field Data.

Converts UofW-format flat CSVs and partner JSON exports into structured,
compressed HDF5 files. Provides PyTorch loading helpers.

Schema:
    /meta          (attrs: nx, ny, nz, nt, domain bounds, units, source)
    /coords/x      (1D float32)
    /coords/y      (1D float32)
    /coords/z      (1D float32 — length 1 for 2D)
    /coords/t      (1D float32)
    /fields/<name> (Nt, Nz, Ny, Nx — chunked by timestep, gzip+shuffle)
"""

import csv
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GZIP_LEVEL = 4  # ~95% of level-9 compression at 3x write speed


# --------------------------------------------------------------------------- #
# Core: CSV → HDF5
# --------------------------------------------------------------------------- #
def csv_to_hdf5(
    csv_path: str,
    hdf5_path: str,
    field_columns: list[str] | None = None,
    coord_columns: list[str] | None = None,
    time_column: str = "time_s",
    source: str = "csv",
    units: str = "mm, bar, s",
) -> dict:
    """
    Convert a UofW-format flat CSV to structured HDF5.

    CSV format: one row per (x, y, z, t) grid point with columns like
    x_mm, y_mm, z_mm, Pressure_bar, time_s.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    hdf5_path : str
        Path for output HDF5 file.
    field_columns : list[str], optional
        Column names to store as fields. Default: auto-detect non-coord columns.
    coord_columns : list[str], optional
        Spatial coordinate column names. Default: ['x_mm', 'y_mm', 'z_mm'].
    time_column : str
        Time column name. Default: 'time_s'.
    source : str
        Source metadata string.
    units : str
        Units metadata string.

    Returns
    -------
    dict with keys: hdf5_size, csv_size, ratio, write_time
    """
    t0 = time.perf_counter()

    if coord_columns is None:
        coord_columns = ["x_mm", "y_mm", "z_mm"]

    # Read CSV
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    all_cols = list(data.dtype.names)

    if field_columns is None:
        skip = set(coord_columns) | {time_column}
        field_columns = [c for c in all_cols if c not in skip]

    # Extract unique sorted coordinates
    xs = np.unique(data[coord_columns[0]])
    ys = np.unique(data[coord_columns[1]])
    if len(coord_columns) >= 3 and coord_columns[2] in all_cols:
        zs = np.unique(data[coord_columns[2]])
    else:
        zs = np.array([0.0])
    ts = np.unique(data[time_column])

    nx, ny, nz, nt = len(xs), len(ys), len(zs), len(ts)

    # Build index maps for fast lookup
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    z_idx = {v: i for i, v in enumerate(zs)}
    t_idx = {v: i for i, v in enumerate(ts)}

    # Reshape fields into 4D arrays
    fields_4d = {}
    for fname in field_columns:
        fields_4d[fname] = np.full((nt, nz, ny, nx), np.nan, dtype=np.float32)

    for row in data:
        xi = x_idx[row[coord_columns[0]]]
        yi = y_idx[row[coord_columns[1]]]
        if len(coord_columns) >= 3 and coord_columns[2] in all_cols:
            zi = z_idx[row[coord_columns[2]]]
        else:
            zi = 0
        ti = t_idx[row[time_column]]
        for fname in field_columns:
            fields_4d[fname][ti, zi, yi, xi] = np.float32(row[fname])

    # Write HDF5
    _write_hdf5(
        hdf5_path, xs, ys, zs, ts, fields_4d,
        nx=nx, ny=ny, nz=nz, nt=nt,
        source=source, units=units,
    )

    write_time = time.perf_counter() - t0
    csv_size = os.path.getsize(csv_path)
    hdf5_size = os.path.getsize(hdf5_path)
    return {
        "csv_size": csv_size,
        "hdf5_size": hdf5_size,
        "ratio": csv_size / hdf5_size if hdf5_size > 0 else float("inf"),
        "write_time": write_time,
    }


# --------------------------------------------------------------------------- #
# Core: JSON → HDF5
# --------------------------------------------------------------------------- #
def json_to_hdf5(
    json_path: str,
    hdf5_path: str,
    source: str = "partner_pinn",
    units: str = "nondim",
) -> dict:
    """
    Convert partner JSON format to structured HDF5.

    JSON schema (from export_flow_fields_json):
        meta: {nx, ny, nt, x_min, x_max, y_min, y_max, t_min, t_max}
        grid: {x: [...], y: [...], t: [...]}
        fields: {u: [nt][ny][nx], v: ..., p: ..., T: ...}

    Returns
    -------
    dict with keys: json_size, hdf5_size, ratio, write_time, fields
    """
    t0 = time.perf_counter()

    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data["meta"]
    nx, ny, nt = meta["nx"], meta["ny"], meta["nt"]
    nz = meta.get("nz", 1)

    xs = np.array(data["grid"]["x"], dtype=np.float32)
    ys = np.array(data["grid"]["y"], dtype=np.float32)
    zs = np.array(data["grid"].get("z", [0.0]), dtype=np.float32)
    ts = np.array(data["grid"]["t"], dtype=np.float32)

    # Map partner field names to our canonical names
    field_map = {
        "u": "velocity_x",
        "v": "velocity_y",
        "w": "velocity_z",
        "p": "pressure",
        "T": "temperature",
    }

    fields_4d = {}
    for src_name, canonical in field_map.items():
        if src_name in data["fields"]:
            arr = np.array(data["fields"][src_name], dtype=np.float32)
            # arr shape: (nt, ny, nx) — insert z dimension
            if arr.ndim == 3:
                arr = arr[:, np.newaxis, :, :]  # (nt, 1, ny, nx)
            fields_4d[canonical] = arr

    _write_hdf5(
        hdf5_path, xs, ys, zs, ts, fields_4d,
        nx=nx, ny=ny, nz=nz, nt=nt,
        x_min=meta.get("x_min"), x_max=meta.get("x_max"),
        y_min=meta.get("y_min"), y_max=meta.get("y_max"),
        t_min=meta.get("t_min"), t_max=meta.get("t_max"),
        source=source, units=units,
    )

    write_time = time.perf_counter() - t0
    json_size = os.path.getsize(json_path)
    hdf5_size = os.path.getsize(hdf5_path)
    return {
        "json_size": json_size,
        "hdf5_size": hdf5_size,
        "ratio": json_size / hdf5_size if hdf5_size > 0 else float("inf"),
        "write_time": write_time,
        "fields": list(fields_4d.keys()),
    }


# --------------------------------------------------------------------------- #
# Core: HDF5 → CSV (roundtrip verification)
# --------------------------------------------------------------------------- #
def hdf5_to_csv(
    hdf5_path: str,
    csv_path: str,
    coord_names: tuple[str, str, str] = ("x_mm", "y_mm", "z_mm"),
    time_name: str = "time_s",
) -> float:
    """
    Convert structured HDF5 back to flat CSV.

    Returns write time in seconds.
    """
    t0 = time.perf_counter()

    with h5py.File(hdf5_path, "r") as hf:
        xs = hf["coords/x"][:]
        ys = hf["coords/y"][:]
        zs = hf["coords/z"][:]
        ts = hf["coords/t"][:]

        field_names = list(hf["fields"].keys())
        fields = {name: hf[f"fields/{name}"][:] for name in field_names}

    nt, nz, ny, nx = fields[field_names[0]].shape

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = list(coord_names) + [time_name] + field_names
        writer.writerow(header)

        for ti in range(nt):
            for zi in range(nz):
                for yi in range(ny):
                    for xi in range(nx):
                        row = [
                            float(xs[xi]),
                            float(ys[yi]),
                            float(zs[zi]),
                            float(ts[ti]),
                        ]
                        for fname in field_names:
                            row.append(float(fields[fname][ti, zi, yi, xi]))
                        writer.writerow(row)

    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Loading: HDF5 → PyTorch
# --------------------------------------------------------------------------- #
def load_hdf5_to_torch(hdf5_path: str, device: str = "cpu") -> dict:
    """
    Load all fields and coordinates from HDF5 as PyTorch tensors.

    Returns
    -------
    dict with keys: 'coords' (dict of 1D tensors), 'fields' (dict of 4D tensors),
    'meta' (dict of attributes)
    """
    import torch

    result = {"coords": {}, "fields": {}, "meta": {}}

    with h5py.File(hdf5_path, "r") as hf:
        for name in hf["coords"]:
            result["coords"][name] = torch.tensor(
                hf[f"coords/{name}"][:], device=device
            )
        for name in hf["fields"]:
            result["fields"][name] = torch.tensor(
                hf[f"fields/{name}"][:], device=device
            )
        for key, val in hf["meta"].attrs.items():
            result["meta"][key] = val

    return result


def load_field_at_timestep(
    hdf5_path: str, field: str, timestep: int, device: str = "cpu"
) -> "torch.Tensor":
    """
    Load a single timestep of a single field. O(chunk) I/O thanks to
    chunked storage.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file.
    field : str
        Field name (e.g. 'pressure', 'velocity_x').
    timestep : int
        Timestep index.
    device : str
        PyTorch device.

    Returns
    -------
    torch.Tensor of shape (Nz, Ny, Nx)
    """
    import torch

    with h5py.File(hdf5_path, "r") as hf:
        data = hf[f"fields/{field}"][timestep]  # reads only one chunk
    return torch.tensor(data, device=device)


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def generate_uofw_csv(
    csv_path: str,
    nx: int = 20,
    ny: int = 10,
    nz: int = 1,
    nt: int = 5,
    fields: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic UofW-format CSV for benchmarking.

    Returns
    -------
    dict with keys: csv_size, nx, ny, nz, nt, n_rows
    """
    if fields is None:
        fields = ["Pressure_bar"]

    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 100, nx)   # mm
    ys = np.linspace(0, 50, ny)    # mm
    zs = np.linspace(0, 10, nz) if nz > 1 else np.array([0.0])
    ts = np.linspace(0, 1, nt)     # s

    n_rows = nx * ny * nz * nt

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["x_mm", "y_mm", "z_mm", "time_s"] + fields
        writer.writerow(header)

        for t in ts:
            for z in zs:
                for y in ys:
                    for x in xs:
                        row = [x, y, z, t]
                        for _ in fields:
                            row.append(rng.standard_normal() * 10 + 50)
                        writer.writerow(row)

    csv_size = os.path.getsize(csv_path)
    return {"csv_size": csv_size, "nx": nx, "ny": ny, "nz": nz, "nt": nt, "n_rows": n_rows}


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #
def print_comparison_table(rows: list[dict], title: str = "Compression Results") -> str:
    """
    Format benchmark results as an ASCII table.

    Each row dict should have keys:
        label, original_size, hdf5_size, ratio, write_time

    Returns the formatted string (also prints it).
    """
    header = f"{'Label':<30} {'Original':>12} {'HDF5':>12} {'Ratio':>8} {'Time':>8}"
    sep = "-" * len(header)

    lines = [f"\n{title}", sep, header, sep]
    for r in rows:
        orig = _fmt_size(r["original_size"])
        hdf5 = _fmt_size(r["hdf5_size"])
        ratio = f"{r['ratio']:.1f}x"
        wtime = f"{r['write_time']:.2f}s"
        lines.append(f"{r['label']:<30} {orig:>12} {hdf5:>12} {ratio:>8} {wtime:>8}")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    return text


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _write_hdf5(
    hdf5_path, xs, ys, zs, ts, fields_4d,
    nx=None, ny=None, nz=None, nt=None,
    x_min=None, x_max=None, y_min=None, y_max=None,
    z_min=None, z_max=None, t_min=None, t_max=None,
    source="", units="",
):
    """Write the standardized HDF5 structure."""
    os.makedirs(os.path.dirname(hdf5_path) or ".", exist_ok=True)

    nx = nx or len(xs)
    ny = ny or len(ys)
    nz = nz or len(zs)
    nt = nt or len(ts)

    with h5py.File(hdf5_path, "w") as hf:
        # Metadata
        meta = hf.create_group("meta")
        meta.attrs["nx"] = nx
        meta.attrs["ny"] = ny
        meta.attrs["nz"] = nz
        meta.attrs["nt"] = nt
        meta.attrs["x_min"] = x_min if x_min is not None else float(xs[0])
        meta.attrs["x_max"] = x_max if x_max is not None else float(xs[-1])
        meta.attrs["y_min"] = y_min if y_min is not None else float(ys[0])
        meta.attrs["y_max"] = y_max if y_max is not None else float(ys[-1])
        meta.attrs["z_min"] = z_min if z_min is not None else float(zs[0])
        meta.attrs["z_max"] = z_max if z_max is not None else float(zs[-1])
        meta.attrs["t_min"] = t_min if t_min is not None else float(ts[0])
        meta.attrs["t_max"] = t_max if t_max is not None else float(ts[-1])
        meta.attrs["source"] = source
        meta.attrs["units"] = units

        # Coordinates (1D, stored once)
        coords = hf.create_group("coords")
        coords.create_dataset("x", data=np.asarray(xs, dtype=np.float32))
        coords.create_dataset("y", data=np.asarray(ys, dtype=np.float32))
        coords.create_dataset("z", data=np.asarray(zs, dtype=np.float32))
        coords.create_dataset("t", data=np.asarray(ts, dtype=np.float32))

        # Fields (4D, chunked by timestep, compressed)
        fgrp = hf.create_group("fields")
        for name, arr in fields_4d.items():
            arr = np.asarray(arr, dtype=np.float32)
            chunk_shape = (1, nz, ny, nx)
            fgrp.create_dataset(
                name, data=arr,
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=GZIP_LEVEL,
                shuffle=True,
            )


def _fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    elif nbytes >= 1e6:
        return f"{nbytes / 1e6:.2f} MB"
    elif nbytes >= 1e3:
        return f"{nbytes / 1e3:.1f} KB"
    else:
        return f"{nbytes} B"
