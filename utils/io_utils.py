import os
import numpy as np


def load_point_cloud(file_path, label_col=None, skip_header=False):
    """Load point cloud from a text file.

    Args:
        file_path: Path to the text file.
        label_col: Column index for labels (e.g. 3 or 6). None means no labels.
        skip_header: Whether to skip the first line.

    Returns:
        points: (N, 3) array of xyz coordinates.
        labels: (N,) integer array if label_col is set, else None.
        full_data: The full loaded array for access to other columns.
    """
    data = np.loadtxt(file_path, skiprows=1 if skip_header else 0)
    if data.ndim == 1:
        data = data[None, :]
    points = data[:, :3]
    labels = None
    if label_col is not None and data.shape[1] > label_col:
        labels = data[:, label_col].astype(int)
    return points, labels, data


def save_point_cloud(file_path, points, labels=None, fmt_coords="%.6f"):
    """Save point cloud to a text file.

    Args:
        file_path: Output path.
        points: (N, 3) xyz coordinates.
        labels: (N,) integer labels, optional.
        fmt_coords: Format string for coordinates.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if labels is not None:
        data = np.column_stack((points, labels.reshape(-1, 1)))
        fmt = f"{fmt_coords} {fmt_coords} {fmt_coords} %d"
    else:
        data = points
        fmt = f"{fmt_coords} {fmt_coords} {fmt_coords}"
    np.savetxt(file_path, data, fmt=fmt)


def save_point_cloud_full(file_path, data, fmt="%.6f"):
    """Save full point cloud data array preserving all columns."""
    np.savetxt(file_path, data, fmt=fmt)


def list_txt_files(directory):
    """List all .txt files in a directory, sorted."""
    return sorted([f for f in os.listdir(directory) if f.endswith(".txt")])
