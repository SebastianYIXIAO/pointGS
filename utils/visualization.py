import os
import struct
import numpy as np

from .io_utils import list_txt_files

S3DIS_COLOR_MAP = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194),
]


def labels_to_colored_ply(input_dir, output_dir, color_map=None):
    """Convert labeled point clouds to colored PLY files for visualization.

    Input format: x y z label (4 columns) or x y z r g b label (7 columns).
    Label is used to assign color from the color map.

    Args:
        input_dir: Directory with labeled .txt point clouds.
        output_dir: Output directory for .ply files.
        color_map: List of (r, g, b) tuples for each class. Defaults to S3DIS colors.
    """
    if color_map is None:
        color_map = S3DIS_COLOR_MAP
    os.makedirs(output_dir, exist_ok=True)

    for filename in list_txt_files(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".ply")

        points = []
        with open(input_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    if len(parts) == 4:
                        x, y, z, label = float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))
                    elif len(parts) >= 7:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        label = int(float(parts[6]))
                    else:
                        continue
                    if 0 <= label < len(color_map):
                        r, g, b = color_map[label]
                        points.append((x, y, z, r, g, b))
                except (ValueError, IndexError):
                    continue

        if not points:
            continue

        with open(output_path, "wb") as ply:
            header = "\n".join([
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {len(points)}",
                "property float x", "property float y", "property float z",
                "property uchar red", "property uchar green", "property uchar blue",
                "end_header",
            ]) + "\n"
            ply.write(header.encode("utf-8"))
            for pt in points:
                ply.write(struct.pack("fffBBB", *pt))

        print(f"Converted {filename} -> {os.path.basename(output_path)} ({len(points)} points)")
