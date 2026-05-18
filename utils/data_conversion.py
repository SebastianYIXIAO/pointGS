import os
import subprocess
import struct
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# PTH to TXT conversion (S3DIS dataset)
# ---------------------------------------------------------------------------

def _read_s3dis_pth(pth_path):
    """Read an S3DIS .pth file and return (points, colors, labels) as numpy."""
    data = torch.load(pth_path, map_location="cpu")

    if isinstance(data, tuple):
        if len(data) < 3:
            raise ValueError(f"PTH tuple too short: {len(data)}")
        points, colors, labels = data[0], data[1], data[2]
        if hasattr(labels, "ndim") and labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.flatten()
    elif isinstance(data, dict):
        points = data.get("coord", data.get("points", data.get("xyz")))
        colors = data.get("color", data.get("colors", data.get("rgb")))
        labels = data.get("semantic_gt", data.get("label", data.get("labels")))
        if points is None:
            raise ValueError("No coordinate data found in PTH file")
        if colors is None:
            colors = np.ones((len(points), 3)) * 127
        if labels is None:
            labels = np.zeros(len(points), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported PTH format: {type(data)}")

    for arr_name in ("points", "colors", "labels"):
        arr = locals()[arr_name]
        if hasattr(arr, "numpy"):
            locals()[arr_name] = arr.numpy()

    points = points.numpy() if hasattr(points, "numpy") else points
    colors = colors.numpy() if hasattr(colors, "numpy") else colors
    labels = labels.numpy() if hasattr(labels, "numpy") else labels
    return points, colors, labels


def _normalize_color(colors):
    """Normalize color values to 0-255 uint8."""
    if colors.dtype in (np.float32, np.float64):
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def convert_pth_to_txt(input_dir, output_dir):
    """Batch convert S3DIS .pth files to .txt format (x y z r g b label)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pth_files = list(input_path.rglob("*.pth"))
    if not pth_files:
        print(f"No .pth files found in {input_dir}")
        return

    print(f"Found {len(pth_files)} .pth files to convert")
    for pth_file in pth_files:
        try:
            points, colors, labels = _read_s3dis_pth(pth_file)
            colors = _normalize_color(colors)
            relative_path = pth_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(".txt")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for i in range(len(points)):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {int(labels[i])}\n")
            print(f"Converted: {pth_file.name} -> {output_file.name} ({len(points)} points)")
        except Exception as e:
            print(f"Failed: {pth_file.name}: {e}")


# ---------------------------------------------------------------------------
# Point cloud to image rendering
# ---------------------------------------------------------------------------

def _plot_perspective(points, centroid, angle, save_path):
    """Render a single perspective view of the point cloud and save as PNG."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 3:6] / 255.0, s=3, alpha=0.7,
    )
    ax.view_init(elev=angle[0], azim=angle[1])
    ax.dist = 8
    ax.set_box_aspect([1, 1, 1])

    max_range = np.ptp(points[:, :3], axis=0).max() / 2.0
    mid_x, mid_y, mid_z = centroid
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.axis("off")
    plt.grid(False)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    del fig, ax


def generate_view_angles(filename, num_views=80):
    """Generate camera angles based on point cloud part name."""
    if "part_1" in filename.lower():
        return [(0 - i * 45 / (num_views - 1), i * 720 / (num_views - 1)) for i in range(num_views)]
    elif "part_2" in filename.lower():
        return [(0 + i * 45 / (num_views - 1), i * 720 / (num_views - 1)) for i in range(num_views)]
    else:
        return [(20 + i * 50 / (num_views - 1), i * 360 / (num_views - 1)) for i in range(num_views)]


def render_views(input_dir, num_views=80):
    """Render multi-view images from point cloud .txt files.

    Creates an 'images/<basename>/' subfolder for each file with numbered PNGs.
    """
    image_dir = os.path.join(input_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} point cloud files to render")

    for idx, filename in enumerate(txt_files):
        print(f"Rendering ({idx + 1}/{len(txt_files)}): {filename}")
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)
        if data is None or len(data) == 0:
            continue

        centroid = data[:, :3].mean(axis=0)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(image_dir, base)
        os.makedirs(folder, exist_ok=True)

        angles = generate_view_angles(filename, num_views)
        for j, angle in enumerate(angles):
            save_path = os.path.join(folder, f"{j:03d}.png")
            _plot_perspective(data, centroid, angle, save_path)
            if (j + 1) % 10 == 0:
                gc.collect()

        del data
        gc.collect()
        print(f"  Generated {len(angles)} views")


# ---------------------------------------------------------------------------
# COLMAP SfM runner
# ---------------------------------------------------------------------------

def _read_images_bin(bin_path):
    """Read COLMAP images.bin and return list of (image_id, qw,qx,qy,qz, tx,ty,tz, camera_id, name)."""
    images = []
    with open(bin_path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)
            images.append((image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name))
    return images


def run_colmap_sfm(colmap_exe, root_dir):
    """Run COLMAP SfM pipeline on rendered images.

    Processes each subfolder in <root_dir>/images/ that contains PNG files.
    Creates COLMAP database and sparse reconstruction for each scene.

    Args:
        colmap_exe: Path to the COLMAP executable.
        root_dir: Root directory containing 'images/<scene_name>/' folders.
    """
    images_root = os.path.join(root_dir, "images")
    if not os.path.isdir(images_root):
        print(f"No images directory found at {images_root}")
        return

    for scene_name in os.listdir(images_root):
        scene_images = os.path.join(images_root, scene_name)
        if not os.path.isdir(scene_images):
            continue

        database_path = os.path.join(scene_images, "database.db")
        sparse_path = os.path.join(scene_images, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        print(f"Running COLMAP on scene: {scene_name}")

        subprocess.run([
            colmap_exe, "feature_extractor",
            "--database_path", database_path,
            "--image_path", scene_images,
        ], check=True)

        subprocess.run([
            colmap_exe, "sequential_matcher",
            "--database_path", database_path,
        ], check=True)

        subprocess.run([
            colmap_exe, "mapper",
            "--database_path", database_path,
            "--image_path", scene_images,
            "--output_path", sparse_path,
        ], check=True)

        points3d_path = os.path.join(sparse_path, "0", "points3D.bin")
        if os.path.exists(points3d_path):
            size_kb = os.path.getsize(points3d_path) / 1024
            if size_kb < 1:
                print(f"  Warning: points3D.bin is very small ({size_kb:.1f} KB), reconstruction may have failed")
            else:
                print(f"  Reconstruction complete: points3D.bin = {size_kb:.1f} KB")
        else:
            print(f"  Warning: points3D.bin not found, reconstruction may have failed")
