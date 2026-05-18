import os
import numpy as np
from scipy.spatial import cKDTree

from .io_utils import load_point_cloud, save_point_cloud, list_txt_files


def transfer_labels_1nn(source_dir, target_dir, output_dir):
    """Transfer labels from source to target point clouds using 1-nearest-neighbor.

    For each matching file pair:
    - Source (A): x y z label (label_col=3)
    - Target (B): x y z (no labels)

    Builds a KDTree on source xyz, queries each target point with k=1,
    and assigns the nearest source label.

    Args:
        source_dir: Directory with labeled point clouds (x y z label).
        target_dir: Directory with unlabeled point clouds (x y z ...).
        output_dir: Output directory for labeled target clouds (x y z label).
    """
    os.makedirs(output_dir, exist_ok=True)
    files = list_txt_files(source_dir)

    for filename in files:
        src_path = os.path.join(source_dir, filename)
        tgt_path = os.path.join(target_dir, filename)

        if not os.path.exists(tgt_path):
            print(f"Skipping {filename}: no matching file in target directory")
            continue

        print(f"1NN label transfer: {filename}")
        pts_src, labels_src, _ = load_point_cloud(src_path, label_col=3)
        pts_tgt, _, _ = load_point_cloud(tgt_path)

        tree = cKDTree(pts_src)
        _, indices = tree.query(pts_tgt, k=1)
        transferred = labels_src[indices]

        save_point_cloud(os.path.join(output_dir, filename), pts_tgt, transferred)
