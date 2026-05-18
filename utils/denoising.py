import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from .io_utils import list_txt_files


def voxel_grid_denoise(input_dir, output_dir, voxel_size=0.15, cube_size=5):
    """Denoise by finding the densest cube_size^3 voxel block.

    For each file, builds a voxel grid and finds the cube_size x cube_size x cube_size
    block containing the most points. Only points within that block are kept.

    Input format: x y z label (with optional header line, skipped).
    """
    os.makedirs(output_dir, exist_ok=True)
    files = list_txt_files(input_dir)

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path, skiprows=1)
        if len(data) == 0:
            continue

        points = data[:, :3]
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        n_voxels = np.ceil((maxs - mins) / voxel_size).astype(int)

        voxel_indices = {}
        for i, pt in enumerate(points):
            vx = int((pt[0] - mins[0]) // voxel_size)
            vy = int((pt[1] - mins[1]) // voxel_size)
            vz = int((pt[2] - mins[2]) // voxel_size)
            key = (vx, vy, vz)
            if key not in voxel_indices:
                voxel_indices[key] = []
            voxel_indices[key].append(i)

        best_count = 0
        best_coord = None
        cs = cube_size
        for x in range(max(n_voxels[0] - cs + 1, 1)):
            for y in range(max(n_voxels[1] - cs + 1, 1)):
                for z in range(max(n_voxels[2] - cs + 1, 1)):
                    count = 0
                    for dx in range(cs):
                        for dy in range(cs):
                            for dz in range(cs):
                                key = (x + dx, y + dy, z + dz)
                                if key in voxel_indices:
                                    count += len(voxel_indices[key])
                    if count > best_count:
                        best_count = count
                        best_coord = (x, y, z)

        if best_coord is None:
            continue

        keep = []
        for dx in range(cs):
            for dy in range(cs):
                for dz in range(cs):
                    key = (best_coord[0] + dx, best_coord[1] + dy, best_coord[2] + dz)
                    if key in voxel_indices:
                        keep.extend(voxel_indices[key])

        denoised = data[np.unique(keep)]
        np.savetxt(os.path.join(output_dir, filename), denoised, fmt="%.6f")
        print(f"Voxel denoise {filename}: {len(data)} -> {len(denoised)}")


def remove_zero_labels(input_dir, output_dir=None):
    """Remove points with label=0. If output_dir is None, modifies files in-place.

    Input format: x y z label (4 columns).
    """
    in_place = output_dir is None
    if not in_place:
        os.makedirs(output_dir, exist_ok=True)

    for filename in list_txt_files(input_dir):
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 4:
            continue

        mask = data[:, 3] != 0
        filtered = data[mask]

        out_path = file_path if in_place else os.path.join(output_dir, filename)
        np.savetxt(out_path, filtered, fmt="%.6f")
        print(f"Remove label-0 {filename}: {len(data)} -> {len(filtered)}")


def kdtree_radius_denoise(input_dir, output_dir, radius=0.007, min_neighbors=35, iterations=2):
    """Remove sparse outlier points using KDTree radius search.

    Points with fewer than min_neighbors within the given radius are removed.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in list_txt_files(input_dir):
        data = np.loadtxt(os.path.join(input_dir, filename))
        print(f"KDTree denoise {filename}: {len(data)} points loaded")

        for it in range(iterations):
            tree = cKDTree(data[:, :3] if data.shape[1] > 3 else data)
            counts = tree.query_ball_point(data[:, :3] if data.shape[1] > 3 else data, radius, return_length=True)
            mask = counts >= min_neighbors
            data = data[mask]
            print(f"  Iteration {it + 1}: {np.sum(mask)} points retained")

        np.savetxt(os.path.join(output_dir, filename), data, fmt="%.6f")


def label_consistency_filter(input_dir, output_dir, k_consistency=20, k_connected=50,
                             min_consistency_ratio=0.8, min_connected_threshold=30):
    """Filter points based on label consistency and local connectivity.

    For each point, checks:
    1. What fraction of its k_consistency nearest neighbors share the same label.
    2. The size of its connected component within the k_connected neighborhood
       (only same-label neighbors are connected).

    Points failing either threshold are removed.
    Input format: x y z label.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in list_txt_files(input_dir):
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)
        points = data[:, :3]
        labels = data[:, 3].astype(int)

        nbrs_c = NearestNeighbors(n_neighbors=k_consistency, algorithm="auto").fit(points)
        _, idx_c = nbrs_c.kneighbors(points)

        nbrs_k = NearestNeighbors(n_neighbors=k_connected, algorithm="auto").fit(points)
        _, idx_k = nbrs_k.kneighbors(points)

        adj = lil_matrix((len(points), len(points)), dtype=int)
        for i in range(len(points)):
            for j in idx_k[i]:
                if labels[i] == labels[j]:
                    adj[i, j] = 1
                    adj[j, i] = 1

        keep = []
        for i in range(len(points)):
            ratio = np.sum(labels[idx_c[i]] == labels[i]) / k_consistency
            subgraph = adj[idx_k[i], :][:, idx_k[i]]
            _, comp_labels = connected_components(csgraph=subgraph, directed=False)
            comp_size = np.sum(comp_labels == comp_labels[0])
            if ratio >= min_consistency_ratio and comp_size >= min_connected_threshold:
                keep.append(i)

        filtered = data[keep]
        np.savetxt(os.path.join(output_dir, filename), filtered, fmt="%.6f %.6f %.6f %d")
        print(f"Label filter {filename}: {len(data)} -> {len(filtered)}")
