import os
import numpy as np
import torch

from .io_utils import load_point_cloud, list_txt_files


def _compute_num_segments(length, min_seg=4.0, max_seg=7.0):
    """Compute how many segments to divide a length into so each is in [min_seg, max_seg]."""
    target = (min_seg + max_seg) / 2.0
    n = max(2, round(length / target))
    while length / n > max_seg:
        n += 1
    while length / n < min_seg and n > 2:
        n -= 1
    return n


def _split_along_shortest_axis(data, base_name, output_dir, fmt):
    """Split a point cloud at the median of its shortest axis into two parts."""
    ranges = np.ptp(data[:, :3], axis=0)
    shortest_axis = int(np.argmin(ranges))
    median_val = np.median(data[:, shortest_axis])

    part1 = data[data[:, shortest_axis] >= median_val]
    part2 = data[data[:, shortest_axis] < median_val]

    np.savetxt(os.path.join(output_dir, f"{base_name}_part_1.txt"), part1, fmt=fmt)
    np.savetxt(os.path.join(output_dir, f"{base_name}_part_2.txt"), part2, fmt=fmt)
    return len(part1), len(part2)


def segment_and_split(input_dir, output_dir, max_length=7.0, min_seg=4.0, max_seg=7.0):
    """Segment large scenes along the longest axis, then split each segment in half.

    For each point cloud:
    - If all axis lengths <= max_length: just split at the shortest-axis median
      -> <name>_part_1.txt, <name>_part_2.txt
    - If any axis > max_length: segment along the longest axis into N pieces
      (each piece avg length in [min_seg, max_seg]), then split each piece
      -> <name>_segment1_part_1.txt, <name>_segment1_part_2.txt, ...

    Input format: x y z r g b label (7 columns).
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = list_txt_files(input_dir)
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    fmt = "%.6f %.6f %.6f %d %d %d %d"

    for filename in txt_files:
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)
        if len(data) == 0:
            continue

        base = os.path.splitext(filename)[0]
        ranges = np.ptp(data[:, :3], axis=0)

        if ranges.max() <= max_length:
            n1, n2 = _split_along_shortest_axis(data, base, output_dir, fmt)
            print(f"  {filename}: no segmentation needed, split -> part_1({n1}), part_2({n2})")
            continue

        longest_axis = int(np.argmax(ranges))
        axis_names = ["X", "Y", "Z"]
        n_seg = _compute_num_segments(ranges[longest_axis], min_seg, max_seg)
        seg_len = ranges[longest_axis] / n_seg
        axis_min = data[:, longest_axis].min()

        print(f"  {filename}: {axis_names[longest_axis]}-axis={ranges[longest_axis]:.2f} "
              f"-> {n_seg} segments (each ~{seg_len:.2f})")

        for seg_idx in range(n_seg):
            seg_lo = axis_min + seg_idx * seg_len
            seg_hi = axis_min + (seg_idx + 1) * seg_len
            if seg_idx == n_seg - 1:
                mask = (data[:, longest_axis] >= seg_lo)
            else:
                mask = (data[:, longest_axis] >= seg_lo) & (data[:, longest_axis] < seg_hi)

            segment_data = data[mask]
            if len(segment_data) == 0:
                continue

            seg_name = f"{base}_segment{seg_idx + 1}"
            n1, n2 = _split_along_shortest_axis(segment_data, seg_name, output_dir, fmt)
            print(f"    segment{seg_idx + 1}: {len(segment_data)} pts -> part_1({n1}), part_2({n2})")


def farthest_point_sampling(points, k):
    """Farthest Point Sampling on a (N, 3) point set. Returns indices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.tensor(points, device=device, dtype=torch.float32)
    N = pts.shape[0]
    k = min(k, N)
    sampled = torch.zeros(k, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    sampled[0] = torch.randint(0, N, (1,), device=device)
    for i in range(1, k):
        dist = torch.norm(pts - pts[sampled[i - 1]].unsqueeze(0), dim=1)
        distances = torch.min(distances, dist)
        sampled[i] = torch.argmax(distances)
    return sampled.cpu().numpy()


def compute_diameter(points):
    """Compute the maximum pairwise distance of a point set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.tensor(points, device=device, dtype=torch.float32)
    dists = torch.cdist(pts, pts, p=2)
    return dists.max().item()


def normalize_scale(contour_dir, reference_dir, full_dir, output_dir, fps_k=1024):
    """Scale Gaussian point clouds to match reference diameter.

    Uses FPS to compute diameters, then scales the full Gaussian point clouds.

    Args:
        contour_dir: Denoised Gaussians (step 2) for diameter computation.
        reference_dir: Reference/GT point clouds for target diameter.
        full_dir: Full Gaussians (step 1) to actually scale.
        output_dir: Output directory for scaled results.
        fps_k: Number of FPS samples for diameter estimation.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = list_txt_files(contour_dir)

    for filename in files:
        path_contour = os.path.join(contour_dir, filename)
        path_ref = os.path.join(reference_dir, filename)
        path_full = os.path.join(full_dir, filename)

        if not (os.path.exists(path_ref) and os.path.exists(path_full)):
            print(f"Skipping {filename}: missing files in reference or full directory")
            continue

        print(f"Scaling {filename}...")
        pts_contour, _, _ = load_point_cloud(path_contour, label_col=3)
        pts_ref, _, _ = load_point_cloud(path_ref, label_col=3)
        pts_full, labels_full, _ = load_point_cloud(path_full, label_col=3)

        idx_contour = farthest_point_sampling(pts_contour, fps_k)
        idx_ref = farthest_point_sampling(pts_ref, fps_k)

        diam_contour = compute_diameter(pts_contour[idx_contour])
        diam_ref = compute_diameter(pts_ref[idx_ref])
        scale = diam_ref / diam_contour

        scaled_pts = pts_full * scale
        print(f"  Diameter: Gaussian={diam_contour:.4f}, Ref={diam_ref:.4f}, Scale={scale:.4f}")

        out_data = np.column_stack((scaled_pts, labels_full.reshape(-1, 1)))
        np.savetxt(os.path.join(output_dir, filename), out_data, fmt="%.6f", delimiter=" ")
