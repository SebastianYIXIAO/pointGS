import os
import re
import shutil
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings

from .io_utils import list_txt_files


# ---------------------------------------------------------------------------
# Step 1: Initial ICP registration with CloudCompare
# ---------------------------------------------------------------------------

def initial_icp(source_dir, reference_dir, output_dir, matrix_dir, cloudcompare_exe):
    """Run initial ICP registration between source and reference point clouds.

    Args:
        source_dir: Scaled Gaussian point clouds.
        reference_dir: Reference point clouds.
        output_dir: Registered point clouds output.
        matrix_dir: Registration matrices output.
        cloudcompare_exe: Path to CloudCompare executable.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)

    ref_files = {os.path.splitext(f)[0]: os.path.join(reference_dir, f)
                 for f in list_txt_files(reference_dir)}
    src_files = {os.path.splitext(f)[0]: os.path.join(source_dir, f)
                 for f in list_txt_files(source_dir)}

    for name, ref_path in ref_files.items():
        if name not in src_files:
            continue
        src_path = src_files[name]
        print(f"Initial ICP: {name}")

        command = [
            cloudcompare_exe, "-SILENT", "-NO_TIMESTAMP",
            "-C_EXPORT_FMT", "ASC",
            "-O", src_path, "-O", ref_path,
            "-ICP", "-SAVE_CLOUDS",
            "-OUT_DIR", source_dir,
        ]
        subprocess.run(command)

        asc_path = os.path.join(source_dir, f"{name}_REGISTERED.asc")
        txt_path = os.path.join(output_dir, f"{name}_REGISTERED.txt")
        if os.path.exists(asc_path):
            os.rename(asc_path, txt_path)

        mat_name = f"{name}_REGISTRATION_MATRIX.txt"
        mat_path = os.path.join(source_dir, mat_name)
        if os.path.exists(mat_path):
            os.rename(mat_path, os.path.join(matrix_dir, mat_name))

        bin_path = os.path.join(reference_dir, f"{name}.bin")
        if os.path.exists(bin_path):
            os.remove(bin_path)


# ---------------------------------------------------------------------------
# Step 2: Generate 24 rotation variants
# ---------------------------------------------------------------------------

def _generate_rotation_matrices():
    """Generate 24 rotation matrices (6 directions x 4 spins)."""
    directions = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    spin_angles = [0, 90, 180, 270]
    normal_vector = np.array([0, 0, 1], dtype=float)
    matrices = []

    for direction in directions:
        direction = np.array(direction, dtype=float)
        nv = normal_vector / np.linalg.norm(normal_vector)
        dn = direction / np.linalg.norm(direction)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if np.allclose(nv, dn, atol=1e-6) or np.allclose(nv, -dn, atol=1e-6):
                initial = R.align_vectors([nv], [dn])[0]
            else:
                ov_src = np.cross(nv, [1, 0, 0])
                if np.linalg.norm(ov_src) < 1e-6:
                    ov_src = np.cross(nv, [0, 1, 0])
                ov_src /= np.linalg.norm(ov_src)
                ov_tgt = np.cross(dn, [1, 0, 0])
                if np.linalg.norm(ov_tgt) < 1e-6:
                    ov_tgt = np.cross(dn, [0, 1, 0])
                ov_tgt /= np.linalg.norm(ov_tgt)
                initial = R.align_vectors([nv, ov_src], [dn, ov_tgt])[0]

        for angle in spin_angles:
            spin = R.from_rotvec(np.radians(angle) * dn)
            combined = initial * spin
            mat = np.eye(4)
            mat[:3, :3] = combined.as_matrix()
            matrices.append(mat)
    return matrices


def generate_24_rotations(input_dir, output_dir):
    """Apply 24 rotation variants to each point cloud file.

    Input format: x y z label.
    Each file produces 24 rotated variants named <base>_00.txt through <base>_23.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    rotation_matrices = _generate_rotation_matrices()

    for filename in list_txt_files(input_dir):
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)
        points = data[:, :3]
        labels = data[:, 3] if data.shape[1] > 3 else np.zeros(len(data))
        center = points.mean(axis=0)

        base = os.path.splitext(filename)[0]
        for i, mat in enumerate(rotation_matrices):
            centered = points - center
            homo = np.hstack((centered, np.ones((len(centered), 1))))
            transformed = (homo @ mat.T)[:, :3] + center
            out = np.column_stack((transformed, labels.reshape(-1, 1)))
            np.savetxt(os.path.join(output_dir, f"{base}_{i:02d}.txt"), out, fmt="%.6f %.6f %.6f %.6f")

        print(f"Generated 24 rotations for {filename}")


# ---------------------------------------------------------------------------
# Step 3: Multi-view ICP with 24 rotations
# ---------------------------------------------------------------------------

def multi_view_icp(reference_dir, rotated_dir, output_dir, matrix_dir, logs_dir, cloudcompare_exe):
    """Run ICP on each of the 24 rotated variants against reference.

    Args:
        reference_dir: Reference point clouds.
        rotated_dir: Directory with 24-rotation variants.
        output_dir: Registered results output.
        matrix_dir: Registration matrices output.
        logs_dir: ICP log files output.
        cloudcompare_exe: Path to CloudCompare executable.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ref_files = {os.path.splitext(f)[0]: os.path.join(reference_dir, f)
                 for f in list_txt_files(reference_dir)}
    rot_files = [f for f in os.listdir(rotated_dir)
                 if f.endswith(".txt") and re.search(r"_\d{2}\.txt$", f)]

    for ref_name, ref_path in ref_files.items():
        matching = [f for f in rot_files if f.startswith(ref_name + "_")]
        if not matching:
            continue

        for rot_file in matching:
            rot_path = os.path.join(rotated_dir, rot_file)
            rot_base = os.path.splitext(rot_file)[0]
            log_path = os.path.join(logs_dir, f"{rot_base}.txt")

            out_asc = os.path.join(rotated_dir, f"{rot_base}_REGISTERED.asc")
            command = [
                cloudcompare_exe, "-SILENT", "-NO_TIMESTAMP",
                "-C_EXPORT_FMT", "ASC",
                "-O", rot_path, "-O", ref_path,
                "-ICP", "-SAVE_CLOUDS", "FILE", out_asc,
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(result.stdout)
                lf.write(result.stderr)

            rms_match = re.search(r"Final RMS difference: ([\d.]+)", result.stdout)
            if rms_match:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\nRMS: {rms_match.group(1)}\n")

            if os.path.exists(out_asc):
                txt_name = f"{rot_base}_REGISTERED.txt"
                os.rename(out_asc, os.path.join(output_dir, txt_name))

            mat_name = f"{rot_base}_REGISTRATION_MATRIX.txt"
            mat_path = os.path.join(rotated_dir, mat_name)
            if os.path.exists(mat_path):
                shutil.move(mat_path, os.path.join(matrix_dir, mat_name))


# ---------------------------------------------------------------------------
# Step 4: Move ASC output files
# ---------------------------------------------------------------------------

def _move_icp_outputs(rotated_dir, output_dir, matrix_dir):
    """Move remaining .asc and matrix files from the rotated directory."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)

    for filename in os.listdir(rotated_dir):
        src = os.path.join(rotated_dir, filename)
        if filename.endswith(".asc"):
            new_name = filename.replace("_REGISTERED", "").replace(".asc", ".txt")
            shutil.move(src, os.path.join(output_dir, new_name))
        elif filename.endswith("_REGISTRATION_MATRIX.txt"):
            shutil.move(src, os.path.join(matrix_dir, filename))


# ---------------------------------------------------------------------------
# Step 5: Select best ICP result by RMS
# ---------------------------------------------------------------------------

def select_best_icp(logs_dir, icp_results_dir, output_dir):
    """Select the best ICP result per scene based on minimum RMS from logs.

    Args:
        logs_dir: Directory containing ICP log files with RMS values.
        icp_results_dir: Directory containing the 24 ICP result files.
        output_dir: Output directory for best results.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_info = {}
    for log_name in os.listdir(logs_dir):
        if not log_name.endswith(".txt"):
            continue
        log_path = os.path.join(logs_dir, log_name)
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"RMS: ([\d.]+)", content)
        if not match:
            continue
        rms = float(match.group(1))

        prefix_match = re.search(r"([^_]+_[^_]+_[^_]+)", log_name)
        if not prefix_match:
            continue
        prefix = prefix_match.group(1)

        if prefix not in file_info:
            file_info[prefix] = []
        file_info[prefix].append((log_name, rms))

    for prefix, entries in file_info.items():
        best_log, best_rms = min(entries, key=lambda x: x[1])
        src_name = best_log.replace("_REGISTERED_", "_")
        src_path = os.path.join(icp_results_dir, src_name)
        if not os.path.exists(src_path):
            print(f"Best ICP file not found: {src_path}")
            continue

        dst_name = re.sub(r"_\d{3}\.txt$", ".txt", src_name)
        if dst_name == src_name:
            dst_name = re.sub(r"_\d{2}\.txt$", ".txt", src_name)
        shutil.copy2(src_path, os.path.join(output_dir, dst_name))
        print(f"Best ICP for {prefix}: RMS={best_rms:.6f} -> {dst_name}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_icp_registration(source_dir, reference_dir, output_dir, cloudcompare_exe, work_dir=None):
    """Run the full ICP registration pipeline.

    Steps:
    1. Initial ICP registration
    2. Generate 24 rotation variants from initial result
    3. Multi-view ICP on all 24 variants
    4. Move output files
    5. Select best result by minimum RMS

    Args:
        source_dir: Scaled Gaussian point clouds.
        reference_dir: Reference point clouds for alignment.
        output_dir: Final best registered point clouds.
        cloudcompare_exe: Path to CloudCompare executable.
        work_dir: Working directory for intermediates (default: output_dir/icp_work).
    """
    if work_dir is None:
        work_dir = os.path.join(output_dir, "icp_work")

    c_dir = os.path.join(work_dir, "C_icp_gauss")
    d_dir = os.path.join(work_dir, "D_24_dirct_gauss")
    m1_dir = os.path.join(work_dir, "M_1st_icp_matrix")
    n_dir = os.path.join(work_dir, "N_24_icp_matrix")
    x_dir = os.path.join(work_dir, "X_24_icp")
    logs_dir = os.path.join(work_dir, "logs")

    print("=== ICP Step 1: Initial registration ===")
    initial_icp(source_dir, reference_dir, c_dir, m1_dir, cloudcompare_exe)

    print("=== ICP Step 2: Generate 24 rotation variants ===")
    generate_24_rotations(c_dir, d_dir)

    print("=== ICP Step 3: Multi-view ICP ===")
    multi_view_icp(reference_dir, d_dir, x_dir, n_dir, logs_dir, cloudcompare_exe)

    print("=== ICP Step 4: Move outputs ===")
    _move_icp_outputs(d_dir, x_dir, n_dir)

    print("=== ICP Step 5: Select best result ===")
    os.makedirs(output_dir, exist_ok=True)
    select_best_icp(logs_dir, x_dir, output_dir)

    print("ICP registration complete.")
