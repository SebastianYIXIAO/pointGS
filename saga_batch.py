"""
SAGA Batch Processing
Convenience script for running the SegAnyGAussians pipeline on multiple scenes.

Prerequisites:
    Clone the SAGA repository: https://github.com/Jumpat/SegAnyGAussians
    Follow their installation instructions to set up the environment.

Usage:
    python saga_batch.py --saga_dir /path/to/SegAnyGAussians --data_dir /path/to/data/scannet
    python saga_batch.py --saga_dir /path/to/SegAnyGAussians --data_dir /path/to/data --sam_ckpt /path/to/sam.pth
"""

import os
import subprocess
import argparse


def run_saga_pipeline(saga_dir, data_dir, sam_checkpoint="sam-ckpt/sam_vit_h_4b8939.pth",
                      downsample=4, iterations=10000, num_sampled_rays=1000):
    """Run the full SAGA pipeline on all scenes in data_dir.

    Steps:
    1. Train 3DGS scene
    2. Extract SAM masks
    3. Compute scale
    4. Train contrastive features
    5. Run SAGA GUI for segmentation

    Args:
        saga_dir: Path to the cloned SegAnyGAussians repository.
        data_dir: Directory containing scene subfolders with images.
        sam_checkpoint: Path to SAM checkpoint file.
        downsample: Downsampling factor for SAM mask extraction.
        iterations: Training iterations for 3DGS and contrastive features.
        num_sampled_rays: Number of sampled rays for contrastive feature training.
    """
    scenes = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))])

    if not scenes:
        print(f"No scene directories found in {data_dir}")
        return

    print(f"Found {len(scenes)} scenes to process")

    for scene in scenes:
        scene_path = os.path.join(data_dir, scene)
        model_path = os.path.join(saga_dir, "output", scene)
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene}")
        print(f"{'='*60}")

        # Step 1: Train 3DGS
        print("\n--- Training 3DGS ---")
        subprocess.run([
            "python", os.path.join(saga_dir, "train_scene.py"),
            "-s", scene_path,
            "--iterations", str(iterations),
        ], cwd=saga_dir)

        # Step 2: Extract SAM masks
        print("\n--- Extracting SAM masks ---")
        subprocess.run([
            "python", os.path.join(saga_dir, "extract_segment_everything_masks.py"),
            "--image_root", scene_path,
            "--sam_checkpoint_path", sam_checkpoint,
            "--downsample", str(downsample),
        ], cwd=saga_dir)

        # Step 3: Compute scale
        print("\n--- Computing scale ---")
        subprocess.run([
            "python", os.path.join(saga_dir, "get_scale.py"),
            "--image_root", scene_path,
            "--model_path", model_path,
        ], cwd=saga_dir)

        # Step 4: Train contrastive features
        print("\n--- Training contrastive features ---")
        subprocess.run([
            "python", os.path.join(saga_dir, "train_contrastive_feature.py"),
            "-m", model_path,
            "--iterations", str(iterations),
            "--num_sampled_rays", str(num_sampled_rays),
        ], cwd=saga_dir)

        # Step 5: Run SAGA GUI
        print("\n--- Running SAGA segmentation ---")
        subprocess.run([
            "python", os.path.join(saga_dir, "saga_gui.py"),
            "-m", model_path,
        ], cwd=saga_dir)

    print(f"\n{'='*60}")
    print("SAGA batch processing complete.")


def main():
    parser = argparse.ArgumentParser(description="SAGA Batch Processing")
    parser.add_argument("--saga_dir", required=True, help="Path to cloned SegAnyGAussians repository")
    parser.add_argument("--data_dir", required=True, help="Directory with scene subfolders")
    parser.add_argument("--sam_ckpt", default="sam-ckpt/sam_vit_h_4b8939.pth", help="SAM checkpoint path")
    parser.add_argument("--downsample", type=int, default=4, help="Downsampling factor for SAM")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--num_sampled_rays", type=int, default=1000, help="Sampled rays for contrastive training")
    args = parser.parse_args()

    run_saga_pipeline(
        saga_dir=args.saga_dir,
        data_dir=args.data_dir,
        sam_checkpoint=args.sam_ckpt,
        downsample=args.downsample,
        iterations=args.iterations,
        num_sampled_rays=args.num_sampled_rays,
    )


if __name__ == "__main__":
    main()
