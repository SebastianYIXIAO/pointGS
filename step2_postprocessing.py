"""
Step 2: Post-processing Pipeline (after SAGA produces labeled Gaussians)
Voxel denoise -> Remove label-0 -> KDTree denoise -> FPS scale ->
ICP registration -> Label consistency filter -> 1NN label transfer -> Greedy label mapping

Usage:
    python step2_postprocessing.py --config configs/default.yaml --saga_output_dir output/saga --reference_dir data/ref
    python step2_postprocessing.py --config configs/default.yaml --step denoise
    python step2_postprocessing.py --config configs/default.yaml --step icp
"""

import os
import argparse
import yaml

from utils.denoising import (
    voxel_grid_denoise,
    remove_zero_labels,
    kdtree_radius_denoise,
    label_consistency_filter,
)
from utils.point_cloud_ops import normalize_scale
from utils.registration import run_icp_registration
from utils.label_transfer import transfer_labels_1nn, greedy_iou_label_mapping


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PointGS Step 2: Post-processing")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--step", default="all",
                        choices=["all", "denoise", "scale", "icp", "filter", "transfer", "match"],
                        help="Which sub-step to run")
    parser.add_argument("--saga_output_dir", default=None, help="SAGA output directory (labeled Gaussians)")
    parser.add_argument("--reference_dir", default=None, help="Reference/GT point clouds directory")
    parser.add_argument("--output_dir", default=None, help="Override final output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    pp = cfg["postprocessing"]

    saga_dir = args.saga_output_dir or paths["saga_output_dir"]
    ref_dir = args.reference_dir or paths["reference_dir"]
    out_root = args.output_dir or paths["final_output_dir"]

    # Intermediate directories
    denoise1_dir = os.path.join(out_root, "denoise_step1")
    denoise2_dir = os.path.join(out_root, "denoise_step2")
    scaled_dir = os.path.join(out_root, "scaled")
    icp_dir = os.path.join(out_root, "icp_registered")
    filtered_dir = os.path.join(out_root, "label_filtered")
    transferred_dir = os.path.join(out_root, "label_transferred")
    mapped_dir = os.path.join(out_root, "label_mapped")

    if args.step in ("all", "denoise"):
        print("=" * 60)
        print("Sub-step 1: Voxel grid denoising")
        print("=" * 60)
        voxel_grid_denoise(saga_dir, denoise1_dir,
                           voxel_size=pp["voxel_size"],
                           cube_size=pp["voxel_cube_size"])

        print("\nSub-step 2: Remove label=0 points")
        remove_zero_labels(denoise1_dir)

        print("\nSub-step 3: KDTree radius denoising")
        kdtree_radius_denoise(denoise1_dir, denoise2_dir,
                              radius=pp["kdtree_radius"],
                              min_neighbors=pp["kdtree_min_neighbors"],
                              iterations=pp["kdtree_iterations"])

    if args.step in ("all", "scale"):
        print("=" * 60)
        print("Sub-step 4: FPS scale normalization")
        print("=" * 60)
        normalize_scale(denoise2_dir, ref_dir, denoise1_dir, scaled_dir,
                        fps_k=pp["fps_sample_size"])

    if args.step in ("all", "icp"):
        print("=" * 60)
        print("Sub-step 5: ICP registration")
        print("=" * 60)
        run_icp_registration(scaled_dir, ref_dir, icp_dir, cfg["cloudcompare_exe"])

    if args.step in ("all", "filter"):
        print("=" * 60)
        print("Sub-step 6: Label consistency filtering")
        print("=" * 60)
        label_consistency_filter(icp_dir, filtered_dir,
                                 k_consistency=pp["k_consistency"],
                                 k_connected=pp["k_connected"],
                                 min_consistency_ratio=pp["min_consistency_ratio"],
                                 min_connected_threshold=pp["min_connected_threshold"])

    if args.step in ("all", "transfer"):
        print("=" * 60)
        print("Sub-step 7: 1NN label transfer")
        print("=" * 60)
        transfer_labels_1nn(filtered_dir, ref_dir, transferred_dir)

    if args.step in ("all", "match"):
        print("=" * 60)
        print("Sub-step 8: Greedy IoU label mapping")
        print("=" * 60)
        eval_cfg = cfg["evaluation"]
        greedy_iou_label_mapping(transferred_dir, ref_dir, mapped_dir,
                                 pred_label_col=eval_cfg["pred_label_col"],
                                 gt_label_col=eval_cfg["gt_label_col"])

    print(f"\nStep 2 complete. Results in: {out_root}")


if __name__ == "__main__":
    main()
