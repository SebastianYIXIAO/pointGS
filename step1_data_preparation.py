"""
Step 1: Data Preparation Pipeline
Converts S3DIS .pth -> .txt -> Z-split -> render multi-view images -> COLMAP SfM

Usage:
    python step1_data_preparation.py --config configs/default.yaml
    python step1_data_preparation.py --config configs/default.yaml --step pth2txt
    python step1_data_preparation.py --config configs/default.yaml --step split --input_dir data/txt --output_dir data/cut
"""

import argparse
import yaml

from utils.data_conversion import convert_pth_to_txt, render_views, run_colmap_sfm
from utils.point_cloud_ops import split_by_z_median


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PointGS Step 1: Data Preparation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--step", default="all", choices=["all", "pth2txt", "split", "render", "colmap"],
                        help="Which sub-step to run")
    parser.add_argument("--input_dir", default=None, help="Override input directory")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    if args.step in ("all", "pth2txt"):
        print("=" * 60)
        print("Sub-step 1: Convert .pth to .txt")
        print("=" * 60)
        inp = args.input_dir or paths["s3dis_pth_dir"]
        out = args.output_dir or paths["s3dis_txt_dir"]
        convert_pth_to_txt(inp, out)

    if args.step in ("all", "split"):
        print("=" * 60)
        print("Sub-step 2: Split point clouds by Z-median")
        print("=" * 60)
        inp = paths["s3dis_txt_dir"] if args.step == "all" else (args.input_dir or paths["s3dis_txt_dir"])
        out = args.output_dir or paths["cut_output_dir"]
        split_by_z_median(inp, out)

    if args.step in ("all", "render"):
        print("=" * 60)
        print("Sub-step 3: Render multi-view images")
        print("=" * 60)
        inp = paths["cut_output_dir"] if args.step == "all" else (args.input_dir or paths["cut_output_dir"])
        render_views(inp, num_views=cfg["data_prep"]["num_render_views"])

    if args.step in ("all", "colmap"):
        print("=" * 60)
        print("Sub-step 4: Run COLMAP SfM")
        print("=" * 60)
        inp = paths["cut_output_dir"] if args.step == "all" else (args.input_dir or paths["cut_output_dir"])
        run_colmap_sfm(cfg["colmap_exe"], inp)

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
