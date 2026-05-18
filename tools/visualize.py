"""
Visualization Tool
Convert labeled point clouds to colored PLY files for viewing in MeshLab, CloudCompare, etc.

Usage:
    python tools/visualize.py --input_dir output/results/label_mapped --output_dir output/visualization
"""

import argparse
from utils.visualization import labels_to_colored_ply


def main():
    parser = argparse.ArgumentParser(description="PointGS Visualization: Labels to Colored PLY")
    parser.add_argument("--input_dir", required=True, help="Directory with labeled .txt point clouds")
    parser.add_argument("--output_dir", required=True, help="Output directory for .ply files")
    args = parser.parse_args()

    labels_to_colored_ply(args.input_dir, args.output_dir)
    print(f"\nVisualization files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
