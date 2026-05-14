"""
Step 3: Evaluation
Compute mIoU, AP, mAcc, oAcc for predicted vs ground truth point clouds.

Usage:
    python step3_evaluation.py --config configs/default.yaml --pred_dir output/results/label_mapped --gt_dir data/reference
    python step3_evaluation.py --pred_dir output/pred --gt_dir output/gt --num_classes 13
"""

import argparse
import yaml

from utils.metrics import evaluate_batch, print_results


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PointGS Step 3: Evaluation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted labels")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground truth labels")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (default: from config)")
    parser.add_argument("--pred_label_col", type=int, default=None, help="Column for predicted labels")
    parser.add_argument("--gt_label_col", type=int, default=None, help="Column for GT labels")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]

    num_classes = args.num_classes or eval_cfg["num_classes"]
    pred_col = args.pred_label_col if args.pred_label_col is not None else eval_cfg["pred_label_col"]
    gt_col = args.gt_label_col if args.gt_label_col is not None else eval_cfg["gt_label_col"]

    print(f"Evaluating: {args.pred_dir} vs {args.gt_dir}")
    print(f"Classes: {num_classes}, Pred col: {pred_col}, GT col: {gt_col}\n")

    results = evaluate_batch(
        args.pred_dir, args.gt_dir,
        num_classes=num_classes,
        pred_label_col=pred_col,
        gt_label_col=gt_col,
    )

    print_results(results, num_classes=num_classes)


if __name__ == "__main__":
    main()
