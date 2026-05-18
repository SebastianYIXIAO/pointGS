# PointGS

[Paper on arXiv](https://arxiv.org/pdf/2605.11520)

**3D Point Cloud Semantic Segmentation via Gaussian Splatting**

PointGS is a pipeline for indoor point cloud semantic segmentation that leverages 3D Gaussian Splatting (3DGS) and the Segment Anything Model (SAM). It projects point clouds to multi-view images, reconstructs 3D Gaussians, segments them using [SegAnyGAussians (SAGA)](https://github.com/Jumpat/SegAnyGAussians), and transfers semantic labels back to the original point cloud.

## Pipeline Overview

```
S3DIS Point Cloud (.pth)
    |
    v
[Step 1] Data Preparation
    |  pth -> txt -> segment & split -> render views -> COLMAP SfM
    v
[SAGA] Gaussian Reconstruction & Segmentation (external)
    |  3DGS training -> SAM masks -> contrastive features -> segmentation
    v
[Step 2] Post-processing
    |  denoise -> scale -> ICP registration -> label transfer
    v
Pseudo-labeled Point Cloud
```

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### External Tools

| Tool | Purpose | Installation |
|------|---------|-------------|
| [COLMAP](https://colmap.github.io/) | Structure-from-Motion reconstruction | [Install guide](https://colmap.github.io/install.html) |
| [CloudCompare](https://www.cloudcompare.org/) | ICP point cloud registration | [Download](https://www.cloudcompare.org/release/index.html) |
| [SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) | 3DGS training & segmentation | See their README |

## Quick Start

### 1. Configure Paths

Edit `configs/default.yaml` to set your data paths and tool locations:

```yaml
colmap_exe: "/path/to/colmap"
cloudcompare_exe: "/path/to/CloudCompare"

paths:
  s3dis_pth_dir: "data/s3dis/Area_5"
  s3dis_txt_dir: "data/s3dis/Area_5_txt"
  cut_output_dir: "data/s3dis/Area_5_cut"
  saga_output_dir: "output/saga_gaussians"
  reference_dir: "data/reference"
  final_output_dir: "output/results"
```

### 2. Step 1: Data Preparation

Convert S3DIS data, segment & split point clouds, render multi-view images, and run COLMAP:

```bash
# Run the full data preparation pipeline
python step1_data_preparation.py --config configs/default.yaml

# Or run individual sub-steps
python step1_data_preparation.py --config configs/default.yaml --step pth2txt
python step1_data_preparation.py --config configs/default.yaml --step split
python step1_data_preparation.py --config configs/default.yaml --step render
python step1_data_preparation.py --config configs/default.yaml --step colmap
```

**Sub-steps:**
1. **pth2txt** - Convert S3DIS `.pth` files to `.txt` format (x y z r g b label)
2. **split** - Segment large scenes and split each piece in half:
   - If all axes <= 7m: split at the shortest-axis median into 2 parts
   - If any axis > 7m: segment along the longest axis into N pieces (each ~4-7m), then split each piece at the shortest-axis median
3. **render** - Render 80 perspective views per scene using matplotlib
4. **colmap** - Run COLMAP feature extraction, matching, and sparse reconstruction

### 3. SAGA: Gaussian Reconstruction & Segmentation

Clone and set up the SAGA repository following their instructions:

```bash
git clone https://github.com/Jumpat/SegAnyGAussians.git
cd SegAnyGAussians
# Follow their installation guide
```

Use the provided batch script to process multiple scenes:

```bash
python saga_batch.py --saga_dir /path/to/SegAnyGAussians --data_dir /path/to/scenes
```

This runs: 3DGS training -> SAM mask extraction -> scale computation -> contrastive feature training -> SAGA segmentation.

### 4. Step 2: Post-processing

After SAGA produces labeled Gaussian point clouds, run the post-processing pipeline:

```bash
# Run the full post-processing pipeline
python step2_postprocessing.py --config configs/default.yaml \
    --saga_output_dir output/saga_gaussians \
    --reference_dir data/reference

# Or run individual sub-steps
python step2_postprocessing.py --config configs/default.yaml --step denoise
python step2_postprocessing.py --config configs/default.yaml --step scale
python step2_postprocessing.py --config configs/default.yaml --step icp
python step2_postprocessing.py --config configs/default.yaml --step filter
python step2_postprocessing.py --config configs/default.yaml --step transfer
```

**Sub-steps:**
1. **denoise** - Voxel grid denoising + remove label-0 + KDTree radius denoising
2. **scale** - FPS-based scale normalization to match reference point cloud diameter
3. **icp** - ICP registration using CloudCompare (initial + 24 rotations + best selection)
4. **filter** - Label consistency filtering with KNN and connected component analysis
5. **transfer** - 1-nearest-neighbor label transfer from Gaussians to original points

### Optional: Visualization

Convert labeled point clouds to colored PLY files:

```bash
python tools/visualize.py --input_dir output/results/label_mapped --output_dir output/visualization
```

## Project Structure

```
PointGS/
├── configs/default.yaml          # Configuration (paths, parameters)
├── step1_data_preparation.py     # Data conversion, segmentation & COLMAP
├── step2_postprocessing.py       # Denoising, registration, label transfer
├── saga_batch.py                 # SAGA batch processing helper
├── utils/
│   ├── io_utils.py               # Point cloud I/O
│   ├── data_conversion.py        # PTH conversion, rendering, COLMAP
│   ├── point_cloud_ops.py        # Scene segmentation, FPS, scaling
│   ├── denoising.py              # Voxel/KDTree denoising, label filtering
│   ├── registration.py           # ICP registration (CloudCompare)
│   ├── label_transfer.py         # 1NN label transfer
│   └── visualization.py          # Label-to-PLY conversion
└── tools/visualize.py            # Visualization CLI
```

## Configuration Parameters

All parameters are configurable via `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_axis_length` | 7.0 | Axis length threshold for scene segmentation |
| `min_segment_length` | 4.0 | Minimum segment length when splitting |
| `max_segment_length` | 7.0 | Maximum segment length when splitting |
| `voxel_size` | 0.15 | Voxel size for grid denoising |
| `voxel_cube_size` | 5 | Cube size for densest region search |
| `kdtree_radius` | 0.007 | Radius for KDTree denoising |
| `kdtree_min_neighbors` | 35 | Minimum neighbors within radius |
| `kdtree_iterations` | 2 | Number of denoising passes |
| `fps_sample_size` | 1024 | FPS sample count for diameter estimation |
| `k_consistency` | 20 | K for label consistency check |
| `k_connected` | 50 | K for connectivity analysis |
| `min_consistency_ratio` | 0.8 | Minimum label agreement ratio |
| `min_connected_threshold` | 30 | Minimum connected component size |

## Acknowledgements

This project builds upon [SegAnyGAussians (SAGA)](https://github.com/Jumpat/SegAnyGAussians) by Cen et al. for 3D Gaussian segmentation. Please cite their work if you use this pipeline:

```bibtex
@inproceedings{cen2025segment,
  title={Segment any 3d gaussians},
  author={Cen, Jiazhong and Fang, Jiemin and Yang, Chen and Xie, Lingxi and Zhang, Xiaopeng and Shen, Wei and Tian, Qi},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={39},
  number={2},
  pages={1971--1979},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
