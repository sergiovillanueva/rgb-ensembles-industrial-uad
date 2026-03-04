# RGB Ensemble Strategies for Unsupervised Industrial Anomaly Detection on the AutoVI Dataset

This repository contains the source code and experimental results for reproducing the experiments presented in our paper.

**Authors:** Sergio Villanueva, Emilio Soria-Olivas, Manuel Sanchez-Montanes

## Overview

We evaluate nine unsupervised anomaly detection (UAD) models on the AutoVI dataset (automotive inspection) and propose late-fusion ensemble strategies to improve detection and localization performance. AutoVI is a publicly available dataset introduced by Carvalho et al. (2024) in *Computers in Industry*.

**Models evaluated:** PatchCore, PaDiM, EfficientAD, FastFlow, Reverse Distillation, DiNomaly, U-Flow, SuperSimpleNet, UniNet

**Ensemble strategies:** Mean, Median, Max, Trimmed Mean, Rank Average (applied at both score and pixel-map levels)

## Requirements

- Python 3.11 (tested with 3.11.x)
- NVIDIA GPU with CUDA 12.x (tested on RTX A5000, 24GB VRAM)

Install dependencies (recommended: [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

If you prefer plain `pip`, install the same dependencies listed in `pyproject.toml`.

## Dataset

Download the AutoVI dataset (Carvalho et al., *Computers in Industry*, 2024) and place it under `data/` following the MVTec-style structure:

```
data/
├── engine_wiring/
│   ├── train/good/
│   ├── test/good/
│   ├── test/<defect_type>/
│   └── ground_truth/<defect_type>/
├── pipe_clip/
├── pipe_staple/
├── tank_screw/
├── underbody_pipes/
└── underbody_screw/
```

For cross-modal experiments, generate pseudo-depth maps using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and place them in `data/<category>_estimate_depth/` with the same structure.

## Reproducing Experiments

### 1. Train individual models

```bash
uv run train_all.py --device 0
```

This trains all models on all 12 datasets (6 RGB + 6 pseudo-depth) with 3 random seeds (1, 2, 42). Results are saved to `results_0/` following this structure:

`results_0/all_models/<dataset>/<seed>/<model>/...`

The paper focuses on a subset of comparisons in some analyses, but the code supports all models listed above.

Training takes approximately 48-72 hours on a single RTX A5000.

### 2. Run within-modality ensembles (RGB)

```bash
uv run ensemble_simple.py
```

Combines model predictions using late-fusion strategies. Results are saved to `ensemble_results/`.

### 3. Run cross-modal ensembles (RGB + Depth)

```bash
uv run cross_modal_ensemble.py
```

Evaluates RGB + pseudo-depth fusion strategies. Results are saved to `cross_modal_results/`.

### 4. Analyze results

```bash
uv run ensemble_analysis.py
uv run data_analysis.py
```

Generates summary tables and statistical analysis.

### 5. Generate paper tables and figures

*(Note: If you have skipped training and ensemble generation due to dataset size, you can directly generate the final tables and figures using the pre-computed CSVs in `results/`)*

```bash
uv run tables_to_latex.py
uv run csv_to_tables.py
uv run graph_ensemble.py
```

## Pre-computed Results

The `results/` directory contains the pre-computed metrics from our experiments (no images are redistributed):

If you only want to regenerate the paper tables/figures from these CSVs (without training and without downloading AutoVI), a lightweight setup is enough:

```bash
pip install numpy pandas scipy matplotlib
python data_analysis.py
python csv_to_tables.py
python tables_to_latex.py
```

| File | Description |
|------|-------------|
| `training_results.csv` | Raw per-seed results for all individual models (semicolon-delimited, comma decimal) |
| `paper_table2_summary_with_std.csv` | Aggregated mean and std across seeds |
| `paper_table1_significant_improvements.csv` | Ensemble improvements over best individual models |
| `paper_table3_best_per_dataset.csv` | Best ensemble configuration per dataset |
| `complete_ensemble_analysis.csv` | Full seed-level ensemble analysis |

## Project Structure

```
├── train_all.py              # Main training script (individual UAD models)
├── custom_metrics.py          # sPRO, AUsPRO, TPR@TNR metric implementations
├── ensemble_simple.py         # Within-modality late-fusion ensembles
├── cross_modal_ensemble.py    # Cross-modal RGB+Depth fusion
├── ensemble_analysis.py       # Post-hoc analysis of ensemble results
├── data_analysis.py           # Statistical analysis and figure data
├── tables_to_latex.py         # LaTeX table generation
├── csv_to_tables.py           # Pareto frontier plots
├── graph_ensemble.py          # Ensemble comparison visualizations
├── export_overlays.py         # Anomaly heatmap overlay generation
├── mosaics.py                 # Multi-panel figure generation
├── pyproject.toml             # Python dependencies and project metadata
└── results/                   # Pre-computed experimental results
```

## Key Configuration

| Parameter | Value |
|-----------|-------|
| Framework | Anomalib 2.1 |
| Seeds | {1, 2, 42} |
| Max epochs | 100 |
| Early stopping | patience=12, min_delta=1e-5 |
| Precision | FP16 (mixed) for most models |
| PatchCore train limit | 128 images |

## Metrics

**Image-level:** AUROC, AP (AUPR), TPR@TNR (at 99%, 95%, 90%)

**Pixel-level:** sPRO@0.05, AUsPRO@0.05 (Saturated Per-Region Overlap at 5% FPR)

## Citation

If you use this code, please cite:

```bibtex
@article{villanueva2025rgb,
  title={RGB Ensemble Strategies for Unsupervised Industrial Anomaly Detection on the {AutoVI} Dataset},
  author={Villanueva, Sergio and Soria-Olivas, Emilio and Sanchez-Montanes, Manuel},
  journal={Computers in Industry},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
