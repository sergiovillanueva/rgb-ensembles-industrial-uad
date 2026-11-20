# RGB Ensemble Strategies for Unsupervised Industrial Anomaly Detection on the AutoVI Dataset

[![Paper](https://img.shields.io/badge/Paper-Electronics%202026-1F3864)](https://doi.org/10.3390/electronics15102077)
[![License: non-commercial, Eines](https://img.shields.io/badge/License-non--commercial%20%C2%B7%20Eines-red.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)

Code and results of the paper

> S. Villanueva López, E. Soria-Olivas, M. Sánchez-Montañés. *RGB Ensemble Strategies for Unsupervised Industrial
> Anomaly Detection on the AutoVI Dataset.* Electronics **15**(10), 2077, 2026.
> [doi:10.3390/electronics15102077](https://doi.org/10.3390/electronics15102077)

Real automotive inspection lines have cluttered backgrounds, changing light and objects left by operators (labels,
cables, tape), and defective samples are too scarce to train a supervised detector. This study measures how far
unsupervised anomaly detection gets on AutoVI, a public benchmark recorded on a car assembly line (Carvalho et al.,
*Computers in Industry*, 2024), and what helps.

## Findings

- **Unified baselines.** RGB and pseudo-depth baselines for seven UAD models under one calibration and evaluation
  policy: AUROC and AP, TPR at fixed TNR (the operating point a plant cares about), and pixel-level sPRO / AUsPRO at a
  5% false-positive budget.
- **Late fusion of calibrated RGB detectors** recovers pixel-level localization that no single model reaches, with no
  extra training and no change to the models. Every flagged improvement has Cohen's d > 2 and is significant across
  three seeds.
- **Negative result worth knowing.** Adding monocular pseudo-depth (Depth Anything V2) through the same fusion makes
  localization worse. The paper traces this to the relative scale of estimated depth combined with a parameter-free
  aggregator.

**Models in the code:** PatchCore, PaDiM, EfficientAD, FastFlow, Reverse Distillation, DiNomaly, U-Flow,
SuperSimpleNet, UniNet (the paper reports a subset of seven).

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

## Configuration

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
@article{villanueva2026rgb,
  title   = {{RGB} Ensemble Strategies for Unsupervised Industrial Anomaly Detection on the {AutoVI} Dataset},
  author  = {Villanueva L{\'o}pez, Sergio and Soria-Olivas, Emilio and S{\'a}nchez-Monta{\~n}{\'e}s, Manuel},
  journal = {Electronics},
  volume  = {15},
  number  = {10},
  pages   = {2077},
  year    = {2026},
  doi     = {10.3390/electronics15102077}
}
```

## License

Code written by Sergio Villanueva López at Eines Vision Systems, within an industrial PhD
with the Universitat de València. The code belongs to Eines: copyright (c) 2026 Eines
Systems S.L.U. It may be used for non-commercial research and teaching and to reproduce the
results of the paper. Commercial use requires prior written permission from Eines. See
[LICENSE](LICENSE).
The datasets keep their own licenses.

## Related repositories

This repository is one of five from my industrial PhD on anomaly detection for visual inspection (Universitat de València and Eines Vision Systems, 2026). All five work on frozen foundation-model features and do not train a network for each product.

| Repository | Paper | Question |
|---|---|---|
| **rgb-ensembles-industrial-uad** (this one) | Electronics, 2026 | Does combining several detectors improve defect localization on a real automotive benchmark? |
| [GaLAD_Gaussian_layered_AD](https://github.com/sergiovillanueva/GaLAD_Gaussian_layered_AD) | under review | How do you set up a new product from one to five good images and still find small defects? |
| [memory-pollution-ad](https://github.com/sergiovillanueva/memory-pollution-ad) | MAKE, 2026 | What goes wrong when many products share one memory bank, and how do you fix it? |
| [patch-purification-fewshot](https://github.com/sergiovillanueva/patch-purification-fewshot) | Electronics, 2026 | What happens if a defective part slips into the reference images at commissioning? |
| [pc-reg](https://github.com/sergiovillanueva/pc-reg) | Electronics, 2026 | How do you detect parts that look fine but are missing or in the wrong place? |

Sergio Villanueva López · [LinkedIn](https://www.linkedin.com/in/sergiovillanuevalopez)
