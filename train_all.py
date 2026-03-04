"""
AutoVI Anomaly Detection Training Script

Trains UAD models on AutoVI dataset with:
- Resumable experiments (skips completed)
- Incremental CSV results
- Detailed logging
- Robust error handling

Usage: uv run train_all.py --device 0
"""
import argparse
import os
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='GPU device ID to use')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
GPU_DEVICE = args.device

import torch
torch.set_float32_matmul_precision('medium')

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_LIGHTNING_DISABLE_WARNINGS"] = "1"
os.environ["TIMM_FUSED_ATTN"] = "0"
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import csv
import gc
import json
import pickle
import random
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom, label as cc_label
from sklearn.metrics import roc_auc_score, average_precision_score

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import (
    Patchcore, Padim, ReverseDistillation, EfficientAd,
    Supersimplenet, Uflow, UniNet, Dinomaly, Fastflow
)
from lightning.pytorch.callbacks import EarlyStopping
from anomalib.callbacks import ModelCheckpoint

# Monkey-patch to fix Windows symlink issue in Anomalib's create_versioned_dir
# (Only applied on Windows; Linux/macOS keep the upstream behavior.)
if os.name == "nt":
    try:
        import anomalib.utils.path as anomalib_path
        import re

        _original_create_versioned_dir = anomalib_path.create_versioned_dir

        def _patched_create_versioned_dir(root_dir):
            """Create versioned directory without symlinks (Windows compatible)."""
            root_dir = Path(root_dir)
            root_dir.mkdir(parents=True, exist_ok=True)

            version_pattern = re.compile(r"^v(\d+)$")
            highest_version = -1
            if root_dir.exists():
                for version_dir in root_dir.iterdir():
                    if version_dir.is_dir():
                        match = version_pattern.match(version_dir.name)
                        if match:
                            highest_version = max(highest_version, int(match.group(1)))

            new_version_number = highest_version + 1
            new_version_dir = root_dir / f"v{new_version_number}"
            new_version_dir.mkdir(exist_ok=True)

            # Skip symlink creation - just return the versioned directory directly
            return new_version_dir

        anomalib_path.create_versioned_dir = _patched_create_versioned_dir
    except Exception:
        # Best-effort: training should still work even if this patch cannot be applied.
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = {
    "engine_wiring": ["test/blue_hoop", "test/cardboard", "test/fastening", "test/obstruction"],
    "pipe_clip": ["test/operator", "test/unclipped"],
    "pipe_staple": ["test/missing"],
    "tank_screw": ["test/missing"],
    "underbody_pipes": ["test/obstruction", "test/operator"],
    "underbody_screw": ["test/missing"],
    "engine_wiring_estimate_depth": ["test/blue_hoop", "test/cardboard", "test/fastening", "test/obstruction"],
    "pipe_clip_estimate_depth": ["test/operator", "test/unclipped"],
    "pipe_staple_estimate_depth": ["test/missing"],
    "tank_screw_estimate_depth": ["test/missing"],
    "underbody_pipes_estimate_depth": ["test/obstruction", "test/operator"],
    "underbody_screw_estimate_depth": ["test/missing"],
}

MODELS = [
    "patchcore",
    "padim",
    "fastflow",
    "reversedistillation",
    "efficientad",
    "dinomaly",
    "uflow",
    "supersimplenet",
    "uninet",
]

SEEDS = [42, 1, 2]
EPOCHS = 100
PATCHCORE_MAX_TRAIN_IMAGES = 128  # Paper: "memory bank is limited to at most 128 training images"

# Paths
OUTPUT_DIR = Path(f"results_{GPU_DEVICE}")
CSV_PATH = OUTPUT_DIR / "training_results.csv"
LOG_PATH = OUTPUT_DIR / "training_log.txt"
ERROR_LOG_PATH = OUTPUT_DIR / "training_errors.txt"

# =============================================================================
# UTILITIES
# =============================================================================

def cuda_cleanup(reason: str = ""):
    """Best-effort CUDA + Python GC cleanup between experiments."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def log(msg):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(dataset, model, seed, exception):
    """Log detailed error info to separate error log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"DATASET:   {dataset}\n")
        f.write(f"MODEL:     {model}\n")
        f.write(f"SEED:      {seed}\n")
        f.write(f"EXCEPTION: {type(exception).__name__}: {str(exception)}\n")
        f.write("TRACEBACK:\n")
        f.write(traceback.format_exc())
        f.write("\n")


def get_completed_experiments():
    """Return set of (dataset, model, seed) already completed."""
    completed = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (row["dataset"], row["model"], int(row["seed"]))
                completed.add(key)
    return completed


def save_result(result):
    """Append result to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists()

    def fmt(v, decimals=4):
        if v is None:
            return ""
        return f"{v:.{decimals}f}"

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if not file_exists:
            writer.writerow([
                "timestamp", "dataset", "model", "seed",
                "height", "width", "batch_size", "epochs", "epochs_trained",
                "n_train", "n_test",
                "img_auroc", "img_ap",
                "tpr_tnr99", "tpr_tnr95", "tpr_tnr90",
                "pix_auroc", "spro_005", "auspro_005",
                "train_time_s", "gpu_mem_gb"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["dataset"],
            result["model"],
            result["seed"],
            result["height"],
            result["width"],
            result["batch_size"],
            result["epochs"],
            result["epochs_trained"],
            result["n_train"],
            result["n_test"],
            fmt(result.get("img_auroc")),
            fmt(result.get("img_ap")),
            fmt(result.get("tpr_tnr99")),
            fmt(result.get("tpr_tnr95")),
            fmt(result.get("tpr_tnr90")),
            fmt(result.get("pix_auroc")),
            fmt(result.get("spro_005")),
            fmt(result.get("auspro_005")),
            f"{result['train_time']:.1f}",
            f"{result['gpu_mem_gb']:.2f}",
        ])


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(model, datamodule, device, model_directory):
    """Compute all metrics for a trained model."""
    model.eval()
    model = model.to(device)

    test_loader = datamodule.test_dataloader()
    train_loader = datamodule.train_dataloader()

    # Collect predictions
    test_scores = []
    test_labels = []
    test_anomaly_maps = []
    test_gt_masks = []
    pixel_labels = []
    train_normal_scores = []

    with torch.no_grad():
        # Test set
        for batch_idx, batch in enumerate(test_loader):
            try:
                batch.image = batch.image.to(device)
                if hasattr(batch, 'gt_label'):
                    batch.gt_label = batch.gt_label.to(device)
                if hasattr(batch, 'gt_mask') and batch.gt_mask is not None:
                    batch.gt_mask = batch.gt_mask.to(device)

                outputs = model.test_step(batch, batch_idx)

                if not hasattr(outputs, 'pred_score'):
                    continue

                labels = batch.gt_label.detach().cpu().numpy().reshape(-1)
                scores = outputs.pred_score.detach().cpu().numpy().reshape(-1)

                test_labels.append(labels)
                test_scores.append(scores)

                # Pixel-level data
                if hasattr(outputs, 'anomaly_map') and outputs.anomaly_map is not None:
                    if hasattr(batch, 'gt_mask') and batch.gt_mask is not None:
                        am_np = outputs.anomaly_map.detach().cpu().numpy()
                        gm_np = batch.gt_mask.detach().cpu().numpy()

                        if am_np.ndim == 4:
                            am_np = am_np[:, 0] if am_np.shape[1] == 1 else am_np.mean(axis=1)
                        if gm_np.ndim == 4:
                            gm_np = gm_np[:, 0] if gm_np.shape[1] == 1 else gm_np.max(axis=1)

                        b = min(am_np.shape[0], gm_np.shape[0])
                        for i in range(b):
                            amap_i = am_np[i].astype('float32')
                            gtm_i = (gm_np[i] > 0).astype('uint8')
                            if amap_i.shape != gtm_i.shape:
                                th, tw = gtm_i.shape[-2], gtm_i.shape[-1]
                                h, w = amap_i.shape[-2], amap_i.shape[-1]
                                zy, zx = th / h, tw / w
                                amap_i = zoom(amap_i, (zy, zx), order=1).astype('float32')
                            test_anomaly_maps.append(amap_i)
                            test_gt_masks.append(gtm_i)
                        pixel_labels.extend(labels.astype(int)[:b])

            except Exception as e:
                log(f"Error in test batch {batch_idx}: {e}")

        # Train set (for normalization)
        for batch_idx, batch in enumerate(train_loader):
            try:
                batch.image = batch.image.to(device)
                outputs = model.test_step(batch, batch_idx)
                if hasattr(outputs, 'pred_score'):
                    scores = outputs.pred_score.detach().cpu().numpy().reshape(-1)
                    train_normal_scores.append(scores)
            except Exception:
                pass

    # Concatenate
    if len(test_scores) == 0 or len(test_labels) == 0:
        return {}

    test_scores = np.concatenate(test_scores)
    test_labels = np.concatenate(test_labels)
    train_normal_scores = np.concatenate(train_normal_scores) if train_normal_scores else test_scores[test_labels == 0]

    # Image-level metrics
    metrics = {}
    try:
        metrics["img_auroc"] = roc_auc_score(test_labels, test_scores)
        metrics["img_ap"] = average_precision_score(test_labels, test_scores)
    except Exception as e:
        log(f"Error computing image metrics: {e}")

    # TPR@TNR using z-score normalization
    try:
        mu = np.mean(train_normal_scores)
        sigma = np.std(train_normal_scores)
        if sigma == 0:
            sigma = 1.0
        z_train = (train_normal_scores - mu) / sigma
        z_test = (test_scores - mu) / sigma
        z_test_anom = z_test[test_labels == 1]

        for tnr, key in [(0.99, "tpr_tnr99"), (0.95, "tpr_tnr95"), (0.90, "tpr_tnr90")]:
            thr = np.quantile(z_train, tnr)
            if len(z_test_anom) > 0:
                metrics[key] = float((z_test_anom > thr).mean())
    except Exception as e:
        log(f"Error computing TPR@TNR: {e}")

    # Pixel-level metrics
    if len(test_anomaly_maps) > 0 and len(test_gt_masks) > 0:
        try:
            # Pixel AUROC
            all_pix_scores = np.concatenate([m.ravel() for m in test_anomaly_maps])
            all_pix_labels = np.concatenate([m.ravel() for m in test_gt_masks])
            if len(np.unique(all_pix_labels)) > 1:
                metrics["pix_auroc"] = roc_auc_score(all_pix_labels, all_pix_scores)
        except Exception as e:
            log(f"Error computing pixel AUROC: {e}")

        # sPRO / AUsPRO
        try:
            metrics.update(compute_spro_auspro(test_anomaly_maps, test_gt_masks, pixel_labels))
        except Exception as e:
            log(f"Error computing sPRO/AUsPRO: {e}")

    # Save raw data for ensemble
    try:
        model_directory.mkdir(parents=True, exist_ok=True)

        test_data = {
            'scores': test_scores.tolist(),
            'labels': test_labels.tolist(),
            'train_normal_scores': train_normal_scores.tolist(),
        }
        with open(model_directory / 'test_results.json', 'w') as f:
            json.dump(test_data, f, indent=2)

        if len(test_anomaly_maps) > 0:
            with open(model_directory / 'anomaly_maps.pkl', 'wb') as f:
                pickle.dump(test_anomaly_maps, f)
            with open(model_directory / 'gt_masks.pkl', 'wb') as f:
                pickle.dump(test_gt_masks, f)
    except Exception as e:
        log(f"Error saving raw data: {e}")

    return metrics


def compute_spro_auspro(anomaly_maps, gt_masks, image_labels, alpha=0.05):
    """Compute sPRO@0.05 and AUsPRO@0.05."""
    image_labels = np.asarray(image_labels, dtype=int)
    normals_idx = np.where(image_labels == 0)[0]
    abnormals_idx = np.where(image_labels == 1)[0]

    if len(normals_idx) == 0 or len(abnormals_idx) == 0:
        return {}

    # Threshold range over ALL test images
    t_min = min(anomaly_maps[i].min() for i in range(len(anomaly_maps)))
    t_max = max(anomaly_maps[i].max() for i in range(len(anomaly_maps)))

    if t_max <= t_min or not (np.isfinite(t_min) and np.isfinite(t_max)):
        return {}

    thresholds = np.linspace(t_max, t_min, num=300, dtype='float32')

    # FPR on normal images
    total_normal_pixels = 0
    normal_fp_counts = np.zeros(len(thresholds), dtype='float64')
    for idx in normals_idx:
        amap = anomaly_maps[idx]
        total_normal_pixels += amap.size
        normal_fp_counts += (amap[None, :, :] >= thresholds[:, None, None]).sum(axis=(1, 2))
    fprs = normal_fp_counts / max(1, total_normal_pixels)

    # Pre-compute connected components
    abnormal_components = []
    for idx in abnormals_idx:
        gtm = gt_masks[idx]
        comp_map, num = cc_label(gtm.astype('uint8'))
        components = []
        for comp_id in range(1, num + 1):
            comp_mask = (comp_map == comp_id)
            comp_size = comp_mask.sum()
            if comp_size > 0:
                components.append((comp_mask, comp_size))
        abnormal_components.append((idx, components))

    # PRO for each threshold
    pros = np.zeros(len(thresholds), dtype='float64')
    for t_i, thr in enumerate(thresholds):
        pro_vals = []
        for idx, components in abnormal_components:
            amap = anomaly_maps[idx]
            pred_mask = (amap >= thr)
            for comp_mask, comp_size in components:
                tp = (pred_mask & comp_mask).sum()
                pro_vals.append(tp / comp_size)
        if pro_vals:
            pros[t_i] = np.mean(pro_vals)

    # Limit to FPR <= alpha
    valid = fprs <= alpha
    if not np.any(valid):
        return {}

    order = np.argsort(fprs[valid])
    xf = fprs[valid][order]
    yf = pros[valid][order]

    if xf[0] > 0:
        xf = np.concatenate([[0], xf])
        yf = np.concatenate([[0], yf])

    area = np.trapezoid(yf, xf) / alpha
    spro_edge = float(yf[-1])

    return {
        'spro_005': spro_edge,
        'auspro_005': float(area)
    }


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def get_image_size(dataset):
    """Get image size for dataset."""
    if dataset in ["engine_wiring", "pipe_clip", "pipe_staple",
                   "engine_wiring_estimate_depth", "pipe_clip_estimate_depth", "pipe_staple_estimate_depth"]:
        return (400, 400)
    elif dataset in ["underbody_pipes", "underbody_screw", "tank_screw",
                     "underbody_pipes_estimate_depth", "underbody_screw_estimate_depth", "tank_screw_estimate_depth"]:
        return (672, 896)
    else:
        return (256, 256)


def configure_model(model_name, size, epochs, visualizer=False):
    """Configure model with appropriate settings."""

    if model_name == "patchcore":
        max_dim = 640
        if size[0] > max_dim or size[1] > max_dim:
            scale = min(max_dim / size[0], max_dim / size[1])
            size = (int(size[0] * scale) // 16 * 16, int(size[1] * scale) // 16 * 16)

        pre_processor = Patchcore.configure_pre_processor(image_size=size)
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            pre_processor=pre_processor,
        )
        return model, size, 1, 1, None  # batch_size=1, epochs=1

    elif model_name == "padim":
        max_dim = 512
        if size[0] > max_dim or size[1] > max_dim:
            scale = min(max_dim / size[0], max_dim / size[1])
            size = (int(size[0] * scale) // 16 * 16, int(size[1] * scale) // 16 * 16)

        pre_processor = Padim.configure_pre_processor(image_size=size)
        model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            pre_processor=pre_processor,
        )
        batch_size = 32 if size[0] < 500 else 16
        return model, size, batch_size, 1, None

    elif model_name == "reversedistillation":
        alignment = 32
        size = ((size[0] // alignment) * alignment, (size[1] // alignment) * alignment)

        pre_processor = ReverseDistillation.configure_pre_processor(image_size=size)
        model = ReverseDistillation(pre_processor=pre_processor)
        batch_size = 32 if size[0] < 500 else 8
        return model, size, batch_size, epochs, "train_loss"

    elif model_name == "efficientad":
        alignment = 32
        size = ((size[0] // alignment) * alignment, (size[1] // alignment) * alignment)

        pre_processor = EfficientAd.configure_pre_processor(image_size=size)
        model = EfficientAd(pre_processor=pre_processor)
        return model, size, 1, epochs, "train_loss"

    elif model_name == "supersimplenet":
        pre_processor = Supersimplenet.configure_pre_processor(image_size=size)
        model = Supersimplenet(pre_processor=pre_processor)
        batch_size = 16 if size[0] < 500 else 4
        return model, size, batch_size, epochs, "train_loss"

    elif model_name == "uflow":
        pre_processor = Uflow.configure_pre_processor(image_size=size)
        model = Uflow(pre_processor=pre_processor)
        batch_size = 32 if size[0] < 500 else 16
        return model, size, batch_size, epochs, "loss"

    elif model_name == "uninet":
        alignment = 32
        max_dim = 736
        if max(size) > max_dim:
            scale = max_dim / max(size)
            size = tuple(int(dim * scale // alignment) * alignment for dim in size)
        else:
            size = ((size[0] // alignment) * alignment, (size[1] // alignment) * alignment)

        pre_processor = UniNet.configure_pre_processor(image_size=size)
        model = UniNet(pre_processor=pre_processor)
        if max(size) <= 384:
            batch_size = 6
        elif max(size) <= 576:
            batch_size = 2
        else:
            batch_size = 1
        return model, size, batch_size, epochs, "train_loss"

    elif model_name == "dinomaly":
        max_dim = 700
        min_dim = 392
        square_size = min(size[0], size[1])
        square_size = max(min_dim, min(max_dim, square_size))
        size = (square_size // 14 * 14, square_size // 14 * 14)

        pre_processor = Dinomaly.configure_pre_processor(image_size=size)
        model = Dinomaly(pre_processor=pre_processor)
        batch_size = 32 if max(size) < 500 else 3
        return model, size, batch_size, epochs, "train_loss"

    elif model_name == "fastflow":
        pre_processor = Fastflow.configure_pre_processor(image_size=size)
        model = Fastflow(
            backbone="resnet18",
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            pre_processor=pre_processor,
        )
        batch_size = 32 if size[0] < 500 else 8
        return model, size, batch_size, epochs, "train_loss"

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# TRAINING
# =============================================================================

def train_single(dataset, abnormal_dirs, model_name, seed, epochs):
    """Train and evaluate a single configuration."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_cleanup("start train_single")

    # Get base image size
    base_size = get_image_size(dataset)

    # Configure model (may adjust size)
    model, size, batch_size, max_epochs, monitor = configure_model(model_name, base_size, epochs)

    # Create datamodule
    dm = Folder(
        root=f"data/{dataset}",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        mask_dir="ground_truth",
        val_split_mode="from_train",
        val_split_ratio=0.1,
        train_batch_size=batch_size,
        eval_batch_size=1 if model_name == "patchcore" else batch_size * 2,
        seed=seed,
        num_workers=0,
        name=f"{model_name}_{dataset}"
    )
    dm.setup()

    # Apply train limit for PatchCore
    if model_name == "patchcore":
        original_size = len(dm.train_data.samples)
        if original_size > PATCHCORE_MAX_TRAIN_IMAGES:
            random.seed(seed)
            indices = random.sample(range(original_size), PATCHCORE_MAX_TRAIN_IMAGES)
            indices.sort()
            dm.train_data.samples = dm.train_data.samples.iloc[indices].reset_index(drop=True)
            log(f"  PatchCore: limited train from {original_size} to {len(dm.train_data.samples)}")

    n_train = len(dm.train_data.samples)
    n_test = len(dm.test_data.samples)

    # Model directory (shared convention across scripts)
    # Expected by ensemble scripts: results_0/all_models/<dataset>/<seed>/<model>/
    model_dir = OUTPUT_DIR / "all_models" / dataset / str(seed) / model_name

    # Callbacks
    callbacks = []
    if monitor is not None:
        callbacks.append(ModelCheckpoint(
            monitor=monitor, mode="min", save_top_k=1,
            save_last=False, dirpath=str(model_dir), filename="model"
        ))
        callbacks.append(EarlyStopping(
            monitor=monitor, mode="min", patience=12, min_delta=1e-5  # Paper values
        ))

    # Engine (symlink issue fixed via monkey-patch at top of file)
    engine = Engine(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=None if model_name in ["patchcore", "padim"] else "16-mixed",
        default_root_dir=str(model_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    # Train
    start_time = datetime.now()
    engine.fit(datamodule=dm, model=model)
    train_time = (datetime.now() - start_time).total_seconds()

    # Get actual epochs trained
    try:
        epochs_trained = engine.trainer.current_epoch + 1
    except Exception:
        epochs_trained = max_epochs

    # Load best checkpoint if available
    if callbacks and hasattr(callbacks[0], 'best_model_path'):
        best_path = callbacks[0].best_model_path
        if best_path and os.path.exists(best_path):
            try:
                state = torch.load(best_path, map_location='cpu', weights_only=False)
                state_dict = state.get('state_dict', state)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                log(f"  Could not load best checkpoint: {e}")

    cuda_cleanup("after fit")

    # Compute metrics
    metrics = compute_metrics(model, dm, device, model_dir)

    gpu_mem = get_gpu_memory_gb()

    # Cleanup
    del model
    del engine
    del dm
    cuda_cleanup("end train_single")

    # Clean up checkpoints to save space
    if model_dir.exists():
        for ckpt in model_dir.glob("*.ckpt"):
            ckpt.unlink()

    return {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "height": size[0],
        "width": size[1],
        "batch_size": batch_size,
        "epochs": max_epochs,
        "epochs_trained": epochs_trained,
        "n_train": n_train,
        "n_test": n_test,
        "train_time": train_time,
        "gpu_mem_gb": gpu_mem,
        **metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("="*60)
    log("AutoVI ANOMALY DETECTION TRAINING")
    log("="*60)
    log(f"GPU Device: {GPU_DEVICE}")
    log(f"Datasets: {len(DATASETS)}")
    log(f"Models: {MODELS}")
    log(f"Seeds: {SEEDS}")
    log(f"Epochs: {EPOCHS}")

    if torch.cuda.is_available():
        log(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Total experiments
    total = len(DATASETS) * len(MODELS) * len(SEEDS)
    log(f"Total experiments: {total}")

    # Get completed
    completed = get_completed_experiments()
    log(f"Already completed: {len(completed)}")

    # Run experiments
    current = 0
    errors = []

    for seed in SEEDS:
        for dataset, abnormal_dirs in DATASETS.items():
            for model_name in MODELS:
                current += 1
                key = (dataset, model_name, seed)

                if key in completed:
                    log(f"[{current}/{total}] SKIP {dataset} | {model_name} | seed={seed}")
                    continue

                log(f"[{current}/{total}] {dataset} | {model_name} | seed={seed}")
                cuda_cleanup("before experiment")

                try:
                    result = train_single(
                        dataset=dataset,
                        abnormal_dirs=abnormal_dirs,
                        model_name=model_name,
                        seed=seed,
                        epochs=EPOCHS,
                    )
                    save_result(result)

                    auroc = result.get('img_auroc', 0)
                    auspro = result.get('auspro_005', 0)
                    log(f"[{current}/{total}] DONE AUROC={auroc:.4f} AUsPRO={auspro:.4f} time={result['train_time']:.1f}s")

                except Exception as e:
                    error_msg = f"{dataset} | {model_name} | seed={seed}: {str(e)}"
                    errors.append(error_msg)
                    log(f"[{current}/{total}] ERROR {error_msg}")
                    log_error(dataset, model_name, seed, e)
                    cuda_cleanup("after error")

    # Summary
    log("="*60)
    log("TRAINING COMPLETED")
    log(f"Total: {total}, Errors: {len(errors)}")
    if errors:
        log("Failed experiments:")
        for e in errors:
            log(f"  - {e}")
    log(f"Results: {CSV_PATH}")
    log("="*60)


if __name__ == "__main__":
    main()
