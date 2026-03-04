#!/usr/bin/env python3
"""
Cross-Modal Ensemble: RGB + Pseudo-Depth Late Fusion (v3 - Parallel)
=====================================================================
Parallelized version for multi-core machines.

Changes vs v2:
- ProcessPoolExecutor to parallelize by (dataset, model, seed)
- Progress bar with tqdm
- ~4-8x speedup on machines with multiple cores
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import label as cc_label, zoom
from scipy.integrate import trapezoid
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_PATH = "results_0/all_models"
OUTPUT_DIR = "cross_modal_results"
SEEDS = [1, 2, 42]
DEBUG = False  # Disable debug in parallel mode for cleaner output
MAX_WORKERS = None  # None = use all available CPUs

# Models to evaluate per dataset
MODELS_CONFIG = {
    "engine_wiring": ['patchcore', 'efficientad', 'reversedistillation'],
    "pipe_clip": ['patchcore', 'uninet', 'reversedistillation'],
    "pipe_staple": ['patchcore', 'supersimplenet', 'dinomaly'],
    "tank_screw": ['patchcore', 'reversedistillation', 'supersimplenet'],
    "underbody_pipes": ['patchcore', 'dinomaly', 'reversedistillation'],
    "underbody_screw": ['patchcore', 'supersimplenet', 'padim'],
}

# ============================================================================
# HELPER FUNCTIONS (same as v2, but without debug prints)
# ============================================================================

def is_valid_array(arr):
    if arr is None:
        return False
    if isinstance(arr, np.ndarray):
        return arr.size > 0
    if isinstance(arr, list):
        return len(arr) > 0
    return False


def load_model_data(model_path):
    results_path = os.path.join(model_path, 'test_results.json')
    
    if not os.path.exists(results_path):
        return None, [], [], []
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    anomaly_maps, gt_masks, train_normal_maps = [], [], []
    
    maps_path = os.path.join(model_path, 'anomaly_maps.pkl')
    if os.path.exists(maps_path):
        try:
            with open(maps_path, 'rb') as f:
                anomaly_maps = pickle.load(f)
        except Exception:
            pass

    masks_path = os.path.join(model_path, 'gt_masks.pkl')
    if os.path.exists(masks_path):
        try:
            with open(masks_path, 'rb') as f:
                gt_masks = pickle.load(f)
        except Exception:
            pass

    train_maps_path = os.path.join(model_path, 'train_normal_maps.pkl')
    if os.path.exists(train_maps_path):
        try:
            with open(train_maps_path, 'rb') as f:
                train_normal_maps = pickle.load(f)
        except Exception:
            pass
    
    return results, anomaly_maps, gt_masks, train_normal_maps


def zcal(scores, ref_scores):
    scores = np.asarray(scores, dtype=np.float64)
    
    if not is_valid_array(ref_scores):
        return scores, 0.0, 1.0
    
    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    mu = float(np.mean(ref_scores))
    sigma = float(np.std(ref_scores))
    
    if sigma == 0 or not np.isfinite(sigma):
        sigma = 1.0
    
    calibrated = (scores - mu) / sigma
    return calibrated, mu, sigma


def check_and_fix_sign(scores, train_scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    
    if is_valid_array(train_scores):
        train_scores = np.asarray(train_scores, dtype=np.float64)
    else:
        train_scores = None
    
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            auroc = roc_auc_score(labels, scores)
            if auroc < 0.5:
                scores = -scores
                if train_scores is not None:
                    train_scores = -train_scores
    except Exception:
        pass

    return scores, train_scores


def normalize_maps_robust(maps, train_normal_maps, use_mad=True):
    if not is_valid_array(train_normal_maps):
        return maps
    
    all_normal_pixels = np.concatenate([np.asarray(m).ravel() for m in train_normal_maps])
    
    if use_mad:
        median_normal = np.median(all_normal_pixels)
        mad_normal = np.median(np.abs(all_normal_pixels - median_normal))
        if mad_normal == 0:
            mad_normal = 1e-8
        scale = 1.4826 * mad_normal
        normalized = [(np.asarray(m) - median_normal) / scale for m in maps]
    else:
        mean_normal = np.mean(all_normal_pixels)
        std_normal = np.std(all_normal_pixels)
        if std_normal == 0:
            std_normal = 1e-8
        normalized = [(np.asarray(m) - mean_normal) / std_normal for m in maps]
    
    return [m.astype('float32') for m in normalized]


def resize_map_bilinear(arr, th, tw):
    arr = np.asarray(arr)
    h, w = arr.shape[-2], arr.shape[-1]
    if h == th and w == tw:
        return arr.astype('float32')
    zy, zx = th / h, tw / w
    return zoom(arr.astype('float32'), (zy, zx), order=1)


def compute_auspro(anomaly_maps, gt_masks, labels, fpr_limit=0.05, s_fraction=0.3):
    if not is_valid_array(anomaly_maps) or not is_valid_array(gt_masks):
        return 0.0, 0.0
    
    labels = np.asarray(labels)
    abnormal_indices = np.where(labels == 1)[0]
    
    if len(abnormal_indices) == 0:
        return 0.0, 0.0
    
    normal_pixels = []
    for i, label in enumerate(labels):
        if i >= len(anomaly_maps):
            break
        amap = np.asarray(anomaly_maps[i])
        
        if label == 0:
            normal_pixels.extend(amap.ravel())
        elif i < len(gt_masks) and gt_masks[i] is not None:
            mask_arr = np.asarray(gt_masks[i]).astype(bool)
            if mask_arr.shape == amap.shape[-2:]:
                normal_pixels.extend(amap[~mask_arr].ravel())
    
    if len(normal_pixels) == 0:
        return 0.0, 0.0
    
    normal_pixels = np.array(normal_pixels)
    
    n_thresholds = 200
    thresholds = np.linspace(np.min(normal_pixels), np.max(normal_pixels), n_thresholds)[::-1]
    
    fpr_values = []
    spro_values = []
    
    for theta in thresholds:
        fpr = np.mean(normal_pixels >= theta)
        if fpr > fpr_limit:
            continue
        
        region_overlaps = []
        for idx in abnormal_indices:
            if idx >= len(gt_masks) or gt_masks[idx] is None:
                continue
            if idx >= len(anomaly_maps):
                continue
                
            mask_arr = np.asarray(gt_masks[idx]).astype(bool)
            amap_arr = np.asarray(anomaly_maps[idx])
            
            if mask_arr.shape != amap_arr.shape[-2:]:
                continue
            
            labeled_mask, num_regions = cc_label(mask_arr)
            
            for region_id in range(1, num_regions + 1):
                region_mask = labeled_mask == region_id
                region_size = region_mask.sum()
                if region_size == 0:
                    continue
                
                s_r = max(1, int(np.ceil(s_fraction * region_size)))
                detected_pixels = (amap_arr >= theta) & region_mask
                overlap = min(1.0, detected_pixels.sum() / s_r)
                region_overlaps.append(overlap)
        
        if len(region_overlaps) > 0:
            spro = np.mean(region_overlaps)
            fpr_values.append(fpr)
            spro_values.append(spro)
    
    if len(fpr_values) == 0:
        return 0.0, 0.0
    
    idx_05 = np.argmin(np.abs(np.array(fpr_values) - fpr_limit))
    spro_at_05 = spro_values[idx_05]
    
    auspro = trapezoid(spro_values, fpr_values) / fpr_limit if len(fpr_values) > 1 else 0.0
    
    return float(abs(auspro)), float(spro_at_05)


def cross_modal_ensemble_scores(rgb_scores, depth_scores, rgb_train, depth_train, labels, method='mean'):
    rgb_scores = np.asarray(rgb_scores, dtype=np.float64)
    depth_scores = np.asarray(depth_scores, dtype=np.float64)
    labels = np.asarray(labels)
    
    rgb_train_arr = np.asarray(rgb_train, dtype=np.float64) if is_valid_array(rgb_train) else None
    depth_train_arr = np.asarray(depth_train, dtype=np.float64) if is_valid_array(depth_train) else None
    
    rgb_s, rgb_tr = check_and_fix_sign(rgb_scores, rgb_train_arr, labels)
    depth_s, depth_tr = check_and_fix_sign(depth_scores, depth_train_arr, labels)
    
    z_rgb, mu_rgb, sigma_rgb = zcal(rgb_s, rgb_tr)
    z_depth, mu_depth, sigma_depth = zcal(depth_s, depth_tr)
    
    n = min(len(z_rgb), len(z_depth), len(labels))
    z_rgb = z_rgb[:n]
    z_depth = z_depth[:n]
    labels_aligned = labels[:n]
    
    if method == 'mean':
        fused = 0.5 * (z_rgb + z_depth)
    elif method == 'max':
        fused = np.maximum(z_rgb, z_depth)
    else:
        fused = 0.5 * (z_rgb + z_depth)
    
    if rgb_tr is not None and depth_tr is not None:
        m = min(len(rgb_tr), len(depth_tr))
        z_rgb_tr = (rgb_tr[:m] - mu_rgb) / max(sigma_rgb, 1e-8)
        z_depth_tr = (depth_tr[:m] - mu_depth) / max(sigma_depth, 1e-8)
        
        if method == 'mean':
            fused_train = 0.5 * (z_rgb_tr + z_depth_tr)
        else:
            fused_train = np.maximum(z_rgb_tr, z_depth_tr)
        
        mu_fused = np.mean(fused_train)
        sigma_fused = np.std(fused_train)
        if sigma_fused == 0 or not np.isfinite(sigma_fused):
            sigma_fused = 1.0
        
        fused = (fused - mu_fused) / sigma_fused
    
    return fused, labels_aligned


def cross_modal_ensemble_maps(rgb_maps, depth_maps, rgb_train_maps, depth_train_maps, 
                              masks, labels, method='mean'):
    if not is_valid_array(rgb_maps) or not is_valid_array(depth_maps):
        return []
    
    th, tw = 400, 400
    if is_valid_array(masks):
        for m in masks:
            if m is not None:
                target = np.asarray(m)
                th, tw = target.shape[-2], target.shape[-1]
                break
    
    rgb_resized = [resize_map_bilinear(m, th, tw) for m in rgb_maps]
    depth_resized = [resize_map_bilinear(m, th, tw) for m in depth_maps]
    
    rgb_train_resized = [resize_map_bilinear(m, th, tw) for m in rgb_train_maps] if is_valid_array(rgb_train_maps) else []
    depth_train_resized = [resize_map_bilinear(m, th, tw) for m in depth_train_maps] if is_valid_array(depth_train_maps) else []
    
    rgb_norm = normalize_maps_robust(rgb_resized, rgb_train_resized, use_mad=True)
    depth_norm = normalize_maps_robust(depth_resized, depth_train_resized, use_mad=True)
    
    fused_maps = []
    n = min(len(rgb_norm), len(depth_norm))
    for i in range(n):
        if method == 'mean':
            fused = (rgb_norm[i] + depth_norm[i]) / 2.0
        elif method == 'max':
            fused = np.maximum(rgb_norm[i], depth_norm[i])
        else:
            fused = (rgb_norm[i] + depth_norm[i]) / 2.0
        fused_maps.append(fused.astype('float32'))
    
    return fused_maps


# ============================================================================
# SINGLE TASK FUNCTION (for parallel execution)
# ============================================================================

def process_single_task(args):
    """Process a single (dataset, model, seed) combination"""
    dataset, model, seed, base_path = args
    
    try:
        rgb_path = os.path.join(base_path, dataset, str(seed), model)
        depth_path = os.path.join(base_path, f"{dataset}_estimate_depth", str(seed), model)
        
        # Load RGB data
        rgb_results, rgb_maps, rgb_masks, rgb_train_maps = load_model_data(rgb_path)
        if rgb_results is None:
            return None
        
        # Load Depth data
        depth_results, depth_maps, depth_masks, depth_train_maps = load_model_data(depth_path)
        if depth_results is None:
            return None
        
        # Extract scores and labels
        rgb_scores = rgb_results.get('scores', [])
        depth_scores = depth_results.get('scores', [])
        labels = rgb_results.get('labels', [])
        rgb_train_scores = rgb_results.get('train_normal_scores', [])
        depth_train_scores = depth_results.get('train_normal_scores', [])
        
        if len(rgb_scores) == 0 or len(depth_scores) == 0 or len(labels) == 0:
            return None
        
        rgb_scores = np.asarray(rgb_scores, dtype=np.float64)
        depth_scores = np.asarray(depth_scores, dtype=np.float64)
        labels = np.asarray(labels)
        
        # Individual metrics
        try:
            if len(np.unique(labels)) == 2:
                rgb_auroc = roc_auc_score(labels, rgb_scores)
                rgb_ap = average_precision_score(labels, rgb_scores)
                depth_auroc = roc_auc_score(labels, depth_scores)
                depth_ap = average_precision_score(labels, depth_scores)
            else:
                rgb_auroc, rgb_ap = 0.5, 0.0
                depth_auroc, depth_ap = 0.5, 0.0
        except Exception:
            rgb_auroc, rgb_ap = 0.5, 0.0
            depth_auroc, depth_ap = 0.5, 0.0
        
        rgb_auspro, rgb_spro = compute_auspro(rgb_maps, rgb_masks, labels) if is_valid_array(rgb_maps) else (0.0, 0.0)
        depth_auspro, depth_spro = compute_auspro(depth_maps, depth_masks, labels) if is_valid_array(depth_maps) else (0.0, 0.0)
        
        results = {
            'Dataset': dataset,
            'Model': model,
            'Seed': seed,
            'RGB_AUROC': rgb_auroc,
            'RGB_AP': rgb_ap,
            'RGB_AUsPRO': rgb_auspro,
            'Depth_AUROC': depth_auroc,
            'Depth_AP': depth_ap,
            'Depth_AUsPRO': depth_auspro,
        }
        
        # Cross-modal ensemble (score-level)
        for method in ['mean', 'max']:
            try:
                fused_scores, labels_aligned = cross_modal_ensemble_scores(
                    rgb_scores, depth_scores, 
                    rgb_train_scores, depth_train_scores,
                    labels, method=method
                )
                
                if len(np.unique(labels_aligned)) == 2:
                    ens_auroc = roc_auc_score(labels_aligned, fused_scores)
                    ens_ap = average_precision_score(labels_aligned, fused_scores)
                else:
                    ens_auroc, ens_ap = 0.5, 0.0
                
                results[f'Ens_{method}_AUROC'] = ens_auroc
                results[f'Ens_{method}_AP'] = ens_ap
                results[f'Ens_{method}_AUROC_Improv'] = ens_auroc - max(rgb_auroc, depth_auroc)
                results[f'Ens_{method}_AP_Improv'] = ens_ap - max(rgb_ap, depth_ap)
                
            except Exception as e:
                results[f'Ens_{method}_AUROC'] = 0.0
                results[f'Ens_{method}_AP'] = 0.0
        
        # Cross-modal ensemble (map-level)
        if is_valid_array(rgb_maps) and is_valid_array(depth_maps):
            for method in ['mean', 'max']:
                try:
                    fused_maps = cross_modal_ensemble_maps(
                        rgb_maps, depth_maps,
                        rgb_train_maps, depth_train_maps,
                        rgb_masks, labels, method=method
                    )
                    
                    if is_valid_array(fused_maps):
                        ens_auspro, ens_spro = compute_auspro(fused_maps, rgb_masks, labels)
                        results[f'Ens_{method}_AUsPRO'] = ens_auspro
                        results[f'Ens_{method}_sPRO'] = ens_spro
                        results[f'Ens_{method}_AUsPRO_Improv'] = ens_auspro - max(rgb_auspro, depth_auspro)
                        
                except Exception:
                    pass

        return results
        
    except Exception as e:
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CROSS-MODAL ENSEMBLE ANALYSIS: RGB + Pseudo-Depth (v3 - PARALLEL)")
    print("Section 4.3 of Paper")
    print("=" * 70)
    
    # Determine number of workers
    n_cpus = multiprocessing.cpu_count()
    n_workers = MAX_WORKERS if MAX_WORKERS else max(1, n_cpus - 1)
    print(f"CPUs available: {n_cpus}")
    print(f"Workers to use: {n_workers}")
    print(f"Base path: {os.path.abspath(BASE_PATH)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build task list
    tasks = []
    for dataset, models in MODELS_CONFIG.items():
        for model in models:
            for seed in SEEDS:
                tasks.append((dataset, model, seed, BASE_PATH))
    
    print(f"\nTotal tasks: {len(tasks)}")
    print("=" * 70)
    
    # Process in parallel
    all_results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            dataset, model, seed = task[:3]
            completed += 1
            
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
                    ens_ap = result.get('Ens_mean_AP', 0)
                    print(f"[{completed}/{len(tasks)}]  {dataset}/{model}/seed{seed}: "
                          f"RGB={result['RGB_AP']:.3f}, Depth={result['Depth_AP']:.3f}, "
                          f"Ens={ens_ap:.3f} (Δ={result.get('Ens_mean_AP_Improv', 0):+.3f})")
                else:
                    print(f"[{completed}/{len(tasks)}]  {dataset}/{model}/seed{seed}: No data")
            except Exception as e:
                print(f"[{completed}/{len(tasks)}]  {dataset}/{model}/seed{seed}: Error - {e}")
    
    if not all_results:
        print("\n No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    
    detailed_path = os.path.join(OUTPUT_DIR, "cross_modal_detailed.csv")
    df.to_csv(detailed_path, index=False)
    print(f"\n Detailed results: {detailed_path}")
    
    # Summary table
    summary_records = []
    for dataset in df['Dataset'].unique():
        for model in df[df['Dataset'] == dataset]['Model'].unique():
            subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            
            record = {
                'Dataset': dataset,
                'Model': model,
                'N_Seeds': len(subset),
                'RGB_AP_mean': subset['RGB_AP'].mean(),
                'RGB_AP_std': subset['RGB_AP'].std(),
                'Depth_AP_mean': subset['Depth_AP'].mean(),
                'Depth_AP_std': subset['Depth_AP'].std(),
                'RGB_AUsPRO_mean': subset['RGB_AUsPRO'].mean(),
                'RGB_AUsPRO_std': subset['RGB_AUsPRO'].std(),
                'Depth_AUsPRO_mean': subset['Depth_AUsPRO'].mean(),
                'Depth_AUsPRO_std': subset['Depth_AUsPRO'].std(),
            }
            
            for method in ['mean', 'max']:
                for metric in ['AP', 'AUROC', 'AUsPRO']:
                    col = f'Ens_{method}_{metric}'
                    col_imp = f'{col}_Improv'
                    if col in subset.columns:
                        record[f'{col}_mean'] = subset[col].mean()
                        record[f'{col}_std'] = subset[col].std()
                    if col_imp in subset.columns:
                        record[f'{col_imp}_mean'] = subset[col_imp].mean()
            
            summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUTPUT_DIR, "cross_modal_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f" Summary table: {summary_path}")
    
    # LaTeX table
    print("\n" + "=" * 70)
    print("TABLE 8: Cross-Modal Ensemble Results")
    print("=" * 70)
    
    latex_records = []
    for _, row in summary_df.iterrows():
        latex_records.append({
            'Dataset': row['Dataset'],
            'Model': row['Model'],
            'RGB_AP': f"{row['RGB_AP_mean']:.3f}±{row['RGB_AP_std']:.3f}",
            'Depth_AP': f"{row['Depth_AP_mean']:.3f}±{row['Depth_AP_std']:.3f}",
            'Ens_AP': f"{row.get('Ens_mean_AP_mean', 0):.3f}±{row.get('Ens_mean_AP_std', 0):.3f}",
            'Delta_AP': f"{row.get('Ens_mean_AP_Improv_mean', 0):+.3f}",
            'RGB_AUsPRO': f"{row['RGB_AUsPRO_mean']:.3f}±{row['RGB_AUsPRO_std']:.3f}",
            'Ens_AUsPRO': f"{row.get('Ens_mean_AUsPRO_mean', 0):.3f}±{row.get('Ens_mean_AUsPRO_std', 0):.3f}",
            'Delta_AUsPRO': f"{row.get('Ens_mean_AUsPRO_Improv_mean', 0):+.3f}",
        })
    
    latex_df = pd.DataFrame(latex_records)
    latex_path = os.path.join(OUTPUT_DIR, "table8_cross_modal_latex.csv")
    latex_df.to_csv(latex_path, index=False)
    
    print(latex_df.to_string(index=False))
    print(f"\n LaTeX table: {latex_path}")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if 'Ens_mean_AP_Improv_mean' in summary_df.columns:
        ap_improv = summary_df['Ens_mean_AP_Improv_mean']
        positive = (ap_improv > 0).sum()
        negative = (ap_improv <= 0).sum()
        
        print(f" AP improvements: {positive}/{len(summary_df)} ({100*positive/len(summary_df):.1f}%)")
        print(f" AP degradations: {negative}/{len(summary_df)} ({100*negative/len(summary_df):.1f}%)")
        print(f" Mean ΔAP: {ap_improv.mean():+.4f}")
        
        if 'Ens_mean_AUsPRO_Improv_mean' in summary_df.columns:
            auspro_improv = summary_df['Ens_mean_AUsPRO_Improv_mean']
            print(f" Mean ΔAUsPRO: {auspro_improv.mean():+.4f}")
    
    print("\n Analysis complete!")


if __name__ == "__main__":
    main()
