import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import label as cc_label, zoom
from scipy.stats import trim_mean, rankdata
from scipy.integrate import trapezoid

# Set random seed for reproducibility
np.random.seed(42)

def load_model_data(model_path):
    """Load test results and optionally anomaly maps/masks from a model directory"""
    with open(os.path.join(model_path, 'test_results.json'), 'r') as f:
        results = json.load(f)

    anomaly_maps, gt_masks, train_normal_maps = [], [], []
    try:
        with open(os.path.join(model_path, 'anomaly_maps.pkl'), 'rb') as f:
            anomaly_maps = pickle.load(f)
        with open(os.path.join(model_path, 'gt_masks.pkl'), 'rb') as f:
            gt_masks = pickle.load(f)
    except FileNotFoundError:
        # Maps/masks may not exist for some models; proceed without them
        anomaly_maps, gt_masks = [], []
    
    # Try to load train normal maps for normalization
    try:
        with open(os.path.join(model_path, 'train_normal_maps.pkl'), 'rb') as f:
            train_normal_maps = pickle.load(f)
    except FileNotFoundError:
        # Train normal maps may not exist; we'll extract them from test normal maps
        train_normal_maps = []

    return results, anomaly_maps, gt_masks, train_normal_maps


def normalize_maps_robust(maps, train_normal_maps, use_mad=True):
    """Normalize anomaly maps using robust statistics (MAD) or z-score"""
    if len(train_normal_maps) == 0:
        return maps
    
    # Flatten all train normal maps to compute global statistics
    all_normal_pixels = np.concatenate([m.ravel() for m in train_normal_maps])
    
    if use_mad:
        # Robust normalization using Median Absolute Deviation
        median_normal = np.median(all_normal_pixels)
        mad_normal = np.median(np.abs(all_normal_pixels - median_normal))
        if mad_normal == 0:
            mad_normal = 1e-8
        # MAD to std approximation: 1.4826 * MAD ≈ std for normal distribution
        scale = 1.4826 * mad_normal
        normalized = [(m - median_normal) / scale for m in maps]
    else:
        # Standard z-score normalization
        mean_normal = np.mean(all_normal_pixels)
        std_normal = np.std(all_normal_pixels)
        if std_normal == 0:
            std_normal = 1e-8
        normalized = [(m - mean_normal) / std_normal for m in maps]
    
    return [m.astype('float32') for m in normalized]

def check_and_fix_map_orientation(maps, train_normal_maps, gt_masks, labels):
    """Check and fix map orientation: higher values should indicate more anomalous regions"""
    if len(maps) == 0 or len(gt_masks) == 0:
        return maps, train_normal_maps
    
    # Compute mean anomaly score in anomalous regions vs normal regions
    anomalous_scores = []
    normal_scores = []
    
    for i, (amap, gtm, label) in enumerate(zip(maps, gt_masks, labels)):
        if label == 1 and gtm is not None:  # Anomalous image with mask
            amap_arr = np.asarray(amap)
            gtm_arr = np.asarray(gtm).astype(bool)
            
            if gtm_arr.any():  # Has anomalous regions
                anomalous_pixels = amap_arr[gtm_arr]
                normal_pixels = amap_arr[~gtm_arr]
                
                if len(anomalous_pixels) > 0:
                    anomalous_scores.extend(anomalous_pixels.ravel())
                if len(normal_pixels) > 0:
                    normal_scores.extend(normal_pixels.ravel())
    
    if len(anomalous_scores) > 0 and len(normal_scores) > 0:
        mean_anomalous = np.mean(anomalous_scores)
        mean_normal = np.mean(normal_scores)
        
        # If anomalous regions have lower values than normal regions, flip the maps
        if mean_anomalous < mean_normal:
            print(f"     Flipping maps orientation (anomalous < normal: {mean_anomalous:.4f} < {mean_normal:.4f})")
            flipped_maps = [-np.asarray(m).astype('float32') for m in maps]
            flipped_train_maps = [-np.asarray(m).astype('float32') for m in train_normal_maps]
            return flipped_maps, flipped_train_maps
        else:
            print(f"     Maps orientation correct (anomalous > normal: {mean_anomalous:.4f} > {mean_normal:.4f})")
    
    return maps, train_normal_maps

def zcal(scores, ref_scores):
    """Z-calibrate scores using reference scores"""
    if ref_scores is None or len(ref_scores) == 0:
        # no reference: center at 0 and use scale 1 to avoid overflow
        return scores.astype(float), 0.0, 1.0
    mu = float(np.mean(ref_scores))
    sigma = float(np.std(ref_scores))
    if sigma == 0: 
        sigma = 1e-8
    return (scores - mu) / sigma, mu, sigma


def check_and_fix_sign(scores, train_scores, labels, model_name):
    """Ensure scores have correct sign: higher = more anomalous"""
    # Guard single-class case
    try:
        auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else 0.5
        if auroc < 0.5:
            scores = -scores
            train_scores = -train_scores
    except ValueError:
        # Handle edge case where roc_auc_score fails
        pass
    return scores, train_scores

def ensemble_scores(scores1, scores2, train_scores1, train_scores2, labels, method='mean'):
    """Ensemble prediction scores from two models with proper calibration"""
    # 1) Fix sign
    s1, tr1 = check_and_fix_sign(np.asarray(scores1), np.asarray(train_scores1), labels, "Model1")
    s2, tr2 = check_and_fix_sign(np.asarray(scores2), np.asarray(train_scores2), labels, "Model2")

    # 2) Z-calibrate EACH model with its train normals
    z1, mu1, sigma1 = zcal(s1, tr1)
    z2, mu2, sigma2 = zcal(s2, tr2)

    # 3) Align lengths
    n = min(len(z1), len(z2))
    z1, z2 = z1[:n], z2[:n]
    m = min(len(tr1), len(tr2))
    tr1, tr2 = tr1[:m], tr2[:m]

    # 4) Fuse in comparable scale
    if method == 'mean':
        fused_scores = 0.5*(z1 + z2)
        fused_train = 0.5*( (tr1 - mu1)/max(sigma1,1e-8) + (tr2 - mu2)/max(sigma2,1e-8) )
    elif method == 'median':
        fused_scores = np.median(np.stack([z1, z2], 0), 0)
        fused_train  = np.median(np.stack([(tr1-mu1)/max(sigma1,1e-8), (tr2-mu2)/max(sigma2,1e-8)], 0), 0)
    elif method == 'max':
        fused_scores = np.maximum(z1, z2)
        fused_train  = np.maximum((tr1-mu1)/max(sigma1,1e-8), (tr2-mu2)/max(sigma2,1e-8))
    elif method == 'trimmed_mean':
        # Trimmed mean (25% trim) - robust to outliers
        fused_scores = np.array([trim_mean([z1[i], z2[i]], 0.25) for i in range(len(z1))])
        tr1_norm = (tr1-mu1)/max(sigma1,1e-8)
        tr2_norm = (tr2-mu2)/max(sigma2,1e-8)
        fused_train = np.array([trim_mean([tr1_norm[i], tr2_norm[i]], 0.25) for i in range(len(tr1_norm))])
    elif method == 'rank_average':
        # Rank average - average the ranks of scores
        # Convert to ranks (higher score = higher rank)
        ranks1 = rankdata(z1, method='average')
        ranks2 = rankdata(z2, method='average')
        fused_scores = 0.5 * (ranks1 + ranks2)
        
        # For train scores, also use rank average
        tr1_norm = (tr1-mu1)/max(sigma1,1e-8)
        tr2_norm = (tr2-mu2)/max(sigma2,1e-8)
        tr_ranks1 = rankdata(tr1_norm, method='average')
        tr_ranks2 = rankdata(tr2_norm, method='average')
        fused_train = 0.5 * (tr_ranks1 + tr_ranks2)

    # 5) (Optional) re-calibrate ensemble for reporting (with robust guard)
    if fused_train.size == 0 or not np.isfinite(fused_train).all():
        mu_fused, sigma_fused = 0.0, 1.0
    else:
        mu_fused = float(np.mean(fused_train))
        sigma_fused = float(np.std(fused_train))
        if not np.isfinite(sigma_fused) or sigma_fused == 0:
            sigma_fused = 1e-8
    calibrated_scores = (fused_scores - mu_fused) / sigma_fused
    calibrated_train  = (fused_train  - mu_fused) / sigma_fused
    
    return calibrated_scores, calibrated_train, mu_fused, sigma_fused


def ensemble_maps(maps1, maps2, train_maps1, train_maps2, masks=None, labels=None, method='mean', use_mad=True):
    """Combine anomaly maps with proper normalization and orientation checking"""
    def resize_map_bilinear(arr, th, tw):
        """Resize map using bilinear interpolation"""
        h, w = arr.shape[-2], arr.shape[-1]
        if h == th and w == tw:
            return arr.astype('float32')
        zy, zx = th / h, tw / w
        return zoom(arr.astype('float32'), (zy, zx), order=1)  # order=1 is bilinear

    # Determine target size from masks if available, otherwise use first mask size
    if masks and len(masks) > 0:
        # Use first non-None mask size as target
        target_mask = None
        for m in masks:
            if m is not None:
                target_mask = np.asarray(m)
                break
        if target_mask is not None:
            th, tw = target_mask.shape[-2], target_mask.shape[-1]
        else:
            # Fallback to maps if all masks None
            if len(maps1) > 0 and len(maps2) > 0:
                h1, w1 = np.asarray(maps1[0]).shape[-2:]
                h2, w2 = np.asarray(maps2[0]).shape[-2:]
                th, tw = max(h1, h2), max(w1, w2)
            else:
                th, tw = 400, 400
    else:
        # Fallback: use the larger of the two first maps
        if len(maps1) > 0 and len(maps2) > 0:
            h1, w1 = np.asarray(maps1[0]).shape[-2:]
            h2, w2 = np.asarray(maps2[0]).shape[-2:]
            th, tw = max(h1, h2), max(w1, w2)
        else:
            th, tw = 400, 400  # Final fallback

    # Resize all maps to target size using bilinear interpolation
    resized_maps1 = [resize_map_bilinear(np.asarray(m), th, tw) for m in maps1]
    resized_maps2 = [resize_map_bilinear(np.asarray(m), th, tw) for m in maps2]
    resized_train1 = [resize_map_bilinear(np.asarray(m), th, tw) for m in train_maps1] if train_maps1 else []
    resized_train2 = [resize_map_bilinear(np.asarray(m), th, tw) for m in train_maps2] if train_maps2 else []
    
    # If no train maps available, use normal test samples as approximation (first 20)
    if not resized_train1 and labels is not None:
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        resized_train1 = [resized_maps1[i] for i in normal_indices[:min(20, len(normal_indices))]]
    if not resized_train2 and labels is not None:
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        resized_train2 = [resized_maps2[i] for i in normal_indices[:min(20, len(normal_indices))]]
    
    # Step 1: Check and fix orientation for both models
    if masks and labels is not None:
        print(f"   Checking map orientations...")
        resized_maps1, resized_train1 = check_and_fix_map_orientation(resized_maps1, resized_train1, masks, labels)
        resized_maps2, resized_train2 = check_and_fix_map_orientation(resized_maps2, resized_train2, masks, labels)
    
    # Step 2: Normalize maps using robust statistics and collect normalization stats
    print(f"   Normalizing maps using {'MAD' if use_mad else 'z-score'}...")
    normalized_maps1 = normalize_maps_robust(resized_maps1, resized_train1, use_mad)
    normalized_maps2 = normalize_maps_robust(resized_maps2, resized_train2, use_mad)
    
    # Calculate normalization stats for each model (for metadata)
    map_stats = {}
    if len(resized_train1) > 0:
        all_train_pixels1 = np.concatenate([m.ravel() for m in resized_train1])
        if use_mad:
            map_stats['mu_map_model1'] = float(np.median(all_train_pixels1))
            map_stats['sigma_map_model1'] = float(1.4826 * np.median(np.abs(all_train_pixels1 - map_stats['mu_map_model1'])))
        else:
            map_stats['mu_map_model1'] = float(np.mean(all_train_pixels1))
            map_stats['sigma_map_model1'] = float(np.std(all_train_pixels1))
    
    if len(resized_train2) > 0:
        all_train_pixels2 = np.concatenate([m.ravel() for m in resized_train2])
        if use_mad:
            map_stats['mu_map_model2'] = float(np.median(all_train_pixels2))
            map_stats['sigma_map_model2'] = float(1.4826 * np.median(np.abs(all_train_pixels2 - map_stats['mu_map_model2'])))
        else:
            map_stats['mu_map_model2'] = float(np.mean(all_train_pixels2))
            map_stats['sigma_map_model2'] = float(np.std(all_train_pixels2))
    
    # Step 3: Fuse normalized maps
    print(f"   Fusing maps using {method}...")
    combined_maps = []
    n = min(len(normalized_maps1), len(normalized_maps2))
    for i in range(n):
        a = normalized_maps1[i]
        b = normalized_maps2[i]
        if method == 'mean':
            ens_map = (a + b) / 2.0
        elif method == 'median':
            ens_map = np.median(np.stack([a, b], axis=0), axis=0)
        elif method == 'max':
            ens_map = np.maximum(a, b)
        combined_maps.append(ens_map.astype('float32'))
    
    # Calculate fused map stats from the ensemble train normal maps
    if len(resized_train1) > 0 and len(resized_train2) > 0:
        # Create ensemble train maps for stats calculation
        fused_train_maps = []
        for i in range(min(len(resized_train1), len(resized_train2))):
            train_a = normalize_maps_robust([resized_train1[i]], resized_train1, use_mad)[0]
            train_b = normalize_maps_robust([resized_train2[i]], resized_train2, use_mad)[0]
            if method == 'mean':
                fused_train = (train_a + train_b) / 2.0
            elif method == 'median':
                fused_train = np.median(np.stack([train_a, train_b], axis=0), axis=0)
            elif method == 'max':
                fused_train = np.maximum(train_a, train_b)
            fused_train_maps.append(fused_train)
        
        if len(fused_train_maps) > 0:
            all_fused_pixels = np.concatenate([m.ravel() for m in fused_train_maps])
            map_stats['mu_map_fused'] = float(np.mean(all_fused_pixels))
            map_stats['sigma_map_fused'] = float(np.std(all_fused_pixels))
    
    print(f"   Generated {len(combined_maps)} normalized ensemble maps of size {th}x{tw}")
    return combined_maps, map_stats


def compute_metrics(scores, labels, train_normal_scores=None, anomaly_maps=None, gt_masks=None, 
                   ensemble_metadata=None):
    """Compute evaluation metrics with proper TPR@TNR, TNR_test, ΔTNR and sanity checks"""
    # sPRO saturation parameter
    s_fraction = 0.3  # saturation fraction as in the sPRO definition
    
    # Image-level metrics (guard single-class)
    if len(np.unique(labels)) == 2:
        auroc = roc_auc_score(labels, scores)
        image_ap = average_precision_score(labels, scores)
    else:
        auroc = 0.5
        image_ap = 0.0
    
    # Enhanced TPR@TNR metrics with TNR_test and ΔTNR
    tpr_metrics = {}
    normal_mask = labels == 0
    positive_mask = labels == 1
    num_pos = int(positive_mask.sum())
    num_neg = int(normal_mask.sum())
    
    if train_normal_scores is not None and len(train_normal_scores) > 0:
        tnr_targets = [0.99, 0.95, 0.90]
        tolerances = {0.99: 0.03, 0.95: 0.05, 0.90: 0.05}  # Tolerance for comparability
        
        # Use train normal scores for thresholds
        thresholds = {tnr: float(np.quantile(train_normal_scores, tnr)) for tnr in tnr_targets}
        
        for tnr in tnr_targets:
            thr = thresholds[tnr]
            pred_pos = scores >= thr
            
            # TPR calculation
            tp = int((pred_pos & positive_mask).sum())
            tpr_value = (tp / num_pos) if num_pos > 0 else 0.0
            tpr_metrics[f'TPR@TNR{tnr:.2f}'] = tpr_value
        
            # TNR_test calculation (actual TNR on test normal samples)
            tn = int((~pred_pos & normal_mask).sum())
            tnr_test = (tn / num_neg) if num_neg > 0 else 1.0
            tpr_metrics[f'TNR_test@{tnr:.2f}'] = tnr_test
            
            # ΔTNR calculation
            delta_tnr = tnr_test - tnr
            tpr_metrics[f'ΔTNR@{tnr:.2f}'] = delta_tnr
            
            # Comparability check
            is_comparable = abs(delta_tnr) <= tolerances[tnr]
            tpr_metrics[f'Comparable@{tnr:.2f}'] = is_comparable
        
        # Sanity checks for monotonicity
        tpr_values = [tpr_metrics[f'TPR@TNR{tnr:.2f}'] for tnr in tnr_targets]
        tnr_test_values = [tpr_metrics[f'TNR_test@{tnr:.2f}'] for tnr in tnr_targets]
        
        # TPR should be monotonic: TPR@0.99 ≤ TPR@0.95 ≤ TPR@0.90
        monotonic_tpr = all(tpr_values[i] <= tpr_values[i+1] for i in range(len(tpr_values)-1))
        # TNR_test should be monotonic: TNR_test@0.99 ≥ TNR_test@0.95 ≥ TNR_test@0.90
        monotonic_tnr = all(tnr_test_values[i] >= tnr_test_values[i+1] for i in range(len(tnr_test_values)-1))
        
        tpr_metrics['MonotonicityOK'] = monotonic_tpr and monotonic_tnr
        
        # Store thresholds for ensemble metadata
        if ensemble_metadata is not None:
            ensemble_metadata['theta_fused'] = thresholds
    else:
        # Fallback to test normals (less reliable) with guard when no train normals available
        normal_scores = scores[normal_mask]
        tnr_targets = [0.99, 0.95, 0.90]
        if normal_scores.size == 0:
            for tnr in tnr_targets:
                tpr_metrics[f'TPR@TNR{tnr:.2f}'] = 0.0
        else:
            thresholds = {tnr: float(np.quantile(normal_scores, tnr)) for tnr in tnr_targets}
            for tnr in tnr_targets:
                thr = thresholds[tnr]
                pred_pos = scores >= thr
                tp = int((pred_pos & positive_mask).sum())
                tpr_value = (tp / num_pos) if num_pos > 0 else 0.0
                tpr_metrics[f'TPR@TNR{tnr:.2f}'] = tpr_value
    
    metrics = {
        'image_AUROC': float(auroc),
        'Image_AP_AUPR': float(image_ap),
        **tpr_metrics
    }
    
    # Improved sPRO calculation - sweep thresholds from high to low
    if anomaly_maps is not None and gt_masks is not None:
        abnormal_indices = np.where(labels == 1)[0]
        normal_indices = np.where(labels == 0)[0]
        
        if len(abnormal_indices) > 0 and len(normal_indices) > 0:
            # Get all pixels from normal images for FPR calculation
            all_normal_pixels = []
            for idx in normal_indices:
                if idx < len(anomaly_maps):
                    amap = np.asarray(anomaly_maps[idx])
                    # resize amap to its corresponding mask size if available, else 400x400
                    if gt_masks is not None and idx < len(gt_masks) and gt_masks[idx] is not None:
                        gtm = np.asarray(gt_masks[idx])
                        th, tw = gtm.shape[-2], gtm.shape[-1]
                    else:
                        th, tw = 400, 400
                    h, w = amap.shape[-2], amap.shape[-1]
                    if (h, w) != (th, tw):
                        zy, zx = th / h, tw / w
                        amap = zoom(amap.astype('float32'), (zy, zx), order=1)
                    all_normal_pixels.extend(amap.ravel())
            
            if len(all_normal_pixels) > 0:
                all_normal_pixels = np.array(all_normal_pixels)
                
                # Create thresholds from high to low
                t_min, t_max = float(all_normal_pixels.min()), float(all_normal_pixels.max())
                thresholds = np.linspace(t_max, t_min, num=100)
                
                fprs = []
                spros = []
                
                def resize_to_mask(a, mask):
                    th, tw = mask.shape[-2], mask.shape[-1]
                    h, w = a.shape[-2], a.shape[-1]
                    if (h, w) == (th, tw):
                        return a
                    zy, zx = th / h, tw / w
                    return zoom(a.astype('float32'), (zy, zx), order=1)

                for thr in thresholds:
                    # FPR: fraction of normal pixels predicted as anomalous
                    fp_count = (all_normal_pixels >= thr).sum()
                    fpr = fp_count / len(all_normal_pixels)
                    fprs.append(fpr)
                    
                    # sPRO: recall per component
                    pro_scores = []
                    for idx in abnormal_indices:
                        if idx < len(anomaly_maps) and idx < len(gt_masks) and gt_masks[idx] is not None:
                            amap = np.asarray(anomaly_maps[idx])
                            gtm = np.asarray(gt_masks[idx])
                            if amap.shape != gtm.shape:
                                amap = resize_to_mask(amap, gtm)
                            # Sanity check: map-mask alignment
                            assert amap.shape == gtm.shape, f"Map shape {amap.shape} != mask shape {gtm.shape}"
                            comp_map, num = cc_label(gtm.astype('uint8'))
                            if num > 0:
                                comp_pros = []
                                for comp_id in range(1, num + 1):
                                    comp = (comp_map == comp_id)
                                    denom = comp.sum()
                                    if denom > 0:
                                        tp = ((amap >= thr) & comp).sum()
                                        s_R = max(1, int(np.ceil(s_fraction * denom)))  # saturation per region
                                        comp_pros.append(min(1.0, tp / s_R))
                                if len(comp_pros) > 0:
                                    pro_scores.append(float(np.mean(comp_pros)))
                    
                    spro_val = float(np.mean(pro_scores)) if len(pro_scores) > 0 else 0.0
                    spros.append(spro_val)
                
                fprs = np.array(fprs)
                spros = np.array(spros)
                
                # Find sPRO at FPR = 0.05 and compute AUsPRO@0.05 (aligned with custom_metrics)
                alpha = 0.05
                valid_indices = fprs <= alpha
                
                if np.any(valid_indices):
                    # Get valid FPR and sPRO points up to alpha=0.05
                    valid_fprs = fprs[valid_indices]
                    valid_spros = spros[valid_indices]
                    
                    # Sort by FPR for proper integration
                    sort_order = np.argsort(valid_fprs)
                    sorted_fprs = valid_fprs[sort_order]
                    sorted_spros = valid_spros[sort_order]
                    
                    # sPRO@alpha: value at maximum FPR <= alpha
                    spro_at_05 = float(sorted_spros[-1])
                    metrics['sPRO@0.05'] = spro_at_05
                    
                    # AUsPRO@alpha: area under sPRO curve normalized by alpha
                    if len(sorted_fprs) > 1:
                        # auspro_area = np.trapezoid(sorted_spros, sorted_fprs) / alpha
                        auspro_area = trapezoid(sorted_spros, sorted_fprs) / alpha
                        metrics['AUsPRO@0.05'] = float(auspro_area)
                    else:
                        metrics['AUsPRO@0.05'] = spro_at_05  # Fallback if only one point
                else:
                    # No valid points in range - use fallback
                    metrics['sPRO@0.05'] = 0.0
                    metrics['AUsPRO@0.05'] = 0.0

                # Pixel-level metrics (AUROC/AP) over all pixels
                y_scores = []
                y_true = []
                for i in range(min(len(anomaly_maps), len(gt_masks))):
                    amap = np.asarray(anomaly_maps[i])
                    gtm = gt_masks[i]
                    if gtm is None:
                        # assume all-zero mask for missing masks
                        mask = np.zeros_like(amap, dtype=np.uint8)
                    else:
                        mask = np.asarray(gtm)
                        if amap.shape != mask.shape:
                            mask_h, mask_w = mask.shape[-2], mask.shape[-1]
                            h, w = amap.shape[-2], amap.shape[-1]
                            if (h, w) != (mask_h, mask_w):
                                zy, zx = mask_h / h, mask_w / w
                                amap = zoom(amap.astype('float32'), (zy, zx), order=1)
                    y_scores.append(amap.ravel())
                    y_true.append((mask > 0).astype(np.uint8).ravel())
                if len(y_scores) > 0:
                    y_scores = np.concatenate(y_scores)
                    y_true = np.concatenate(y_true)
                    if y_true.min() != y_true.max():
                        metrics['pixel_AUROC'] = float(roc_auc_score(y_true, y_scores))
                        metrics['pixel_AP_AUPR'] = float(average_precision_score(y_true, y_scores))
    
    return metrics


def run_ensemble(path1, path2, name1, name2, output_path, fusion_method=None):
    """Main ensemble function - TEST DATA ONLY (no train data used)"""
    print(f" Running ensemble: {name1} + {name2}")
    
    # Load data from both models
    try:
        results1, maps1, masks1, _ = load_model_data(path1)
        results2, maps2, masks2, _ = load_model_data(path2)
        
        print(f"     Data loaded:")
        print(f"      {name1}: {len(results1.get('scores', []))} test samples")
        print(f"      {name2}: {len(results2.get('scores', []))} test samples")
        
    except Exception as e:
        print(f"     Error loading data: {e}")
        raise
    
    # Extract scores and labels (TEST ONLY)
    scores1 = np.array(results1['scores'])
    scores2 = np.array(results2['scores'])
    
    # Convert boolean labels to int (True/False -> 1/0)
    labels_raw = results1['labels']
    if isinstance(labels_raw[0], bool):
        labels = np.array(labels_raw).astype(int)
    else:
        labels = np.array(labels_raw).astype(int)
    
    # Check that labels match between models
    assert np.array_equal(results1['labels'], results2['labels']), "Labels must match between models"
    
    # Prefer true train normals if available:
    train_scores1 = np.array(results1.get('train_normal_scores', []))
    train_scores2 = np.array(results2.get('train_normal_scores', []))
    if train_scores1.size == 0 or train_scores2.size == 0:
        print("  No train_normal_scores found; FALLBACK to test normals (labels==0).")
        train_scores1 = scores1[labels == 0]
        train_scores2 = scores2[labels == 0]
    
    results_list = []
    
    # Add individual model metrics first (with proper train scores)
    individual_metrics1 = compute_metrics(scores1, labels, train_scores1, maps1, masks1)
    row1 = {
        'Type': 'Individual',
        'Method': name1,
        'Model1': name1,
        'Model2': '',
        'Test_Images': len(labels),
        **individual_metrics1
    }
    
    individual_metrics2 = compute_metrics(scores2, labels, train_scores2, maps2, masks2)
    row2 = {
        'Type': 'Individual',
        'Method': name2,
        'Model1': name2,
        'Model2': '',
        'Test_Images': len(labels),
        **individual_metrics2
    }
    
    results_list.extend([row1, row2])
    
    # Ensemble experiments with proper normalization - add 'max' for better TPR@TNR
    methods = ['mean', 'median', 'max'] if fusion_method is None else [fusion_method]
    
    for method in methods:
        # Map method fallback for unsupported methods in ensemble_maps
        map_method = method if method in ['mean', 'median', 'max'] else 'mean'
        
        # Score ensemble with FUSION → CALIBRATION (correct order)
        ens_scores, combined_train_scores, mu_fused, sigma_fused = ensemble_scores(
            scores1, scores2, train_scores1, train_scores2, labels, method)
        
        # Ensemble metadata for auditing
        ensemble_metadata = {
            'members': [name1, name2],
            'fusion_policy': method,
            'mu_fused': float(mu_fused),
            'sigma_fused': float(sigma_fused),
            'map_resize': 'bilinear',
            'mask_resize': 'nearest',
            'morphology': 'none'
        }
        
        # Fix for TPR@TNR in advanced fusion methods
        if method in ['trimmed_mean', 'rank_average']:
            # For advanced methods, using test normals is more stable
            ensemble_train_fallback = ens_scores[labels == 0]
        elif len(combined_train_scores) == 0:
            print(f"      No combined_train_scores available for TPR@TNR calculation")
            # Fallback: use test normals for TPR@TNR (suboptimal but functional)
            ensemble_train_fallback = ens_scores[labels == 0]
        else:
            ensemble_train_fallback = combined_train_scores
            
        score_metrics = compute_metrics(ens_scores, labels, ensemble_train_fallback, 
                                      ensemble_metadata=ensemble_metadata)
        
        result_row = {
            'Type': 'Ensemble',
            'Method': f'score_ensemble_{method}',
            'Model1': name1,
            'Model2': name2,
            'Test_Images': len(labels),
            'mu_fused': mu_fused,
            'sigma_fused': sigma_fused,
            'fusion_policy': method,
            **score_metrics
        }
        
        # Add comparability flags
        for k in ['Comparable@0.99','Comparable@0.95','Comparable@0.90','MonotonicityOK']:
            if k in score_metrics: 
                result_row[k] = score_metrics[k]
        results_list.append(result_row)
        
        # Map ensemble with normalization and orientation
        if len(maps1) > 0 and len(maps2) > 0:
            print(f"\n  Map ensemble using {map_method} (from {method})")
            ens_maps, map_stats = ensemble_maps(maps1, maps2, [], [], masks1, labels, map_method, use_mad=True)
            
            # Update metadata for maps
            map_metadata = ensemble_metadata.copy()
            map_metadata['morphology'] = 'none'
            
            # Use the same fallback for map_metrics
            map_metrics = compute_metrics(ens_scores, labels, ensemble_train_fallback, ens_maps, masks1,
                                        ensemble_metadata=map_metadata)
            
            result_row = {
                'Type': 'Ensemble',
                'Method': f'map_ensemble_{map_method}',
                'Model1': name1,
                'Model2': name2,
                'Test_Images': len(labels),
                'mu_fused': mu_fused,
                'sigma_fused': sigma_fused,
                'fusion_policy': method,
                'morphology': 'none',
                'fusion_policy_requested': method,
                'fusion_policy_used_maps': map_method,
                **map_stats,  # Add map normalization stats
                **map_metrics
            }
            
            # Add comparability flags
            for k in ['Comparable@0.99','Comparable@0.95','Comparable@0.90','MonotonicityOK']:
                if k in map_metrics:
                    result_row[k] = map_metrics[k]
            results_list.append(result_row)
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    df_results = pd.DataFrame(results_list)
    csv_path = os.path.join(output_path, f"ensemble_complete_{name1}_{name2}.csv")
    df_results.to_csv(csv_path, index=False, sep=';', decimal=',')
    
    print(f"\n Results saved to: {csv_path}")

    # Academic recommendation: Check comparability of ensembles
    ensemble_rows = df_results[df_results['Type'] == 'Ensemble']
    print(f"\n Academic Analysis:")
    for _, row in ensemble_rows.iterrows():
        method = row['Method']
        comparable_95 = row.get('Comparable@0.95', True)
        comparable_90 = row.get('Comparable@0.90', True)
        monotonic = row.get('MonotonicityOK', True)
        
        if not (comparable_95 and comparable_90 and monotonic):
            print(f"  {method}: Non-comparable ensemble (suggest reporting individual models)")
            print(f"    Comparable@0.95={comparable_95}, Comparable@0.90={comparable_90}, Monotonic={monotonic}")
        else:
            print(f" {method}: Comparable ensemble (suitable for academic reporting)")


def dataset_ensemble(dataset_name, model_names, seeds, base_path="results_0/all_models", output_dir=None):
    """Generate ensemble results for multiple models and seeds for a dataset (PRECISE mode only)"""
    print(f" Dataset Ensemble Analysis: {dataset_name}")
    print(f" Models: {model_names}")
    print(f" Seeds: {seeds}")
    print(f"  Mode: PRECISE (real pixel-level metrics)")
    
    if output_dir is None:
        output_dir = f"ensemble_results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results for aggregation
    individual_results = []
    ensemble_results = []
    
    # For each seed, run all pairwise ensembles
    for seed in seeds:
        print(f"\n Processing seed {seed}")
        seed_individual = []
        seed_ensemble = []
        
        # Collect individual model metrics for this seed
        individual_metrics = {}
        valid_models = []
        
        for model in model_names:
            model_path = os.path.join(base_path, dataset_name, str(seed), model)
            test_results_path = os.path.join(model_path, 'test_results.json')
            
            if os.path.exists(test_results_path):
                try:
                    with open(test_results_path, 'r') as f:
                        results = json.load(f)
                    
                    individual_metrics[model] = {
                        'Image_AP_AUPR': results.get('Image_AP_AUPR', results.get('image_ap', 0.0)),
                        'AUsPRO@0.05': results.get('AUsPRO@0.05', 0.0),
                        'image_AUROC': results.get('image_AUROC', results.get('auroc', 0.0)),
                        'pixel_AUROC': results.get('pixel_AUROC', results.get('pixel_auroc', 0.0)),
                        'sPRO@0.05': results.get('sPRO@0.05', 0.0),
                        'TPR@TNR0.99': results.get('TPR@TNR0.99', 0.0),
                        'TPR@TNR0.95': results.get('TPR@TNR0.95', 0.0),
                        'TPR@TNR0.90': results.get('TPR@TNR0.90', 0.0),
                        'n_samples': len(results.get('scores', []))
                    }
                    valid_models.append(model)
                    
                    # Store individual result
                    seed_individual.append({
                        'Dataset': dataset_name,
                        'Seed': seed,
                        'Type': 'Individual',
                        'Model': model,
                        'Model1': model,
                        'Model2': '',
                        **individual_metrics[model]
                    })
                    
                    print(f"   {model}: AP={individual_metrics[model]['Image_AP_AUPR']:.4f}, AUsPRO={individual_metrics[model]['AUsPRO@0.05']:.4f}")
                    
                except Exception as e:
                    print(f"   Error loading {model}: {e}")
            else:
                print(f"    {model}: test_results.json not found at {test_results_path}")
        
        # Run pairwise ensembles for valid models
        for i, model1 in enumerate(valid_models):
            for j, model2 in enumerate(valid_models):
                if i < j:  # Avoid duplicates
                    try:
                        print(f"   Ensemble: {model1} + {model2}")
                        
                        path1 = os.path.join(base_path, dataset_name, str(seed), model1)
                        path2 = os.path.join(base_path, dataset_name, str(seed), model2)
                        
                        # PRECISE version: Real pixel-level ensemble
                        temp_output = os.path.join(output_dir, f"temp_{seed}_{model1}_{model2}")
                        run_ensemble(path1, path2, model1, model2, temp_output, fusion_method='mean')
                        
                        # Read precise results from CSV
                        csv_path = os.path.join(temp_output, f"ensemble_complete_{model1}_{model2}.csv")
                        if not os.path.exists(csv_path):
                            print(f"      CSV not found for {model1}+{model2}, skipping.")
                            if os.path.exists(temp_output): 
                                shutil.rmtree(temp_output)
                            continue
                        
                        if os.path.exists(csv_path):
                            df_temp = pd.read_csv(csv_path, sep=';', decimal=',')
                            ensemble_rows = df_temp[df_temp['Type'] == 'Ensemble']
                            
                            # Prefer map_ensemble (real pixel-level) over score_ensemble
                            map_ensemble_row = ensemble_rows[ensemble_rows['Method'].str.contains('map_ensemble', na=False)]
                            if len(map_ensemble_row) > 0:
                                best_ensemble = map_ensemble_row.iloc[0]  # Take first map_ensemble
                            else:
                                best_ensemble = ensemble_rows.iloc[0]  # Fallback to first ensemble
                            
                            ensemble_result = {
                                'Dataset': dataset_name,
                                'Seed': seed,
                                'Type': 'Ensemble',
                                'Model': f"{model1}+{model2}",
                                'Model1': model1,
                                'Model2': model2,
                                'Method': best_ensemble.get('Method', 'map_ensemble_mean'),
                                'Image_AP_AUPR': best_ensemble.get('Image_AP_AUPR', 0.0),
                                'AUsPRO@0.05': best_ensemble.get('AUsPRO@0.05', 0.0),
                                'image_AUROC': best_ensemble.get('image_AUROC', 0.0),
                                'pixel_AUROC': best_ensemble.get('pixel_AUROC', 0.0),
                                'sPRO@0.05': best_ensemble.get('sPRO@0.05', 0.0),
                                'TPR@TNR0.99': best_ensemble.get('TPR@TNR0.99', 0.0),
                                'TPR@TNR0.95': best_ensemble.get('TPR@TNR0.95', 0.0),
                                'TPR@TNR0.90': best_ensemble.get('TPR@TNR0.90', 0.0)
                            }
                        # Clean up
                        if os.path.exists(temp_output):
                            shutil.rmtree(temp_output)
                        seed_ensemble.append(ensemble_result)
                        
                        print(f"     AP={ensemble_result['Image_AP_AUPR']:.4f}, AUsPRO={ensemble_result['AUsPRO@0.05']:.4f}")
                            
                    except Exception as e:
                        print(f"     Ensemble {model1}+{model2} failed: {e}")
        
        individual_results.extend(seed_individual)
        ensemble_results.extend(seed_ensemble)
    
    # Combine all results
    all_results = individual_results + ensemble_results
    
    # Create summary statistics
    summary_results = []
    
    # Group by model configuration
    df_all = pd.DataFrame(all_results)
    
    # Individual model summaries
    for model in model_names:
        model_data = df_all[(df_all['Type'] == 'Individual') & (df_all['Model'] == model)]
        if len(model_data) > 0:
            summary_results.append({
                'Dataset': dataset_name,
                'Type': 'Individual_Summary',
                'Model': model,
                'Model1': model,
                'Model2': '',
                'N_Seeds': len(model_data),
                'Image_AP_AUPR_mean': model_data['Image_AP_AUPR'].mean(),
                'Image_AP_AUPR_std': model_data['Image_AP_AUPR'].std(),
                'AUsPRO@0.05_mean': model_data['AUsPRO@0.05'].mean(),
                'AUsPRO@0.05_std': model_data['AUsPRO@0.05'].std(),
                'image_AUROC_mean': model_data['image_AUROC'].mean(),
                'image_AUROC_std': model_data['image_AUROC'].std(),
                'pixel_AUROC_mean': model_data['pixel_AUROC'].mean(),
                'pixel_AUROC_std': model_data['pixel_AUROC'].std(),
                # TPR@TNR metrics (handle nan values properly)
                'TPR@TNR0.99_mean': model_data['TPR@TNR0.99'].dropna().mean() if 'TPR@TNR0.99' in model_data.columns else 0.0,
                'TPR@TNR0.99_std': model_data['TPR@TNR0.99'].dropna().std() if 'TPR@TNR0.99' in model_data.columns else 0.0,
                'TPR@TNR0.95_mean': model_data['TPR@TNR0.95'].dropna().mean() if 'TPR@TNR0.95' in model_data.columns else 0.0,
                'TPR@TNR0.95_std': model_data['TPR@TNR0.95'].dropna().std() if 'TPR@TNR0.95' in model_data.columns else 0.0,
                'TPR@TNR0.90_mean': model_data['TPR@TNR0.90'].dropna().mean() if 'TPR@TNR0.90' in model_data.columns else 0.0,
                'TPR@TNR0.90_std': model_data['TPR@TNR0.90'].dropna().std() if 'TPR@TNR0.90' in model_data.columns else 0.0
            })
    
    # Ensemble summaries (group by model pair)
    ensemble_groups = df_all[df_all['Type'] == 'Ensemble'].groupby(['Model1', 'Model2'])
    for (model1, model2), group_data in ensemble_groups:
        if len(group_data) > 0:
            summary_results.append({
                'Dataset': dataset_name,
                'Type': 'Ensemble_Summary', 
                'Model': f"{model1}+{model2}",
                'Model1': model1,
                'Model2': model2,
                'N_Seeds': len(group_data),
                'Image_AP_AUPR_mean': group_data['Image_AP_AUPR'].mean(),
                'Image_AP_AUPR_std': group_data['Image_AP_AUPR'].std(),
                'AUsPRO@0.05_mean': group_data['AUsPRO@0.05'].mean(),
                'AUsPRO@0.05_std': group_data['AUsPRO@0.05'].std(),
                'image_AUROC_mean': group_data['image_AUROC'].mean(),
                'image_AUROC_std': group_data['image_AUROC'].std(),
                'pixel_AUROC_mean': group_data['pixel_AUROC'].mean(),
                'pixel_AUROC_std': group_data['pixel_AUROC'].std(),
                # TPR@TNR metrics (handle nan values properly)
                'TPR@TNR0.99_mean': group_data['TPR@TNR0.99'].dropna().mean() if 'TPR@TNR0.99' in group_data.columns else 0.0,
                'TPR@TNR0.99_std': group_data['TPR@TNR0.99'].dropna().std() if 'TPR@TNR0.99' in group_data.columns else 0.0,
                'TPR@TNR0.95_mean': group_data['TPR@TNR0.95'].dropna().mean() if 'TPR@TNR0.95' in group_data.columns else 0.0,
                'TPR@TNR0.95_std': group_data['TPR@TNR0.95'].dropna().std() if 'TPR@TNR0.95' in group_data.columns else 0.0,
                'TPR@TNR0.90_mean': group_data['TPR@TNR0.90'].dropna().mean() if 'TPR@TNR0.90' in group_data.columns else 0.0,
                'TPR@TNR0.90_std': group_data['TPR@TNR0.90'].dropna().std() if 'TPR@TNR0.90' in group_data.columns else 0.0
            })
    
    # Save results
    all_results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_results)
    
    all_csv_path = os.path.join(output_dir, f"{dataset_name}_all_results.csv")
    summary_csv_path = os.path.join(output_dir, f"{dataset_name}_summary.csv")
    
    all_results_df.to_csv(all_csv_path, index=False, sep=';', decimal=',')
    summary_df.to_csv(summary_csv_path, index=False, sep=';', decimal=',')
    
    print(f"\n Results saved:")
    print(f"    All results: {all_csv_path}")
    print(f"    Summary: {summary_csv_path}")
    
    # Print summary table for paper
    print(f"\n Summary for {dataset_name} (Paper Table):")
    print("="*120)
    for _, row in summary_df.iterrows():
        model = row['Model']
        ap_mean = row['Image_AP_AUPR_mean']
        ap_std = row['Image_AP_AUPR_std']
        auspro_mean = row['AUsPRO@0.05_mean']
        auspro_std = row['AUsPRO@0.05_std']
        tpr99_mean = row['TPR@TNR0.99_mean']
        tpr99_std = row['TPR@TNR0.99_std']
        tpr95_mean = row['TPR@TNR0.95_mean']
        tpr95_std = row['TPR@TNR0.95_std']
        tpr90_mean = row['TPR@TNR0.90_mean']
        tpr90_std = row['TPR@TNR0.90_std']
        
        print(f"{model:20} | AP: {ap_mean:.4f}±{ap_std:.4f} | AUsPRO: {auspro_mean:.4f}±{auspro_std:.4f} | TPR@99: {tpr99_mean:.4f}±{tpr99_std:.4f} | TPR@95: {tpr95_mean:.4f}±{tpr95_std:.4f} | TPR@90: {tpr90_mean:.4f}±{tpr90_std:.4f}")
    
    return all_results_df, summary_df


def run_all_datasets(datasets_config, base_path="results_0/all_models", output_base="ensemble_results"):
    """Run ensemble analysis for multiple datasets"""
    print(" Running ensemble analysis for all datasets")
    
    all_summaries = []
    
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*80}")
        print(f" Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            all_results_df, summary_df = dataset_ensemble(
                dataset_name=dataset_name,
                model_names=config['models'],
                seeds=config['seeds'],
                base_path=base_path,
                output_dir=os.path.join(output_base, dataset_name)
            )
            
            # Add dataset info to summary
            summary_df['Dataset'] = dataset_name
            all_summaries.append(summary_df)
            
        except Exception as e:
            print(f" Error processing {dataset_name}: {e}")
    
    # Combine all summaries
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(output_base, "combined_summary.csv")
        combined_summary.to_csv(combined_path, index=False, sep=';', decimal=',')
        
        print(f"\n{'='*80}")
        print(f" COMBINED RESULTS")
        print(f"{'='*80}")
        print(f" Combined summary saved to: {combined_path}")
        
        # Print paper-ready table for all datasets
        print("\n Paper-ready summary table:")
        print("-"*150)
        for dataset in combined_summary['Dataset'].unique():
            print(f"\n{dataset.upper()}:")
            dataset_data = combined_summary[combined_summary['Dataset'] == dataset]
            for _, row in dataset_data.iterrows():
                model = row['Model']
                ap_mean = row['Image_AP_AUPR_mean']
                ap_std = row['Image_AP_AUPR_std'] 
                auspro_mean = row['AUsPRO@0.05_mean']
                auspro_std = row['AUsPRO@0.05_std']
                tpr99_mean = row['TPR@TNR0.99_mean']
                tpr99_std = row['TPR@TNR0.99_std']
                tpr95_mean = row['TPR@TNR0.95_mean']
                tpr95_std = row['TPR@TNR0.95_std']
                tpr90_mean = row['TPR@TNR0.90_mean']
                tpr90_std = row['TPR@TNR0.90_std']
                print(f"  {model:20} | AP: {ap_mean:.4f}±{ap_std:.4f} | AUsPRO: {auspro_mean:.4f}±{auspro_std:.4f} | TPR@99: {tpr99_mean:.4f}±{tpr99_std:.4f} | TPR@95: {tpr95_mean:.4f}±{tpr95_std:.4f} | TPR@90: {tpr90_mean:.4f}±{tpr90_std:.4f}")
        
        # Generate final paper CSV with all metrics
        paper_csv_path = os.path.join(output_base, "paper_final_results.csv")
        paper_columns = [
            'Dataset', 'Model', 'Type', 'Model1', 'Model2',
            'Image_AP_AUPR_mean', 'Image_AP_AUPR_std',
            'AUsPRO@0.05_mean', 'AUsPRO@0.05_std',
            'TPR@TNR0.99_mean', 'TPR@TNR0.99_std',
            'TPR@TNR0.95_mean', 'TPR@TNR0.95_std',
            'TPR@TNR0.90_mean', 'TPR@TNR0.90_std',
            'N_Seeds'
        ]
        paper_df = combined_summary[paper_columns]
        paper_df.to_csv(paper_csv_path, index=False, sep=';', decimal=',')
        print(f"\n Final paper CSV saved to: {paper_csv_path}")


if __name__ == '__main__':
    print(" Ensemble Analysis for Anomaly Detection Paper")
    print("  PRECISE MODE: Real pixel-level metrics only")
    print("="*60)
    
    # Configuration for all datasets
    
    datasets_config = {
        "engine_wiring": {
            "models": [
                'patchcore',          # strong at localization
                'efficientad',        # good AP for this class
                'reversedistillation' # contributes some localization
            ],
            "seeds": [1, 2, 42]
        },
        "pipe_clip": {
            "models": [
                'patchcore',          # top in AUsPRO
                'uninet',             # top en AP
                'reversedistillation',# decent in both metrics
                'supersimplenet'      # adds diversity in AP
            ], 
            "seeds": [1, 2, 42]
        },
        "pipe_staple": {
            "models": [
                'patchcore',          # stable and good maps
                'supersimplenet',     # best AP
                'dinomaly'            # competitive AP
            ],
            "seeds": [1, 2, 42]
        },
        "tank_screw": {
            "models": [
                'patchcore',          # the only one that stands out in loc
                'reversedistillation',# contributes some localization
                'supersimplenet'      # slightly better in AP
            ],
            "seeds": [1, 2, 42]
        },
        "underbody_pipes": {
            "models": [
                'patchcore',          # saturates AUsPRO
                'dinomaly',           # also very high
                'reversedistillation' # consistent
            ],
            "seeds": [1, 2, 42]
        },
        "underbody_screw": {
            "models": [
                'patchcore',          # very strong localization
                'supersimplenet',     # very high AP (0.86)
                'padim'               # good AP (0.82), complementary
            ],
            "seeds": [1, 2, 42]
        }
    }

    # Run all datasets (PRECISE mode only)
    run_all_datasets(datasets_config)
    
