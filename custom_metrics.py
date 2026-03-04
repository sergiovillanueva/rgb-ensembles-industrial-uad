
from scipy.ndimage import label as cc_label
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from scipy.ndimage import zoom
import pickle
import torch
import os
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def compute_custom_metrics(datamodule, model, model_directory):

    model.eval()
    
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_dataloader = datamodule.test_dataloader()
    train_dataloader = datamodule.train_dataloader()

    test_scores = []
    test_labels = []
    test_anomaly_maps = []
    test_gt_masks = []
    pixel_labels = []
    train_normal_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            try:
                # Move key tensors to GPU
                if hasattr(batch.image, 'to'):
                    batch.image = batch.image.to(device)
                if hasattr(batch, 'gt_label') and hasattr(batch.gt_label, 'to'):
                    batch.gt_label = batch.gt_label.to(device)
                if hasattr(batch, 'gt_mask') and hasattr(batch.gt_mask, 'to'):
                    batch.gt_mask = batch.gt_mask.to(device)
                
                outputs = model.test_step(batch, batch_idx)
                
                # Check if outputs has expected attributes
                if not hasattr(outputs, 'pred_score'):
                    logger.error(f"Test batch {batch_idx}: outputs missing pred_score attribute. Available: {dir(outputs)}")
                    continue

                # Extract labels and scores (vectorized)
                labels = batch.gt_label.detach().cpu().numpy().reshape(-1)
                scores = outputs.pred_score.detach().cpu().numpy().reshape(-1)
                
                test_labels.append(labels)
                test_scores.append(scores)

                # Extract pixel-level data (simplified)
                if hasattr(outputs, 'anomaly_map') and outputs.anomaly_map is not None and hasattr(batch, 'gt_mask') and batch.gt_mask is not None:
                    am_np = outputs.anomaly_map.detach().cpu().numpy()
                    gm_np = batch.gt_mask.detach().cpu().numpy()

                    # Simplify shapes: handle B x H x W or B x C x H x W
                    if am_np.ndim == 4:
                        am_np = am_np[:, 0] if am_np.shape[1] == 1 else am_np.mean(axis=1)
                    if gm_np.ndim == 4:
                        gm_np = gm_np[:, 0] if gm_np.shape[1] == 1 else gm_np.max(axis=1)

                    # Align anomaly map to mask size per sample
                    b = min(am_np.shape[0], gm_np.shape[0])
                    for i in range(b):
                        amap_i = am_np[i].astype('float32')
                        gtm_i = (gm_np[i] > 0).astype('uint8')
                        if amap_i.shape != gtm_i.shape:
                            try:
                                th, tw = gtm_i.shape[-2], gtm_i.shape[-1]
                                h, w = amap_i.shape[-2], amap_i.shape[-1]
                                zy, zx = th / h, tw / w
                                # Anomaly map: bilinear interpolation (order=1)
                                amap_i = zoom(amap_i, (zy, zx), order=1).astype('float32')
                            except Exception as rez_e:
                                logger.error(f"Resize failed for sample {i}: {rez_e}")
                        # Safety check: ensure map and mask have same shape
                        assert amap_i.shape == gtm_i.shape, f"Shape mismatch: amap {amap_i.shape} vs mask {gtm_i.shape}"
                        test_anomaly_maps.append(amap_i)
                        test_gt_masks.append(gtm_i)
                    pixel_labels.extend(labels.astype(int)[:b])
            except Exception as e:
                # Special handling for dinomaly reshape errors
                if "shape" in str(e) and "invalid for input of size" in str(e):
                    if hasattr(batch, 'image') and hasattr(batch.image, 'shape'):
                        logger.error(f"DINOMALY RESHAPE ERROR in test batch {batch_idx}: {e}")
                        logger.error(f"Image shape was: {batch.image.shape}")
                        logger.error(f"Image size: {batch.image.numel() if hasattr(batch.image, 'numel') else 'unknown'}")
                    else:
                        logger.error(f"DINOMALY RESHAPE ERROR in test batch {batch_idx}: {e} (no image shape available)")
                else:
                    logger.error(f"Error in test batch {batch_idx}: {e}")

        # Train (only normal): scores for thresholds
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Move key tensors to GPU
                if hasattr(batch.image, 'to'):
                    batch.image = batch.image.to(device)
                if hasattr(batch, 'gt_label') and hasattr(batch.gt_label, 'to'):
                    batch.gt_label = batch.gt_label.to(device)
                if hasattr(batch, 'gt_mask') and hasattr(batch.gt_mask, 'to'):
                    batch.gt_mask = batch.gt_mask.to(device)
                    
                outputs = model.test_step(batch, batch_idx)
                
                # Check if outputs has expected attributes  
                if not hasattr(outputs, 'pred_score'):
                    logger.error(f"Train batch {batch_idx}: outputs missing pred_score attribute. Available: {dir(outputs)}")
                    continue
                
                scores = outputs.pred_score.detach().cpu().numpy().reshape(-1)
                if hasattr(batch, 'gt_label'):
                    labels_train = batch.gt_label.detach().cpu().numpy().reshape(-1).astype(int)
                    scores = scores[labels_train == 0]
                if scores.size > 0:
                    train_normal_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in train batch {batch_idx}: {e}")
        
    # Critical checkpoint: verify we have required data
    if len(test_scores) == 0 or len(test_labels) == 0 or len(train_normal_scores) == 0:
        logger.error(f"CRITICAL: Missing required data - test_scores: {len(test_scores)}, test_labels: {len(test_labels)}, train_normal_scores: {len(train_normal_scores)}")
        logger.error("This will cause metrics to return None values")
        return {
            'image_AUROC': None,
            'pixel_AUROC': None,
            'Image_AP_AUPR': None,
            'TPR@TNR0.99': None,
            'TPR@TNR0.95': None,
            'TPR@TNR0.90': None
        }

    # Concatenate all batches efficiently
    test_scores = np.concatenate(test_scores)
    test_labels = np.concatenate(test_labels)
    train_normal_scores = np.concatenate(train_normal_scores)

    try:
        # Anomalib always returns higher score = more anomalous, no inversion needed
        auroc = roc_auc_score(test_labels, test_scores)
        image_ap = average_precision_score(test_labels, test_scores)

        # Z-score normalization using train-normal
        mu = float(np.mean(train_normal_scores))
        sigma = float(np.std(train_normal_scores))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        z_train_normal = (train_normal_scores - mu) / sigma
        z_test_all = (test_scores - mu) / sigma
        test_normal_mask = (test_labels == 0)
        test_anom_mask = (test_labels == 1)
        z_test_normal = z_test_all[test_normal_mask]
        z_test_anom = z_test_all[test_anom_mask]

        # TPR@TNR with thresholds from z_train_normal
        tnr_targets = [0.99, 0.95, 0.90]
        tpr_metrics = {}
        z_thresholds = {}

        for tnr_target in tnr_targets:
            thr = np.quantile(z_train_normal, tnr_target)
            z_thresholds[f"{tnr_target:.2f}"] = float(thr)
            tnr_train = float(((z_train_normal <= thr).sum()) / max(1, len(z_train_normal)))

            tnr_test_value = None
            if z_test_normal.size > 0:
                tnr_test_value = float((z_test_normal <= thr).mean())

            tpr_value = None
            if z_test_anom.size > 0:
                tpr_value = float((z_test_anom > thr).mean())

            # comparability
            tol = 0.03 if abs(tnr_target - 0.99) < 1e-9 else 0.05
            comparable = (tnr_test_value is not None) and (abs(tnr_test_value - tnr_target) <= tol)
            
            # Calculate ΔTNR for traceability
            delta_tnr = (tnr_test_value - tnr_target) if tnr_test_value is not None else None

            tpr_metrics[f'TPR@TNR{tnr_target:.2f}'] = tpr_value if tpr_value is not None else None
            tpr_metrics[f'TNR_test@TNR{tnr_target:.2f}'] = tnr_test_value if tnr_test_value is not None else None
            tpr_metrics[f'Comparable@TNR{tnr_target:.2f}'] = bool(comparable)
            tpr_metrics[f'delta_TNR_{tnr_target:.2f}'] = delta_tnr
            print(f"  Target TNR: {tnr_target:.2f}, Train TNR: {tnr_train:.4f}, Test TNR: {tnr_test_value if tnr_test_value is not None else np.nan:.4f}, Test TPR: {tpr_value if tpr_value is not None else np.nan:.4f}, thr(z): {thr:.4f}")

        # monotonicity sanity
        try:
            tprs = [tpr_metrics.get(f'TPR@TNR{t:.2f}') for t in tnr_targets]
            tnrs = [tpr_metrics.get(f'TNR_test@TNR{t:.2f}') for t in tnr_targets]
            if all(v is not None for v in tprs) and all(v is not None for v in tnrs):
                mono_ok = (tprs[0] <= tprs[1] <= tprs[2]) and (tnrs[0] >= tnrs[1] >= tnrs[2])
                tpr_metrics['Monotonic_OK'] = bool(mono_ok)
                if not mono_ok:
                    print("   Monotonicity check failed. Review orientation/normalization.")
        except Exception:
            pass

        # Comparable semaphore (green/amber/red)
        comparable_flags = [tpr_metrics.get(f'Comparable@TNR{t:.2f}', False) for t in tnr_targets]
        comparable_count = sum(comparable_flags)
        if comparable_count == 3:
            semaphore = 'green'  # All comparable
        elif comparable_count > 0:
            semaphore = 'amber'  # Some comparable
        else:
            semaphore = 'red'    # None comparable
        tpr_metrics['Comparable_semaphore'] = semaphore
        print(f"   Comparable semaphore: {semaphore} ({comparable_count}/3 τ values comparable)")

        # Quick NOK map sanity check
        try:
            if len(test_anomaly_maps) > 0 and len(test_anomaly_maps) == len(test_gt_masks):
                nok_idx = np.where(np.asarray(pixel_labels, dtype=int) == 1)[0]
                if len(nok_idx) > 0:
                    diffs = []
                    for idx in nok_idx:
                        amap = test_anomaly_maps[idx]
                        gtm = test_gt_masks[idx]
                        if amap.shape != gtm.shape:
                            print(f"[WARN] NOK shape mismatch {idx}: {amap.shape} vs {gtm.shape}")
                            continue
                        in_mask = gtm.astype(bool)
                        out_mask = ~in_mask
                        in_mean = float(amap[in_mask].mean()) if in_mask.any() else 0.0
                        out_mean = float(amap[out_mask].mean()) if out_mask.any() else 0.0
                        diffs.append(in_mean - out_mean)
                    if len(diffs) > 0:
                        diffs_np = np.asarray(diffs)
                        print(f"[NOK maps] mean(in-out)={np.mean(diffs_np):.4f}, >0 frac={(diffs_np>0).mean():.2f}")
                else:
                    print("[INFO] No NOK samples with maps.")
            else:
                print(f"[INFO] No maps/masks for sPRO. maps={len(test_anomaly_maps)}, masks={len(test_gt_masks)}")
        except Exception as e:
            print(f"[INFO] NOK sanity skipped: {e}")

        # sPRO/AUsPRO (optional, if maps and masks available)
        # Following MVTec/AutoVI protocol: thresholds sweep over ALL test images,
        # FPR computed on normal test images, PRO computed on abnormal test images
        spro_metrics = {}
        try:
            if len(test_anomaly_maps) == len(test_gt_masks) and len(test_anomaly_maps) > 0:
                alpha = 0.05
                image_labels = np.asarray(pixel_labels, dtype=int)
                normals_idx = np.where(image_labels == 0)[0]
                abnormals_idx = np.where(image_labels == 1)[0]

                if len(normals_idx) > 0 and len(abnormals_idx) > 0:
                    # Compute threshold range over ALL test images (standard protocol)
                    t_min = np.inf
                    t_max = -np.inf
                    for idx in range(len(test_anomaly_maps)):
                        amap = test_anomaly_maps[idx]
                        t_min = min(t_min, float(amap.min()))
                        t_max = max(t_max, float(amap.max()))

                    # Validate threshold range
                    if not (np.isfinite(t_min) and np.isfinite(t_max)):
                        logger.error(f"Invalid threshold range: t_min={t_min}, t_max={t_max}")
                        raise ValueError("Non-finite threshold range")

                    if t_max <= t_min:
                        logger.error(f"Degenerate threshold range: t_max={t_max} <= t_min={t_min}")
                        raise ValueError("Degenerate threshold range")

                    # 300 thresholds from high to low for smooth curve
                    thresholds = np.linspace(t_max, t_min, num=300, dtype='float32')
                    print(f"   Threshold range: [{t_min:.4f}, {t_max:.4f}]")

                    # Calculate FPR on normal test images only
                    total_normal_pixels = 0
                    normal_fp_counts = np.zeros(len(thresholds), dtype='float64')
                    for idx in normals_idx:
                        amap = test_anomaly_maps[idx]
                        total_normal_pixels += amap.size
                        # Vectorized: for each threshold, count pixels >= threshold (false positives)
                        normal_fp_counts += (amap[None, :, :] >= thresholds[:, None, None]).sum(axis=(1, 2))

                    fprs = normal_fp_counts / max(1, total_normal_pixels)

                    # PRO calculation on abnormal test images
                    # Pre-compute connected components for efficiency
                    abnormal_components = []
                    for idx in abnormals_idx:
                        gtm = test_gt_masks[idx]
                        comp_map, num = cc_label(gtm.astype('uint8'))
                        components = []
                        for comp_id in range(1, num + 1):
                            comp_mask = (comp_map == comp_id)
                            comp_size = comp_mask.sum()
                            if comp_size > 0:
                                components.append((comp_mask, comp_size))
                        abnormal_components.append((idx, components))

                    # Calculate PRO for each threshold
                    pros = np.zeros(len(thresholds), dtype='float64')
                    for t_i, thr in enumerate(thresholds):
                        pro_vals = []
                        for idx, components in abnormal_components:
                            amap = test_anomaly_maps[idx]
                            pred_mask = (amap >= thr)
                            for comp_mask, comp_size in components:
                                tp = (pred_mask & comp_mask).sum()
                                pro_vals.append(tp / comp_size)

                        if len(pro_vals) > 0:
                            pros[t_i] = np.mean(pro_vals)

                    # Limit to FPR <= alpha and sort by FPR for proper curve
                    valid = fprs <= alpha
                    if np.any(valid):
                        # Sort by FPR for proper integration
                        order = np.argsort(fprs[valid])
                        xf = fprs[valid][order]
                        yf = pros[valid][order]

                        # Ensure curve starts at (0, 0) for proper AUC calculation
                        if xf[0] > 0:
                            xf = np.concatenate([[0], xf])
                            yf = np.concatenate([[0], yf])

                        # AUsPRO: Area under PRO curve normalized by alpha
                        area = np.trapezoid(yf, xf) / alpha

                        # sPRO@0.05: PRO value at FPR closest to (but not exceeding) alpha
                        spro_edge = float(yf[-1])

                        spro_metrics = {
                            'sPRO@0.05': spro_edge,
                            'AUsPRO@0.05': float(area)
                        }
                        print(f"   sPRO@0.05: {spro_edge:.4f}, AUsPRO@0.05: {area:.4f}")
                        print(f"   PRO curve points: {len(xf)}, FPR range in curve: [{xf[0]:.6f}, {xf[-1]:.6f}]")
                    else:
                        logger.error(f"No valid FPR points <= {alpha} for sPRO calculation")
                        logger.error(f"FPR range: [{fprs.min():.6f}, {fprs.max():.6f}]")
                else:
                    logger.error(f"No normal ({len(normals_idx)}) or abnormal ({len(abnormals_idx)}) samples for sPRO")
        except Exception as e:
            logger.error(f"sPRO/AUsPRO not computed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Pixel-AUROC (optional)
        pixel_auroc = None
        try:
            if len(test_anomaly_maps) == len(test_gt_masks) and len(test_anomaly_maps) > 0:
                # Flatten all anomaly maps and gt masks for pixel-level AUROC
                all_pixel_scores = []
                all_pixel_labels = []
                for i, (amap, gtm) in enumerate(zip(test_anomaly_maps, test_gt_masks)):
                    all_pixel_scores.append(amap.ravel())
                    all_pixel_labels.append(gtm.ravel())
                
                if len(all_pixel_scores) > 0:
                    all_pixel_scores = np.concatenate(all_pixel_scores)
                    all_pixel_labels = np.concatenate(all_pixel_labels)
                    
                    # Calculate pixel-level AUROC
                    if len(np.unique(all_pixel_labels)) > 1:  # Need both classes
                        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
                        print(f"   Pixel-AUROC: {pixel_auroc:.4f}")
        except Exception as e:
            logger.error(f"Pixel-AUROC not computed: {e}")

        # Save test results for ensemble
        test_data = {
            'scores': test_scores.tolist(),
            'labels': test_labels.tolist(),
            'train_normal_scores': train_normal_scores.tolist(),  # For normalization in ensemble
            'image_ap': float(image_ap),
            'auroc': float(auroc),
            'zscore_mu': mu,
            'zscore_sigma': sigma,
            'calibrated_after_fusion': False,  # Flag for ensemble tracking
            'z_thresholds': z_thresholds,
            **{k: (
                v if isinstance(v, (bool, np.bool_)) else (
                    None if v is None else (
                        float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                    )
                )
            ) for k, v in tpr_metrics.items()},
            **{k: float(v) for k, v in spro_metrics.items()}
        }
        
        if pixel_auroc is not None:
            test_data['pixel_auroc'] = float(pixel_auroc)
        
        # Save test data
        test_results_path = os.path.join(model_directory, 'test_results.json')
        with open(test_results_path, 'w') as f:
            import json
            json.dump(test_data, f, indent=2)
        
        # Save anomaly maps if available
        if len(test_anomaly_maps) > 0:
            anomaly_maps_path = os.path.join(model_directory, 'anomaly_maps.pkl')
            gt_masks_path = os.path.join(model_directory, 'gt_masks.pkl')
            with open(anomaly_maps_path, 'wb') as f:
                pickle.dump(test_anomaly_maps, f)
            with open(gt_masks_path, 'wb') as f:
                pickle.dump(test_gt_masks, f)
        else:
            logger.error(f"No anomaly maps to save")
        
        # Calculate basic metrics too (avoid duplicate engine.test())
        result_metrics = {
            'image_AUROC': float(auroc),
            'pixel_AUROC': float(pixel_auroc) if pixel_auroc is not None else None,
            'Image_AP_AUPR': image_ap,
            **tpr_metrics,
            **spro_metrics
        }
            
        return result_metrics

    except Exception as e:
        logger.error(f"CRITICAL ERROR in compute_custom_metrics: {e}")
        logger.error(f"Error occurred for model in {model_directory}")
        print(f"Error computing metrics: {e}")
        return {
            'image_AUROC': None,
            'pixel_AUROC': None,
            'Image_AP_AUPR': None,
            'TPR@TNR0.99': None,
            'TPR@TNR0.95': None,
            'TPR@TNR0.90': None
        }