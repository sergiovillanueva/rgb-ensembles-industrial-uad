import os
import json
import pickle
import numpy as np
from PIL import Image

try:
    from matplotlib import colormaps as mpl_cmaps
except Exception:
    mpl_cmaps = None


def load_results(model_dir):
    with open(os.path.join(model_dir, 'test_results.json'), 'r') as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, 'anomaly_maps.pkl'), 'rb') as f:
        maps = pickle.load(f)
    masks = []
    try:
        with open(os.path.join(model_dir, 'gt_masks.pkl'), 'rb') as f:
            masks = pickle.load(f)
    except FileNotFoundError:
        masks = []
    return meta, maps, masks


def list_abnormal_images(dataset_root, specific_classes=None):
    test_dir = os.path.join(dataset_root, 'test')
    items = []
    items_by_class = {}
    
    for cls in sorted(os.listdir(test_dir)):
        if cls.lower() == 'good':
            continue
        if specific_classes and cls not in specific_classes:
            continue
            
        cls_dir = os.path.join(test_dir, cls)
        class_items = []
        for n in sorted(os.listdir(cls_dir)):
            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(cls_dir, n)
                items.append(full_path)
                class_items.append(full_path)
        items_by_class[cls] = class_items
        
    return items, items_by_class


def to_uint8_img(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def apply_colormap(amap, cmap_name='inferno'):
    amap = np.asarray(amap, dtype=np.float32)
    a_min, a_max = float(amap.min()), float(amap.max())
    d = (a_max - a_min) if (a_max - a_min) != 0 else 1e-8
    norm = (amap - a_min) / d
    if mpl_cmaps is None:
        # simple red overlay if matplotlib is unavailable
        heat = np.zeros((amap.shape[0], amap.shape[1], 3), dtype=np.float32)
        heat[..., 0] = norm
        return heat
    colors = mpl_cmaps.get_cmap(cmap_name)(norm)[..., :3]
    return colors.astype(np.float32)


def overlay_image(img, heat, alpha):
    img = np.asarray(img, dtype=np.float32) / 255.0
    if heat.shape[:2] != img.shape[:2]:
        heat = np.array(Image.fromarray((heat*255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR)) / 255.0
    out = (1 - alpha) * img + alpha * heat
    return to_uint8_img(out)


def draw_contour_over_rgb(rgb_img, gt_mask, color=(0, 255, 255), thickness=4):
    """Draw a thick contour of gt_mask on top of rgb_img.
    - rgb_img: HxWx3 uint8
    - gt_mask: HxW {0,1} or {0,255}
    """
    from PIL import ImageDraw
    from scipy import ndimage

    rgb = np.asarray(rgb_img, dtype=np.uint8).copy()
    if gt_mask is None:
        return rgb
    mask = np.asarray(gt_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bin = (mask > 0).astype(np.uint8)
    # Edge and dilation for thickness
    edges = ndimage.binary_erosion(mask_bin) ^ mask_bin
    edges = ndimage.binary_dilation(edges, iterations=max(1, thickness // 2))
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    ys, xs = np.where(edges)
    for y, x in zip(ys, xs):
        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < pil_img.height and 0 <= nx < pil_img.width:
                    draw.point((nx, ny), fill=tuple(int(c) for c in color))
    return np.array(pil_img)


def export_overlays(model_dir, dataset_root=None, out=None, num=3, alpha=0.5, cmap='inferno', specific_classes=None):
    meta, maps, masks = load_results(model_dir)
    labels = np.array(meta.get('labels', []), dtype=int)

    # infer dataset root from model_dir if not provided
    if dataset_root is None:
        # results_*/<dt>/<dataset>/<seed>/<model>
        parts = os.path.normpath(model_dir).split(os.sep)
        dataset = parts[-3] if len(parts) >= 3 else None
        if dataset is None:
            raise SystemExit('dataset_root required')
        dataset_root = os.path.join('data', dataset)

    abn_images, abn_by_class = list_abnormal_images(dataset_root, specific_classes)
    if len(abn_images) == 0:
        raise SystemExit('no abnormal images found')

    out_dir = out or os.path.join(model_dir, 'paper_overlays')
    os.makedirs(out_dir, exist_ok=True)

    abn_idx = np.where(labels == 1)[0]
    if len(abn_idx) == 0:
        abn_idx = np.arange(min(len(maps), len(abn_images)))

    used = 0
    for k, mi in enumerate(abn_idx):
        if mi >= len(maps):
            break
        img_path = abn_images[k % len(abn_images)]
        img = Image.open(img_path).convert('RGB')
        amap = maps[mi]
        # resize image to map size
        h, w = np.asarray(amap).shape[-2:]
        img_res = img.resize((w, h), Image.BILINEAR)
        heat = apply_colormap(amap, cmap)
        over = overlay_image(img_res, heat, alpha)

        base = os.path.splitext(os.path.basename(img_path))[0]
        # Use mi (map index) instead of k to avoid overwriting files
        Image.fromarray(over).save(os.path.join(out_dir, f'{mi:03d}_{base}_overlay.png'))
        Image.fromarray(to_uint8_img(heat)).save(os.path.join(out_dir, f'{mi:03d}_{base}_heat.png'))
        img_res.save(os.path.join(out_dir, f'{mi:03d}_{base}_img.png'))

        # Save gt mask if available
        if mi < len(masks) and masks[mi] is not None:
            gtm = np.asarray(masks[mi]).astype(np.uint8)
            if gtm.ndim == 3:
                gtm = gtm[0]
            # resize mask to map size
            gtm_img = Image.fromarray((gtm * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            gtm_img.save(os.path.join(out_dir, f'{mi:03d}_{base}_mask.png'))
        used += 1
        if used >= num:
            break

    print(f'saved {used} overlays to {out_dir}')


def build_model_dir(results_root, dataset, seed, model):
    return os.path.join(results_root, dataset, str(seed), model)


def export_paper_overlays(dataset, seed, model, results_root='results_0/all_models',
                          data_root='data', out=None, max_per_defect=1,
                          alpha=0.5, cmap='inferno',
                          contour_color=(255, 0, 0), contour_thickness=3):
    """Export overlay and GT-contour images for a given dataset/seed/model.
    Saves two images per selected abnormal image: *_overlay.png and *_contour.png.
    """
    model_dir = build_model_dir(results_root, dataset, seed, model)
    if not os.path.exists(model_dir):
        raise SystemExit(f'model_dir not found: {model_dir}')

    # Load results (maps, masks, labels)
    meta, maps, masks = load_results(model_dir)
    labels = np.array(meta.get('labels', []), dtype=int)
    abn_idx = np.where(labels == 1)[0]
    if len(abn_idx) == 0:
        abn_idx = np.arange(min(len(maps), labels.size))

    # List all abnormal images on disk (filesystem order)
    dataset_root = os.path.join(data_root, dataset)
    abn_images, abn_by_class = list_abnormal_images(dataset_root)
    if len(abn_images) == 0:
        raise SystemExit(f'no abnormal images found in {dataset_root}')

    # Map image index -> map index using abn_idx convention
    # abn_images[k] corresponds to maps[abn_idx[k]]
    def get_map_by_image_index(k):
        mi = abn_idx[k] if k < len(abn_idx) else k
        if mi >= len(maps):
            return None
        return maps[mi], (masks[mi] if mi < len(masks) else None)

    # Defect folders per dataset (strip 'test/' prefix)
    defect_map = {
        'engine_wiring': ['blue_hoop', 'cardboard', 'fastening', 'obstruction'],
        'pipe_clip': ['operator', 'unclipped'],
        'pipe_staple': ['missing'],
        'tank_screw': ['missing'],
        'underbody_pipes': ['obstruction', 'operator'],
        'underbody_screw': ['missing'],
    }
    defects = defect_map.get(dataset, [])

    # Output directory
    out_dir = out or os.path.join('images_mosaic', 'paper_overlays', dataset, str(seed), model)
    os.makedirs(out_dir, exist_ok=True)

    used = 0
    for defect in defects:
        # abn_by_class keys correspond to folder names directly under test/
        cls_images = abn_by_class.get(defect, [])
        if len(cls_images) == 0:
            continue
        for img_path in cls_images[:max_per_defect]:
            try:
                # Find image index k in abn_images list
                k = abn_images.index(img_path)
            except ValueError:
                continue
            mm = get_map_by_image_index(k)
            if mm is None:
                continue
            amap, gtmask_from_pkl = mm

            # Load original image
            img = Image.open(img_path).convert('RGB')
            h, w = np.asarray(amap).shape[-2:]
            img_res = img.resize((w, h), Image.BILINEAR)
            heat = apply_colormap(amap, cmap)
            over = overlay_image(img_res, heat, alpha)

            # Load GT mask from dataset (prefer ground_truth on disk)
            # Convert test path to ground_truth path
            gt_path = img_path.replace(os.sep + 'test' + os.sep, os.sep + 'ground_truth' + os.sep)
            if os.path.exists(gt_path):
                gt_mask_img = Image.open(gt_path).convert('L')
                gt_mask_arr = np.array(gt_mask_img.resize((w, h), Image.NEAREST))
            else:
                gt_mask_arr = (gtmask_from_pkl * 255).astype(np.uint8) if gtmask_from_pkl is not None else None

            contour = draw_contour_over_rgb(
                np.array(img_res), gt_mask_arr,
                color=contour_color, thickness=contour_thickness
            )

            base = os.path.splitext(os.path.basename(img_path))[0]
            Image.fromarray(over).save(os.path.join(out_dir, f'{base}_{dataset}_{model}_overlay.png'))
            Image.fromarray(contour).save(os.path.join(out_dir, f'{base}_{dataset}_{model}_contour.png'))
            used += 2

    print(f'paper overlays saved to {out_dir} (files: {used})')

def export_figure3_images():
    """Export images specifically for Figure 3: anomaly maps comparison."""
    target_classes = ['underbody_pipes', 'engine_wiring', 'tank_screw']  # easy, medium, hard
    models = {
        'patchcore': 'patchcore',  # Model name to directory name mapping
        'padim': 'padim'
    }
    
    # Base directory containing trained model outputs
    base_results_dir = 'results_0/all_models'
    
    print("Generating images for Figure 3...")
    print("Target classes:", target_classes)
    print("Models:", list(models.keys()))
    
    for model_name, model_pattern in models.items():
        print(f"\nProcessing model: {model_name}")
        
        for class_name in target_classes:
            # Resolve model directory path from components
            model_dir = os.path.join(base_results_dir, class_name, '0', model_pattern)
            
            if not os.path.exists(model_dir):
                print(f"  Warning: {model_dir} not found, trying different patterns...")
                # Try alternative patterns
                alternatives = [
                    os.path.join(base_results_dir, class_name, '0', model_name),
                    os.path.join(base_results_dir, class_name, model_pattern),
                    os.path.join(base_results_dir, class_name, model_name)
                ]
                found = False
                for alt in alternatives:
                    if os.path.exists(alt):
                        model_dir = alt
                        found = True
                        break
                
                if not found:
                    print(f"  Skipping {model_name} for {class_name} - directory not found")
                    continue
            
            print(f"  Exporting: {class_name} with {model_name}")
            
            # Export overlays for this class
            out_dir = f'images_mosaic/figure3_raw/{model_name}_{class_name}'
            try:
                export_overlays(
                    model_dir=model_dir,
                    dataset_root=None,  # will be inferred
                    out=out_dir,
                    num=2,  # export 5 examples per class
                    alpha=0.5,
                    cmap='inferno',
                    specific_classes=[class_name]
                )
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    print("\n" + "="*60)
    print("Raw images exported to images_mosaic/figure3_raw/")
    print("Next steps:")
    print("1. Review the exported images")
    print("2. Choose the best representative image for each class/model combination")
    print("3. Copy chosen images to images_mosaic/figure3/ with standardized naming")
    print("4. Run python mosaics.py to generate Figure 3")

def main():
    # Set parameters manually here
    model_dir = 'results_0/all_models/engine_wiring/42/patchcore'
    dataset_root = None  # will be inferred from model_dir
    out = None  # will use default: model_dir/paper_overlays
    num = 2
    
    export_overlays(model_dir, dataset_root, out, num)


if __name__ == '__main__':
    # Choose function to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'figure3':
        export_figure3_images()
    else:
        # Batch export for paper overlays - configure here
        datasets = ['underbody_screw', 'pipe_clip', 'tank_screw', 'underbody_pipes']
        models = ['patchcore', 'reversedistillation', 'fastflow']
        seed = 42
        out_dir = 'images_mosaic/paper_overlays_batch'
        max_per_defect = 3
        alpha = 0.5
        cmap = 'inferno'
        
        for ds in datasets:
            for m in models:
                print(f"Exporting: dataset={ds}, seed={seed}, model={m}")
                export_paper_overlays(
                    dataset=ds,
                    seed=seed,
                    model=m,
                    out=out_dir,
                    max_per_defect=max_per_defect,
                    alpha=alpha,
                    cmap=cmap
                )


