import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

# Configure matplotlib for academic papers
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_image_safe(path):
    """Load image safely, handle different formats."""
    try:
        if os.path.exists(path):
            return np.array(Image.open(path).convert('RGB'))
        else:
            print(f"Warning: Image not found: {path}")
            return None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_mask_safe(path):
    """Load mask safely, convert to binary."""
    try:
        if os.path.exists(path):
            mask = np.array(Image.open(path).convert('L'))
            return mask > 128  # Binary threshold
        else:
            print(f"Warning: Mask not found: {path}")
            return None
    except Exception as e:
        print(f"Error loading mask {path}: {e}")
        return None

def apply_contour_overlay(rgb_img, mask, contour_color='white', thickness=6):
    """Apply GT contour overlay to RGB image."""
    if rgb_img is None or mask is None:
        return rgb_img
    
    # Create PIL image for drawing
    pil_img = Image.fromarray(rgb_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    
    # Find contours (simplified edge detection with dilation for thickness)
    from scipy import ndimage
    # Create thicker edges by dilating the mask difference
    edges = ndimage.binary_erosion(mask) ^ mask
    # Dilate edges to make them thicker
    thick_edges = ndimage.binary_dilation(edges, iterations=thickness//2)
    edge_coords = np.where(thick_edges)
    
    # Draw contour pixels with high contrast
    color_rgb = (0, 255, 255) if contour_color == 'white' else (255, 0, 0)
    for y, x in zip(edge_coords[0], edge_coords[1]):
        # Draw thick square for better visibility
        for dy in range(-thickness//2, thickness//2 + 1):
            for dx in range(-thickness//2, thickness//2 + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < pil_img.height and 0 <= nx < pil_img.width:
                    draw.point((nx, ny), fill=color_rgb)
    
    return np.array(pil_img)

def create_figure1_overview():
    """
    Figure 1: Dataset Overview (RGB + GT)
    
    Expected folder structure:
    images_mosaic/figure1/
    ├── engine_wiring_normal.png
    ├── engine_wiring_defect.png  
    ├── engine_wiring_defect_mask.png
    ├── pipe_clip_normal.png
    ├── pipe_clip_defect.png
    ├── pipe_clip_defect_mask.png
    ├── ... (similar for all 6 classes)
    """
    
    classes = ['engine_wiring', 'pipe_clip', 'pipe_staple', 
               'tank_screw', 'underbody_pipes', 'underbody_screw']
    
    base_path = Path('images_mosaic/figure1/')
    
    # Check if folder exists
    if not base_path.exists():
        print(f"Error: Folder {base_path} not found!")
        print("Please create the folder structure as described above.")
        return None
    
    # Create figure: 2 rows × 6 columns
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.5))  # Double column width
    
    for i, class_name in enumerate(classes):
        # Load images
        normal_path = base_path / f"{class_name}_normal.png"
        defect_path = base_path / f"{class_name}_defect.png" 
        mask_path = base_path / f"{class_name}_defect_mask.png"
        
        normal_img = load_image_safe(normal_path)
        defect_img = load_image_safe(defect_path)
        defect_mask = load_mask_safe(mask_path)
        
        # Top row: Normal images
        if normal_img is not None:
            axes[0, i].imshow(normal_img)
        else:
            axes[0, i].text(0.5, 0.5, 'Missing\nNormal', ha='center', va='center',
                           transform=axes[0, i].transAxes, fontsize=8)
        
        axes[0, i].set_title(f"{class_name.replace('_', ' ').title()}\n(Normal)", 
                            fontsize=10, pad=5)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Bottom row: Defect images with GT overlay
        if defect_img is not None:
            if defect_mask is not None:
                defect_with_contour = apply_contour_overlay(defect_img, defect_mask, 
                                                           contour_color='red', thickness=6)
                axes[1, i].imshow(defect_with_contour)
            else:
                axes[1, i].imshow(defect_img)
        else:
            axes[1, i].text(0.5, 0.5, 'Missing\nDefect', ha='center', va='center',
                           transform=axes[1, i].transAxes, fontsize=8)
        
        axes[1, i].set_title("Defective", fontsize=10, pad=5)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Use tight layout for compact figure
    plt.tight_layout()
    
    # Save
    os.makedirs('output/figures', exist_ok=True)
    output_path = 'output/figures/Figure1_dataset_overview'
    plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
    
    print(f"Figure 1 saved to: {output_path}.(pdf|png|eps)")
    return fig

def create_figure2_rgb_vs_depth():
    """
    Figure 2: RGB vs Pseudo-depth
    
    Expected folder structure:
    images_mosaic/figure2/
    ├── engine_wiring_rgb.png
    ├── engine_wiring_depth.png
    ├── pipe_clip_rgb.png
    ├── pipe_clip_depth.png
    ├── tank_screw_rgb.png  
    ├── tank_screw_depth.png
    """
    
    classes = ['engine_wiring', 'pipe_clip', 'tank_screw']
    base_path = Path('images_mosaic/figure2/')
    
    # Check if folder exists
    if not base_path.exists():
        print(f"Error: Folder {base_path} not found!")
        print("Please create the folder structure as described above.")
        return None
    
    # Create figure: 3 rows × 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(6, 8))  # Single column width, tall
    
    for i, class_name in enumerate(classes):
        # Load images
        rgb_path = base_path / f"{class_name}_rgb.png"
        depth_path = base_path / f"{class_name}_depth.png"
        
        rgb_img = load_image_safe(rgb_path)
        depth_img = load_image_safe(depth_path)
        
        # Left column: RGB
        if rgb_img is not None:
            axes[i, 0].imshow(rgb_img)
        else:
            axes[i, 0].text(0.5, 0.5, 'Missing\nRGB', ha='center', va='center',
                           transform=axes[i, 0].transAxes, fontsize=8)
        
        # Right column: Pseudo-depth
        if depth_img is not None:
            # If it's grayscale depth, apply colormap
            if len(depth_img.shape) == 2 or (len(depth_img.shape) == 3 and depth_img.shape[2] == 1):
                axes[i, 1].imshow(depth_img, cmap='viridis')
            else:
                axes[i, 1].imshow(depth_img)
        else:
            axes[i, 1].text(0.5, 0.5, 'Missing\nDepth', ha='center', va='center',
                           transform=axes[i, 1].transAxes, fontsize=8)
        
        # Labels
        if i == 0:  # Header row
            axes[i, 0].set_title("RGB", fontsize=12, fontweight='bold', pad=10)
            axes[i, 1].set_title("Pseudo-Depth", fontsize=12, fontweight='bold', pad=10)
        
        # Class labels on left
        axes[i, 0].set_ylabel(class_name.replace('_', ' ').title(), 
                              fontsize=11, fontweight='bold')
        
        # Remove ticks
        for j in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    # Use tight layout for compact figure
    plt.tight_layout()
    
    # Save
    os.makedirs('output/figures', exist_ok=True)
    output_path = 'output/figures/Figure2_rgb_vs_depth'
    plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
    
    print(f"Figure 2 saved to: {output_path}.(pdf|png|eps)")
    return fig

def create_figure3_anomaly_maps():
    """
    Figure 3: Anomaly Maps (4x4 layout)
    
    Expected folder structure:
    images_mosaic/figure3/
    ├── pipe_clip_contour.png
    ├── pipe_clip_patchcore.png
    ├── pipe_clip_reversedistillation.png
    ├── pipe_clip_fastflow.png
    ├── tank_screw_fastflow_contour.png
    ├── tank_screw_patchcore.png
    ├── tank_screw_reversedistillation.png
    ├── tank_screw_fastflow.png
    ├── underbody_pipes_fastflow_contour.png
    ├── underbody_pipes_patchcore.png
    ├── underbody_pipes_reversedistillation.png
    ├── underbody_pipes_fastflow.png
    ├── underbody_screw_fastflow_contour.png
    ├── underbody_screw_patchcore.png
    ├── underbody_screw_reversedistillation.png
    └── underbody_screw_fastflow.png
    """
    
    classes = ['pipe_clip', 'tank_screw', 'underbody_pipes', 'underbody_screw']
    class_labels = ['Pipe Clip', 'Tank Screw', 'Underbody Pipes', 'Underbody Screw']
    column_labels = ['GT Contour', 'PatchCore', 'Reverse Distillation', 'UniNet']
    
    base_path = Path('images_mosaic/figure3/')
    
    # Check if folder exists
    if not base_path.exists():
        print(f"Error: Folder {base_path} not found!")
        print("Please create the folder structure as described above.")
        return None
    
    # Create figure: 4 rows × 4 columns
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # Layout for 4 columns
    
    for i, (class_name, class_label) in enumerate(zip(classes, class_labels)):
        # Determine file naming pattern for contour (some have different patterns)
        if class_name == 'pipe_clip':
            contour_path = base_path / f"{class_name}_contour.png"
        else:
            contour_path = base_path / f"{class_name}_fastflow_contour.png"
        
        # Load images for this class
        patchcore_path = base_path / f"{class_name}_patchcore.png"
        reverse_path = base_path / f"{class_name}_reversedistillation.png"
        uninet_path = base_path / f"{class_name}_uninet.png"

        image_paths = [contour_path, patchcore_path, reverse_path, uninet_path]
        
        for j, img_path in enumerate(image_paths):
            ax = axes[i, j]
            img = load_image_safe(img_path)
            
            if img is not None:
                if j == 0:  # Contour - show as is
                    ax.imshow(img)
                else:  # Heatmaps - apply colormap if grayscale
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        # Normalize to 0-1 range if needed
                        if img.max() > 1.0:
                            img_norm = img / 255.0
                        else:
                            img_norm = img
                        ax.imshow(img_norm, cmap='inferno', vmin=0, vmax=1)
                        # Add small colorbar annotation
                        ax.text(0.95, 0.95, '1', transform=ax.transAxes, fontsize=8, 
                               ha='right', va='top', color='white', fontweight='bold')
                        ax.text(0.95, 0.05, '0', transform=ax.transAxes, fontsize=8,
                               ha='right', va='bottom', color='white', fontweight='bold')
                    else:
                        ax.imshow(img)
            else:
                # Show missing placeholder
                ax.text(0.5, 0.5, f'Missing\n{column_labels[j]}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='red')
                ax.set_facecolor('lightgray')
            
            # Column headers (only first row)
            if i == 0:
                ax.set_title(column_labels[j], fontsize=16, fontweight='bold', pad=8)
            
            # Row labels (only first column)
            if j == 0:
                ax.set_ylabel(class_label, fontsize=16, fontweight='bold')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add thin border
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')
    
    # Use tight layout for compact figure
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)  # Compact spacing
    
    # Save
    os.makedirs('output/figures', exist_ok=True)
    output_path = 'output/figures/Figure3_anomaly_maps'
    plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
    
    print(f"Figure 3 saved to: {output_path}.(pdf|png|eps)")
    return fig

def create_figure4_pseudodepth_variability():
    """
    Figure 4: Pseudo-depth Variability
    
    Expected folder structure:
    images_mosaic/figure4/
    ├── pipe_staple/
    │   ├── image1.png (any names, 3 files required)
    │   ├── image2.png
    │   └── image3.png
    ├── tank_screw/
    │   ├── depth1.png (any names, 3 files required)
    │   ├── depth2.png
    │   └── depth3.png
    └── underbody_screw/
        ├── example1.png (any names, 3 files required)
        ├── example2.png
        └── example3.png
    """
    
    classes = ['pipe_staple', 'tank_screw', 'underbody_screw']
    class_labels = ['Pipe Staple', 'Tank Screw', 'Underbody Screw']
    
    base_path = Path('images_mosaic/figure4/')
    
    # Check if folder exists
    if not base_path.exists():
        print(f"Error: Folder {base_path} not found!")
        print("Please create the folder structure as described above.")
        return None
    
    # Load images for each class
    all_images = []
    for class_name in classes:
        class_path = base_path / class_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found!")
            all_images.append([None, None, None])
            continue
            
        # Get all PNG files in the class folder
        png_files = list(class_path.glob('*.png'))
        if len(png_files) < 3:
            print(f"Warning: {class_name} needs 3 PNG files, found {len(png_files)}")
            # Pad with None if not enough images
            class_images = [load_image_safe(f) if f else None for f in png_files]
            while len(class_images) < 3:
                class_images.append(None)
        else:
            # Take first 3 images for the 3×3 grid
            class_images = [load_image_safe(png_files[i]) for i in range(3)]
        
        all_images.append(class_images[:3])  # Ensure we only take 3 images per class
    
    # Create figure: 3 rows × 3 columns  
    fig, axes = plt.subplots(3, 3, figsize=(7, 9))  # Single column width, taller for 3x3
    
    for j, (class_name, class_label) in enumerate(zip(classes, class_labels)):
        for i in range(3):  # 3 rows
            ax = axes[i, j]
            
            # Get image for this position
            if j < len(all_images) and i < len(all_images[j]):
                img = all_images[j][i]
            else:
                img = None
            
            if img is not None:
                # Apply viridis colormap for depth visualization
                if len(img.shape) == 2:  # Grayscale depth
                    ax.imshow(img, cmap='viridis')
                elif len(img.shape) == 3 and img.shape[2] == 1:  # Single channel
                    ax.imshow(img.squeeze(), cmap='viridis')
                else:  # RGB or already colored
                    ax.imshow(img)
            else:
                # Show missing placeholder
                ax.text(0.5, 0.5, f'Missing\nImage {i+1}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=9, color='red')
                ax.set_facecolor('lightgray')
            
            # Column headers (only first row)
            if i == 0:
                ax.set_title(class_label, fontsize=12, fontweight='bold', pad=8)
            
            # Row labels (only first column)  
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}', fontsize=10, fontweight='bold')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add thin border
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')
    
    # Use tight layout for compact figure
    plt.tight_layout()
    
    # Save
    os.makedirs('output/figures', exist_ok=True)
    output_path = 'output/figures/Figure4_pseudodepth_variability'
    plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
    
    print(f"Figure 4 saved to: {output_path}.(pdf|png|eps)")
    return fig

def create_all_mosaics():
    """Generate all mosaic figures for the paper."""
    print("Creating mosaic figures for academic paper...")
    print("=" * 60)
    
    # Check base directory
    if not os.path.exists('images_mosaic/'):
        print("Error: 'images_mosaic/' folder not found!")
        print("\nPlease create the following folder structure:")
        print_folder_structure()
        return
    
    # Figure 3: Anomaly Maps
    try:
        if Path('images_mosaic/figure3/').exists():
            create_figure3_anomaly_maps()
            print(" Figure 3: Anomaly maps comparison created")
        else:
            print(" Figure 3 skipped: images_mosaic/figure3/ not found")
    except Exception as e:
        print(f" Error creating Figure 3: {e}")
    

def print_folder_structure():
    """Print expected folder structure."""
    print("""
EXPECTED FOLDER STRUCTURE:

images_mosaic/
├── figure1/                    # Figure 1: Dataset Overview
│   ├── engine_wiring_normal.png
│   ├── engine_wiring_defect.png
│   ├── engine_wiring_defect_mask.png
│   ├── pipe_clip_normal.png
│   ├── pipe_clip_defect.png
│   ├── pipe_clip_defect_mask.png
│   ├── pipe_staple_normal.png
│   ├── pipe_staple_defect.png
│   ├── pipe_staple_defect_mask.png
│   ├── tank_screw_normal.png
│   ├── tank_screw_defect.png
│   ├── tank_screw_defect_mask.png
│   ├── underbody_pipes_normal.png
│   ├── underbody_pipes_defect.png
│   ├── underbody_pipes_defect_mask.png
│   ├── underbody_screw_normal.png
│   ├── underbody_screw_defect.png
│   └── underbody_screw_defect_mask.png
│
├── figure2/                    # Figure 2: RGB vs Depth
│   ├── engine_wiring_rgb.png
│   ├── engine_wiring_depth.png
│   ├── pipe_clip_rgb.png
│   ├── pipe_clip_depth.png
│   ├── tank_screw_rgb.png
│   └── tank_screw_depth.png
│
├── figure3/                    # Figure 3: Anomaly Maps
│   ├── underbody_pipes_rgb.png
│   ├── underbody_pipes_mask.png
│   ├── underbody_pipes_patchcore.png
│   ├── underbody_pipes_reverse.png
│   ├── underbody_pipes_overlay.png
│   ├── engine_wiring_rgb.png
│   ├── engine_wiring_mask.png
│   ├── engine_wiring_patchcore.png
│   ├── engine_wiring_reverse.png
│   ├── engine_wiring_overlay.png
│   ├── tank_screw_rgb.png
│   ├── tank_screw_mask.png
│   ├── tank_screw_patchcore.png
│   ├── tank_screw_reverse.png
│   └── tank_screw_overlay.png
│
└── figure4/                    # Figure 4: Pseudo-depth Variability
    ├── pipe_staple/
    │   ├── any_name1.png (exactly 3 PNG files required)
    │   ├── any_name2.png
    │   └── any_name3.png
    ├── tank_screw/
    │   ├── sample1.png (exactly 3 PNG files required)
    │   ├── sample2.png
    │   └── sample3.png
    └── underbody_screw/
        ├── example1.png (exactly 3 PNG files required)
        ├── example2.png
        └── example3.png

NAMING CONVENTIONS:
- RGB images: .png or .png format
- GT masks: .png format (binary: black=normal, white=defect)
- Depth images: .png or .png (normalized 0-255 or RGB colorized)

SIZING RECOMMENDATIONS:
- All images: same dimensions (e.g., 512×512 or 256×256)
- GT masks: exact same size as corresponding RGB images
""")

def verify_images():
    """Verify that all expected images are present."""
    print("Verifying image availability...")
    
    # Figure 1 requirements
    classes = ['engine_wiring', 'pipe_clip', 'pipe_staple', 
               'tank_screw', 'underbody_pipes', 'underbody_screw']
    
    fig1_missing = []
    base_path = Path('images_mosaic/figure1/')
    
    for class_name in classes:
        for img_type in ['normal', 'defect']:
            img_path = base_path / f"{class_name}_{img_type}.png"
            if not img_path.exists():
                fig1_missing.append(str(img_path))
        
        # Check mask
        mask_path = base_path / f"{class_name}_defect_mask.png"
        if not mask_path.exists():
            fig1_missing.append(str(mask_path))
    
    # Figure 2 requirements
    classes_fig2 = ['engine_wiring','pipe_clip', 'tank_screw']
    fig2_missing = []
    base_path2 = Path('images_mosaic/figure2/')
    
    for class_name in classes_fig2:
        for img_type in ['rgb', 'depth']:
            img_path = base_path2 / f"{class_name}_{img_type}.png"
            if not img_path.exists():
                fig2_missing.append(str(img_path))
                
    # Figure 3 requirements
    classes_fig3 = ['underbody_pipes', 'engine_wiring', 'tank_screw']
    fig3_missing = []
    base_path3 = Path('images_mosaic/figure3/')
    
    for class_name in classes_fig3:
        for img_type in ['rgb', 'mask', 'patchcore', 'reverse', 'overlay']:
            img_path = base_path3 / f"{class_name}_{img_type}.png"
            if not img_path.exists():
                fig3_missing.append(str(img_path))
                
    # Figure 4 requirements
    classes_fig4 = ['pipe_staple', 'tank_screw', 'underbody_screw']
    fig4_missing = []
    base_path4 = Path('images_mosaic/figure4/')
    
    for class_name in classes_fig4:
        class_path = base_path4 / class_name
        if not class_path.exists():
            fig4_missing.append(f"Directory: {class_path}")
        else:
            # Check for exactly 3 PNG files
            png_files = list(class_path.glob('*.png'))
            if len(png_files) < 3:
                fig4_missing.append(f"{class_name}: needs 3 PNG files, found {len(png_files)}")
    
    # Report results
    print(f"Figure 1: {len(fig1_missing)} missing files")
    if fig1_missing:
        print("  Missing:", fig1_missing[:5], "..." if len(fig1_missing) > 5 else "")
    
    print(f"Figure 2: {len(fig2_missing)} missing files")  
    if fig2_missing:
        print("  Missing:", fig2_missing)
        
    print(f"Figure 3: {len(fig3_missing)} missing files")
    if fig3_missing:
        print("  Missing:", fig3_missing[:8], "..." if len(fig3_missing) > 8 else "")
        
    print(f"Figure 4: {len(fig4_missing)} missing requirements")
    if fig4_missing:
        print("  Issues:", fig4_missing)
    
    total_missing = len(fig1_missing) + len(fig2_missing) + len(fig3_missing) + len(fig4_missing)
    if total_missing == 0:
        print(" All images found! Ready to generate mosaics.")
    else:
        print(" Some images missing. Please add them before running.")
        if len(fig3_missing) > 0:
            print("\nTo generate Figure 3 images, run: python export_overlays.py")
        if len(fig4_missing) > 0:
            print("For Figure 4: Place exactly 3 PNG files in each class subfolder (pipe_staple, tank_screw, underbody_screw)")
    
    return total_missing == 0

def test_single_image(class_name='pipe_clip', img_type='normal'):
    """Test loading and displaying a single image."""
    print(f"Testing image loading: {class_name}_{img_type}")
    
    if img_type == 'normal':
        img_path = Path(f'images_mosaic/figure1/{class_name}_normal.png')
        img = load_image_safe(img_path)
        
        if img is not None:
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"Test: {class_name} normal")
            plt.axis('off')
            plt.savefig('test_image.png', dpi=150, bbox_inches='tight')
            print(" Test image saved as test_image.png")
        else:
            print(" Failed to load test image")
    
    elif img_type == 'defect_overlay':
        img_path = Path(f'images_mosaic/figure1/{class_name}_defect.png')
        mask_path = Path(f'images_mosaic/figure1/{class_name}_defect_mask.png')
        
        img = load_image_safe(img_path)
        mask = load_mask_safe(mask_path)
        
        if img is not None and mask is not None:
            img_with_contour = apply_contour_overlay(img, mask, contour_color='white', thickness=8)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(img_with_contour)
            plt.title(f"Test: {class_name} defect + GT overlay")
            plt.axis('off')
            plt.savefig('test_defect_overlay.png', dpi=150, bbox_inches='tight')
            print(" Test defect overlay saved as test_defect_overlay.png")
        else:
            print(" Failed to load test images or mask")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'verify':
            verify_images()
        elif sys.argv[1] == 'structure':
            print_folder_structure()
        elif sys.argv[1] == 'test':
            test_class = sys.argv[2] if len(sys.argv) > 2 else 'pipe_clip'
            test_type = sys.argv[3] if len(sys.argv) > 3 else 'normal'
            test_single_image(test_class, test_type)
        else:
            print("Usage:")
            print("  python mosaics.py verify    # Check images")
            print("  python mosaics.py structure # Show expected folders")
            print("  python mosaics.py test [class] [type] # Test single image")
            print("  python mosaics.py          # Generate all mosaics")
    else:
        create_all_mosaics()
