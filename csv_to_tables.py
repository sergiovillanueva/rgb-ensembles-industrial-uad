import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for academic papers
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.linewidth': 0.5,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

# Color palette for models (professional, colorblind-friendly)
MODEL_COLORS = {
    'patchcore': '#1f77b4',      # Blue
    'efficientad': '#ff7f0e',    # Orange
    'reversedistillation': '#2ca02c',  # Green
    'supersimplenet': '#d62728', # Red
    'dinomaly': '#9467bd',       # Purple
    'padim': '#8c564b',          # Brown
    'uninet': '#e377c2',         # Pink
}

# Display names for legends (preserving official capitalization)
MODEL_DISPLAY_NAMES = {
    'patchcore': 'PatchCore',
    'efficientad': 'EfficientAD',
    'reversedistillation': 'Reverse Distillation',
    'supersimplenet': 'SuperSimpleNet',
    'dinomaly': 'DiNomaly',
    'padim': 'PaDiM',
    'uninet': 'UniNet',
}

# Markers for better distinction
MODEL_MARKERS = {
    'patchcore': 'o',
    'efficientad': 's',
    'reversedistillation': '^',
    'supersimplenet': 'D',
    'dinomaly': 'v',
    'padim': 'p',
    'uninet': 'h',
}

def setup_figure_style():
    """Setup professional figure style for two-column academic papers."""
    # Two column width: ~3.5 inches, single column: ~7 inches
    return {
        'single_width': 3.5,    # Single column width in inches
        'double_width': 7.0,    # Double column width in inches  
        'height_ratio': 0.75,   # Height as ratio of width (golden ratio ≈ 0.618)
        'dpi': 300
    }

def create_f1_pareto_figure():
    """Create F1: Pareto frontier plot (AUROC vs AUsPRO)."""
    if not os.path.exists('figures/F1_pareto_auroc_auspro.csv'):
        print("Error: F1 data not found. Run data_analysis.py first.")
        return
    
    # Load data
    df = pd.read_csv('figures/F1_pareto_auroc_auspro.csv', sep=';')

    # Filter out removed models (FastFlow and U-Flow)
    df = df[~df['Model'].isin(['fastflow', 'uflow'])]
    
    # Setup figure - make it slightly wider since no title
    style = setup_figure_style()
    fig, ax = plt.subplots(figsize=(style['single_width'], 
                                   style['single_width'] * 0.85))  # Slightly wider aspect ratio
    
    # Create scatter plot
    for model in sorted(df['Model'].unique()):
        model_data = df[df['Model'] == model]
        
        color = MODEL_COLORS.get(model, '#333333')
        marker = MODEL_MARKERS.get(model, 'o')
        
        ax.scatter(model_data['AUROC'], 
                  model_data['AUsPRO'],
                  color=color,
                  marker=marker, 
                  s=16,  # Marker size
                  alpha=0.8,
                  label=MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title()),
                  edgecolors='white',
                  linewidth=0.5)
    
    # Customize axes
    ax.set_xlabel('Image AUROC', fontweight='bold')
    ax.set_ylabel('Pixel AUsPRO@0.05', fontweight='bold')
    
    # Set optimized limits for better space usage
    auroc_min, auroc_max = df['AUROC'].min(), df['AUROC'].max()
    auspro_min, auspro_max = df['AUsPRO'].min(), df['AUsPRO'].max()
    
    # Start AUROC from a reasonable minimum (not 0) for better space usage
    auroc_start = max(0.45, auroc_min - 0.02)  # Never go below 0.45 (below random)
    auroc_end = min(1.05, auroc_max + 0.02)    # Never go above 1.05
    
    ax.set_xlim(auroc_start, auroc_end)
    ax.set_ylim(max(0, auspro_min - 0.01), auspro_max + 0.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Legend with better positioning
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      frameon=True, fancybox=False, shadow=False,
                      borderaxespad=0)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.5)
    
    # Add annotation for Pareto frontier concept
    ax.annotate('Better →', xy=(0.8, 0.95), xycoords='axes fraction', 
                fontsize=7, ha='center', style='italic', color='gray')
    ax.annotate('↑\nBetter', xy=(0.12, 0.90), xycoords='axes fraction', 
                fontsize=7, ha='center', va='center', style='italic', color='gray')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    output_path = 'output/figures/F1_pareto_auroc_auspro'
    
    # Save in multiple formats
    plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
    
    print(f"F1 figure saved to: {output_path}.(pdf|png|eps)")
    
    # Show stats
    print(f"  Data points: {len(df)}")
    print(f"  Models: {len(df['Model'].unique())}")
    print(f"  Datasets: {len(df['Dataset'].unique())}")
    
    # Find Pareto efficient models
    pareto_efficient = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        avg_auroc = model_data['AUROC'].mean()
        avg_auspro = model_data['AUsPRO'].mean()
        
        # Simple Pareto efficiency: no other model dominates in both metrics
        dominated = False
        for other_model in df['Model'].unique():
            if other_model != model:
                other_data = df[df['Model'] == other_model]
                other_auroc = other_data['AUROC'].mean()
                other_auspro = other_data['AUsPRO'].mean()
                
                if other_auroc >= avg_auroc and other_auspro >= avg_auspro:
                    if other_auroc > avg_auroc or other_auspro > avg_auspro:
                        dominated = True
                        break
        
        if not dominated:
            pareto_efficient.append((model, avg_auroc, avg_auspro))
    
    print(f"  Pareto efficient models: {[m[0] for m in pareto_efficient]}")
    
    return fig

def create_all_figures():
    """Generate all figures for the paper."""
    print("Generating figures for academic paper...")
    print("=" * 50)
    
    # Check if figure data exists
    if not os.path.exists('figures/'):
        print("Error: 'figures/' folder not found.")
        print("Please run data_analysis.py first to generate figure data.")
        return
    
    # F1: Pareto frontier
    try:
        create_f1_pareto_figure()
        print(" F1: Pareto frontier plot created")
    except Exception as e:
        print(f" Error creating F1: {e}")
    
    print("=" * 50)
    print("Figure generation completed!")
    print("Figures saved to 'output/figures/' in PDF, PNG, and EPS formats")

if __name__ == '__main__':
    create_all_figures()
