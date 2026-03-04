import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for academic papers (same as csv_to_tables.py)
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

# Color palette for models (same as csv_to_tables.py)
MODEL_COLORS = {
    'patchcore': '#1f77b4',      # Blue
    'efficientad': '#ff7f0e',    # Orange  
    'reversedistillation': '#2ca02c',  # Green
    'supersimplenet': '#d62728', # Red
    'dinomaly': '#9467bd',       # Purple
    'padim': '#8c564b',          # Brown
    'uninet': '#e377c2',         # Pink
    'fastflow': '#7f7f7f',       # Gray
    'uflow': '#bcbd22'           # Olive
}

# Ensemble colors (different shades/styles)
ENSEMBLE_COLORS = {
    'patch+effad': '#1f77b4',
    'patch+revdist': '#2ca02c', 
    'patch+SSN': '#d62728',
    'patch+dinomaly': '#9467bd',
    'patch+padim': '#8c564b',
    'patch+uni': '#e377c2',
    'efficientad+reversedistillation': '#ff7f0e',
    'SSN+dinomaly': '#d62728',
    'SSN+padim': '#8c564b',
    'uninet+reversedistillation': '#e377c2',
    'dinomaly+reversedistillation': '#9467bd',
    'default': '#333333'  # Default gray for other ensembles
}

def setup_figure_style():
    """Setup professional figure style for two-column academic papers."""
    return {
        'single_width': 3.5,    # Single column width in inches
        'double_width': 7.0,    # Double column width in inches  
        'height_ratio': 0.75,   # Height as ratio of width
        'dpi': 300
    }

def load_ensemble_data():
    """Load data from ensemble analysis results"""
    data_files = {
        'summary': 'results/paper_table2_summary_with_std.csv',
        'improvements': 'results/paper_table1_significant_improvements.csv',
        'best_per_dataset': 'results/paper_table3_best_per_dataset.csv'
    }
    
    loaded_data = {}
    for key, filename in data_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                loaded_data[key] = df
                print(f" Loaded {filename}: {len(df)} rows")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        else:
            print(f"  File not found: {filename}")
    
    return loaded_data

def prepare_plot_data(summary_df):
    """Prepare data for Pareto plot"""
    if summary_df is None or len(summary_df) == 0:
        print(" No summary data available")
        return None
    
    plot_data = []
    
    for _, row in summary_df.iterrows():
        # Extract metrics
        ap_mean = row.get('Image_AP_AUPR_mean', 0)
        ap_std = row.get('Image_AP_AUPR_std', 0)
        auspro_mean = row.get('AUsPRO@0.05_mean', 0)
        auspro_std = row.get('AUsPRO@0.05_std', 0)
        auroc_mean = row.get('image_AUROC_mean', 0)
        auroc_std = row.get('image_AUROC_std', 0)
        
        # Determine if individual or ensemble
        is_ensemble = row.get('Type', '') == 'Ensemble_Summary'
        model_name = row.get('Model', '')
        model1 = row.get('Model1', '')
        model2 = row.get('Model2', '')
        dataset = row.get('Dataset', '')
        
        # Create ensemble identifier for coloring
        if is_ensemble and model1 and model2:
            # Sort model names for consistent color mapping
            sorted_models = '+'.join(sorted([model1, model2]))
            ensemble_key = sorted_models
        else:
            ensemble_key = model_name
        
        plot_data.append({
            'Dataset': dataset,
            'Model': model_name,
            'Model1': model1,
            'Model2': model2,
            'Type': 'Ensemble' if is_ensemble else 'Individual',
            'AP_mean': ap_mean,
            'AP_std': ap_std,
            'AUsPRO_mean': auspro_mean,
            'AUsPRO_std': auspro_std,
            'AUROC_mean': auroc_mean,
            'AUROC_std': auroc_std,
            'Color_Key': ensemble_key,
            'Display_Name': f"{model1}+{model2}" if is_ensemble else model_name
        })
    
    return pd.DataFrame(plot_data)

def create_ensemble_pareto_plot(plot_data, x_metric='AP_mean', y_metric='AUsPRO_mean', 
                               x_label='Image AP (AUPR)', y_label='Pixel AUsPRO@0.05'):
    """Create Pareto plot showing ensemble improvements over best individual models per dataset"""
    
    if plot_data is None or len(plot_data) == 0:
        print(" No data to plot")
        return None
    
    # Setup figure (same style as csv_to_tables.py)
    style = setup_figure_style()
    fig, ax = plt.subplots(figsize=(style['single_width'], 
                                   style['single_width'] * 0.85))
    
    # Separate individual and ensemble data
    individual_data = plot_data[plot_data['Type'] == 'Individual']
    ensemble_data = plot_data[plot_data['Type'] == 'Ensemble']
    
    # Get best individual per dataset (not per model)
    best_individuals_per_dataset = individual_data.groupby('Dataset').apply(
        lambda x: x.loc[x[x_metric].idxmax()]
    ).reset_index(drop=True)
    
    # Get all ensembles
    all_ensembles = ensemble_data.copy()
    
    print(f" Plotting {len(best_individuals_per_dataset)} best individual models per dataset and {len(all_ensembles)} ensembles")
    
    # Short model names for legend
    short_names = {
        'patchcore': 'PatchCore',
        'efficientad': 'EfficientAD', 
        'reversedistillation': 'RevDist',
        'supersimplenet': 'SSN',
        'dinomaly': 'DiNomaly',
        'padim': 'PaDiM',
        'uninet': 'UniNet',
        'fastflow': 'FastFlow',
        'uflow': 'U-Flow'
    }
    
    # Short ensemble names for legend
    short_ensemble_names = {
        'patchcore+efficientad': 'Patch+EffAD',
        'patchcore+reversedistillation': 'Patch+RevDist',
        'patchcore+supersimplenet': 'Patch+SSN',
        'patchcore+dinomaly': 'Patch+Dino',
        'patchcore+padim': 'Patch+PaDiM',
        'patchcore+uninet': 'Patch+UniNet',
        'efficientad+reversedistillation': 'EffAD+RevDist',
        'supersimplenet+dinomaly': 'SSN+Dino',
        'supersimplenet+padim': 'SSN+PaDiM',
        'uninet+reversedistillation': 'UniNet+RevDist',
        'dinomaly+reversedistillation': 'Dino+RevDist',
        'reversedistillation+supersimplenet': 'RevDist+SSN',
        'reversedistillation+dinomaly': 'RevDist+Dino',
        'reversedistillation+padim': 'RevDist+PaDiM',
        'reversedistillation+uninet': 'RevDist+UniNet',
        'dinomaly+supersimplenet': 'Dino+SSN',
        'dinomaly+padim': 'Dino+PaDiM',
        'dinomaly+uninet': 'Dino+UniNet',
        'uninet+supersimplenet': 'UniNet+SSN',
    }
    
    # Only plot most important individual models (top performers)
    important_models = ['patchcore', 'efficientad', 'supersimplenet', 'reversedistillation', 'dinomaly']
    
    plotted_models = set()
    for _, row in best_individuals_per_dataset.iterrows():
        model = row['Model']
        if model in plotted_models or model not in important_models:
            continue
        plotted_models.add(model)
        
        color = MODEL_COLORS.get(model, '#666666')
        
        ax.scatter(row[x_metric], row[y_metric],
                  color=color, marker='o', s=24, alpha=0.9,
                  label=short_names.get(model, model),
                  edgecolors='white', linewidth=0.8)
    
    # Only plot best performing ensembles (top 6-8)
    ensemble_performance = all_ensembles.groupby('Display_Name')[x_metric].mean().sort_values(ascending=False)
    top_ensembles = ensemble_performance.head(8).index.tolist()
    
    # Define distinct colors for ensembles (high contrast, colorblind-friendly)
    ensemble_colors = [
        '#e74c3c',  # Red
        '#f39c12',  # Orange  
        '#9b59b6',  # Purple
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#1abc9c',  # Teal
        '#34495e',  # Dark blue-gray
        '#e67e22'   # Dark orange
    ]
    
    plotted_ensembles = set()
    improvement_count = 0
    
    for _, ensemble_row in all_ensembles.iterrows():
        ensemble_name = ensemble_row['Display_Name']
        dataset = ensemble_row['Dataset']
        
        # Only plot top ensembles
        if ensemble_name not in top_ensembles:
            continue
            
        # Find best individual for this dataset
        best_individual = best_individuals_per_dataset[best_individuals_per_dataset['Dataset'] == dataset]
        
        if len(best_individual) > 0:
            best_ind = best_individual.iloc[0]
            
            # Calculate improvement
            x_improvement = ensemble_row[x_metric] - best_ind[x_metric]
            y_improvement = ensemble_row[y_metric] - best_ind[y_metric]
            
            # Plot ensemble if it's new and has improvement
            if ensemble_name not in plotted_ensembles and (x_improvement > 0 or y_improvement > 0):
                plotted_ensembles.add(ensemble_name)
                color_idx = len(plotted_ensembles) - 1
                color = ensemble_colors[color_idx % len(ensemble_colors)]
                
                # Plot ensemble
                ax.scatter(ensemble_row[x_metric], ensemble_row[y_metric],
                          color=color, marker='D', s=28, alpha=0.9,
                          label=short_ensemble_names.get(ensemble_name, ensemble_name.replace('_', '')),
                          edgecolors='white', linewidth=1.0)
                
                # Draw improvement arrow for ALL improvements
                ax.annotate('', xy=(ensemble_row[x_metric], ensemble_row[y_metric]),
                           xytext=(best_ind[x_metric], best_ind[y_metric]),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=1.8))
                
                # Add improvement text for significant improvements
                if x_improvement > 0.005:  # Lower threshold to see more improvements
                    mid_x = (ensemble_row[x_metric] + best_ind[x_metric]) / 2
                    mid_y = (ensemble_row[y_metric] + best_ind[y_metric]) / 2
                    ax.text(mid_x, mid_y, f'+{x_improvement:.3f}', 
                           fontsize=5, ha='center', va='bottom', color=color, alpha=0.9)
    
    # Customize axes (same as csv_to_tables.py)
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    
    # Set optimized limits for better space usage
    all_data = pd.concat([best_individuals_per_dataset, all_ensembles])
    x_values = all_data[x_metric].dropna()
    y_values = all_data[y_metric].dropna()
    
    if len(x_values) > 0 and len(y_values) > 0:
        x_min, x_max = x_values.min(), x_values.max()
        y_min, y_max = y_values.min(), y_values.max()
        
        # Add some padding for arrows
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    # Add grid (same style as csv_to_tables.py)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Legend with better positioning and smaller font
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      frameon=True, fancybox=False, shadow=False,
                      borderaxespad=0, fontsize=6, markerscale=0.8)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.5)
    
    # Add annotation for improvement direction
    ax.annotate('Better →', xy=(0.8, 0.95), xycoords='axes fraction', 
                fontsize=7, ha='center', style='italic', color='gray')
    ax.annotate('↑\nBetter', xy=(0.12, 0.90), xycoords='axes fraction',
                fontsize=7, ha='center', va='center', style='italic', color='gray')

    # Tight layout
    plt.tight_layout()
    
    return fig

def find_pareto_efficient_points(plot_data, x_metric='AP_mean', y_metric='AUsPRO_mean'):
    """Find Pareto efficient points (models that are not dominated by others)"""
    pareto_efficient = []
    
    for i, row in plot_data.iterrows():
        x_val = row[x_metric]
        y_val = row[y_metric]
        
        # Check if this point is dominated by any other point
        dominated = False
        for j, other_row in plot_data.iterrows():
            if i != j:
                other_x = other_row[x_metric]
                other_y = other_row[y_metric]
                
                # Point is dominated if another point is better in both metrics
                if other_x >= x_val and other_y >= y_val:
                    if other_x > x_val or other_y > y_val:
                        dominated = True
                        break
        
        if not dominated:
            pareto_efficient.append({
                'Model': row['Model'],
                'Type': row['Type'],
                'Dataset': row['Dataset'],
                'AP': x_val,
                'AUsPRO': y_val,
                'Display_Name': row['Display_Name']
            })
    
    return pareto_efficient

def generate_ensemble_comparison_plots():
    """Generate clean ensemble comparison plot for paper"""
    print(" GENERATING ENSEMBLE COMPARISON PLOT FOR PAPER")
    print("="*60)
    
    # Load data
    ensemble_data = load_ensemble_data()
    
    if 'summary' not in ensemble_data:
        print(" No summary data found. Please run ensemble_analysis.py first.")
        return
    
    # Prepare plot data
    plot_data = prepare_plot_data(ensemble_data['summary'])
    
    if plot_data is None:
        print(" Failed to prepare plot data")
        return
    
    print(f" Prepared data: {len(plot_data)} total points")
    print(f"   - Individual models: {len(plot_data[plot_data['Type'] == 'Individual'])}")
    print(f"   - Ensemble models: {len(plot_data[plot_data['Type'] == 'Ensemble'])}")
    
    # Create output directory
    os.makedirs('figures_ensemble_comparison', exist_ok=True)
    
    # Determine which X metric to use (prefer AP over AUROC for anomaly detection)
    if 'AP_mean' in plot_data.columns and plot_data['AP_mean'].notna().any():
        x_metric = 'AP_mean'
        x_label = 'Image AP (AUPR)'
        plot_name = 'ap_auspro'
        print(" Using Image AP vs AUsPRO (preferred for anomaly detection)")
    elif 'AUROC_mean' in plot_data.columns and plot_data['AUROC_mean'].notna().any():
        x_metric = 'AUROC_mean'
        x_label = 'Image AUROC'
        plot_name = 'auroc_auspro'
        print(" Using Image AUROC vs AUsPRO (fallback)")
    else:
        print(" No suitable X metric found (need AP or AUROC)")
        return
    
    # Create the main comparison plot
    fig = create_ensemble_pareto_plot(plot_data, 
                                     x_metric=x_metric, 
                                     y_metric='AUsPRO_mean',
                                     x_label=x_label,
                                     y_label='Pixel AUsPRO@0.05')
    
    if fig:
        output_path = f'figures_ensemble_comparison/ensemble_comparison_{plot_name}'
        fig.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(f'{output_path}.png', format='png', bbox_inches='tight')
        fig.savefig(f'{output_path}.eps', format='eps', bbox_inches='tight')
        print(f" Saved: {output_path}.(pdf|png|eps)")
        
        # Find and report Pareto efficient points
        pareto_points = find_pareto_efficient_points(plot_data, x_metric, 'AUsPRO_mean')
        print(f"\n Pareto efficient models ({len(pareto_points)}):")
        for point in pareto_points:
            x_val = point.get('AP', point.get('AUROC', 0))
            print(f"   - {point['Display_Name']} ({point['Type']}): {x_metric.replace('_mean','')}={x_val:.3f}, AUsPRO={point['AUsPRO']:.3f}")
    
    # Print summary statistics
    print(f"\n SUMMARY STATISTICS:")
    print(f"   - Total datasets: {plot_data['Dataset'].nunique()}")
    print(f"   - Individual models tested: {len(plot_data[plot_data['Type'] == 'Individual'])}")
    print(f"   - Ensemble combinations: {len(plot_data[plot_data['Type'] == 'Ensemble'])}")
    
    # Best performers analysis
    if len(plot_data) > 0:
        x_col = x_metric
        individual_data = plot_data[plot_data['Type'] == 'Individual']
        ensemble_data = plot_data[plot_data['Type'] == 'Ensemble']
        
        if len(individual_data) > 0 and len(ensemble_data) > 0:
            best_individual_x = individual_data[x_col].max()
            best_ensemble_x = ensemble_data[x_col].max()
            best_individual_auspro = individual_data['AUsPRO_mean'].max()
            best_ensemble_auspro = ensemble_data['AUsPRO_mean'].max()
            
            metric_name = x_label.split()[1] if len(x_label.split()) > 1 else x_label
            print(f"\n BEST PERFORMERS:")
            print(f"   - Best Individual {metric_name}: {best_individual_x:.4f}")
            print(f"   - Best Ensemble {metric_name}: {best_ensemble_x:.4f} (+{best_ensemble_x-best_individual_x:.4f})")
            print(f"   - Best Individual AUsPRO: {best_individual_auspro:.4f}")
            print(f"   - Best Ensemble AUsPRO: {best_ensemble_auspro:.4f} (+{best_ensemble_auspro-best_individual_auspro:.4f})")
    
    plt.close('all')  # Clean up
    
    print(f"\n Ensemble comparison plot generated successfully!")
    print(f" Files saved to: figures_ensemble_comparison/")
    print(f" Ready for paper inclusion!")

if __name__ == "__main__":
    generate_ensemble_comparison_plots()
