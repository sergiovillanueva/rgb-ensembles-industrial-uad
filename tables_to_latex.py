import pandas as pd
import os
import glob

def get_table_info(table_name):
    """Get table-specific caption and notes."""
    table_info = {
        'auspro_0_05_rgb': {
            'caption': 'Pixel-level segmentation results (AUsPRO@0.05) for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. AUsPRO measures pixel-level localization performance. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'auspro_0_05_depth': {
            'caption': 'Pixel-level segmentation results (AUsPRO@0.05) for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps as additional input. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'image_auroc_rgb': {
            'caption': 'Image-level classification results (AUROC) for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. AUROC measures discriminative power between normal and anomalous images. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'image_auroc_depth': {
            'caption': 'Image-level classification results (AUROC) for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps as additional input. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'image_ap_aupr_rgb': {
            'caption': 'Image-level classification results (AP-AUPR) for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Average Precision under Precision-Recall curve for imbalanced anomaly detection. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'image_ap_aupr_depth': {
            'caption': 'Image-level classification results (AP-AUPR) for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps as additional input. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_99_rgb': {
            'caption': 'True Positive Rate at 99\\% True Negative Rate for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Operating point with strict false positive control (1\\% FPR). * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_99_depth': {
            'caption': 'True Positive Rate at 99\\% True Negative Rate for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps. Operating point with 1\\% FPR. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_95_rgb': {
            'caption': 'True Positive Rate at 95\\% True Negative Rate for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Operating point with moderate false positive control (5\\% FPR). * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_95_depth': {
            'caption': 'True Positive Rate at 95\\% True Negative Rate for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps. Operating point with 5\\% FPR. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_90_rgb': {
            'caption': 'True Positive Rate at 90\\% True Negative Rate for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Operating point with relaxed false positive control (10\\% FPR). * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'tpr_tnr0_90_depth': {
            'caption': 'True Positive Rate at 90\\% True Negative Rate for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps. Operating point with 10\\% FPR. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'spro_0_05_rgb': {
            'caption': 'Standardized Per-Region Overlap (sPRO@0.05) for RGB images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. sPRO measures pixel-level localization with region-based evaluation. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'spro_0_05_depth': {
            'caption': 'Standardized Per-Region Overlap (sPRO@0.05) for depth-enhanced images. Higher is better.',
            'notes': 'Values are mean $\\pm$ std across $n$ seeds. Results using estimated depth maps as additional input. * = Best result (not significantly different at 5\\% level using Welch t-test).'
        },
        'calibration_transfer': {
            'caption': 'Calibration transfer analysis: $\\Delta$TNR between training and test sets.',
            'notes': 'Values show mean $\\Delta$TNR = Test\\_TNR - Target\\_TNR across seeds. Comparable: $\\checkmark$ if $|\\Delta\\text{TNR}| \\leq 0.03$ (99\\%) or $\\leq 0.05$ (95\\%/90\\%), $\\times$ otherwise.'
        },
        'monotonicity_checks': {
            'caption': 'Monotonicity verification for operating points and test TNR values.',
            'notes': 'Monotonic TPR: $\\checkmark$ if TPR@90\\% $\\geq$ TPR@95\\% $\\geq$ TPR@99\\%. Monotonic TNR: $\\checkmark$ if Test\\_TNR(99\\%) $\\geq$ Test\\_TNR(95\\%) $\\geq$ Test\\_TNR(90\\%). All OK: both conditions satisfied.'
        },
        'operational_sensitivity': {
            'caption': 'Aggregated operational sensitivity across multiple operating points.',
            'notes': 'Mean-TPR@TNR computed as average of TPR values at 99\\%, 95\\%, and 90\\% TNR operating points. Higher values indicate better overall sensitivity under false positive control.'
        },
        'seed_robustness': {
            'caption': 'Robustness analysis across random seeds by model.',
            'notes': 'Standard deviation (Std) and coefficient of variation (CV = std/mean) computed across all datasets and seeds. Lower values indicate more stable performance.'
        },
        'computational_efficiency': {
            'caption': 'Computational requirements and efficiency metrics.',
            'notes': 'Time per image in seconds, performance per minute (AUsPRO/training\\_time), and peak GPU memory usage in GB. Lower time and memory with higher performance indicates better efficiency.'
        },
        'class_difficulty_ranking': {
            'caption': 'Dataset difficulty ranking based on model performance.',
            'notes': 'Average rank computed across all models for each dataset using AUsPRO@0.05. Higher average rank indicates more challenging datasets. Best/worst scores show performance range.'
        },
        'image_pixel_correlation': {
            'caption': 'Correlation between image-level and pixel-level performance metrics.',
            'notes': 'Spearman correlation coefficients between image AUROC/AP and pixel AUsPRO computed across models per dataset. Higher correlations indicate consistent ranking between detection and localization.'
        },
        'winners_by_class': {
            'caption': 'Best performing models per dataset with significance testing.',
            'notes': 'Winner shows best model per metric. Score format: mean$\\pm$std. $\\dagger$ indicates significant difference from second-best (simplified threshold). Multiple metrics evaluated per dataset.'
        },
        
        # REVIEWER REQUESTED TABLES
        'r1_rgb_image_main': {
            'caption': 'R1: RGB baseline results for image-level detection (macro-averaged over datasets)',
            'notes': 'Values show macro-mean $\\pm$ macro-std across datasets for each model. Mean-TPR@TNR averaged over 99\\%, 95\\%, and 90\\% operating points. $\\textbf{Best}$ (rank 1) and second-best (rank 2) marked with *.'
        },
        'r2_rgb_pixel_main': {
            'caption': 'R2: RGB baseline results for pixel-level segmentation (macro-averaged over datasets)',
            'notes': 'Values show macro-mean $\\pm$ macro-std across datasets for each model. AUsPRO and sPRO at 5\\% FPR threshold. $\\textbf{Best}$ (rank 1) and second-best (rank 2) marked with *.'
        },
        'r3_calibration_main': {
            'caption': 'R3: Calibration analysis with operating point stability (macro-averaged over datasets)',
            'notes': '$\\Delta$TNR = Test\\_TNR - Target\\_TNR shows calibration drift. Comp\\%: percentage of datasets with comparable thresholds ($|\\Delta$TNR$| \\leq 0.03$ for 99\\%, $\\leq 0.05$ for 95\\%/90\\%). Mono\\%: percentage with monotonic TPR/TNR trends across operating points.'
        },
        'r4_depth_deltas_main': {
            'caption': 'R4: Pseudo-depth enhancement effects (macro-averaged deltas over datasets)',  
            'notes': '$\\Delta$ = Depth\\_enhanced - RGB\\_baseline, macro-averaged across datasets. Positive values indicate improvement with pseudo-depth. Values show mean $\\pm$ std of per-dataset deltas. Negative values suggest depth estimation may introduce noise.'
        },
        
        # MACRO-AVERAGED INDIVIDUAL METRICS
        'image_ap_aupr_macro_rgb': {
            'caption': 'Image-level Average Precision (AP-AUPR) results for RGB images (macro-averaged over datasets)',
            'notes': 'Values show macro-mean $\\pm$ macro-std across datasets for each model. Average Precision under Precision-Recall curve for imbalanced anomaly detection. $\\textbf{Best}$ (rank 1) and second-best (rank 2) marked with *.'
        },
        'auspro_0_05_macro_rgb': {
            'caption': 'Pixel-level segmentation (AUsPRO@0.05) results for RGB images (macro-averaged over datasets)',
            'notes': 'Values show macro-mean $\\pm$ macro-std across datasets for each model. AUsPRO measures pixel-level localization performance at 5\\% FPR threshold. $\\textbf{Best}$ (rank 1) and second-best (rank 2) marked with *.'
        }
    }
    
    # Default fallback
    default_info = {
        'caption': f'{table_name.replace("_", " ").title()} results.',
        'notes': 'See text for details.'
    }
    
    return table_info.get(table_name.lower(), default_info)

def csv_to_latex_table(csv_path, table_name):
    """Convert CSV table to LaTeX format."""
    df = pd.read_csv(csv_path, sep=';')
    
    # Handle reviewer-requested tables and macro tables with special formatting
    if table_name.lower().startswith(('r1_', 'r2_', 'r3_', 'r4_')) or '_macro_' in table_name.lower():
        return reviewer_table_to_latex(df, table_name)
    # Handle different table types
    elif 'Model' in df.columns:
        # Standard model comparison tables
        return model_table_to_latex(df, table_name)
    elif 'Dataset' in df.columns and 'Model' in df.columns:
        # Dataset x Model tables
        return dataset_model_table_to_latex(df, table_name)
    else:
        # Generic tables
        return generic_table_to_latex(df, table_name)

def model_table_to_latex(df, table_name):
    """Convert model comparison table to LaTeX."""
    models = df['Model'].tolist()
    
    # Get dataset columns (excluding Model)
    dataset_cols = [col for col in df.columns if col != 'Model']
    
    # Start LaTeX table
    latex = []
    
    # Determine if table needs double column
    num_cols = len(dataset_cols) + 1  # +1 for Model column
    is_wide = num_cols > 6
    
    if is_wide:
        latex.append("\\begin{table*}[t]")
    else:
        latex.append("\\begin{table}[t]")
    
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\setlength{\\tabcolsep}{5pt}")
    # Get table-specific info
    table_info = get_table_info(table_name)
    
    latex.append("\\begin{threeparttable}")
    latex.append(f"\\caption{{{table_info['caption']}}}")
    latex.append(f"\\label{{tab:{table_name.lower().replace('_', '-')}}}")
    
    # Create column specification
    col_spec = "l" + "c" * len(dataset_cols)
    if is_wide:
        latex.append(f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} {col_spec}}}")
    else:
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    
    latex.append("\\toprule")
    
    # Header
    header = "Model & " + " & ".join([col.replace('_', '\\_') for col in dataset_cols]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        model = row['Model'].replace('_', '\\_')
        values = []
        
        for col in dataset_cols:
            val = str(row[col])
            if val == 'N/A' or val == 'nan':
                values.append("N/A")
            else:
                # Convert asterisk to bold and remove asterisk
                if val.endswith('*'):
                    val = val[:-1]  # Remove asterisk
                    # Extract mean±std
                    if '±' in val:
                        val = f"\\textbf{{{val}}}"
                    else:
                        val = f"\\textbf{{{val}}}"
                # Convert ± symbol
                val = val.replace('±', '$\\pm$')
                values.append(val)
        
        row_str = model + " & " + " & ".join(values) + " \\\\"
        latex.append(row_str)
    
    latex.append("\\bottomrule")
    
    if is_wide:
        latex.append("\\end{tabular*}")
    else:
        latex.append("\\end{tabular}")
    
    # Add notes
    latex.append("\\begin{tablenotes}[flushleft]")
    latex.append("\\footnotesize")
    latex.append(f"\\item Notes: {table_info['notes']}")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{threeparttable}")
    
    if is_wide:
        latex.append("\\end{table*}")
    else:
        latex.append("\\end{table}")
    
    return "\n".join(latex)

def dataset_model_table_to_latex(df, table_name):
    """Convert dataset-model table to LaTeX."""
    # Pivot table if needed
    if 'Dataset' in df.columns and 'Model' in df.columns:
        # Check if it's already in the right format or needs pivoting
        other_cols = [col for col in df.columns if col not in ['Dataset', 'Model']]
        if len(other_cols) == 1:
            # Need to pivot
            df_pivot = df.pivot(index='Model', columns='Dataset', values=other_cols[0])
            df_pivot = df_pivot.reset_index()
            return model_table_to_latex(df_pivot, table_name)
    
    return generic_table_to_latex(df, table_name)

def generic_table_to_latex(df, table_name):
    """Convert generic table to LaTeX."""
    latex = []
    
    num_cols = len(df.columns)
    is_wide = num_cols > 6
    
    if is_wide:
        latex.append("\\begin{table*}[t]")
    else:
        latex.append("\\begin{table}[t]")
    
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\setlength{\\tabcolsep}{5pt}")
    
    # Get table-specific info
    table_info = get_table_info(table_name)
    
    latex.append("\\begin{threeparttable}")
    latex.append(f"\\caption{{{table_info['caption']}}}")
    latex.append(f"\\label{{tab:{table_name.lower().replace('_', '-')}}}")
    
    # Column specification
    col_spec = "l" + "c" * (num_cols - 1)
    if is_wide:
        latex.append(f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} {col_spec}}}")
    else:
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    
    latex.append("\\toprule")
    
    # Header
    header = " & ".join([col.replace('_', '\\_').replace('@', '\\@') for col in df.columns]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = str(row[col])
            if val == 'nan' or val == 'N/A':
                values.append("N/A")
            else:
                # Handle special symbols
                val = val.replace('_', '\\_')
                val = val.replace('±', '$\\pm$')
                val = val.replace('', '$\\checkmark$')
                val = val.replace('', '$\\times$')
                val = val.replace('†', '$\\dagger$')
                values.append(val)
        
        row_str = " & ".join(values) + " \\\\"
        latex.append(row_str)
    
    latex.append("\\bottomrule")
    
    if is_wide:
        latex.append("\\end{tabular*}")
    else:
        latex.append("\\end{tabular}")
    
    latex.append("\\begin{tablenotes}[flushleft]")
    latex.append("\\footnotesize")
    latex.append(f"\\item Notes: {table_info['notes']}")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{threeparttable}")
    
    if is_wide:
        latex.append("\\end{table*}")
    else:
        latex.append("\\end{table}")
    
    return "\n".join(latex)

def reviewer_table_to_latex(df, table_name):
    """Convert reviewer-requested table to LaTeX with special formatting."""
    latex = []
    
    num_cols = len(df.columns)
    is_wide = num_cols > 6
    
    if is_wide:
        latex.append("\\begin{table*}[t]")
    else:
        latex.append("\\begin{table}[t]")
    
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\setlength{\\tabcolsep}{5pt}")
    
    # Get table-specific info
    table_info = get_table_info(table_name)
    
    latex.append("\\begin{threeparttable}")
    latex.append(f"\\caption{{{table_info['caption']}}}")
    latex.append(f"\\label{{tab:{table_name.lower().replace('_', '-')}}}")
    
    # Determine which columns to show based on table type
    if table_name.lower().startswith('r1_'):
        display_cols = ['Model', 'Image_AUROC', 'Image_AP_AUPR', 'Mean_TPR@TNR']
    elif table_name.lower().startswith('r2_'):
        display_cols = ['Model', 'AUsPRO@0.05', 'sPRO@0.05']
    elif table_name.lower().startswith('r3_'):
        display_cols = ['Model', 'ΔTNR@0.99', 'ΔTNR@0.95', 'ΔTNR@0.90', 'Comp_99%', 'Comp_95%', 'Comp_90%', 'Mono_TPR%', 'Mono_TNR%']
    elif table_name.lower().startswith('r4_'):
        display_cols = ['Model', 'ΔAUROC', 'ΔAP', 'ΔAUsPRO']
    elif 'image_ap_aupr_macro' in table_name.lower():
        display_cols = ['Model', 'Image_AP_AUPR']
    elif 'auspro_0_05_macro' in table_name.lower():
        display_cols = ['Model', 'AUsPRO@0.05']
    else:
        display_cols = df.columns.tolist()
    
    # Filter columns that exist in the DataFrame
    display_cols = [col for col in display_cols if col in df.columns]
    
    # Column specification
    col_spec = "l" + "c" * (len(display_cols) - 1)
    if is_wide:
        latex.append(f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} {col_spec}}}")
    else:
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    
    latex.append("\\toprule")
    
    # Header - clean up column names for display
    clean_headers = []
    for col in display_cols:
        clean_col = col.replace('_', '\\_').replace('@', '\\@').replace('Δ', '$\\Delta$')
        if 'TPR@TNR' in col:
            clean_col = 'Mean-TPR\\@TNR'
        elif 'Comp_' in col:
            clean_col = col.replace('Comp_', 'Comp ').replace('%', '\\%')
        elif 'Mono_' in col:
            clean_col = col.replace('Mono_', 'Mono ').replace('%', '\\%')
        clean_headers.append(clean_col)
    
    header = " & ".join(clean_headers) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows with special formatting
    for _, row in df.iterrows():
        values = []
        for col in display_cols:
            val = str(row[col])
            
            if col == 'Model':
                values.append(val.replace('_', '\\_'))
            elif val == 'nan' or val == 'N/A':
                values.append("N/A")
            else:
                # Handle different formatting based on table type and column
                if table_name.lower().startswith(('r1_', 'r2_')) or '_macro_' in table_name.lower():
                    # R1/R2/Macro: Format mean±std with ranking
                    if col in ['Image_AUROC', 'Image_AP_AUPR', 'Mean_TPR@TNR', 'AUsPRO@0.05', 'sPRO@0.05']:
                        rank_col = f'{col}_rank'
                        if rank_col in df.columns:
                            rank = row[rank_col]
                            std_col = f'{col}_std'
                            if std_col in df.columns:
                                mean_val = float(row[col])
                                std_val = float(row[std_col])
                                
                                # Format with ranking
                                if rank == 1:
                                    formatted = f"\\textbf{{{mean_val:.3f}$\\pm${std_val:.3f}}}"
                                elif rank == 2:
                                    formatted = f"{mean_val:.3f}$\\pm${std_val:.3f}*"
                                else:
                                    formatted = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                                values.append(formatted)
                            else:
                                values.append(f"{float(val):.3f}")
                        else:
                            values.append(f"{float(val):.3f}")
                    else:
                        values.append(val.replace('±', '$\\pm$'))
                        
                elif table_name.lower().startswith('r3_'):
                    # R3: Format ΔTNR with ± and percentages
                    if col.startswith('ΔTNR@'):
                        std_col = f'{col}_std'
                        if std_col in df.columns:
                            mean_val = float(row[col])
                            std_val = float(row[std_col])
                            formatted = f"{mean_val:+.3f}$\\pm${std_val:.3f}"
                            values.append(formatted)
                        else:
                            values.append(f"{float(val):+.3f}")
                    elif col.endswith('%'):
                        values.append(f"{float(val):.1f}\\%")
                    else:
                        values.append(val.replace('±', '$\\pm$'))
                        
                elif table_name.lower().startswith('r4_'):
                    # R4: Format deltas with ± and sign indicators
                    if col.startswith('Δ'):
                        std_col = f'{col}_std'
                        if std_col in df.columns:
                            mean_val = float(row[col])
                            std_val = float(row[std_col])
                            
                            # Color negative values
                            if mean_val < -0.001:
                                formatted = f"\\textcolor{{red}}{{{mean_val:+.3f}$\\pm${std_val:.3f}}}"
                            elif mean_val > 0.001:
                                formatted = f"\\textcolor{{green}}{{{mean_val:+.3f}$\\pm${std_val:.3f}}}"
                            else:
                                formatted = f"{mean_val:+.3f}$\\pm${std_val:.3f}"
                            values.append(formatted)
                        else:
                            mean_val = float(val)
                            if mean_val < -0.001:
                                formatted = f"\\textcolor{{red}}{{{mean_val:+.3f}}}"
                            elif mean_val > 0.001:
                                formatted = f"\\textcolor{{green}}{{{mean_val:+.3f}}}"
                            else:
                                formatted = f"{mean_val:+.3f}"
                            values.append(formatted)
                    else:
                        values.append(val.replace('±', '$\\pm$'))
                else:
                    # Default formatting
                    values.append(val.replace('±', '$\\pm$').replace('', '$\\checkmark$').replace('', '$\\times$'))
        
        row_str = " & ".join(values) + " \\\\"
        latex.append(row_str)
    
    latex.append("\\bottomrule")
    
    if is_wide:
        latex.append("\\end{tabular*}")
    else:
        latex.append("\\end{tabular}")
    
    latex.append("\\begin{tablenotes}[flushleft]")
    latex.append("\\footnotesize")
    latex.append(f"\\item Notes: {table_info['notes']}")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{threeparttable}")
    
    if is_wide:
        latex.append("\\end{table*}")
    else:
        latex.append("\\end{table}")
    
    return "\n".join(latex)

def process_all_tables():
    """Process all CSV files in tables/ directory."""
    if not os.path.exists('tables/'):
        print("No tables/ directory found!")
        return
    
    csv_files = glob.glob('tables/*.csv')
    if not csv_files:
        print("No CSV files found in tables/ directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    all_latex = []
    all_latex.append("% LaTeX tables generated from CSV files")
    all_latex.append("% Requires packages: booktabs, threeparttable")
    all_latex.append("")
    
    for csv_file in sorted(csv_files):
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"Processing {table_name}...")
        
        try:
            latex_table = csv_to_latex_table(csv_file, table_name)
            all_latex.append(f"% Table: {table_name}")
            all_latex.append(latex_table)
            all_latex.append("")
            all_latex.append("")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Save to file
    output_file = 'latex_tables.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_latex))
    
    print(f"\nLaTeX tables saved to {output_file}")
    print(f"Processed {len(csv_files)} CSV files")

if __name__ == '__main__':
    process_all_tables()
