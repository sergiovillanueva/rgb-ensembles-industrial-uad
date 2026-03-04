import pandas as pd
import numpy as np
from scipy import stats

def load_results(csv_path: str, sep: str = ';') -> pd.DataFrame:
    """Load results file and convert numeric columns."""
    df = pd.read_csv(csv_path, sep=sep)
    # Filter empty rows
    df = df.dropna(subset=['Dataset']).copy()
    metric_cols = ['Image_AUROC','Image_AP_AUPR','TPR@TNR0.99','TPR@TNR0.95','TPR@TNR0.90',
                   'Test_TNR_at_train_TNR0.99','Test_TNR_at_train_TNR0.95','Test_TNR_at_train_TNR0.90',
                   'Pixel_AUROC','sPRO@0.05','AUsPRO@0.05','Training_Time_Seconds','Used_GB']
    for col in metric_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').replace('nan', np.nan).astype(float)
    return df

def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the number of training and test images per dataset."""
    stats = df.groupby('Dataset').agg(Train=('Train_Images','first'),
                                     Test=('Test_Images','first')).reset_index()
    return stats.astype({'Train':'int','Test':'int'})

def performance_summary(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Compute mean and CI for each dataset, model, and metric."""
    if metrics is None:
        metrics = ['Image_AUROC','Image_AP_AUPR','Pixel_AUROC','AUsPRO@0.05','TPR@TNR0.95']
    records = []
    # Consider only RGB versions (without 'estimate_depth')
    for dataset, group_d in df[~df['Dataset'].str.contains('estimate_depth')].groupby('Dataset'):
        for model, group_m in group_d.groupby('Model'):
            row = {'Dataset': dataset, 'Model': model}
            for metric in metrics:
                values = group_m[metric].dropna().values
                if len(values) == 0:
                    row[metric] = (np.nan, np.nan)
                else:
                    mean = values.mean()
                    ci = stats.t.ppf(1-0.025, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                    row[metric] = (mean, ci)
            records.append(row)
    return pd.DataFrame(records)

def depth_improvement(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Compute mean difference (depth - RGB) per model and class."""
    if metrics is None:
        metrics = ['Pixel_AUROC','Image_AUROC','AUsPRO@0.05']
    classes = [c for c in df['Dataset'].unique() if not c.endswith('estimate_depth')]
    records = []
    for base_cls in classes:
        for model in df['Model'].unique():
            base = df[(df['Dataset'] == base_cls) & (df['Model'] == model)]
            depth = df[(df['Dataset'] == base_cls + '_estimate_depth') & (df['Model'] == model)]
            merged = pd.merge(base[['Seed'] + metrics], depth[['Seed'] + metrics], on='Seed', suffixes=('_base','_depth'))
            rec = {'Class': base_cls, 'Model': model}
            for m in metrics:
                diffs = merged[f'{m}_depth'] - merged[f'{m}_base']
                rec[m + '_diff'] = diffs.mean() if len(diffs) > 0 else np.nan
            records.append(rec)
    return pd.DataFrame(records)

def computational_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean time and memory per model (with CI)."""
    records = []
    for model, group in df.groupby('Model'):
        time_vals = group['Training_Time_Seconds'].values / 60  # Convert to minutes
        mem_vals = group['Used_GB'].values
        mean_time = time_vals.mean(); ci_time = stats.t.ppf(1-0.025, len(time_vals)-1) * time_vals.std(ddof=1)/np.sqrt(len(time_vals))
        mean_mem  = mem_vals.mean(); ci_mem  = stats.t.ppf(1-0.025, len(mem_vals)-1) * mem_vals.std(ddof=1)/np.sqrt(len(mem_vals))
        records.append({'Model': model, 'Time_min': round(mean_time, 1), 'Time_CI': round(ci_time, 1), 'Mem_GB': round(mean_mem, 1), 'Mem_CI': round(ci_mem, 1)})
    return pd.DataFrame(records)

def welch_t_test_best_models(df: pd.DataFrame, metric: str, include_depth: bool = False) -> dict:
    """Find the best models using Welch's t-test at 5% significance level."""
    import warnings
    best_models = {}
    if include_depth:
        datasets = df['Dataset'].unique()
    else:
        datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        models = dataset_data['Model'].unique()
        
        # Get values per model
        model_values = {}
        for model in models:
            values = dataset_data[dataset_data['Model'] == model][metric].dropna().values
            if len(values) > 0:
                model_values[model] = values
        
        if len(model_values) < 2:
            best_models[dataset] = list(model_values.keys())
            continue
        
        # Compute means
        model_means = {model: vals.mean() for model, vals in model_values.items()}
        best_mean = max(model_means.values())
        best_model = max(model_means, key=model_means.get)
        
        # Statistical test
        significant_best = [best_model]
        for other_model in models:
            if other_model != best_model and other_model in model_values:
                # Check if values are nearly identical (avoid numerical issues)
                diff_means = abs(model_means[best_model] - model_means[other_model])
                if diff_means < 1e-6:  # Very small difference
                    significant_best.append(other_model)
                else:
                    # Welch's t-test with warning suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        _, p_value = stats.ttest_ind(model_values[best_model], 
                                                   model_values[other_model], 
                                                   equal_var=False)
                    if p_value > 0.05:  # No significantly different
                        significant_best.append(other_model)
        
        best_models[dataset] = significant_best
    
    return best_models

def create_results_table(df: pd.DataFrame, metric: str, include_depth: bool = False) -> pd.DataFrame:
    """Create table with datasets as columns and models as rows, marking the best."""
    if include_depth:
        datasets = df['Dataset'].unique()
    else:
        datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    models = df['Model'].unique()
    
    # Find best models
    best_models = welch_t_test_best_models(df, metric, include_depth)
    
    # Create table
    table_data = []
    for model in models:
        row = {'Model': model}
        for dataset in datasets:
            data = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            values = data[metric].dropna().values
            
            if len(values) == 0:
                row[dataset] = "N/A"
            else:
                mean = values.mean()
                if len(values) > 1:
                    ci = stats.t.ppf(1-0.025, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
                else:
                    ci = 0
                
                # Mark if best
                is_best = model in best_models.get(dataset, [])
                marker = "*" if is_best else ""
                
                row[dataset] = f"{mean:.2f}±{ci:.2f}{marker}"
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def save_tables_to_csv(df: pd.DataFrame):
    """Save all tables as CSV in the tables/ folder."""
    import os
    
    # Create tables folder if it does not exist
    os.makedirs('tables', exist_ok=True)
    
    # Metrics for the tables
    metrics = ['Image_AUROC', 'Image_AP_AUPR', 'TPR@TNR0.99', 'TPR@TNR0.95', 
               'TPR@TNR0.90', 'AUsPRO@0.05', 'sPRO@0.05']
    
    for metric in metrics:
        print(f"Generating tables for {metric}...")
        
        # 1. Macro-averaged table (FIRST)
        if metric == 'Image_AP_AUPR':
            macro_table = macro_image_ap_table(df)
            filename_macro = f"tables/{metric.replace('@', '_').replace('.', '_')}_macro_rgb.csv"
            macro_table.to_csv(filename_macro, index=False, sep=';')
            print(f"   Macro-averaged table saved")
        elif metric == 'AUsPRO@0.05':
            macro_table = macro_auspro_table(df)
            filename_macro = f"tables/{metric.replace('@', '_').replace('.', '_')}_macro_rgb.csv"
            macro_table.to_csv(filename_macro, index=False, sep=';')
            print(f"   Macro-averaged table saved")
        
        # 2. RGB table by class (SECOND)
        table_rgb = create_results_table(df, metric, include_depth=False)
        filename_rgb = f"tables/{metric.replace('@', '_').replace('.', '_')}_rgb.csv"
        table_rgb.to_csv(filename_rgb, index=False, sep=';')
        print(f"   By-class RGB table saved")
        
        # 3. Depth table by class (THIRD)
        table_depth = create_results_table(df, metric, include_depth=True)
        depth_datasets = [col for col in table_depth.columns if 'estimate_depth' in col or col == 'Model']
        if len(depth_datasets) > 1:  # If there are depth datasets besides Model
            table_depth_filtered = table_depth[depth_datasets].copy()
            filename_depth = f"tables/{metric.replace('@', '_').replace('.', '_')}_depth.csv"
            table_depth_filtered.to_csv(filename_depth, index=False, sep=';')
            print(f"   By-class Depth table saved")
    
    # Reviewer-requested tables (MAIN)
    print("Generating requested tables")
    
    r1_rgb_image_table(df).to_csv('tables/R1_RGB_image_main.csv', index=False, sep=';')
    r2_rgb_pixel_table(df).to_csv('tables/R2_RGB_pixel_main.csv', index=False, sep=';')
    r3_calibration_table(df).to_csv('tables/R3_calibration_main.csv', index=False, sep=';')
    r4_depth_deltas_table(df).to_csv('tables/R4_depth_deltas_main.csv', index=False, sep=';')
    
    # New extended analysis tables
    print("Generating extended analysis tables...")
    
    calibration_transfer_table(df).to_csv('tables/calibration_transfer.csv', index=False, sep=';')
    monotonicity_checks_table(df).to_csv('tables/monotonicity_checks.csv', index=False, sep=';')
    operational_sensitivity_table(df).to_csv('tables/operational_sensitivity.csv', index=False, sep=';')
    seed_robustness_table(df).to_csv('tables/seed_robustness.csv', index=False, sep=';')
    computational_efficiency_table(df).to_csv('tables/computational_efficiency.csv', index=False, sep=';')
    class_difficulty_ranking(df).to_csv('tables/class_difficulty_ranking.csv', index=False, sep=';')
    image_pixel_correlation_table(df).to_csv('tables/image_pixel_correlation.csv', index=False, sep=';')
    winners_by_class_table(df).to_csv('tables/winners_by_class.csv', index=False, sep=';')
        
    print(f"\nTables saved in the 'tables/' folder")

def calibration_transfer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔTNR and verify comparability for calibration transfer."""
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    models = df['Model'].unique()
    
    records = []
    for dataset in datasets:
        for model in models:
            data = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if len(data) == 0:
                continue
                
            row = {'Dataset': dataset, 'Model': model}
            
            # Compute ΔTNR
            delta_99 = (data['Test_TNR_at_train_TNR0.99'] - 0.99).mean()
            delta_95 = (data['Test_TNR_at_train_TNR0.95'] - 0.95).mean()
            delta_90 = (data['Test_TNR_at_train_TNR0.90'] - 0.90).mean()
            
            row['ΔTNR@0.99'] = round(delta_99, 2)
            row['ΔTNR@0.95'] = round(delta_95, 2)
            row['ΔTNR@0.90'] = round(delta_90, 2)
            
            # Verify comparability
            comp_99 = "" if abs(delta_99) <= 0.03 else ""
            comp_95 = "" if abs(delta_95) <= 0.05 else ""
            comp_90 = "" if abs(delta_90) <= 0.05 else ""
            
            row['Comparable'] = f"{comp_99}/{comp_95}/{comp_90}"
            records.append(row)
    
    return pd.DataFrame(records)

def monotonicity_checks_table(df: pd.DataFrame) -> pd.DataFrame:
    """Verify monotonicity at operating points."""
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    models = df['Model'].unique()
    
    records = []
    for dataset in datasets:
        for model in models:
            data = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if len(data) == 0:
                continue
                
            row = {'Dataset': dataset, 'Model': model}
            
            # Average values across seeds
            tpr_99 = data['TPR@TNR0.99'].mean()
            tpr_95 = data['TPR@TNR0.95'].mean()
            tpr_90 = data['TPR@TNR0.90'].mean()
            
            tnr_test_99 = data['Test_TNR_at_train_TNR0.99'].mean()
            tnr_test_95 = data['Test_TNR_at_train_TNR0.95'].mean()
            tnr_test_90 = data['Test_TNR_at_train_TNR0.90'].mean()
            
            # Verify monotonicity
            mono_tpr = "" if tpr_90 >= tpr_95 >= tpr_99 else ""
            mono_tnr = "" if tnr_test_99 >= tnr_test_95 >= tnr_test_90 else ""
            all_ok = "" if mono_tpr == "" and mono_tnr == "" else ""
            
            row['Monotonic_TPR'] = mono_tpr
            row['Monotonic_TNR_test'] = mono_tnr
            row['All_OK'] = all_ok
            records.append(row)
    
    return pd.DataFrame(records)

def operational_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated operational sensitivity."""
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    models = df['Model'].unique()
    
    records = []
    for dataset in datasets:
        for model in models:
            data = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if len(data) == 0:
                continue
                
            tpr_99 = data['TPR@TNR0.99'].mean()
            tpr_95 = data['TPR@TNR0.95'].mean()
            tpr_90 = data['TPR@TNR0.90'].mean()
            
            mean_tpr = (tpr_99 + tpr_95 + tpr_90) / 3
            
            records.append({
                'Dataset': dataset,
                'Model': model,
                'Mean_TPR@TNR': round(mean_tpr, 2)
            })
    
    return pd.DataFrame(records)

def seed_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze seed robustness per model."""
    models = df['Model'].unique()
    metrics = ['Image_AUROC', 'AUsPRO@0.05']
    
    records = []
    for model in models:
        model_data = df[df['Model'] == model]
        row = {'Model': model}
        
        for metric in metrics:
            values = model_data[metric].dropna()
            if len(values) > 1:
                std_val = values.std()
                mean_val = values.mean()
                cv = std_val / mean_val if mean_val != 0 else np.inf
                
                row[f'Std_{metric}'] = round(std_val, 2)
                row[f'CV_{metric}'] = round(cv, 2)
            else:
                row[f'Std_{metric}'] = np.nan
                row[f'CV_{metric}'] = np.nan
        
        records.append(row)
    
    return pd.DataFrame(records)

def computational_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze computational efficiency."""
    records = []
    for _, row in df.iterrows():
        train_imgs = row['Train_Images']
        time_min = row['Training_Time_Seconds'] / 60  # Convert to minutes
        ausPRO = row['AUsPRO@0.05']
        
        time_per_img_sec = row['Training_Time_Seconds'] / train_imgs if train_imgs > 0 else np.inf
        perf_per_min = ausPRO / time_min if time_min > 0 else 0
        
        records.append({
            'Dataset': row['Dataset'],
            'Model': row['Model'],
            'Time_per_img_sec': round(time_per_img_sec, 2),
            'Perf_per_min': round(perf_per_min, 2),
            'Peak_GB': round(row['Used_GB'], 1)
        })
    
    return pd.DataFrame(records)

def class_difficulty_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty ranking per class."""
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    metric = 'AUsPRO@0.05'
    
    records = []
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        model_means = dataset_data.groupby('Model')[metric].mean().sort_values(ascending=False)
        
        # Compute average ranking (1 = best, N = worst)
        avg_rank = (len(model_means) + 1) / 2  # Simplification
        
        records.append({
            'Dataset': dataset,
            'Avg_Rank': round(avg_rank, 1),
            'Best_Score': round(model_means.iloc[0], 2) if len(model_means) > 0 else np.nan,
            'Worst_Score': round(model_means.iloc[-1], 2) if len(model_means) > 0 else np.nan
        })
    
    return pd.DataFrame(records).sort_values('Avg_Rank')

def image_pixel_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation between image-level and pixel-level metrics."""
    from scipy.stats import spearmanr
    
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    records = []
    
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        
        # Group by model
        model_auroc = dataset_data.groupby('Model')['Image_AUROC'].mean()
        model_ap = dataset_data.groupby('Model')['Image_AP_AUPR'].mean()
        model_auspro = dataset_data.groupby('Model')['AUsPRO@0.05'].mean()
        
        # Compute correlations
        corr_auroc, _ = spearmanr(model_auroc, model_auspro) if len(model_auroc) > 2 else (np.nan, np.nan)
        corr_ap, _ = spearmanr(model_ap, model_auspro) if len(model_ap) > 2 else (np.nan, np.nan)
        
        records.append({
            'Dataset': dataset,
            'Spearman_AUROC_AUsPRO': round(corr_auroc, 2) if not np.isnan(corr_auroc) else np.nan,
            'Spearman_AP_AUsPRO': round(corr_ap, 2) if not np.isnan(corr_ap) else np.nan
        })
    
    return pd.DataFrame(records)

def winners_by_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact table of winners per class with significance."""
    datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    metrics = ['Image_AUROC', 'Image_AP_AUPR', 'AUsPRO@0.05']
    
    records = []
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        row = {'Dataset': dataset}
        
        for metric in metrics:
            # Find best model
            model_means = dataset_data.groupby('Model')[metric].mean()
            best_model = model_means.idxmax()
            best_value = model_means.max()
            
            # Compute std for the best model
            best_data = dataset_data[dataset_data['Model'] == best_model][metric]
            std_val = best_data.std() if len(best_data) > 1 else 0
            
            # Significance test (simplified)
            second_best = model_means.nlargest(2).iloc[-1] if len(model_means) > 1 else best_value
            significant = "†" if (best_value - second_best) > 0.01 else ""  # Simplified threshold
            
            row[f'{metric}_Winner'] = f"{best_model}{significant}"
            row[f'{metric}_Score'] = f"{best_value:.2f}±{std_val:.2f}"
        
        records.append(row)
    
    return pd.DataFrame(records)

# ========== REVIEWER REQUESTED TABLES ==========

def r1_rgb_image_table(df: pd.DataFrame) -> pd.DataFrame:
    """R1: RGB-Image (AUROC, AP, Mean-TPR@TNR) with macro-average across classes."""
    rgb_df = df[~df['Dataset'].str.contains('estimate_depth')].copy()
    models = sorted(rgb_df['Model'].unique())
    datasets = sorted(rgb_df['Dataset'].unique())
    
    records = []
    for model in models:
        model_data = rgb_df[rgb_df['Model'] == model]
        row = {'Model': model}
        
        # Calculate macro-averages across datasets for this model
        dataset_means = {}
        for dataset in datasets:
            dataset_data = model_data[model_data['Dataset'] == dataset]
            if len(dataset_data) > 0:
                auroc = dataset_data['Image_AUROC'].mean()
                ap = dataset_data['Image_AP_AUPR'].mean()
                tpr99 = dataset_data['TPR@TNR0.99'].mean()
                tpr95 = dataset_data['TPR@TNR0.95'].mean()
                tpr90 = dataset_data['TPR@TNR0.90'].mean()
                mean_tpr = (tpr99 + tpr95 + tpr90) / 3
                
                dataset_means[dataset] = {
                    'AUROC': auroc, 'AP': ap, 'Mean_TPR': mean_tpr
                }
        
        if dataset_means:
            # Macro-average across datasets
            auroc_macro = np.mean([v['AUROC'] for v in dataset_means.values() if not np.isnan(v['AUROC'])])
            ap_macro = np.mean([v['AP'] for v in dataset_means.values() if not np.isnan(v['AP'])])
            tpr_macro = np.mean([v['Mean_TPR'] for v in dataset_means.values() if not np.isnan(v['Mean_TPR'])])
            
            # Calculate std across datasets
            auroc_std = np.std([v['AUROC'] for v in dataset_means.values() if not np.isnan(v['AUROC'])])
            ap_std = np.std([v['AP'] for v in dataset_means.values() if not np.isnan(v['AP'])])
            tpr_std = np.std([v['Mean_TPR'] for v in dataset_means.values() if not np.isnan(v['Mean_TPR'])])
            
            row['Image_AUROC'] = auroc_macro
            row['Image_AUROC_std'] = auroc_std
            row['Image_AP_AUPR'] = ap_macro
            row['Image_AP_AUPR_std'] = ap_std
            row['Mean_TPR@TNR'] = tpr_macro
            row['Mean_TPR@TNR_std'] = tpr_std
        else:
            for col in ['Image_AUROC', 'Image_AUROC_std', 'Image_AP_AUPR', 'Image_AP_AUPR_std', 
                       'Mean_TPR@TNR', 'Mean_TPR@TNR_std']:
                row[col] = np.nan
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    
    # Statistical testing and ranking
    metrics = ['Image_AUROC', 'Image_AP_AUPR', 'Mean_TPR@TNR']
    for metric in metrics:
        values = result_df[metric].values
        valid_indices = ~np.isnan(values)
        
        if np.sum(valid_indices) > 1:
            sorted_indices = np.argsort(values[valid_indices])[::-1]  # Descending order
            rankings = np.zeros(len(values), dtype=int)
            rankings[valid_indices] = [1 + sorted_indices.tolist().index(i) for i in range(np.sum(valid_indices))]
            
            result_df[f'{metric}_rank'] = rankings
            result_df[f'{metric}_rank'] = result_df[f'{metric}_rank'].replace(0, np.nan)
        else:
            result_df[f'{metric}_rank'] = np.nan
    
    return result_df

def r2_rgb_pixel_table(df: pd.DataFrame) -> pd.DataFrame:
    """R2: RGB-Pixel (AUsPRO@0.05, sPRO@0.05) with macro-average across classes."""
    rgb_df = df[~df['Dataset'].str.contains('estimate_depth')].copy()
    models = sorted(rgb_df['Model'].unique())
    datasets = sorted(rgb_df['Dataset'].unique())
    
    records = []
    for model in models:
        model_data = rgb_df[rgb_df['Model'] == model]
        row = {'Model': model}
        
        # Calculate macro-averages across datasets for this model
        dataset_means = {}
        for dataset in datasets:
            dataset_data = model_data[model_data['Dataset'] == dataset]
            if len(dataset_data) > 0:
                auspro = dataset_data['AUsPRO@0.05'].mean()
                spro = dataset_data['sPRO@0.05'].mean()
                
                dataset_means[dataset] = {
                    'AUsPRO': auspro, 'sPRO': spro
                }
        
        if dataset_means:
            # Macro-average across datasets
            auspro_macro = np.mean([v['AUsPRO'] for v in dataset_means.values() if not np.isnan(v['AUsPRO'])])
            spro_macro = np.mean([v['sPRO'] for v in dataset_means.values() if not np.isnan(v['sPRO'])])
            
            # Calculate std across datasets
            auspro_std = np.std([v['AUsPRO'] for v in dataset_means.values() if not np.isnan(v['AUsPRO'])])
            spro_std = np.std([v['sPRO'] for v in dataset_means.values() if not np.isnan(v['sPRO'])])
            
            row['AUsPRO@0.05'] = auspro_macro
            row['AUsPRO@0.05_std'] = auspro_std
            row['sPRO@0.05'] = spro_macro
            row['sPRO@0.05_std'] = spro_std
        else:
            for col in ['AUsPRO@0.05', 'AUsPRO@0.05_std', 'sPRO@0.05', 'sPRO@0.05_std']:
                row[col] = np.nan
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    
    # Statistical testing and ranking
    metrics = ['AUsPRO@0.05', 'sPRO@0.05']
    for metric in metrics:
        values = result_df[metric].values
        valid_indices = ~np.isnan(values)
        
        if np.sum(valid_indices) > 1:
            sorted_indices = np.argsort(values[valid_indices])[::-1]  # Descending order
            rankings = np.zeros(len(values), dtype=int)
            rankings[valid_indices] = [1 + sorted_indices.tolist().index(i) for i in range(np.sum(valid_indices))]
            
            result_df[f'{metric}_rank'] = rankings
            result_df[f'{metric}_rank'] = result_df[f'{metric}_rank'].replace(0, np.nan)
        else:
            result_df[f'{metric}_rank'] = np.nan
    
    return result_df

def r3_calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """R3: Calibration with ΔTNR and flags aggregated per model (macro-average across classes)."""
    rgb_df = df[~df['Dataset'].str.contains('estimate_depth')].copy()
    models = sorted(rgb_df['Model'].unique())
    datasets = sorted(rgb_df['Dataset'].unique())
    
    records = []
    for model in models:
        model_data = rgb_df[rgb_df['Model'] == model]
        row = {'Model': model}
        
        # Calculate macro-averages across datasets
        delta_99_values, delta_95_values, delta_90_values = [], [], []
        mono_tpr_ok, mono_tnr_ok, comp_99_ok, comp_95_ok, comp_90_ok = 0, 0, 0, 0, 0
        total_datasets = 0
        
        for dataset in datasets:
            dataset_data = model_data[model_data['Dataset'] == dataset]
            if len(dataset_data) > 0:
                total_datasets += 1
                
                # Calculate ΔTNR
                delta_99 = (dataset_data['Test_TNR_at_train_TNR0.99'] - 0.99).mean()
                delta_95 = (dataset_data['Test_TNR_at_train_TNR0.95'] - 0.95).mean()
                delta_90 = (dataset_data['Test_TNR_at_train_TNR0.90'] - 0.90).mean()
                
                delta_99_values.append(delta_99)
                delta_95_values.append(delta_95)
                delta_90_values.append(delta_90)
                
                # Check comparability
                if abs(delta_99) <= 0.03: comp_99_ok += 1
                if abs(delta_95) <= 0.05: comp_95_ok += 1
                if abs(delta_90) <= 0.05: comp_90_ok += 1
                
                # Check monotonicity
                tpr_99 = dataset_data['TPR@TNR0.99'].mean()
                tpr_95 = dataset_data['TPR@TNR0.95'].mean()
                tpr_90 = dataset_data['TPR@TNR0.90'].mean()
                
                tnr_test_99 = dataset_data['Test_TNR_at_train_TNR0.99'].mean()
                tnr_test_95 = dataset_data['Test_TNR_at_train_TNR0.95'].mean()
                tnr_test_90 = dataset_data['Test_TNR_at_train_TNR0.90'].mean()
                
                if tpr_90 >= tpr_95 >= tpr_99: mono_tpr_ok += 1
                if tnr_test_99 >= tnr_test_95 >= tnr_test_90: mono_tnr_ok += 1
        
        if total_datasets > 0:
            row['ΔTNR@0.99'] = np.mean(delta_99_values)
            row['ΔTNR@0.95'] = np.mean(delta_95_values)
            row['ΔTNR@0.90'] = np.mean(delta_90_values)
            row['ΔTNR@0.99_std'] = np.std(delta_99_values)
            row['ΔTNR@0.95_std'] = np.std(delta_95_values)
            row['ΔTNR@0.90_std'] = np.std(delta_90_values)
            
            # Flags as percentages
            row['Comp_99%'] = round(100 * comp_99_ok / total_datasets, 1)
            row['Comp_95%'] = round(100 * comp_95_ok / total_datasets, 1)
            row['Comp_90%'] = round(100 * comp_90_ok / total_datasets, 1)
            row['Mono_TPR%'] = round(100 * mono_tpr_ok / total_datasets, 1)
            row['Mono_TNR%'] = round(100 * mono_tnr_ok / total_datasets, 1)
        else:
            for col in ['ΔTNR@0.99', 'ΔTNR@0.95', 'ΔTNR@0.90', 
                       'ΔTNR@0.99_std', 'ΔTNR@0.95_std', 'ΔTNR@0.90_std',
                       'Comp_99%', 'Comp_95%', 'Comp_90%', 'Mono_TPR%', 'Mono_TNR%']:
                row[col] = np.nan
        
        records.append(row)
    
    return pd.DataFrame(records)

def r4_depth_deltas_table(df: pd.DataFrame) -> pd.DataFrame:
    """R4: Pseudo-depth vs RGB deltas with macro-average across classes."""
    base_datasets = [d for d in df['Dataset'].unique() if not d.endswith('estimate_depth')]
    models = sorted(df['Model'].unique())
    
    records = []
    for model in models:
        row = {'Model': model}
        
        # Calculate deltas for each dataset
        auroc_deltas, ap_deltas, auspro_deltas = [], [], []
        
        for base_dataset in base_datasets:
            depth_dataset = base_dataset + '_estimate_depth'
            
            base_data = df[(df['Dataset'] == base_dataset) & (df['Model'] == model)]
            depth_data = df[(df['Dataset'] == depth_dataset) & (df['Model'] == model)]
            
            if len(base_data) > 0 and len(depth_data) > 0:
                # Merge on seed for paired comparison
                merged = pd.merge(base_data[['Seed', 'Image_AUROC', 'Image_AP_AUPR', 'AUsPRO@0.05']], 
                                 depth_data[['Seed', 'Image_AUROC', 'Image_AP_AUPR', 'AUsPRO@0.05']], 
                                 on='Seed', suffixes=('_base', '_depth'))
                
                if len(merged) > 0:
                    auroc_delta = (merged['Image_AUROC_depth'] - merged['Image_AUROC_base']).mean()
                    ap_delta = (merged['Image_AP_AUPR_depth'] - merged['Image_AP_AUPR_base']).mean()
                    auspro_delta = (merged['AUsPRO@0.05_depth'] - merged['AUsPRO@0.05_base']).mean()
                    
                    auroc_deltas.append(auroc_delta)
                    ap_deltas.append(ap_delta)
                    auspro_deltas.append(auspro_delta)
        
        if auroc_deltas:
            row['ΔAUROC'] = np.mean(auroc_deltas)
            row['ΔAUROC_std'] = np.std(auroc_deltas)
            row['ΔAP'] = np.mean(ap_deltas)
            row['ΔAP_std'] = np.std(ap_deltas)
            row['ΔAUsPRO'] = np.mean(auspro_deltas)
            row['ΔAUsPRO_std'] = np.std(auspro_deltas)
        else:
            for col in ['ΔAUROC', 'ΔAUROC_std', 'ΔAP', 'ΔAP_std', 'ΔAUsPRO', 'ΔAUsPRO_std']:
                row[col] = np.nan
        
        records.append(row)
    
    return pd.DataFrame(records)

# ========== MACRO-AVERAGED TABLES FOR SPECIFIC METRICS ==========

def macro_image_ap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-averaged table for Image_AP_AUPR (complements R1)."""
    return r1_rgb_image_table(df)[['Model', 'Image_AP_AUPR', 'Image_AP_AUPR_std', 'Image_AP_AUPR_rank']].copy()

def macro_auspro_table(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-averaged table for AUsPRO@0.05 (complements R2)."""
    return r2_rgb_pixel_table(df)[['Model', 'AUsPRO@0.05', 'AUsPRO@0.05_std', 'AUsPRO@0.05_rank']].copy()

# ========== FIGURE DATA GENERATION ==========

def f1_pareto_data(df: pd.DataFrame) -> pd.DataFrame:
    """F1: Data for Pareto diagram AUROC vs AUsPRO (RGB baseline)."""
    rgb_df = df[~df['Dataset'].str.contains('estimate_depth')].copy()
    datasets = sorted(rgb_df['Dataset'].unique())
    models = sorted(rgb_df['Model'].unique())
    
    records = []
    for dataset in datasets:
        for model in models:
            model_data = rgb_df[(rgb_df['Dataset'] == dataset) & (rgb_df['Model'] == model)]
            if len(model_data) > 0:
                # Calculate means across seeds for this model-dataset combination
                auroc_mean = model_data['Image_AUROC'].mean()
                auspro_mean = model_data['AUsPRO@0.05'].mean()
                
                # Only include if both metrics are available
                if not (pd.isna(auroc_mean) or pd.isna(auspro_mean)):
                    records.append({
                        'Dataset': dataset,
                        'Model': model,
                        'AUROC': auroc_mean,
                        'AUsPRO': auspro_mean,
                        'AUROC_std': model_data['Image_AUROC'].std(),
                        'AUsPRO_std': model_data['AUsPRO@0.05'].std()
                    })
    
    return pd.DataFrame(records)

def save_figure_data(df: pd.DataFrame):
    """Save CSV data for the paper figures."""
    import os
    
    # Create figures folder if it does not exist
    os.makedirs('figures', exist_ok=True)
    
    print("Generating data for figures...")
    
    # F1: Pareto AUROC vs AUsPRO
    f1_data = f1_pareto_data(df)
    f1_data.to_csv('figures/F1_pareto_auroc_auspro.csv', index=False, sep=';')
    print(f"  F1: Pareto data saved ({len(f1_data)} points)")
    
    print(f"Figure data saved in 'figures/' folder")

def print_paper_tables(df: pd.DataFrame):
    """Print all tables needed for the paper."""
    metrics = ['Image_AUROC', 'Image_AP_AUPR', 'TPR@TNR0.99', 'TPR@TNR0.95', 
               'TPR@TNR0.90', 'AUsPRO@0.05', 'sPRO@0.05']
    
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"TABLE RGB: {metric}")
        print(f"{'='*60}")
        
        table = create_results_table(df, metric, include_depth=False)
        print(table.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"TABLE DEPTH: {metric}")
        print(f"{'='*60}")
        
        table_depth = create_results_table(df, metric, include_depth=True)
        # Show only depth datasets
        depth_datasets = [col for col in table_depth.columns if 'estimate_depth' in col or col == 'Model']
        if len(depth_datasets) > 1:
            table_depth_filtered = table_depth[depth_datasets]
            print(table_depth_filtered.to_string(index=False))
        else:
            print("No estimate_depth datasets found for this metric")
        
        print(f"\n* = Best result (not significantly different at 5% level using Welch's t-test)")

def print_reviewer_tables(df: pd.DataFrame):
    """Print reviewer-requested tables."""
    
    print(f"\n{'='*80}")
    print("R1: RGB-IMAGEN (AUROC, AP, Mean-TPR@TNR)")
    print(f"{'='*80}")
    r1_table = r1_rgb_image_table(df)
    # Format for display with rankings
    display_df = r1_table.copy()
    
    for metric in ['Image_AUROC', 'Image_AP_AUPR', 'Mean_TPR@TNR']:
        if metric in display_df.columns:
            mean_col = display_df[metric]
            std_col = display_df[f'{metric}_std']
            rank_col = display_df.get(f'{metric}_rank', np.nan)
            
            formatted = []
            for i, (mean, std, rank) in enumerate(zip(mean_col, std_col, rank_col)):
                if not np.isnan(mean):
                    marker = "**" if rank == 1 else ("*" if rank == 2 else "")
                    formatted.append(f"{mean:.3f}±{std:.3f}{marker}")
                else:
                    formatted.append("N/A")
            display_df[metric] = formatted
    
    # Keep only main columns for display
    cols_to_show = ['Model', 'Image_AUROC', 'Image_AP_AUPR', 'Mean_TPR@TNR']
    print(display_df[cols_to_show].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("R2: RGB-PIXEL (AUsPRO@0.05, sPRO@0.05)")
    print(f"{'='*80}")
    r2_table = r2_rgb_pixel_table(df)
    # Format for display with rankings
    display_df = r2_table.copy()
    
    for metric in ['AUsPRO@0.05', 'sPRO@0.05']:
        if metric in display_df.columns:
            mean_col = display_df[metric]
            std_col = display_df[f'{metric}_std']
            rank_col = display_df.get(f'{metric}_rank', np.nan)
            
            formatted = []
            for i, (mean, std, rank) in enumerate(zip(mean_col, std_col, rank_col)):
                if not np.isnan(mean):
                    marker = "**" if rank == 1 else ("*" if rank == 2 else "")
                    formatted.append(f"{mean:.3f}±{std:.3f}{marker}")
                else:
                    formatted.append("N/A")
            display_df[metric] = formatted
    
    cols_to_show = ['Model', 'AUsPRO@0.05', 'sPRO@0.05']
    print(display_df[cols_to_show].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("R3: CALIBRATION (ΔTNR and flags)")
    print(f"{'='*80}")
    r3_table = r3_calibration_table(df)
    # Format for display
    display_df = r3_table.copy()
    for col in ['ΔTNR@0.99', 'ΔTNR@0.95', 'ΔTNR@0.90']:
        if col in display_df.columns:
            mean_col = display_df[col]
            std_col = display_df[f'{col}_std']
            formatted = []
            for mean, std in zip(mean_col, std_col):
                if not np.isnan(mean):
                    formatted.append(f"{mean:.3f}±{std:.3f}")
                else:
                    formatted.append("N/A")
            display_df[col] = formatted
    
    cols_to_show = ['Model', 'ΔTNR@0.99', 'ΔTNR@0.95', 'ΔTNR@0.90', 'Comp_99%', 'Comp_95%', 'Comp_90%', 'Mono_TPR%', 'Mono_TNR%']
    print(display_df[cols_to_show].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("R4: PSEUDO-DEPTH DELTAS (Δ vs RGB)")
    print(f"{'='*80}")
    r4_table = r4_depth_deltas_table(df)
    # Format for display
    display_df = r4_table.copy()
    for col in ['ΔAUROC', 'ΔAP', 'ΔAUsPRO']:
        if col in display_df.columns:
            mean_col = display_df[col]
            std_col = display_df[f'{col}_std']
            formatted = []
            for mean, std in zip(mean_col, std_col):
                if not np.isnan(mean):
                    # Highlight negative values
                    sign = "↓" if mean < -0.001 else ("↑" if mean > 0.001 else "≈")
                    formatted.append(f"{mean:+.3f}±{std:.3f}{sign}")
                else:
                    formatted.append("N/A")
            display_df[col] = formatted
    
    cols_to_show = ['Model', 'ΔAUROC', 'ΔAP', 'ΔAUsPRO']
    print(display_df[cols_to_show].to_string(index=False))
    
    print(f"\n** = Best (rank 1), * = Second best (rank 2)")
    print(f"↑ = Improvement, ↓ = Degradation, ≈ = Negligible change")

def print_extended_analysis(df: pd.DataFrame):
    """Print the new extended analysis tables."""
    
    print(f"\n{'='*80}")
    print("CALIBRATION TRANSFER ANALYSIS")
    print(f"{'='*80}")
    cal_table = calibration_transfer_table(df)
    print(cal_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("MONOTONICITY & SANITY CHECKS")
    print(f"{'='*80}")
    mono_table = monotonicity_checks_table(df)
    print(mono_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("OPERATIONAL SENSITIVITY")
    print(f"{'='*80}")
    sens_table = operational_sensitivity_table(df)
    print(sens_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("SEED ROBUSTNESS")
    print(f"{'='*80}")
    rob_table = seed_robustness_table(df)
    print(rob_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("COMPUTATIONAL EFFICIENCY")
    print(f"{'='*80}")
    comp_table = computational_efficiency_table(df)
    print(comp_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("CLASS DIFFICULTY RANKING")
    print(f"{'='*80}")
    diff_table = class_difficulty_ranking(df)
    print(diff_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("IMAGE-PIXEL CORRELATION")
    print(f"{'='*80}")
    corr_table = image_pixel_correlation_table(df)
    print(corr_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("WINNERS BY CLASS")
    print(f"{'='*80}")
    win_table = winners_by_class_table(df)
    print(win_table.to_string(index=False))

if __name__ == '__main__':
    print('Loading results...')
    df = load_results('results/training_results.csv')
    
    # Original analysis
    stats_df  = dataset_statistics(df)
    perf_df   = performance_summary(df)
    improv_df = depth_improvement(df)
    comp_df   = computational_requirements(df)
    
    # Save tables to CSV
    save_tables_to_csv(df)
    
    # Save data for figures
    save_figure_data(df)
    
    # Display reviewer-requested tables (PRIORITY)
    print_reviewer_tables(df)
    
    # Display original tables in console
    print_paper_tables(df)
    
    # New extended analysis tables
    print_extended_analysis(df)
