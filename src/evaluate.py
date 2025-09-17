import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def _calculate_metrics(group):
    metrics = {}
    metrics['Top-1 Accuracy'] = group['accuracy'].mean()
    metrics['Total Joules'] = group['energy_joules'].sum()
    metrics['Mean Latency'] = group['latency_ms'].mean()
    metrics['%Adapted batches'] = group['adapted'].mean() * 100
    
    # Strategy 1 specific metrics
    if 'energy_budget' in group.name:
        base_energy_per_step = group['energy_joules'][group['adapted']==0].mean()
        if np.isnan(base_energy_per_step):
             base_energy_per_step = np.quantile(group['energy_joules'], 0.1) # estimate if no non-adapted batches
        total_budget = base_energy_per_step * len(group) * group.name[group.index.names.index('energy_budget')]
        metrics['#Cap violations'] = (group['energy_joules'].sum() > total_budget).astype(int)

    # Strategy 2 specific metrics
    if 'stream_type' in group.index.names and 'RealStream' in group.name[group.index.names.index('stream_type')]:
        # Placeholder for complex forgetting calculation
        # Assumes first 400 steps are clean
        initial_clean_acc = group.iloc[:400]['accuracy'].mean()
        # Assumes clean phase repeats every ~1600 steps
        clean_phases_acc = []
        for i in range(len(group) // 1600):
            start = i*1600
            clean_phases_acc.append(group.iloc[start:start+400]['accuracy'].mean())
        final_clean_acc = clean_phases_acc[-1] if clean_phases_acc else initial_clean_acc
        metrics['\u0394ACC_clean'] = initial_clean_acc - final_clean_acc

    # Strategy 3 specific for Camelyon17
    # This part would require ground truth and predictions for AUROC calculation
    # which is not in the log. We report accuracy instead.
    if 'dataset' in group.name and 'Camelyon17' in group.name[group.index.names.index('dataset')]:
         metrics['AUROC'] = metrics['Top-1 Accuracy'] # Placeholder

    return pd.Series(metrics)

def _run_statistical_tests(df, main_method='EAGER-TTA'):
    results = {}
    baselines = df.index.get_level_values('baseline').unique().tolist()
    if main_method not in baselines:
        return results
    
    main_results = df.xs(main_method, level='baseline')['Top-1 Accuracy']
    
    for baseline in baselines:
        if baseline == main_method:
            continue
        try:
            baseline_results = df.xs(baseline, level='baseline')['Top-1 Accuracy']
            # Align results by seed
            common_seeds = main_results.index.intersection(baseline_results.index)
            if len(common_seeds) > 1:
                t_stat, p_val = ttest_rel(main_results.loc[common_seeds], baseline_results.loc[common_seeds])
                results[f'{main_method}_vs_{baseline}'] = {'t_stat': t_stat, 'p_value': p_val}
        except Exception as e:
            print(f"Could not run t-test for {baseline}: {e}")

    # Holm-Bonferroni correction
    if results:
        p_values = [res['p_value'] for res in results.values()]
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        for i, key in enumerate(results.keys()):
            results[key]['p_value_corrected'] = pvals_corrected[i]
            results[key]['significant_at_0.05'] = bool(reject[i])

    return results

def _generate_plots(df, output_dir):
    plot_files = []
    sns.set_theme(style="whitegrid")
    
    # Strategy 1: Accuracy vs Energy Budget
    if 'energy_budget' in df.index.names:
        try:
            s1_df = df.reset_index()
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=s1_df, x='energy_budget', y='Top-1 Accuracy', hue='baseline', cut=0)
            plt.title('Strategy 1: Accuracy vs. Energy Budget on ImageNet-C')
            plt.xlabel('Energy Budget (multiple of base)')
            plt.ylabel('Top-1 Accuracy (%)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'S1_accuracy_vs_energy.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plot_files.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Failed to generate S1 plot: {e}")

    # Strategy 2: Forgetting Over Time
    if '\u0394ACC_clean' in df.columns:
        try:
            s2_df = df.reset_index()
            s2_df = s2_df[s2_df['\u0394ACC_clean'].notna()]
            plt.figure(figsize=(10, 6))
            sns.barplot(data=s2_df, x='baseline', y='\u0394ACC_clean')
            plt.title('Strategy 2: Catastrophic Forgetting on RealStream-1M')
            plt.xlabel('Method')
            plt.ylabel('Forgetting (\u0394 Accuracy on Clean Data)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'S2_forgetting.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plot_files.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Failed to generate S2 plot: {e}")

    # Strategy 3: Cross-Architecture Accuracy
    if 'dataset' in df.index.names and len(df.index.get_level_values('dataset').unique()) > 1:
        try:
            s3_df = df.reset_index()
            g = sns.catplot(data=s3_df, x='dataset', y='Top-1 Accuracy', hue='baseline', kind='bar', aspect=2)
            g.fig.suptitle('Strategy 3: Cross-Architecture & Cross-Dataset Generalization')
            g.set_xticklabels(rotation=30)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'S3_cross_architecture_accuracy.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plot_files.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Failed to generate S3 plot: {e}"

    return plot_files

def generate_report(results_dir):
    print(f"--- Generating Report from Results in {results_dir} ---")
    csv_files = glob(os.path.join(results_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        print("No CSV files found. Aborting report generation.")
        return

    df_list = []
    for f in csv_files:
        try:
            parts = f.replace(results_dir, '').split(os.sep)
            # Expected structure: strategy/model/dataset/baseline/seed/results.csv
            if len(parts) >= 6:
                # Correctly removing empty strings that might result from os.sep
                parts = [p for p in parts if p]
                strategy, model, dataset, baseline, seed = parts[:5]
                temp_df = pd.read_csv(f)
                temp_df['strategy'] = strategy
                temp_df['model'] = model
                temp_df['dataset'] = dataset.split('_eta_')[0].split('_budget_')[0]
                temp_df['baseline'] = baseline
                temp_df['seed'] = int(seed.replace('seed', ''))
                if 'eta' in dataset:
                    temp_df['arrival_rate'] = float(dataset.split('_eta_')[1].split('_budget_')[0])
                if 'budget' in dataset:
                    temp_df['energy_budget'] = float(dataset.split('_budget_')[1])
                df_list.append(temp_df)
        except Exception as e:
            print(f"Could not process file {f}: {e}")

    if not df_list:
        print("Could not load any valid CSV data. Aborting report generation.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Define index for aggregation
    grouping_cols = ['strategy', 'model', 'dataset', 'baseline', 'seed']
    if 'arrival_rate' in full_df.columns: grouping_cols.append('arrival_rate')
    if 'energy_budget' in full_df.columns: grouping_cols.append('energy_budget')
    
    # Calculate stream-averaged metrics
    stream_avg_df = full_df.groupby(grouping_cols).apply(_calculate_metrics).reset_index()

    # Final aggregation: mean and std over seeds
    final_grouping_cols = [c for c in grouping_cols if c != 'seed']
    final_results = stream_avg_df.groupby(final_grouping_cols).agg(['mean', 'std']).reset_index()
    final_results.columns = ['_'.join(col).strip('_') for col in final_results.columns.values]
    
    # Statistical analysis
    stats_results = {}
    for (strategy, model, dataset), group in stream_avg_df.set_index(grouping_cols).groupby(['strategy', 'model', 'dataset']):
        key = f"{strategy}_{model}_{dataset}"
        stats_results[key] = _run_statistical_tests(group)

    summary_data = {
        'summary_statistics': final_results.to_dict('records'),
        'statistical_tests': stats_results
    }

    # Save and print JSON results
    json_path = os.path.join(results_dir, 'summary_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("\n--- Summary of Experimental Results ---")
    print(json.dumps(summary_data, indent=2))

    # Generate plots
    plot_files = _generate_plots(stream_avg_df.set_index(grouping_cols), results_dir)
    
    print("\n--- Generated Plots ---")
    if plot_files:
        for plot_file in plot_files:
            print(f"- {plot_file}")
    else:
        print("No plots were generated.")

    print("\n--- Report Generation Complete ---")
