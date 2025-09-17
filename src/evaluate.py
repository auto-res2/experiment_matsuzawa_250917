import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

def plot_line_chart(df, x_col, y_cols, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    for y_col in y_cols:
        sns.lineplot(data=df, x=x_col, y=y_col, label=y_col.replace('_', ' ').title(), errorbar=('ci', 95))
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_bar_chart(data, x_col, y_col, hue_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(12, 7))
    sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, errorbar=('ci', 95))
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Plot saved to {output_path}")

def aggregate_results(log_files):
    all_dfs = []
    for log_file in log_files:
        df = pd.read_csv(log_file)
        config_name = os.path.basename(log_file).split('_log.csv')[0]
        parts = config_name.split('_')
        df['model'] = parts[-3]
        df['dataset'] = parts[-2]
        df['seed'] = parts[-1]
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def process_summary(summary_files):
    summary_data = []
    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            data = json.load(f)
            config = data['config']
            exp_name = config['experiment_name']
            parts = exp_name.split('_')
            summary_data.append({
                'model': parts[-3],
                'dataset': parts[-2],
                'seed': parts[-1],
                'best_val_acc': data['best_val_acc'],
                'energy_consumed': data['final_emissions']['energy_consumed'],
                'co2_emissions': data['final_emissions']['co2_emissions_kg']
            })
    return pd.DataFrame(summary_data)

def run_experiment_1(config, results_dir):
    print("\n--- Running Evaluation for Experiment 1: Benchmarking ---")
    log_files = glob.glob(os.path.join(results_dir, "exp1_*_log.csv"))
    summary_files = glob.glob(os.path.join(results_dir, "exp1_*_summary.json"))
    
    if not log_files:
        print("No log files found for experiment 1.")
        return

    full_df = aggregate_results(log_files)
    summary_df = process_summary(summary_files)

    # Generate plots
    plots_dir = os.path.join(results_dir, "images")
    os.makedirs(plots_dir, exist_ok=True)

    # Accuracy curves for each dataset
    for dataset in full_df['dataset'].unique():
        df_subset = full_df[full_df['dataset'] == dataset]
        # Pivot for plotting
        df_pivot = df_subset.pivot_table(index=['epoch', 'seed'], columns='model', values='val_acc').reset_index()
        plot_line_chart(df_pivot, 'epoch', df_pivot.columns[2:], 
                        f'Validation Accuracy on {dataset}', 'Epoch', 'Validation Accuracy',
                        os.path.join(plots_dir, f'exp1_accuracy_{dataset}.pdf'))

    # Bar charts for performance metrics
    perf_metrics = {
        'epoch_time_s': 'Mean Epoch Time (s)',
        'peak_gpu_mem_gb': 'Peak GPU Memory (GB)',
        'energy_consumed': 'Total Energy Consumed (kWh)',
        'co2_emissions': 'Total CO2 Emissions (kg)'
    }
    
    # Aggregate time and memory from epoch logs (avg over epochs 51-150)
    stable_epochs_df = full_df[(full_df['epoch'] >= 51) & (full_df['epoch'] <= 150)]
    perf_df = stable_epochs_df.groupby(['model', 'dataset', 'seed'])[['epoch_time_s', 'peak_gpu_mem_gb']].mean().reset_index()
    # Merge with summary data for energy/co2
    final_perf_df = pd.merge(perf_df, summary_df, on=['model', 'dataset', 'seed'])

    for metric, title in perf_metrics.items():
        plot_bar_chart(final_perf_df, 'dataset', metric, 'model', 
                       title, 'Dataset', title.split(' (')[0],
                       os.path.join(plots_dir, f'exp1_{metric}_comparison.pdf'))

    # Final results aggregation
    final_summary = final_perf_df.groupby(['dataset', 'model']).agg(
        mean_val_acc=('best_val_acc', 'mean'),
        ci_val_acc=('best_val_acc', lambda x: 1.96 * x.std() / np.sqrt(len(x))),
        mean_time=('epoch_time_s', 'mean'),
        mean_mem=('peak_gpu_mem_gb', 'mean'),
        mean_energy=('energy_consumed', 'mean'),
        mean_co2=('co2_emissions', 'mean')
    ).reset_index()

    results_json = final_summary.to_dict('records')
    with open(os.path.join(results_dir, 'experiment_1_results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)

    print("\n--- Experiment 1 Final Results ---")
    print(json.dumps(results_json, indent=4))
    print("--- End Experiment 1 ---")

def run_experiment_2(config, results_dir):
    print("\n--- Running Evaluation for Experiment 2: Robustness ---")
    # Placeholder as perturbation runs are not fully specified in train.py
    print("Experiment 2 evaluation is a placeholder.")
    results_json = {"status": "Placeholder - robustness tests need specific training runs."}
    with open(os.path.join(results_dir, 'experiment_2_results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)
    print(json.dumps(results_json, indent=4))
    print("--- End Experiment 2 ---")

def run_experiment_3(config, results_dir):
    print("\n--- Running Evaluation for Experiment 3: Ablation ---")
    log_files = glob.glob(os.path.join(results_dir, "exp3_*_log.csv"))
    summary_files = glob.glob(os.path.join(results_dir, "exp3_*_summary.json"))

    if not log_files:
        print("No log files found for experiment 3.")
        return
    
    full_df = aggregate_results(log_files)
    summary_df = process_summary(summary_files)
    
    stable_epochs_df = full_df[(full_df['epoch'] >= 51) & (full_df['epoch'] <= 150)]
    perf_df = stable_epochs_df.groupby(['model', 'dataset', 'seed'])[['epoch_time_s', 'peak_gpu_mem_gb']].mean().reset_index()
    final_perf_df = pd.merge(perf_df, summary_df, on=['model', 'dataset', 'seed'])
    
    final_summary = final_perf_df.groupby(['dataset', 'model']).agg(
        mean_val_acc=('best_val_acc', 'mean'),
        mean_time=('epoch_time_s', 'mean'),
        mean_mem=('peak_gpu_mem_gb', 'mean')
    ).reset_index()

    plots_dir = os.path.join(results_dir, "images")
    os.makedirs(plots_dir, exist_ok=True)
    
    for dataset in final_summary['dataset'].unique():
        df_subset = final_summary[final_summary['dataset'] == dataset]
        plot_bar_chart(df_subset, 'model', 'mean_val_acc', None, 
                       f'Ablation Study Accuracy on {dataset}', 'Model Variant', 'Validation Accuracy',
                       os.path.join(plots_dir, f'exp3_accuracy_{dataset}.pdf'))

    results_json = final_summary.to_dict('records')
    with open(os.path.join(results_dir, 'experiment_3_results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)

    print("\n--- Experiment 3 Final Results ---")
    print(json.dumps(results_json, indent=4))
    print("--- End Experiment 3 ---")

def generate_results(config):
    output_dir = config['globals']['output_dir']
    
    if config.get('experiment_1'):
        run_experiment_1(config['experiment_1'], output_dir)
    
    if config.get('experiment_2'):
        run_experiment_2(config['experiment_2'], output_dir)
        
    if config.get('experiment_3'):
        run_experiment_3(config['experiment_3'], output_dir)

