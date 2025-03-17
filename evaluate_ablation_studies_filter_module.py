#!/usr/bin/env python3
# evaluate_advanced_ablation_studies.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate advanced ablation study results')
    parser.add_argument('--results_dir', type=str, default='./ablation_filter_module_results',
                        help='Directory containing all ablation results')
    parser.add_argument('--output_dir', type=str, default='./ablation_filter_module_evaluation',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def get_ablation_configs():
    """Get all possible ablation configurations"""
    ranks = [1, 2, 3]
    filter_types = ['similarity_only', 'domain_only', 'both']
    
    configs = []
    for rank in ranks:
        for filter_type in filter_types:
            config_id = f"rank_{rank}_{filter_type}"
            configs.append({
                'id': config_id,
                'rank': rank,
                'filter_type': filter_type
            })
    
    return configs

def load_results(results_dir, ablation_id):
    """Load all results for a specific ablation configuration"""
    ablation_dir = os.path.join(results_dir, ablation_id)
    results = []
    
    if not os.path.exists(ablation_dir):
        print(f"Warning: Directory not found: {ablation_dir}")
        return results
    
    for filename in os.listdir(ablation_dir):
        if filename.startswith('result_') and filename.endswith('.json'):
            file_path = os.path.join(ablation_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return results

def calculate_metrics(results):
    """Calculate evaluation metrics for the results"""
    if not results:
        return {
            'count': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'avg_inference_time': 0,
            'avg_evidence_count': 0,
            'check_type_distribution': {}
        }
    
    y_true = []
    y_pred = []
    inference_times = []
    evidence_counts = []
    check_types = []
    
    for result in results:
        try:
            # Get ground truth label
            ground_truth = result.get('ground_truth')
            if isinstance(ground_truth, str):
                ground_truth = ground_truth.lower()
            
            # Get prediction
            final_result = result.get('final_result', {})
            prediction = final_result.get('OOC', "")
            if isinstance(prediction, str):
                prediction = prediction.lower()
            
            # Map labels to binary values
            y_true.append(ground_truth)
            y_pred.append(prediction)
            
            # Extract inference time
            inference_times.append(result.get('inference_time', 0))
            
            # Extract evidence count
            check_result = result.get('check_result', {})
            evidences = check_result.get('evidences', [])
            evidence_counts.append(len(evidences))
            
            # Extract check type
            check_types.append(check_result.get('check_type', ''))
            
        except Exception as e:
            print(f"Error processing result: {e}")
            continue
    
    # Calculate metrics
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'count': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'avg_inference_time': 0,
            'avg_evidence_count': 0,
            'check_type_distribution': {}
        }
    
    metrics = {
        'count': len(y_true),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'avg_inference_time': np.mean(inference_times),
        'avg_evidence_count': np.mean(evidence_counts),
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else [[0, 0], [0, 0]],
        'check_type_distribution': {k: check_types.count(k) for k in set(check_types)}
    }
    
    return metrics

def generate_heatmap(metrics_by_config, metric_name, output_dir):
    """Generate a heatmap visualization for a specific metric across all configurations"""
    # Create a dataframe with ranks as rows and filter_types as columns
    ranks = [1, 2, 3]
    filter_types = ['similarity_only', 'domain_only', 'both']
    
    data = np.zeros((len(ranks), len(filter_types)))
    
    for config in metrics_by_config:
        if 'rank' in config and 'filter_type' in config:
            rank_idx = ranks.index(config['rank'])
            filter_idx = filter_types.index(config['filter_type'])
            
            metric_value = metrics_by_config[config['id']].get(metric_name, 0)
            data[rank_idx, filter_idx] = metric_value
    
    plt.figure(figsize=(10, 6))
    
    # Create heatmap
    ax = sns.heatmap(
        data, 
        annot=True, 
        fmt='.4f' if metric_name in ['accuracy', 'precision', 'recall', 'f1'] else '.2f',
        cmap='YlGnBu', 
        xticklabels=[ft.replace('_', ' ').title() for ft in filter_types],
        yticklabels=ranks,
        vmin=0,
        vmax=1 if metric_name in ['accuracy', 'precision', 'recall', 'f1'] else None
    )
    
    plt.title(f'{metric_name.replace("_", " ").title()} by Rank and Filter Type')
    plt.xlabel('Filter Type')
    plt.ylabel('Rank')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{metric_name}.png'))
    plt.close()

def plot_bar_comparison(metrics_by_config, output_dir):
    """Create bar plots comparing performance metrics across different configurations"""
    # Group configurations by rank
    rank_groups = defaultdict(list)
    for config in metrics_by_config.keys():
        for cfg in get_ablation_configs():
            if cfg['id'] == config:
                rank_groups[cfg['rank']].append({
                    'id': cfg['id'],
                    'filter_type': cfg['filter_type']
                })
    
    # Plot for each performance metric
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'avg_inference_time', 'avg_evidence_count']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(15, 8))
        
        bar_width = 0.25
        index = np.arange(3)  # 3 filter types
        
        for i, rank in enumerate([1, 2, 3]):
            values = []
            for config in rank_groups[rank]:
                values.append(metrics_by_config[config['id']].get(metric, 0))
            
            plt.bar(index + i*bar_width, values, bar_width, 
                    label=f'Rank {rank}')
        
        plt.xlabel('Filter Type')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Configurations')
        plt.xticks(index + bar_width, ['Similarity Only', 'Domain Only', 'Both'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bar_comparison_{metric}.png'))
        plt.close()

def create_summary_table(metrics_by_config, output_dir):
    """Create a summary table of all metrics across configurations"""
    data = []
    
    for config in get_ablation_configs():
        if config['id'] in metrics_by_config:
            metrics = metrics_by_config[config['id']]
            data.append({
                'Rank': config['rank'],
                'Filter Type': config['filter_type'].replace('_', ' ').title(),
                'Sample Count': metrics['count'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'Avg Inference Time (s)': f"{metrics['avg_inference_time']:.2f}",
                'Avg Evidence Count': f"{metrics['avg_evidence_count']:.2f}"
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    
    # Create a styled HTML table
    styled_df = df.style.background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], cmap='YlGnBu')
    with open(os.path.join(output_dir, 'summary_table.html'), 'w') as f:
        f.write(styled_df.to_html())
    
    return df

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all ablation configurations
    configs = get_ablation_configs()
    
    # Calculate metrics for each ablation configuration
    metrics_by_config = {}
    config_info = {}
    
    for config in configs:
        print(f"Evaluating {config['id']}...")
        results = load_results(args.results_dir, config['id'])
        
        if results:
            metrics = calculate_metrics(results)
            metrics_by_config[config['id']] = metrics
            config_info[config['id']] = config
            print(f"  - Processed {metrics['count']} results")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"  - No results found for {config['id']}")
    
    # Generate heatmaps for key metrics
    if metrics_by_config:
        print("Generating visualizations...")
        metrics_to_visualize = ['accuracy', 'precision', 'recall', 'f1', 'avg_inference_time', 'avg_evidence_count']
        
        for metric in metrics_to_visualize:
            generate_heatmap(metrics_by_config, metric, args.output_dir)
        
        # Generate bar comparison plots
        plot_bar_comparison(metrics_by_config, args.output_dir)
        
        # Create summary table
        summary_df = create_summary_table(metrics_by_config, args.output_dir)
        print(f"Summary table:\n{summary_df}")
        
        # Save full metrics to JSON
        metrics_json = {}
        for config_id, metrics in metrics_by_config.items():
            metrics_json[config_id] = {
                **metrics,
                'rank': config_info[config_id]['rank'],
                'filter_type': config_info[config_id]['filter_type']
            }
        
        with open(os.path.join(args.output_dir, 'all_metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Evaluation results saved to {args.output_dir}")
    else:
        print("No metrics to evaluate")

if __name__ == "__main__":
    main()