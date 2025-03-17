#!/usr/bin/env python3
# evaluate_ablation_studies.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ablation study results')
    parser.add_argument('--results_dir', type=str, default='./ablation_results',
                        help='Directory containing all ablation results')
    parser.add_argument('--output_dir', type=str, default='./ablation_evaluation',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def load_results(results_dir, ablation_type):
    """Load all results for a specific ablation type"""
    ablation_dir = os.path.join(results_dir, ablation_type)
    results = []
    
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
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'check_type_distribution': {k: check_types.count(k) for k in set(check_types)}
    }
    
    return metrics

def plot_metrics(metrics_dict, output_dir):
    """Create visualizations of the metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for bar charts
    ablation_types = list(metrics_dict.keys())
    accuracy = [metrics_dict[t]['accuracy'] for t in ablation_types]
    precision = [metrics_dict[t]['precision'] for t in ablation_types]
    recall = [metrics_dict[t]['recall'] for t in ablation_types]
    f1 = [metrics_dict[t]['f1'] for t in ablation_types]
    inference_times = [metrics_dict[t]['avg_inference_time'] for t in ablation_types]
    evidence_counts = [metrics_dict[t]['avg_evidence_count'] for t in ablation_types]
    
    # Plot accuracy, precision, recall, F1
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    x = np.arange(len(ablation_types))
    
    plt.bar(x - 1.5*bar_width, accuracy, width=bar_width, label='Accuracy')
    plt.bar(x - 0.5*bar_width, precision, width=bar_width, label='Precision')
    plt.bar(x + 0.5*bar_width, recall, width=bar_width, label='Recall')
    plt.bar(x + 1.5*bar_width, f1, width=bar_width, label='F1')
    
    plt.xlabel('Ablation Type')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Ablation Type')
    plt.xticks(x, ablation_types, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
    plt.close()
    
    # Plot inference time
    plt.figure(figsize=(10, 5))
    plt.bar(ablation_types, inference_times)
    plt.xlabel('Ablation Type')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Average Inference Time by Ablation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time.png'))
    plt.close()
    
    # Plot evidence counts
    plt.figure(figsize=(10, 5))
    plt.bar(ablation_types, evidence_counts)
    plt.xlabel('Ablation Type')
    plt.ylabel('Average Evidence Count')
    plt.title('Average Evidence Count by Ablation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evidence_count.png'))
    plt.close()
    
    # Plot confusion matrices for each ablation type
    for ablation_type in ablation_types:
        conf_matrix = metrics_dict[ablation_type]['conf_matrix']
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {ablation_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{ablation_type}.png'))
        plt.close()
    
    # Create a summary table
    summary_data = []
    for ablation_type in ablation_types:
        m = metrics_dict[ablation_type]
        summary_data.append({
            'Ablation Type': ablation_type,
            'Sample Count': m['count'],
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1 Score': f"{m['f1']:.4f}",
            'Avg Inference Time': f"{m['avg_inference_time']:.2f}s",
            'Avg Evidence Count': f"{m['avg_evidence_count']:.2f}"
        })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of ablation types to evaluate
    ablation_types = ['no_image_evidences', 'no_text_evidences', 'no_filters']
    
    # Calculate metrics for each ablation type
    metrics_dict = {}
    for ablation_type in ablation_types:
        print(f"Evaluating {ablation_type}...")
        results = load_results(args.results_dir, ablation_type)
        if results:
            metrics = calculate_metrics(results)
            metrics_dict[ablation_type] = metrics
            print(f"  - Processed {metrics['count']} results")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"  - No results found for {ablation_type}")
    
    # Plot and save metrics
    if metrics_dict:
        print("Generating plots and saving results...")
        plot_metrics(metrics_dict, args.output_dir)
        print(f"Evaluation results saved to {args.output_dir}")
    else:
        print("No metrics to evaluate")

if __name__ == "__main__":
    main()