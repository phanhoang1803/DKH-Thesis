import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Simple evaluation of debate verification results')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result JSON files')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help='Output file for evaluation metrics')
    parser.add_argument('--output_dir', type=str, default="evaluation_output", help='Directory to save all output files (defaults to same directory as output_file)')
    return parser.parse_args()

def evaluate_results(result_dir, output_file):
    """Evaluate results from a directory of JSON files and calculate basic metrics."""
    
    # Initialize lists to collect data
    ground_truths = []
    predictions = []
    confidences = []
    processing_times = []
    indices = []
    correct_indices = []
    incorrect_indices = []
    
    # Track counts
    total_count = 0
    converged_count = 0
    
    print(f"Evaluating results from: {result_dir}")
    
    # Process all result files
    for filename in os.listdir(result_dir):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(result_dir, filename)
        try:
            # Extract index from filename
            if 'result_' in filename:
                index = int(filename.replace('result_', '').replace('.json', ''))
            else:
                index = int(os.path.splitext(filename)[0])
                
            indices.append(index)
            
            # Load the result file
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            if result['evidence']['evidences'] == []:
                continue
            
            total_count += 1
            
            # Extract prediction (convert YES/NO to 1/0)
            predicted = 1 if result['verdict']['verdict'] == 'YES' else 0
            predictions.append(predicted)
            
            # Extract ground truth (assuming it's stored as a boolean)
            ground_truth = 1 if result.get('ground_truth', {}).get('label', False) else 0
            ground_truths.append(ground_truth)
            
            # Extract confidence
            confidence = result['verdict'].get('confidence', 0.0)
            confidences.append(confidence)
            
            # Check if prediction is correct or incorrect
            if predicted == ground_truth:
                correct_indices.append(index)
            else:
                incorrect_indices.append(index)
            
            # Extract processing time if available
            if 'metadata' in result and 'processing_time' in result['metadata']:
                processing_times.append(result['metadata']['processing_time'])
                
            # Check if debate converged
            if len(result.get('debate_history', [])) > 1:
                last_round = result['debate_history'][-1]
                if 'round' in last_round and last_round['round'] < 3:  # Assuming max rounds is 3
                    converged_count += 1
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Calculate metrics if we have data
    if not predictions or not ground_truths:
        print("No valid data found for evaluation")
        return
    
    # Calculate basic metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average='binary', zero_division=0
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truths, predictions, labels=[0, 1]).ravel()
    
    # Calculate per-class metrics
    true_metrics = {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }
    
    false_metrics = {
        "precision": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "recall": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0
    }
    
    # Calculate average confidence
    avg_confidence = np.mean(confidences) if confidences else 0
    
    # Calculate average processing time
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    # Prepare results
    evaluation_results = {
        "total_samples": total_count,
        "processed_samples": len(predictions),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "class_metrics": {
            "true": true_metrics,
            "false": false_metrics
        },
        "avg_confidence": avg_confidence,
        "avg_processing_time": avg_processing_time,
        "convergence_rate": converged_count / total_count if total_count > 0 else 0
    }
    
    # Print summary
    print("\n===== Evaluation Results =====")
    print(f"Total samples: {total_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Convergence Rate: {evaluation_results['convergence_rate']:.4f}")
    print(f"Average Processing Time: {avg_processing_time:.2f} seconds")
    
    print("\nConfusion Matrix:")
    print(f"True Negative: {tn}, False Positive: {fp}")
    print(f"False Negative: {fn}, True Positive: {tp}")
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save incorrect indices to file
    incorrect_indices.sort()
    incorrect_indices_file = os.path.join(os.path.dirname(output_file), 'incorrect_indices.txt')
    with open(incorrect_indices_file, 'w') as f:
        for idx in incorrect_indices:
            f.write(f"{idx}\n")
    
    # Save correct indices to file
    correct_indices.sort()
    correct_indices_file = os.path.join(os.path.dirname(output_file), 'correct_indices.txt')
    with open(correct_indices_file, 'w') as f:
        for idx in correct_indices:
            f.write(f"{idx}\n")
    
    print(f"\nResults saved to {output_file}")
    print(f"Incorrect indices saved to {incorrect_indices_file}")
    print(f"Correct indices saved to {correct_indices_file}")
    
    return evaluation_results

if __name__ == "__main__":
    args = parse_args()
    
    # If output_dir is specified, create it if it doesn't exist
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine output file path
    output_file = args.output_file
    if args.output_dir:
        output_file = os.path.join(args.output_dir, os.path.basename(args.output_file))
        
    evaluate_results(args.result_dir, output_file)