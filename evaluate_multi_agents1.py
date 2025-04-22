import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Simple evaluation for news image verification results')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='evaluation_output_multi_agents1', help='Directory to save evaluation results')
    return parser.parse_args()

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class SimpleEvaluator:
    def __init__(self, result_dir, output_dir):
        self.result_dir = result_dir
        self.output_dir = output_dir
        ensure_directory(output_dir)
        
        # Create indices directory
        self.indices_dir = os.path.join(output_dir, 'indices')
        ensure_directory(self.indices_dir)
        
        # Data containers
        self.results = []
        self.true_labels = []
        self.predicted_labels = []
        self.confidence_levels = []
        self.evidence_types = []
        self.inference_times = []
        
        # Indices for analysis
        self.correct_indices = []
        self.incorrect_indices = []
        self.no_evidence_indices = []
        self.context_analysis_indices = []
        
        # Load and process results
        self.load_results()
    
    def load_results(self):
        """Load all result files from the result directory."""
        print(f"Loading results from {self.result_dir}")
        result_files = [f for f in os.listdir(self.result_dir) if f.endswith('.json')]
        print(f"Found {len(result_files)} result files")
        
        for filename in result_files:
            file_path = os.path.join(self.result_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if result.get("has_evidence", True):
                    pass
                
                # Extract index from filename
                index_match = re.search(r'result_(\d+)\.json', filename)
                if index_match:
                    index = int(index_match.group(1))
                else:
                    try:
                        index = int(filename.split('_')[-1].split('.')[0])
                    except:
                        # If we can't extract a numeric index, use a hash of the filename
                        index = hash(filename) % 10000
                
                # Extract core information
                ground_truth = 1 if result.get('ground_truth', False) else 0
                
                # Extract prediction based on verification result structure
                predicted = None
                if 'verification_result' in result:
                    verification = result['verification_result']
                    
                    # Try different field names that might indicate context validity
                    if 'is_in_context' in verification:
                        # Invert because is_in_context=False means OOC (1)
                        predicted = 0 if verification['is_in_context'] else 1
                    elif 'contextual_validity' in verification and 'in_context' in verification['contextual_validity']:
                        predicted = 0 if verification['contextual_validity']['in_context'] else 1
                    elif 'context_classification' in verification:
                        # If context_classification is misleading_context, it's OOC
                        predicted = 1 if verification['context_classification'] == 'misleading_context' else 0
                
                # Fallback to final_result if needed
                if predicted is None and 'final_result' in result and 'OOC' in result['final_result']:
                    predicted = 1 if result['final_result']['OOC'] else 0
                
                if predicted is None:
                    print(f"Warning: Could not extract prediction from {filename}")
                    continue
                
                # Determine evidence type
                evidence_type = "no_evidence"
                if 'evidence' in result and result['evidence']:
                    source = result['evidence'].get('source', '')
                    if 'TextEvidencesModule' in source:
                        evidence_type = "text_evidence"
                    elif 'ImageEvidencesModule' in source:
                        evidence_type = "image_evidence"
                
                # Extract confidence level
                confidence = "Medium"
                if 'verification_result' in result and 'confidence' in result['verification_result']:
                    confidence = result['verification_result']['confidence']
                
                # Track indices for the result types
                if evidence_type == "no_evidence":
                    self.no_evidence_indices.append(index)
                
                if 'context_analysis' in result and result['context_analysis'] is not None:
                    self.context_analysis_indices.append(index)
                
                if ground_truth == predicted:
                    self.correct_indices.append(index)
                else:
                    self.incorrect_indices.append(index)
                
                # Store data
                self.results.append({
                    'file': filename,
                    'index': index,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'confidence': confidence,
                    'evidence_type': evidence_type,
                    'inference_time': result.get('inference_time', 0)
                })
                
                self.true_labels.append(ground_truth)
                self.predicted_labels.append(predicted)
                self.confidence_levels.append(confidence)
                self.evidence_types.append(evidence_type)
                self.inference_times.append(result.get('inference_time', 0))
                
            except Exception as e:
                # print(f"Error processing {filename}: {str(e)}")
                pass
    
    def calculate_basic_metrics(self):
        """Calculate basic evaluation metrics."""
        if not self.results:
            print("No results to evaluate")
            return {}
        
        # Convert to numpy arrays for easier manipulation
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predicted_labels)
        
        # Calculate basic metrics
        correct = y_true == y_pred
        accuracy = np.mean(correct)
        
        # Calculate class-specific metrics
        class_counts = {
            'OOC (1)': np.sum(y_true == 1),
            'NOOC (0)': np.sum(y_true == 0)
        }
        
        class_accuracies = {
            'OOC (1)': np.mean(correct[y_true == 1]) if np.any(y_true == 1) else 0,
            'NOOC (0)': np.mean(correct[y_true == 0]) if np.any(y_true == 0) else 0
        }
        
        # Calculate metrics by evidence type
        evidence_types = set(self.evidence_types)
        evidence_metrics = {}
        
        for ev_type in evidence_types:
            indices = [i for i, t in enumerate(self.evidence_types) if t == ev_type]
            if indices:
                ev_true = [self.true_labels[i] for i in indices]
                ev_pred = [self.predicted_labels[i] for i in indices]
                ev_correct = np.array(ev_true) == np.array(ev_pred)
                
                evidence_metrics[ev_type] = {
                    'count': len(indices),
                    'accuracy': float(np.mean(ev_correct)),
                    'percentage': len(indices) / len(self.results)
                }
        
        # Calculate metrics by confidence level
        confidence_metrics = {}
        confidence_levels = set(self.confidence_levels)
        
        for level in confidence_levels:
            indices = [i for i, l in enumerate(self.confidence_levels) if l == level]
            if indices:
                level_true = [self.true_labels[i] for i in indices]
                level_pred = [self.predicted_labels[i] for i in indices]
                level_correct = np.array(level_true) == np.array(level_pred)
                
                confidence_metrics[level] = {
                    'count': len(indices),
                    'accuracy': float(np.mean(level_correct)),
                    'percentage': len(indices) / len(self.results)
                }
        
        # Compile metrics
        metrics = {
            'total_samples': len(self.results),
            'overall_accuracy': float(accuracy),
            'class_distribution': class_counts,
            'class_accuracies': class_accuracies,
            'evidence_metrics': evidence_metrics,
            'confidence_metrics': confidence_metrics,
            'average_inference_time': float(np.mean(self.inference_times))
        }
        
        return metrics
    
    def generate_confusion_matrix(self):
        """Generate confusion matrix."""
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["NOOC (0)", "OOC (1)"],
                   yticklabels=["NOOC (0)", "OOC (1)"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for News Image Verification')
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm.tolist()
    
    def generate_class_report(self):
        """Generate classification report."""
        report = classification_report(
            self.true_labels, 
            self.predicted_labels,
            target_names=["NOOC (0)", "OOC (1)"],
            output_dict=True
        )
        
        return report
    
    def save_incorrect_predictions(self):
        """Save details of incorrect predictions for further analysis."""
        incorrect = []
        
        for i, result in enumerate(self.results):
            if result['ground_truth'] != result['predicted']:
                incorrect.append({
                    'file': result['file'],
                    'index': result['index'],
                    'ground_truth': result['ground_truth'],
                    'predicted': result['predicted'],
                    'confidence': result['confidence'],
                    'evidence_type': result['evidence_type']
                })
        
        return incorrect
    
    def save_indices(self):
        """Save various indices to files for further analysis."""
        # Sort indices
        self.correct_indices.sort()
        self.incorrect_indices.sort()
        self.no_evidence_indices.sort()
        self.context_analysis_indices.sort()
        
        # Save correct indices
        with open(os.path.join(self.indices_dir, 'correct_indices.txt'), 'w') as f:
            for idx in self.correct_indices:
                f.write(f"{idx}\n")
        
        # Save incorrect indices
        with open(os.path.join(self.indices_dir, 'incorrect_indices.txt'), 'w') as f:
            for idx in self.incorrect_indices:
                f.write(f"{idx}\n")
        
        # Save no evidence indices
        with open(os.path.join(self.indices_dir, 'no_evidence_indices.txt'), 'w') as f:
            for idx in self.no_evidence_indices:
                f.write(f"{idx}\n")
        
        # Save context analysis indices
        with open(os.path.join(self.indices_dir, 'context_analysis_indices.txt'), 'w') as f:
            for idx in self.context_analysis_indices:
                f.write(f"{idx}\n")
        
        print(f"Saved indices to {self.indices_dir}/")
    
    def run_evaluation(self):
        """Run the full evaluation and save results."""
        print(f"Running evaluation on {len(self.results)} results...")
        
        # Calculate basic metrics
        metrics = self.calculate_basic_metrics()
        
        # Generate confusion matrix
        confusion = self.generate_confusion_matrix()
        metrics['confusion_matrix'] = confusion
        
        # Generate classification report
        report = self.generate_class_report()
        metrics['classification_report'] = report
        
        # Get incorrect predictions
        incorrect = self.save_incorrect_predictions()
        
        self.save_indices()
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"OOC (1) accuracy: {metrics['class_accuracies']['OOC (1)']:.4f}")
        print(f"NOOC (0) accuracy: {metrics['class_accuracies']['NOOC (0)']:.4f}")
        print(f"Average inference time: {metrics['average_inference_time']:.2f} seconds")
        
        print("\n=== Evidence Type Performance ===")
        for ev_type, ev_metrics in metrics['evidence_metrics'].items():
            print(f"{ev_type}: {ev_metrics['count']} samples, {ev_metrics['accuracy']:.4f} accuracy")
        
        print("\n=== Incorrect Predictions ===")
        print(f"Found {len(incorrect)} incorrect predictions")
        
        # Save incorrect predictions to file
        with open(os.path.join(self.output_dir, 'incorrect_predictions.json'), 'w') as f:
            json.dump(incorrect, f, indent=2)
        
        # Create dataframe for analysis
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_dir, 'results_summary.csv'), index=False)
        
        print(f"\nEvaluation complete. Results saved to {self.output_dir}")
        return metrics

def main():
    args = parse_args()
    evaluator = SimpleEvaluator(args.result_dir, args.output_dir)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()