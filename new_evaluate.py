import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from typing import Dict, List, Any
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate verification results')
    parser.add_argument('--result_dir', type=str, default='result_ranking_lion', help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Directory to save evaluation results')
    parser.add_argument('--skip_non_candidates', '-s', action='store_true', help='Skip evaluation of results without evidence')
    parser.add_argument('--confidence_threshold', type=float, default=None, help='Confidence threshold for adjusting predictions')
    parser.add_argument('--compare_dir', type=str, default=None, help='Optional directory for result comparison')
    return parser.parse_args()

def ensure_directory(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

class ResultEvaluator:
    def __init__(self, result_dir: str, output_dir: str, skip_non_candidates: bool = False, confidence_threshold = None):
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.skip_non_candidates = skip_non_candidates
        self.confidence_threshold = confidence_threshold
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Initialize data containers
        self.captions = []
        self.ground_truth = []
        self.predicted = []
        self.confidence_scores = []
        self.inference_times = []
        self.evidence_present = []
        self.evidence_types = []
        self.evidence_scores = []
        self.confidence_levels = []
        self.verification_methods = []
        
        # Initialize statistics containers
        self.incorrect_indices = []
        self.correct_indices = []
        self.no_evidence_indices = []
        self.context_analysis_indices = []
        self.evidence_stats = {
            "total": 0,
            "no_evidence": 0,
            "with_evidence": 0,
            "text_evidence": 0,
            "image_evidence": 0
        }
        
        # Process results
        self.process_results()
        
    def extract_confidence_score(self, result: Dict) -> float:
        """Extract confidence score from result, either direct value or by parsing confidence level."""
        # If no direct confidence score, derive from confidence level
        confidence_level = result['verification_result']['confidence_level']
        if confidence_level == "High":
            return 0.9
        elif confidence_level == "Medium":
            return 0.7
        elif confidence_level == "Low":
            return 0.5
        return 0.5  # Default
    
    def extract_verification_methods(self, result: Dict) -> List[str]:
        """Extract verification methods from the result."""
        if 'verification_result' in result and 'authenticity_assessment' in result['verification_result']:
            return result['verification_result']['authenticity_assessment'].get('verification_methods', [])
        return []
    
    def extract_evidence_type(self, result: Dict) -> str:
        """Determine the type of evidence used in the result."""
        if 'evidence' not in result or result['evidence'] is None:
            return "no_evidence"
        
        evidence = result['evidence']
        source = evidence.get('source', '')
        if 'TextEvidencesModule' in source:
            return "text_evidence"
        elif 'ImageEvidencesModule' in source:
            return "image_evidence"
        return "unknown_evidence"
    
    def extract_evidence_score(self, result: Dict) -> float:
        """Extract the combined evidence score if available."""
        if 'evidence' in result and result['evidence'] is not None:
            return result['evidence'].get('combined_score', 0.0)
        return 0.0
    
    def extract_prediction(self, result: Dict) -> int:
        """Extract the prediction (OOC or not) from the result."""
        # Try new format first
        if 'verification_result' in result:
            # In the new format, this might be in the contextual_accuracy or in other fields
            contextual_accuracy = result['verification_result'].get('contextual_accuracy', {})
            if 'in_context' in contextual_accuracy:
                # Invert because in_context=True means not out of context (NOOC)
                return 0 if contextual_accuracy['in_context'] else 1
        
        # Fall back to old format
        if 'final_result' in result and 'OOC' in result['final_result']:
            # Convert boolean to int (True -> 1, False -> 0)
            return 1 if result['final_result']['OOC'] else 0
        
        # If we can't determine, return None
        return None
    
    def process_results(self):
        """Process all result files in the result directory."""
        result_files = [f for f in os.listdir(self.result_dir) if f.endswith('.json')]
        
        for filename in result_files:
            file_path = os.path.join(self.result_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Extract index from filename
                index_match = re.search(r'result_(\d+)\.json', filename)
                if index_match:
                    index = int(index_match.group(1))
                else:
                    index = int(filename.split('_')[-1].split('.')[0])
                
                # Check if evidence is present
                has_evidence = 'evidence' in result and result['evidence'] is not None
                evidence_type = self.extract_evidence_type(result)
                self.evidence_stats['total'] += 1
                
                if evidence_type == "no_evidence":
                    self.evidence_stats['no_evidence'] += 1
                    self.no_evidence_indices.append(index)
                    if self.skip_non_candidates:
                        continue
                else:
                    self.evidence_stats['with_evidence'] += 1
                    if evidence_type == "text_evidence":
                        self.evidence_stats['text_evidence'] += 1
                    elif evidence_type == "image_evidence":
                        self.evidence_stats['image_evidence'] += 1
                
                # Check if context analysis was used
                used_context = 'context_analysis' in result and result['context_analysis'] is not None
                if used_context:
                    self.context_analysis_indices.append(index)
                
                # Extract prediction
                prediction = self.extract_prediction(result)
                if prediction is None:
                    print(f"Warning: Could not extract prediction from {filename}")
                    continue
                
                # Extract ground truth
                ground_truth = 1 if result['ground_truth'] else 0
                
                # Record data
                self.captions.append(result.get('caption', ''))
                self.ground_truth.append(ground_truth)
                self.predicted.append(prediction)
                self.confidence_scores.append(self.extract_confidence_score(result))
                self.inference_times.append(result.get('inference_time', 0.0))
                self.evidence_present.append(has_evidence)
                self.evidence_types.append(evidence_type)
                self.evidence_scores.append(self.extract_evidence_score(result))
                
                # Extract confidence level and verification methods from new format
                if 'verification_result' in result:
                    self.confidence_levels.append(result['verification_result'].get('confidence_level', 'Unknown'))
                    self.verification_methods.append(self.extract_verification_methods(result))
                else:
                    self.confidence_levels.append('Unknown')
                    self.verification_methods.append([])
                
                # Record correct/incorrect
                if ground_truth == prediction:
                    self.correct_indices.append(index)
                else:
                    self.incorrect_indices.append(index)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    def create_dataframe(self):
        """Create a pandas DataFrame from the processed results."""
        data = {
            'caption': self.captions,
            'ground_truth': self.ground_truth,
            'predicted': self.predicted,
            'confidence_score': self.confidence_scores,
            'inference_time': self.inference_times,
            'has_evidence': self.evidence_present,
            'evidence_type': self.evidence_types,
            'evidence_score': self.evidence_scores,
            'confidence_level': self.confidence_levels
        }
        
        df = pd.DataFrame(data)
        
        # Add adjusted predictions if confidence threshold is provided
        if self.confidence_threshold is not None:
            df['adjusted_predicted'] = df.apply(
                lambda row: row['predicted'] if row['confidence_score'] >= self.confidence_threshold 
                           else 1 - row['predicted'], 
                axis=1
            )
        
        return df
    
    def save_indices(self):
        """Save various indices to files for further analysis."""
        indices_dir = os.path.join(self.output_dir, 'indices')
        ensure_directory(indices_dir)
        
        # Save incorrect indices
        self.incorrect_indices.sort()
        with open(os.path.join(indices_dir, 'incorrect_indices.txt'), 'w') as f:
            for idx in self.incorrect_indices:
                f.write(f"{idx}\n")
        
        # Save correct indices
        self.correct_indices.sort()
        with open(os.path.join(indices_dir, 'correct_indices.txt'), 'w') as f:
            for idx in self.correct_indices:
                f.write(f"{idx}\n")
        
        # Save no evidence indices
        self.no_evidence_indices.sort()
        with open(os.path.join(indices_dir, 'no_evidence_indices.txt'), 'w') as f:
            for idx in self.no_evidence_indices:
                f.write(f"{idx}\n")
        
        # Save context analysis indices
        self.context_analysis_indices.sort()
        with open(os.path.join(indices_dir, 'context_analysis_indices.txt'), 'w') as f:
            for idx in self.context_analysis_indices:
                f.write(f"{idx}\n")
    
    def generate_basic_statistics(self, df):
        """Generate and print basic statistics."""
        stats = {
            "Total samples": len(df),
            "OOC samples (ground truth)": df['ground_truth'].sum(),
            "NOOC samples (ground truth)": len(df) - df['ground_truth'].sum(),
            "OOC predictions": df['predicted'].sum(),
            "NOOC predictions": len(df) - df['predicted'].sum(),
            "Correct predictions": (df['ground_truth'] == df['predicted']).sum(),
            "Incorrect predictions": (df['ground_truth'] != df['predicted']).sum(),
            "Accuracy": (df['ground_truth'] == df['predicted']).mean(),
            "With evidence": sum(df['has_evidence']),
            "Without evidence": len(df) - sum(df['has_evidence']),
            "Text evidence": sum(df['evidence_type'] == 'text_evidence'),
            "Image evidence": sum(df['evidence_type'] == 'image_evidence'),
            "Average inference time": df['inference_time'].mean(),
            "Min inference time": df['inference_time'].min(),
            "Max inference time": df['inference_time'].max(),
            "Average evidence score": df[df['evidence_score'] > 0]['evidence_score'].mean() if any(df['evidence_score'] > 0) else 0
        }
        
        # Print statistics
        print("\n=== Basic Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # # Save statistics to file
        # with open(os.path.join(self.output_dir, 'basic_statistics.json'), 'w') as f:
        #     json.dump(stats, f, indent=4)
        
        return stats
    
    def generate_classification_report(self, df, adjusted=False):
        """Generate and print classification report."""
        class_names = ["NOOC", "OOC"]
        
        if adjusted and 'adjusted_predicted' in df.columns:
            predictions = df['adjusted_predicted']
        else:
            predictions = df['predicted']
        
        # Generate Classification Report
        report = classification_report(df['ground_truth'], predictions, target_names=class_names, output_dict=True)
        report_text = classification_report(df['ground_truth'], predictions, target_names=class_names)
        
        print("\n=== Classification Report ===")
        print(report_text)
        
        # Calculate confusion matrix
        cm = confusion_matrix(df['ground_truth'], predictions)
        
        # Calculate per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        print("\n=== Per-Class Accuracy ===")
        for class_name, acc in zip(class_names, class_accuracies):
            print(f"{class_name}: {acc:.4f}")
        
        # Add per-class accuracy to report
        for i, class_name in enumerate(class_names):
            report[class_name]['accuracy'] = float(class_accuracies[i])
        
        # Save classification report to file
        filename = 'classification_report_adjusted.json' if adjusted else 'classification_report.json'
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(report, f, indent=4)
        
        return report, cm
    
    def plot_confusion_matrix(self, cm, adjusted=False):
        """Plot and save confusion matrix."""
        class_names = ["NOOC", "OOC"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix' + (' (Adjusted)' if adjusted else ''))
        
        # Save plot
        filename = 'confusion_matrix_adjusted.png' if adjusted else 'confusion_matrix.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_evidence_impact(self, df):
        """Analyze the impact of evidence on prediction accuracy."""
        # Separate data with and without evidence
        with_evidence = df[df['has_evidence']]
        without_evidence = df[~df['has_evidence']]
        
        # Calculate accuracy for each group
        accuracy_with_evidence = (with_evidence['ground_truth'] == with_evidence['predicted']).mean() if len(with_evidence) > 0 else 0
        accuracy_without_evidence = (without_evidence['ground_truth'] == without_evidence['predicted']).mean() if len(without_evidence) > 0 else 0
        
        # Calculate metrics for each evidence type
        evidence_type_metrics = {}
        for evidence_type in df['evidence_type'].unique():
            if evidence_type == 'no_evidence':
                continue
            
            type_df = df[df['evidence_type'] == evidence_type]
            if len(type_df) == 0:
                continue
                
            accuracy = (type_df['ground_truth'] == type_df['predicted']).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                type_df['ground_truth'], 
                type_df['predicted'], 
                average='weighted'
            )
            
            evidence_type_metrics[evidence_type] = {
                'count': len(type_df),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'average_score': float(type_df['evidence_score'].mean())
            }
        
        # Compile results
        evidence_impact = {
            'with_evidence': {
                'count': len(with_evidence),
                'accuracy': float(accuracy_with_evidence)
            },
            'without_evidence': {
                'count': len(without_evidence),
                'accuracy': float(accuracy_without_evidence)
            },
            'evidence_types': evidence_type_metrics,
            'overall': {
                'total_samples': len(df),
                'with_evidence_percentage': float(len(with_evidence) / len(df)) if len(df) > 0 else 0
            }
        }
        
        # Print evidence impact analysis
        print("\n=== Evidence Impact Analysis ===")
        print(f"Accuracy with evidence ({len(with_evidence)} samples): {accuracy_with_evidence:.4f}")
        print(f"Accuracy without evidence ({len(without_evidence)} samples): {accuracy_without_evidence:.4f}")
        
        print("\n=== Evidence Type Analysis ===")
        for evidence_type, metrics in evidence_type_metrics.items():
            print(f"{evidence_type} ({metrics['count']} samples):")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Average Score: {metrics['average_score']:.4f}")
        
        # Save evidence impact analysis to file
        with open(os.path.join(self.output_dir, 'evidence_impact_analysis.json'), 'w') as f:
            json.dump(evidence_impact, f, indent=4)
        
        return evidence_impact
    
    def analyze_confidence_correlation(self, df):
        """Analyze the correlation between confidence scores and accuracy."""
        # Create confidence bins
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        df['confidence_bin'] = pd.cut(df['confidence_score'], bins)
        
        # Calculate accuracy per bin
        bin_accuracy = df.groupby('confidence_bin').apply(
            lambda x: (x['ground_truth'] == x['predicted']).mean()
        ).reset_index(name='accuracy')
        
        bin_count = df.groupby('confidence_bin').size().reset_index(name='count')
        bin_analysis = pd.merge(bin_accuracy, bin_count, on='confidence_bin')
        
        # Calculate correlation
        correlation = df['confidence_score'].corr(
            (df['ground_truth'] == df['predicted']).astype(int)
        )
        
        # Plot confidence vs. accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x=bin_analysis['confidence_bin'].astype(str), y=bin_analysis['accuracy'])
        plt.xlabel('Confidence Score Bin')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Confidence Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add count labels
        for i, count in enumerate(bin_analysis['count']):
            plt.text(i, bin_analysis['accuracy'].iloc[i] + 0.02, f'n={count}', 
                    ha='center', va='bottom', fontsize=9)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'confidence_vs_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confidence level analysis
        if 'confidence_level' in df.columns:
            level_accuracy = df.groupby('confidence_level').apply(
                lambda x: (x['ground_truth'] == x['predicted']).mean()
            ).reset_index(name='accuracy')
            
            level_count = df.groupby('confidence_level').size().reset_index(name='count')
            level_analysis = pd.merge(level_accuracy, level_count, on='confidence_level')
            
            print("\n=== Confidence Level Analysis ===")
            for _, row in level_analysis.iterrows():
                print(f"{row['confidence_level']} ({row['count']} samples): Accuracy = {row['accuracy']:.4f}")
        
        # Prepare results
        confidence_analysis = {
            'correlation': float(correlation),
            'bin_analysis': bin_analysis.to_dict('records')
        }
        
        if 'confidence_level' in df.columns:
            confidence_analysis['level_analysis'] = level_analysis.to_dict('records')
        
        # Save confidence analysis to file
        # with open(os.path.join(self.output_dir, 'confidence_analysis.json'), 'w') as f:
        #     json.dump(confidence_analysis, f, indent=4)
        
        return confidence_analysis
    
    def analyze_verification_methods(self, df):
        """Analyze the verification methods used and their impact."""
        if not any(self.verification_methods):
            print("No verification methods data available")
            return None
        
        # Get all unique verification methods
        all_methods = set()
        for methods in self.verification_methods:
            all_methods.update(methods)
        
        # Calculate statistics for each method
        method_stats = {}
        for method in all_methods:
            # Create mask for samples using this method
            mask = [method in methods for methods in self.verification_methods]
            method_df = df[mask]
            
            if len(method_df) == 0:
                continue
                
            # Calculate metrics
            accuracy = (method_df['ground_truth'] == method_df['predicted']).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                method_df['ground_truth'], 
                method_df['predicted'], 
                average='weighted'
            )
            
            method_stats[method] = {
                'count': int(sum(mask)),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'percentage': float(sum(mask) / len(df)) if len(df) > 0 else 0
            }
        
        # Print verification methods analysis
        print("\n=== Verification Methods Analysis ===")
        for method, stats in method_stats.items():
            print(f"{method} ({stats['count']} samples, {stats['percentage']:.1%}):")
            print(f"  Accuracy: {stats['accuracy']:.4f}")
            print(f"  F1 Score: {stats['f1']:.4f}")
        
        # Calculate method co-occurrence
        method_combinations = {}
        for methods in self.verification_methods:
            if len(methods) < 2:
                continue
                
            methods_tuple = tuple(sorted(methods))
            if methods_tuple in method_combinations:
                method_combinations[methods_tuple] += 1
            else:
                method_combinations[methods_tuple] = 1
        
        # Save verification methods analysis to file
        verification_analysis = {
            'method_statistics': method_stats,
            'method_combinations': {','.join(combo): count for combo, count in method_combinations.items()},
            'average_methods_per_result': sum(len(methods) for methods in self.verification_methods) / len(self.verification_methods) if self.verification_methods else 0
        }
        
        with open(os.path.join(self.output_dir, 'verification_methods_analysis.json'), 'w') as f:
            json.dump(verification_analysis, f, indent=4)
        
        return verification_analysis
    
    def run_evaluation(self):
        """Run the full evaluation process."""
        print(f"Evaluating results from {self.result_dir}")
        print(f"Total results: {self.evidence_stats['total']}")
        print(f"Results with evidence: {self.evidence_stats['with_evidence']}")
        print(f"Results without evidence: {self.evidence_stats['no_evidence']}")
        
        # Create dataframe from processed results
        df = self.create_dataframe()
        
        # Generate basic statistics
        self.generate_basic_statistics(df)
        
        # Generate classification report
        report, cm = self.generate_classification_report(df)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # If confidence threshold is provided, also evaluate adjusted predictions
        if self.confidence_threshold is not None:
            print(f"\n=== Adjusted Predictions (Threshold: {self.confidence_threshold}) ===")
            adjusted_report, adjusted_cm = self.generate_classification_report(df, adjusted=True)
            self.plot_confusion_matrix(adjusted_cm, adjusted=True)
        
        # Analyze impact of evidence on accuracy
        # self.analyze_evidence_impact(df)
        
        # Analyze correlation between confidence scores and accuracy
        # self.analyze_confidence_correlation(df)
        
        # Analyze verification methods
        # self.analyze_verification_methods(df)
        
        # Save indices for further analysis
        self.save_indices()
        
        # Save dataframe
        df.to_csv(os.path.join(self.output_dir, 'evaluation_data.csv'), index=False)
        
        print(f"\nEvaluation complete. Results saved to {self.output_dir}")
        return df

def main():
    args = parse_arguments()
    
    # Initialize evaluator and run evaluation
    evaluator = ResultEvaluator(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        skip_non_candidates=args.skip_non_candidates,
        confidence_threshold=args.confidence_threshold
    )
    
    df = evaluator.run_evaluation()
    
    # If comparison directory is provided, compare results
    if args.compare_dir and os.path.exists(args.compare_dir):
        print(f"\nComparing with results in {args.compare_dir}")
        compare_evaluator = ResultEvaluator(
            result_dir=args.compare_dir,
            output_dir=os.path.join(args.output_dir, 'comparison'),
            skip_non_candidates=args.skip_non_candidates,
            confidence_threshold=args.confidence_threshold
        )
        
        compare_df = compare_evaluator.run_evaluation()
        
        # TODO: Add detailed comparison between the two result sets

if __name__ == "__main__":
    main()