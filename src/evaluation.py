"""
Evaluation & Metrics Module
Computes performance metrics and comparison analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and compares anomaly detection models"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
    
    def compute_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            cm: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n{model_name} - Confusion Matrix:")
        logger.info(f"\n{cm}")
        return cm
    
    def compute_metrics(self, y_true, y_pred, y_scores=None, model_name='Model'):
        """
        Compute comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            model_name: Name of the model
            
        Returns:
            metrics: Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        # Additional metrics
        if tp + fp > 0:
            metrics['fpr'] = fp / (fp + tn)
        else:
            metrics['fpr'] = 0
        
        if tp + fn > 0:
            metrics['fnr'] = fn / (fn + tp)
        else:
            metrics['fnr'] = 0
        
        # ROC-AUC if scores are provided
        if y_scores is not None:
            try:
                metrics['roc_auc'] = auc(
                    *roc_curve(y_true, y_scores)[:2]
                )
            except:
                metrics['roc_auc'] = None
        
        logger.info(f"\n{model_name} - Performance Metrics:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        self.results[model_name] = metrics
        return metrics
    
    def get_roc_curve(self, y_true, y_scores):
        """
        Compute ROC curve
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            
        Returns:
            fpr, tpr, thresholds, auc_score
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_score
    
    def get_precision_recall_curve(self, y_true, y_scores):
        """
        Compute Precision-Recall curve
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            
        Returns:
            precision, recall, thresholds
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        return precision, recall, thresholds
    
    def compare_models(self, models_results):
        """
        Compare performance across multiple models
        
        Args:
            models_results: Dictionary of model results
            
        Returns:
            comparison_df: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'TP': metrics.get('tp', 0),
                'FP': metrics.get('fp', 0),
                'TN': metrics.get('tn', 0),
                'FN': metrics.get('fn', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
    
    def generate_evaluation_report(self, models_results, output_file='evaluation_report.txt'):
        """
        Generate comprehensive evaluation report
        
        Args:
            models_results: Dictionary of model results
            output_file: Output filename
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANOMALY DETECTION - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, metrics in models_results.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
                f.write(f"F1-Score:  {metrics.get('f1_score', 0):.4f}\n")
                f.write(f"ROC-AUC:   {metrics.get('roc_auc', 'N/A')}\n\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"  TP: {metrics.get('tp', 0)}, FP: {metrics.get('fp', 0)}\n")
                f.write(f"  FN: {metrics.get('fn', 0)}, TN: {metrics.get('tn', 0)}\n\n")
            
            # Summary and recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY & RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")
            f.write("Compare the F1-scores and ROC-AUC values to identify the best model.\n")
            f.write("Consider the balance between precision and recall based on your use case.\n")
            f.write("Precision is critical when false alarms are costly.\n")
            f.write("Recall is critical when missing anomalies is dangerous.\n")
        
        logger.info(f"Evaluation report saved to {output_file}")
