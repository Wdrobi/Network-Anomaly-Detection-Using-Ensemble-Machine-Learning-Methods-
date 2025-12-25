"""
Model Comparison & Evaluation Metrics Module
Comprehensive evaluation and comparison of anomaly detection models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, output_dir='results/'):
        """Initialize evaluator with output directory"""
        self.output_dir = output_dir
        self.metrics_dict = {}
        self.roc_data = {}
        self.pr_data = {}
    
    def calculate_metrics(self, y_true, y_pred, y_scores=None, model_name='Model'):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            model_name: Name of the model
            
        Returns:
            metrics_dict: Dictionary of all metrics
        """
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if scores are provided
        if y_scores is not None:
            try:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['ROC-AUC'] = np.nan
        else:
            metrics['ROC-AUC'] = np.nan
        
        # Store for later comparison
        self.metrics_dict[model_name] = metrics
        
        # Store ROC data if scores available
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            self.roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 
                                         'auc': metrics['ROC-AUC']}
            
            # Store PR data
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            self.pr_data[model_name] = {'precision': precision, 'recall': recall}
        
        logger.info(f"Metrics calculated for {model_name}")
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """Plot confusion matrix for a model"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}confusion_matrix_{model_name.lower()}.png', dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix saved for {model_name}")
    
    def plot_all_confusion_matrices(self, predictions_dict, y_true):
        """
        Plot all confusion matrices in a grid
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True labels
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Normal', 'Anomaly'],
                        yticklabels=['Normal', 'Anomaly'], cbar=True)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("All confusion matrices saved")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        if not self.roc_data:
            logger.warning("No ROC data available")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in self.roc_data.items():
            plt.plot(data['fpr'], data['tpr'], linewidth=2.5,
                    label=f"{model_name} (AUC = {data['auc']:.3f})")
        
        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("ROC curves saved")
    
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for all models"""
        if not self.pr_data:
            logger.warning("No precision-recall data available")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in self.pr_data.items():
            plt.plot(data['recall'], data['precision'], linewidth=2.5,
                    label=f"{model_name}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Precision-recall curves saved")
    
    def plot_metrics_comparison(self):
        """Plot bar chart comparing metrics across models"""
        if not self.metrics_dict:
            logger.warning("No metrics to compare")
            return
        
        df_metrics = pd.DataFrame(self.metrics_dict).T
        
        # Select numeric columns
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        df_plot = df_metrics[metric_cols].dropna(axis=1)
        
        fig, axes = plt.subplots(1, len(df_plot.columns), figsize=(16, 5))
        
        for idx, metric in enumerate(df_plot.columns):
            df_plot[metric].plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].set_xlabel('')
            axes[idx].set_ylim([0, 1])
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Metrics comparison chart saved")
    
    def create_comparison_table(self):
        """Create and save comparison table as CSV"""
        if not self.metrics_dict:
            logger.warning("No metrics to save")
            return
        
        df = pd.DataFrame(self.metrics_dict).T
        df = df.round(4)
        
        # Save to CSV
        df.to_csv(f'{self.output_dir}model_comparison.csv')
        
        # Create formatted text report
        report_text = "=" * 80 + "\n"
        report_text += "MODEL COMPARISON REPORT\n"
        report_text += "=" * 80 + "\n\n"
        report_text += df.to_string()
        report_text += "\n\n" + "=" * 80 + "\n"
        
        # Add detailed metrics
        report_text += "\nDETAILED METRICS INTERPRETATION:\n"
        report_text += "-" * 80 + "\n"
        report_text += "Accuracy:   Proportion of correct predictions\n"
        report_text += "Precision:  Of predicted anomalies, how many are actually anomalies?\n"
        report_text += "Recall:     Of actual anomalies, how many did we detect?\n"
        report_text += "F1-Score:   Harmonic mean of Precision and Recall\n"
        report_text += "ROC-AUC:    Area under the ROC curve (0.5=random, 1.0=perfect)\n"
        report_text += "=" * 80 + "\n"
        
        # Save text report
        with open(f'{self.output_dir}model_comparison_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("Comparison table and report saved")
        return df
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.metrics_dict:
            return None
        
        df = pd.DataFrame(self.metrics_dict).T
        
        report = "\n" + "=" * 80 + "\n"
        report += "ANOMALY DETECTION - MODEL COMPARISON SUMMARY\n"
        report += "=" * 80 + "\n\n"
        
        # Best performers
        report += "BEST PERFORMERS:\n"
        report += "-" * 80 + "\n"
        
        metrics_to_check = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in metrics_to_check:
            if metric in df.columns:
                best_model = df[metric].idxmax()
                best_score = df[metric].max()
                report += f"  • Best {metric}: {best_model} ({best_score:.4f})\n"
        
        report += "\n" + "-" * 80 + "\n"
        report += "DETAILED COMPARISON:\n"
        report += "-" * 80 + "\n"
        report += df.to_string()
        
        report += "\n\n" + "=" * 80 + "\n"
        report += "RECOMMENDATIONS:\n"
        report += "-" * 80 + "\n"
        
        # Generate recommendations
        if 'F1-Score' in df.columns:
            best_f1 = df['F1-Score'].idxmax()
            report += f"  • For balanced performance: Use {best_f1}\n"
        
        if 'Precision' in df.columns:
            best_prec = df['Precision'].idxmax()
            report += f"  • For low false alarms: Use {best_prec}\n"
        
        if 'Recall' in df.columns:
            best_rec = df['Recall'].idxmax()
            report += f"  • For detecting anomalies: Use {best_rec}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
