"""
Visualization & Reporting Module
Creates plots and visualizations for analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Creates visualizations for anomaly detection results"""
    
    def __init__(self, output_dir='results/'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        sns.set_style("whitegrid")
    
    def plot_confusion_matrices(self, models_cm, figsize=(15, 5)):
        """
        Plot confusion matrices for multiple models
        
        Args:
            models_cm: Dictionary of model_name -> confusion_matrix
            figsize: Figure size
        """
        n_models = len(models_cm)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, cm) in enumerate(models_cm.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}confusion_matrices.png', dpi=300)
        logger.info("Confusion matrices plot saved")
        plt.close()
    
    def plot_roc_curves(self, models_roc, figsize=(10, 8)):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_roc: Dictionary of model_name -> (fpr, tpr, auc_score)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, (fpr, tpr, auc_score) in models_roc.items():
            plt.plot(fpr, tpr, marker='o', label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}roc_curves.png', dpi=300)
        logger.info("ROC curves plot saved")
        plt.close()
    
    def plot_reconstruction_error(self, reconstruction_errors, y_true, threshold, 
                                 model_name='Autoencoder', figsize=(12, 6)):
        """
        Plot reconstruction error distribution
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            y_true: True labels
            threshold: Anomaly threshold
            model_name: Name of the model
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram of reconstruction errors
        axes[0].hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        axes[0].set_xlabel('Reconstruction Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{model_name} - Error Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot: Error vs True Label
        normal_errors = reconstruction_errors[y_true == 0]
        anomaly_errors = reconstruction_errors[y_true == 1]
        
        axes[1].scatter(range(len(normal_errors)), normal_errors, alpha=0.5, label='Normal', s=20)
        axes[1].scatter(range(len(normal_errors), len(normal_errors) + len(anomaly_errors)), 
                       anomaly_errors, alpha=0.5, label='Anomaly', s=20, color='red')
        axes[1].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('Reconstruction Error', fontsize=12)
        axes[1].set_title(f'{model_name} - Errors by Sample', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}reconstruction_error_{model_name.lower()}.png', dpi=300)
        logger.info(f"Reconstruction error plot saved for {model_name}")
        plt.close()
    
    def plot_anomaly_scores(self, models_scores, y_true, figsize=(15, 5)):
        """
        Plot anomaly scores for multiple models
        
        Args:
            models_scores: Dictionary of model_name -> anomaly_scores
            y_true: True labels
            figsize: Figure size
        """
        n_models = len(models_scores)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, scores) in enumerate(models_scores.items()):
            axes[idx].scatter(range(len(scores)), scores, c=y_true, cmap='RdYlGn_r', alpha=0.6, s=20)
            axes[idx].set_title(f'{model_name} - Anomaly Scores', fontweight='bold')
            axes[idx].set_ylabel('Anomaly Score')
            axes[idx].set_xlabel('Sample Index')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}anomaly_scores.png', dpi=300)
        logger.info("Anomaly scores plot saved")
        plt.close()
    
    def plot_model_comparison(self, comparison_df, figsize=(12, 6)):
        """
        Plot model performance comparison
        
        Args:
            comparison_df: Comparison dataframe
            figsize: Figure size
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax.bar(x + i*width, comparison_df[metric], width, label=metric)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}model_comparison.png', dpi=300)
        logger.info("Model comparison plot saved")
        plt.close()
    
    def plot_pca_anomalies(self, X, y_pred, y_true=None, figsize=(12, 5)):
        """
        Visualize anomalies in PCA space
        
        Args:
            X: Feature matrix
            y_pred: Predicted labels
            y_true: True labels (optional)
            figsize: Figure size
        """
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2 if y_true is not None else 1, figsize=figsize)
        
        if y_true is None:
            axes = [axes]
        
        # Plot predictions
        scatter1 = axes[0].scatter(X_pca[y_pred == 0, 0], X_pca[y_pred == 0, 1], 
                                  label='Normal (Pred)', alpha=0.6, s=30, color='green')
        scatter2 = axes[0].scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], 
                                  label='Anomaly (Pred)', alpha=0.6, s=30, color='red')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[0].set_title('Predicted Anomalies (PCA)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot ground truth if provided
        if y_true is not None:
            axes[1].scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1], 
                           label='Normal (True)', alpha=0.6, s=30, color='green')
            axes[1].scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], 
                           label='Anomaly (True)', alpha=0.6, s=30, color='red')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            axes[1].set_title('True Anomalies (PCA)', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}pca_anomalies.png', dpi=300)
        logger.info("PCA anomalies plot saved")
        plt.close()
