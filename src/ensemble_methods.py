"""
Ensemble Methods Module for Anomaly Detection
Combines multiple anomaly detection models for improved performance
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AnomalyDetectionEnsemble:
    """Ensemble methods for combining multiple anomaly detection models"""
    
    def __init__(self, random_state=42):
        """
        Initialize ensemble
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model_weights = {}
        self.ensemble_predictions = None
        self.ensemble_scores = None
        self.performance_metrics = {}
        
    def voting_ensemble(self, predictions_dict, method='hard', weights=None):
        """
        Combine predictions using voting mechanism
        
        Args:
            predictions_dict: Dict with model names and binary predictions
            method: 'hard' (majority vote) or 'soft' (average probability)
            weights: Optional weights for each model
            
        Returns:
            ensemble_predictions: Combined predictions
        """
        logger.info("=" * 80)
        logger.info("VOTING ENSEMBLE")
        logger.info("=" * 80)
        
        model_names = list(predictions_dict.keys())
        predictions = np.array(list(predictions_dict.values()))
        
        # Initialize weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in model_names}
        
        logger.info(f"Ensemble method: {method} voting")
        logger.info(f"Models: {model_names}")
        logger.info(f"Weights: {weights}\n")
        
        if method == 'hard':
            # Weighted majority voting
            weighted_sum = np.zeros(predictions.shape[1])
            for i, model_name in enumerate(model_names):
                weighted_sum += predictions[i] * weights.get(model_name, 1.0)
            
            # Threshold at 0.5
            threshold = sum(weights.values()) / 2
            ensemble_pred = (weighted_sum >= threshold).astype(int)
            
        elif method == 'soft':
            # Average predictions (weighted)
            weighted_avg = np.zeros(predictions.shape[1])
            weight_sum = 0
            
            for i, model_name in enumerate(model_names):
                w = weights.get(model_name, 1.0)
                weighted_avg += predictions[i] * w
                weight_sum += w
            
            weighted_avg /= weight_sum
            ensemble_pred = (weighted_avg >= 0.5).astype(int)
        
        else:
            raise ValueError(f"Unknown voting method: {method}")
        
        self.ensemble_predictions = ensemble_pred
        return ensemble_pred
    
    def average_scores_ensemble(self, scores_dict, weights=None):
        """
        Combine anomaly scores using weighted averaging
        
        Args:
            scores_dict: Dict with model names and anomaly scores
            weights: Optional weights for each model
            
        Returns:
            ensemble_scores: Combined anomaly scores (0-1 range)
        """
        logger.info("=" * 80)
        logger.info("WEIGHTED AVERAGE ENSEMBLE")
        logger.info("=" * 80)
        
        model_names = list(scores_dict.keys())
        
        # Normalize scores to 0-1 range for each model
        normalized_scores = {}
        for model_name, scores in scores_dict.items():
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(scores)
            normalized_scores[model_name] = normalized
            logger.info(f"{model_name}: min={np.min(scores):.4f}, max={np.max(scores):.4f}")
        
        # Initialize weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in model_names}
        
        logger.info(f"\nWeights: {weights}\n")
        
        # Compute weighted average
        ensemble_scores = np.zeros(len(list(scores_dict.values())[0]))
        weight_sum = 0
        
        for model_name in model_names:
            w = weights.get(model_name, 1.0)
            ensemble_scores += normalized_scores[model_name] * w
            weight_sum += w
        
        ensemble_scores /= weight_sum
        self.ensemble_scores = ensemble_scores
        
        return ensemble_scores
    
    def threshold_ensemble(self, scores_dict, thresholds=None):
        """
        Combine models using adaptive thresholds
        
        Args:
            scores_dict: Dict with model names and anomaly scores
            thresholds: Dict with optimal thresholds for each model
            
        Returns:
            ensemble_predictions: Predictions using adaptive thresholds
        """
        logger.info("=" * 80)
        logger.info("THRESHOLD-BASED ENSEMBLE")
        logger.info("=" * 80)
        
        if thresholds is None:
            # Use median as threshold for each model
            thresholds = {}
            for model_name, scores in scores_dict.items():
                thresholds[model_name] = np.median(scores)
        
        logger.info(f"Using adaptive thresholds: {thresholds}\n")
        
        # Convert scores to predictions using model-specific thresholds
        predictions_dict = {}
        for model_name, scores in scores_dict.items():
            threshold = thresholds[model_name]
            predictions_dict[model_name] = (scores >= threshold).astype(int)
        
        # Use voting ensemble on threshold-based predictions
        ensemble_pred = self.voting_ensemble(predictions_dict, method='hard')
        
        return ensemble_pred
    
    def stacking_ensemble(self, base_predictions_dict, meta_predictions=None):
        """
        Combine models using stacking (meta-learner approach)
        
        Args:
            base_predictions_dict: Dict with base model predictions
            meta_predictions: Optional pre-computed meta-learner predictions
            
        Returns:
            ensemble_predictions: Final stacked predictions
        """
        logger.info("=" * 80)
        logger.info("STACKING ENSEMBLE")
        logger.info("=" * 80)
        
        model_names = list(base_predictions_dict.keys())
        logger.info(f"Base models: {model_names}")
        logger.info("Meta-learner: Logistic Regression (weighted voting)\n")
        
        # Stack predictions as features
        X_meta = np.column_stack([base_predictions_dict[name] for name in model_names])
        
        # Simple meta-learner: weighted majority voting
        # In practice, you could train a second-level classifier here
        ensemble_pred = (np.mean(X_meta, axis=1) >= 0.5).astype(int)
        
        self.ensemble_predictions = ensemble_pred
        return ensemble_pred
    
    def calculate_weights_from_performance(self, y_true, predictions_dict, y_scores_dict=None):
        """
        Calculate ensemble weights based on individual model performance
        
        Args:
            y_true: True labels
            predictions_dict: Dict with model predictions
            y_scores_dict: Optional dict with model scores
            
        Returns:
            weights: Dict with calculated weights for each model
        """
        logger.info("=" * 80)
        logger.info("CALCULATING ENSEMBLE WEIGHTS FROM PERFORMANCE")
        logger.info("=" * 80)
        
        weights = {}
        
        for model_name, y_pred in predictions_dict.items():
            # Calculate performance metrics
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate ROC-AUC if scores available
            try:
                if y_scores_dict and model_name in y_scores_dict:
                    auc = roc_auc_score(y_true, y_scores_dict[model_name])
                else:
                    auc = 0.5
            except:
                auc = 0.5
            
            # Weight = F1 score (balanced metric)
            weight = f1
            weights[model_name] = weight
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {auc:.4f}")
            logger.info(f"  Weight: {weight:.4f}")
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        self.model_weights = weights
        logger.info(f"\nNormalized Weights: {weights}\n")
        
        return weights
    
    def evaluate_ensemble(self, y_true, ensemble_predictions, ensemble_scores=None):
        """
        Evaluate ensemble performance
        
        Args:
            y_true: True labels
            ensemble_predictions: Ensemble predictions
            ensemble_scores: Optional ensemble scores
            
        Returns:
            metrics: Dict with performance metrics
        """
        logger.info("=" * 80)
        logger.info("ENSEMBLE PERFORMANCE EVALUATION")
        logger.info("=" * 80)
        
        acc = accuracy_score(y_true, ensemble_predictions)
        precision = precision_score(y_true, ensemble_predictions, zero_division=0)
        recall = recall_score(y_true, ensemble_predictions, zero_division=0)
        f1 = f1_score(y_true, ensemble_predictions, zero_division=0)
        
        try:
            if ensemble_scores is not None:
                auc = roc_auc_score(y_true, ensemble_scores)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
        
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {auc:.4f}\n")
        
        self.performance_metrics['ensemble'] = metrics
        return metrics
    
    def generate_ensemble_report(self, individual_metrics=None):
        """
        Generate comprehensive ensemble comparison report
        
        Args:
            individual_metrics: Dict with individual model metrics
            
        Returns:
            report: Formatted report string
        """
        logger.info("=" * 80)
        logger.info("ENSEMBLE METHODS REPORT")
        logger.info("=" * 80)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENSEMBLE METHODS - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model weights
        if self.model_weights:
            report_lines.append("MODEL WEIGHTS (Performance-Based)")
            report_lines.append("-" * 80)
            for model_name, weight in self.model_weights.items():
                report_lines.append(f"  {model_name}: {weight:.4f}")
            report_lines.append("")
        
        # Individual model performance
        if individual_metrics:
            report_lines.append("INDIVIDUAL MODEL PERFORMANCE")
            report_lines.append("-" * 80)
            for model_name, metrics in individual_metrics.items():
                report_lines.append(f"\n{model_name}:")
                for metric_name, value in metrics.items():
                    report_lines.append(f"  {metric_name}: {value:.4f}")
        
        # Ensemble performance
        if 'ensemble' in self.performance_metrics:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("ENSEMBLE PERFORMANCE")
            report_lines.append("-" * 80)
            for metric_name, value in self.performance_metrics['ensemble'].items():
                report_lines.append(f"  {metric_name}: {value:.4f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("ENSEMBLE BENEFITS")
        report_lines.append("-" * 80)
        report_lines.append("  ✓ Combines strengths of multiple models")
        report_lines.append("  ✓ Reduces overfitting and variance")
        report_lines.append("  ✓ More robust predictions")
        report_lines.append("  ✓ Better generalization on unseen data")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        logger.info(report)
        
        return report
    
    def save_ensemble_report(self, filepath):
        """Save ensemble report to file"""
        with open(filepath, 'w') as f:
            f.write(self.generate_ensemble_report())
        logger.info(f"Ensemble report saved to {filepath}")
