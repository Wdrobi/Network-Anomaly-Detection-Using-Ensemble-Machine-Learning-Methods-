"""
Local Outlier Factor (LOF) Module
Implements density-based anomaly detection using LOF algorithm
"""

from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LOFAnomalyDetector:
    """Local Outlier Factor based anomaly detector"""
    
    def __init__(self, n_neighbors=10, contamination=0.1, novelty=False, **kwargs):
        """
        Initialize LOF detector
        
        Args:
            n_neighbors: Number of neighbors for LOF calculation (reduced for performance)
            contamination: Expected proportion of anomalies
            novelty: Whether to use novelty mode
            **kwargs: Additional parameters for LocalOutlierFactor
        """
        self.n_neighbors = min(n_neighbors, 10)  # Cap at 10 neighbors for performance
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=contamination,
            novelty=novelty,
            n_jobs=-1,  # Use all cores
            **kwargs
        )
        self.predictions = None
        self.lof_scores = None
    
    def fit_predict(self, X):
        """
        Fit LOF model and predict anomalies
        
        Args:
            X: Input features
            
        Returns:
            predictions: Binary predictions (0: normal, 1: anomaly)
        """
        logger.info("Fitting and predicting with LOF model...")
        predictions = self.model.fit_predict(X)
        # Convert to binary format (0: normal, 1: anomaly)
        self.predictions = ((predictions == -1) * 1).astype(int)
        logger.info(f"LOF fit-predict completed. Anomalies detected: {np.sum(self.predictions)}")
        return self.predictions
    
    def get_lof_scores(self, X=None):
        """
        Get Local Outlier Factor scores
        
        Args:
            X: Input features (if None, uses training data)
            
        Returns:
            lof_scores: LOF scores (higher = more anomalous)
        """
        if X is None:
            # Use negative_outlier_factor_ from training
            self.lof_scores = -self.model.negative_outlier_factor_
        else:
            # For new samples - need to use training data as neighbors
            lof_new = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                novelty=True
            )
            lof_new.fit(self.model._fit_X)
            self.lof_scores = -lof_new.score_samples(X)
        
        return self.lof_scores
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"LOF model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"LOF model loaded from {filepath}")
    
    def get_model_info(self):
        """Get model parameters and information"""
        info = {
            'algorithm': 'Local Outlier Factor',
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'n_features': self.model.n_features_in_
        }
        return info
