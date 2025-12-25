"""
Isolation Forest Module
Implements unsupervised anomaly detection using Isolation Forest algorithm
"""

from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Isolation Forest based anomaly detector"""
    
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100, max_samples='auto', **kwargs):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
            n_estimators: Number of estimators
            max_samples: Number of samples for each estimator
            **kwargs: Additional parameters for IsolationForest
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            **kwargs
        )
        self.predictions = None
        self.anomaly_scores = None
    
    def fit(self, X_train):
        """
        Train Isolation Forest model
        
        Args:
            X_train: Training features
        """
        logger.info("Training Isolation Forest model...")
        self.model.fit(X_train)
        logger.info("Isolation Forest model trained successfully")
    
    def predict(self, X):
        """
        Predict anomalies using trained model
        
        Args:
            X: Input features
            
        Returns:
            predictions: Binary predictions (-1: anomaly, 1: normal)
        """
        logger.info(f"Making predictions on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        # Convert to binary format (0: normal, 1: anomaly)
        self.predictions = ((predictions == -1) * 1).astype(int)
        logger.info(f"Predictions completed. Anomalies detected: {np.sum(self.predictions)}")
        return self.predictions
    
    def get_anomaly_scores(self, X):
        """
        Get anomaly scores (negative of decision function)
        
        Args:
            X: Input features
            
        Returns:
            anomaly_scores: Anomaly scores (higher = more anomalous)
        """
        # Get decision function scores
        decision_scores = self.model.decision_function(X)
        # Convert to anomaly scores (negate so higher = more anomalous)
        self.anomaly_scores = -decision_scores
        return self.anomaly_scores
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get model parameters and information"""
        info = {
            'algorithm': 'Isolation Forest',
            'contamination': self.model.contamination,
            'n_estimators': self.model.n_estimators,
            'max_samples': self.model.max_samples,
            'max_features': self.model.max_features,
            'n_features': self.model.n_features_in_
        }
        return info
