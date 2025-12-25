"""
Deep Autoencoder Module
Implements anomaly detection using PyTorch neural network autoencoders
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")




class Autoencoder(nn.Module):
    """PyTorch Autoencoder Architecture"""
    
    def __init__(self, input_dim, encoding_dim=8):
        """Initialize autoencoder layers"""
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class DeepAutoencoder:
    """Deep Autoencoder for unsupervised anomaly detection"""
    
    def __init__(self, input_dim, encoding_dim=8, learning_rate=0.001):
        """
        Initialize Deep Autoencoder
        
        Args:
            input_dim: Dimension of input features
            encoding_dim: Dimension of encoding layer
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self.reconstruction_errors = None
        self.predictions = None
    
    def build_model(self):
        """Build encoder-decoder architecture"""
        logger.info(f"Building autoencoder with input_dim={self.input_dim}, encoding_dim={self.encoding_dim}")
        
        self.model = Autoencoder(self.input_dim, self.encoding_dim).to(device)
        
        logger.info("Model built successfully")
        logger.info(self.model)
    
    def train(self, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the autoencoder on normal data
        
        Args:
            X_train: Training data (ideally only normal samples)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data proportion
            verbose: Verbosity level
        """
        logger.info("Starting training...")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train).to(device)
        dataset = TensorDataset(X_tensor)
        
        # Split into train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            self.model.train()
            for batch in train_loader:
                X_batch = batch[0].to(device)
                
                optimizer.zero_grad()
                output, _ = self.model(X_batch)
                loss = criterion(output, X_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch[0].to(device)
                    output, _ = self.model(X_batch)
                    loss = criterion(output, X_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed")
        return history
    
    def get_reconstruction_error(self, X):
        """
        Calculate reconstruction error for samples
        
        Args:
            X: Input samples
            
        Returns:
            errors: Mean squared error for each sample
        """
        X_tensor = torch.FloatTensor(X).to(device)
        self.model.eval()
        
        with torch.no_grad():
            output, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - output) ** 2, dim=1).cpu().numpy()
        
        return errors
    
    def set_threshold(self, X_normal, percentile=95):
        """
        Set anomaly threshold based on normal data reconstruction error
        
        Args:
            X_normal: Known normal samples
            percentile: Percentile for threshold
        """
        errors = self.get_reconstruction_error(X_normal)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"Threshold set to {self.threshold:.4f} (percentile: {percentile})")
    
    def predict(self, X, threshold=None):
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Input samples
            threshold: Anomaly threshold (uses set threshold if None)
            
        Returns:
            predictions: Binary predictions (0: normal, 1: anomaly)
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            threshold = self.threshold
        
        errors = self.get_reconstruction_error(X)
        self.reconstruction_errors = errors
        self.predictions = (errors > threshold).astype(int)
        
        logger.info(f"Predictions made. Anomalies detected: {np.sum(self.predictions)}")
        return self.predictions
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        self.model.load_state_dict(torch.load(filepath, map_location=device))
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get model parameters and information"""
        info = {
            'algorithm': 'Deep Autoencoder (PyTorch)',
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'learning_rate': self.learning_rate,
            'threshold': self.threshold
        }
        return info
