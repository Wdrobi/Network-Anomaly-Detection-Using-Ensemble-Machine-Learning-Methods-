"""
Data Acquisition & Preprocessing Module
Handles data loading, cleaning, encoding, and preparation for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading and preprocessing for NSL-KDD dataset"""
    
    def __init__(self, train_path=None, test_path=None):
        """
        Initialize the preprocessor
        
        Args:
            train_path: Path to training data
            test_path: Path to testing data
        """
        self.train_path = train_path
        self.test_path = test_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_nsl_kdd_dataset(self, train_path, test_path):
        """
        Load NSL-KDD dataset from CSV files
        
        Args:
            train_path: Path to training CSV
            test_path: Path to testing CSV
            
        Returns:
            train_df, test_df: Loaded dataframes
        """
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, header=None)
        
        logger.info(f"Loading testing data from {test_path}")
        test_df = pd.read_csv(test_path, header=None)
        
        # NSL-KDD has 41 features (columns 0-40) + 1 label (column 41)
        # Rename columns appropriately
        columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                   'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                   'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                   'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                   'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
        
        # Only rename columns that exist
        rename_dict = {i: col for i, col in enumerate(columns) if i < len(train_df.columns)}
        train_df = train_df.rename(columns=rename_dict)
        test_df = test_df.rename(columns=rename_dict)
        
        # Ensure all column names are strings
        train_df.columns = train_df.columns.astype(str)
        test_df.columns = test_df.columns.astype(str)
        
        # Strip whitespace from object columns
        for col in train_df.select_dtypes(include=['object']).columns:
            train_df[col] = train_df[col].astype(str).str.strip()
            test_df[col] = test_df[col].astype(str).str.strip()
        
        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(f"Testing set shape: {test_df.shape}")
        
        return train_df, test_df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Cleaned dataframe
        """
        missing_count = df.isnull().sum().sum()
        logger.info(f"Missing values found: {missing_count}")
        
        # Drop rows with missing values
        if missing_count > 0:
            df = df.dropna()
            logger.info(f"Shape after removing NaN values: {df.shape}")
        
        return df
    
    def encode_categorical_features(self, X_train, X_test, categorical_columns):
        """
        Encode categorical features using LabelEncoder
        
        Args:
            X_train: Training features
            X_test: Testing features
            categorical_columns: List of categorical column names
            
        Returns:
            X_train_encoded, X_test_encoded: Encoded dataframes
        """
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for col in categorical_columns:
            if col in X_train_encoded.columns:
                try:
                    le = LabelEncoder()
                    # Fit on combined data to avoid unseen labels
                    combined = pd.concat([X_train_encoded[col], X_test_encoded[col]])
                    le.fit(combined.astype(str))
                    
                    X_train_encoded[col] = le.transform(X_train_encoded[col].astype(str))
                    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
                    
                    self.label_encoders[col] = le
                    logger.info(f"Encoded categorical feature: {col}")
                except Exception as e:
                    logger.warning(f"Could not encode {col}: {str(e)}, keeping original")
        
        return X_train_encoded, X_test_encoded
    
    def scale_numerical_features(self, X_train, X_test):
        """
        Scale numerical features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            X_train_scaled, X_test_scaled: Scaled features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Numerical features scaled successfully")
        
        return X_train_scaled, X_test_scaled
    
    def convert_labels_to_binary(self, y, normal_label='normal'):
        """
        Convert labels to binary classification (0: normal, 1: anomaly)
        
        Args:
            y: Target labels
            normal_label: Label representing normal traffic
            
        Returns:
            y_binary: Binary labels
        """
        # Strip whitespace from labels for proper matching
        y_clean = y.astype(str).str.strip()
        y_binary = (y_clean != normal_label).astype(int)
        logger.info(f"Class distribution after binarization:")
        logger.info(f"Normal: {(y_binary == 0).sum()}, Anomaly: {(y_binary == 1).sum()}")
        
        return y_binary
    
    def preprocess_pipeline(self, df, target_col, categorical_cols, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            categorical_cols: List of categorical column names
            test_size: Test set proportion
            random_state: Random state for reproducibility
            
        Returns:
            X_train_scaled, X_test_scaled, y_train_binary, y_test_binary
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Convert labels to binary
        y_binary = self.convert_labels_to_binary(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Encode categorical features
        X_train_encoded, X_test_encoded = self.encode_categorical_features(
            X_train, X_test, categorical_cols
        )
        
        # Scale numerical features
        X_train_scaled, X_test_scaled = self.scale_numerical_features(
            X_train_encoded, X_test_encoded
        )
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
