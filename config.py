"""
Configuration file for Anomaly Detection Project
Centralized settings for all models and parameters
"""

# Data Configuration
DATA_CONFIG = {
    'train_data_path': 'data/KDDTrain+.csv',
    'test_data_path': 'data/KDDTest+.csv',
    'target_column': 'label',
    'normal_label': 'normal',
    'test_size': 0.2,
    'random_state': 42
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'handle_missing_values': True,
    'scaling_method': 'StandardScaler',
    'encoding_method': 'LabelEncoder',
    'remove_duplicates': True
}

# Isolation Forest Configuration
ISOLATION_FOREST_CONFIG = {
    'contamination': 0.1,
    'random_state': 42,
    'n_estimators': 100,
    'max_samples': 'auto',
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1
}

# Local Outlier Factor Configuration
LOF_CONFIG = {
    'n_neighbors': 20,
    'contamination': 0.1,
    'algorithm': 'auto',
    'leaf_size': 30,
    'metric': 'minkowski',
    'p': 2
}

# Deep Autoencoder Configuration
AUTOENCODER_CONFIG = {
    'input_dim': None,  # Will be set based on data
    'encoding_dim': 8,
    'encoder_layers': [64, 32, 16],
    'decoder_layers': [16, 32, 64],
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss_function': 'mse',
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.1,
    'early_stopping_patience': 5,
    'threshold_percentile': 95
}

# EDA Configuration
EDA_CONFIG = {
    'correlation_figsize': (12, 10),
    'pca_components': 2,
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'max_features_to_plot': 10,
    'output_format': 'png',
    'dpi': 300
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'compute_roc_auc': True,
    'compute_confusion_matrix': True,
    'compute_precision_recall': True,
    'cross_validation_folds': 5
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figsize_default': (12, 6),
    'figsize_large': (15, 10),
    'style': 'whitegrid',
    'dpi': 300,
    'font_size': 12,
    'save_format': 'png',
    'color_palette': 'RdYlGn_r'
}

# Output Paths
OUTPUT_PATHS = {
    'models_dir': 'models/',
    'results_dir': 'results/',
    'notebooks_dir': 'notebooks/',
    'data_dir': 'data/'
}

# Model Saving Configuration
MODEL_SAVE_CONFIG = {
    'isolation_forest': 'models/isolation_forest_model.pkl',
    'lof': 'models/lof_model.pkl',
    'autoencoder': 'models/autoencoder_model.h5',
    'scaler': 'models/scaler.pkl',
    'encoders': 'models/label_encoders.pkl'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Hyperparameter Tuning Configuration (for GridSearch)
HYPERPARAMETER_GRIDS = {
    'isolation_forest': {
        'contamination': [0.05, 0.1, 0.15],
        'n_estimators': [50, 100, 200]
    },
    'lof': {
        'n_neighbors': [10, 20, 30],
        'contamination': [0.05, 0.1, 0.15]
    },
    'autoencoder': {
        'encoding_dim': [4, 8, 16],
        'learning_rate': [0.0001, 0.001, 0.01],
        'epochs': [30, 50, 100]
    }
}

# Thresholding Configuration
THRESHOLD_CONFIG = {
    'isolation_forest_percentile': 90,
    'lof_percentile': 90,
    'autoencoder_percentile': 95,
    'dynamic_threshold': False  # If True, adjust threshold based on validation set
}

# Feature Engineering Configuration
FEATURE_ENGINEERING_CONFIG = {
    'use_pca': False,
    'pca_variance_threshold': 0.95,
    'use_feature_selection': False,
    'feature_selection_method': 'mutual_info_classif',  # for supervised selection
    'n_features_to_select': None  # If None, use all features
}

# Advanced Configuration
ADVANCED_CONFIG = {
    'use_gpu': True,  # For TensorFlow
    'mixed_precision': False,
    'ensemble_voting': False,  # Combine predictions from all models
    'voting_method': 'hard',  # 'hard' or 'soft'
    'class_weight': 'balanced'  # For imbalanced datasets
}


def get_config(section):
    """
    Get configuration for a specific section
    
    Args:
        section: Configuration section name
        
    Returns:
        Configuration dictionary for the section
    """
    configs = {
        'data': DATA_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'isolation_forest': ISOLATION_FOREST_CONFIG,
        'lof': LOF_CONFIG,
        'autoencoder': AUTOENCODER_CONFIG,
        'eda': EDA_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'output_paths': OUTPUT_PATHS,
        'logging': LOGGING_CONFIG,
        'hyperparameters': HYPERPARAMETER_GRIDS,
        'thresholding': THRESHOLD_CONFIG,
        'feature_engineering': FEATURE_ENGINEERING_CONFIG,
        'advanced': ADVANCED_CONFIG
    }
    
    return configs.get(section, {})


def print_all_config():
    """Print all configuration settings"""
    all_configs = {
        'Data': DATA_CONFIG,
        'Preprocessing': PREPROCESSING_CONFIG,
        'Isolation Forest': ISOLATION_FOREST_CONFIG,
        'LOF': LOF_CONFIG,
        'Autoencoder': AUTOENCODER_CONFIG,
        'EDA': EDA_CONFIG,
        'Evaluation': EVALUATION_CONFIG,
        'Visualization': VISUALIZATION_CONFIG,
        'Output Paths': OUTPUT_PATHS,
        'Logging': LOGGING_CONFIG,
        'Thresholding': THRESHOLD_CONFIG,
        'Feature Engineering': FEATURE_ENGINEERING_CONFIG,
        'Advanced': ADVANCED_CONFIG
    }
    
    print("="*80)
    print("PROJECT CONFIGURATION")
    print("="*80)
    
    for section, config in all_configs.items():
        print(f"\n{section}:")
        print("-"*80)
        for key, value in config.items():
            print(f"  {key:30s}: {value}")


if __name__ == '__main__':
    print_all_config()
