# Anomaly Detection in Network Traffic Using Unsupervised Machine Learning

## Project Overview

This project implements an unsupervised machine learning system for detecting anomalous patterns in network traffic using the NSL-KDD dataset. The system employs three distinct anomaly detection algorithms:

1. **Isolation Forest** - Tree-based anomaly detection
2. **Local Outlier Factor (LOF)** - Density-based anomaly detection  
3. **Deep Autoencoders** - Neural network-based reconstruction error method

## Project Structure

```
Anomaly_detection/
├── data/                          # Dataset storage
│   ├── KDDTrain+.csv             # Training data
│   └── KDDTest+.csv              # Testing data
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading and preprocessing
│   ├── eda.py                    # Exploratory data analysis
│   ├── isolation_forest_model.py # Isolation Forest implementation
│   ├── lof_model.py              # LOF implementation
│   ├── autoencoder_model.py      # Deep Autoencoder implementation
│   ├── evaluation.py             # Metrics and evaluation
│   └── visualization.py          # Plotting and visualization
├── models/                       # Trained model storage
├── results/                      # Analysis results and plots
├── notebooks/                    # Jupyter notebooks for exploration
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dependencies

- Python 3.7+
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-Learn - Machine learning algorithms
- TensorFlow/Keras - Deep learning framework
- Matplotlib - Plotting library
- Seaborn - Statistical visualization
- SciPy - Scientific computing

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup & Installation

### 1. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
- Download NSL-KDD dataset from: https://www.kaggle.com/datasets/hassan06/nslkdd
- Place the following files in the `data/` directory:
  - `KDDTrain+.csv`
  - `KDDTest+.csv`

## Usage

### Run Complete Pipeline
```bash
python main.py
```

This will:
1. Load and preprocess the NSL-KDD dataset
2. Perform exploratory data analysis with visualizations
3. Train Isolation Forest, LOF, and Deep Autoencoder models
4. Evaluate all models with comprehensive metrics
5. Generate comparison visualizations and reports
6. Save all results and trained models

### Output Files
After execution, you'll find:
- **Models**: `models/isolation_forest_model.pkl`, `models/lof_model.pkl`, `models/autoencoder_model.h5`
- **Reports**: `results/evaluation_report.txt`, `results/eda_report.txt`
- **Visualizations**: 
  - `class_distribution.png` - Class imbalance visualization
  - `pca_visualization.png` - PCA projection of traffic
  - `tsne_visualization.png` - t-SNE projection
  - `confusion_matrices.png` - Confusion matrices for all models
  - `roc_curves.png` - ROC curve comparison
  - `anomaly_scores.png` - Anomaly score distributions
  - `reconstruction_error_deep_autoencoder.png` - Autoencoder error analysis
  - `model_comparison.png` - Performance metrics comparison
  - `pca_anomalies.png` - Detected anomalies in PCA space

## Module Documentation

### DataPreprocessor (preprocessing.py)
Handles data loading, cleaning, encoding, and scaling.

**Key Methods:**
- `load_nsl_kdd_dataset()` - Load NSL-KDD CSV files
- `handle_missing_values()` - Clean missing data
- `encode_categorical_features()` - Encode categorical variables
- `scale_numerical_features()` - Normalize numerical features
- `preprocess_pipeline()` - Complete preprocessing workflow

### EDAAnalyzer (eda.py)
Performs exploratory data analysis and generates insights.

**Key Methods:**
- `analyze_class_distribution()` - Visualize class imbalance
- `analyze_feature_distribution()` - Plot feature distributions
- `analyze_correlation_matrix()` - Create correlation heatmap
- `perform_pca_analysis()` - PCA dimensionality reduction
- `perform_tsne_analysis()` - t-SNE visualization
- `generate_eda_report()` - Create comprehensive EDA report

### IsolationForestAnomalyDetector (isolation_forest_model.py)
Tree-based anomaly detection using Isolation Forest.

**Key Methods:**
- `fit()` - Train on data
- `predict()` - Detect anomalies
- `get_anomaly_scores()` - Get anomaly scores
- `save_model()` / `load_model()` - Model persistence

### LOFAnomalyDetector (lof_model.py)
Density-based anomaly detection using Local Outlier Factor.

**Key Methods:**
- `fit_predict()` - Train and predict in one step
- `get_lof_scores()` - Get outlier factor scores
- `save_model()` / `load_model()` - Model persistence

### DeepAutoencoder (autoencoder_model.py)
Neural network autoencoder for unsupervised anomaly detection.

**Key Methods:**
- `build_model()` - Create encoder-decoder architecture
- `train()` - Train on normal data only
- `set_threshold()` - Set anomaly detection threshold
- `predict()` - Detect anomalies based on reconstruction error
- `save_model()` / `load_model()` - Model persistence

### ModelEvaluator (evaluation.py)
Comprehensive model evaluation and comparison.

**Key Methods:**
- `compute_metrics()` - Calculate accuracy, precision, recall, F1, ROC-AUC
- `compute_confusion_matrix()` - Generate confusion matrix
- `get_roc_curve()` - Calculate ROC curve
- `compare_models()` - Comparative analysis
- `generate_evaluation_report()` - Create evaluation report

### ResultVisualizer (visualization.py)
Create publication-quality visualizations.

**Key Methods:**
- `plot_confusion_matrices()` - Compare confusion matrices
- `plot_roc_curves()` - Plot ROC curves
- `plot_anomaly_scores()` - Visualize anomaly scores
- `plot_reconstruction_error()` - Plot reconstruction error distribution
- `plot_model_comparison()` - Compare model performance
- `plot_pca_anomalies()` - Show anomalies in PCA space

## Performance Metrics Explained

### Confusion Matrix
- **True Positives (TP)**: Correctly identified anomalies
- **True Negatives (TN)**: Correctly identified normal traffic
- **False Positives (FP)**: Normal traffic misidentified as anomalies
- **False Negatives (FN)**: Anomalies missed by the model

### Key Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - Reliability of positive predictions
- **Recall**: TP / (TP + FN) - Ability to find all anomalies
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### Trade-offs
- **High Precision**: Fewer false alarms but may miss anomalies
- **High Recall**: Catches most anomalies but more false alarms
- Choose based on use case: Cost of false positives vs false negatives

## Configuration & Parameters

### Isolation Forest
- `contamination=0.1` - Expected anomaly proportion
- `n_estimators=100` - Number of trees
- Adjust contamination based on expected anomaly rate

### LOF
- `n_neighbors=20` - Neighbors for local density calculation
- `contamination=0.1` - Expected anomaly proportion
- Increase n_neighbors for smoother boundaries

### Deep Autoencoder
- `encoding_dim=8` - Latent space dimension
- `learning_rate=0.001` - Adam optimizer learning rate
- `epochs=50` - Training iterations
- `percentile=95` - Threshold percentile for anomaly detection
- Adjust architecture for better performance

## Tips for Best Results

1. **Data Quality**: Ensure NSL-KDD dataset is properly downloaded
2. **Feature Scaling**: Use StandardScaler for numerical features
3. **Categorical Encoding**: Use LabelEncoder for categorical variables
4. **Class Imbalance**: Adjust contamination parameter to reflect actual anomaly rate
5. **Threshold Tuning**: Experiment with reconstruction error percentiles
6. **Cross-validation**: Consider implementing k-fold validation

## Extending the Project

### Add New Models
Create a new module in `src/` following the pattern of existing models.

### Feature Engineering
Enhance `preprocessing.py` with domain-specific feature engineering.

### Hyperparameter Tuning
Use GridSearchCV or RandomizedSearchCV for parameter optimization.

### Real-time Detection
Adapt models for streaming/online anomaly detection.

### Ensemble Methods
Combine predictions from multiple models for improved performance.

## Troubleshooting

### Out of Memory Error
- Reduce batch size in autoencoder training
- Use data subset for t-SNE analysis
- Process data in chunks

### Slow t-SNE
- Use smaller dataset subset
- Reduce perplexity parameter
- Use approximate t-SNE (openTSNE library)

### Poor Model Performance
- Adjust contamination parameter
- Experiment with different preprocessing techniques
- Tune model hyperparameters
- Ensure quality dataset

## References

- Gogoi et al. (2012). NSL-KDD Dataset
- Liu et al. (2008). Isolation Forest - IEEE ICDM
- Breunig et al. (2000). LOF - ACM SIGMOD
- Hinton & Salakhutdinov (2006). Autoencoders

## Author Notes

This project demonstrates practical implementation of unsupervised anomaly detection techniques for cybersecurity. The combination of tree-based, density-based, and neural network approaches provides comprehensive coverage of different anomaly detection paradigms.

## License

This project is provided for educational purposes.

## Contact & Support

For questions or issues, please refer to the code documentation and comments.

---

**Last Updated**: December 2024
**Python Version**: 3.7+
**Status**: Production Ready
