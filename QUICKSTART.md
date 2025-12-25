# Quick Start Guide - Anomaly Detection Project

## Overview
This guide will help you get started with the Anomaly Detection in Network Traffic project in 5 minutes.

## Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Internet connection (for downloading dependencies and dataset)

## Step 1: Clone/Extract Project Files
Ensure all project files are in: `E:/green university/9th Semester/DM lab/Anomaly_detection/`

## Step 2: Create Virtual Environment (Recommended)

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- numpy, pandas, scikit-learn
- tensorflow, keras
- matplotlib, seaborn
- scipy, joblib

## Step 4: Download NSL-KDD Dataset

1. Visit: https://www.unb.ca/cic/datasets/nsl-kdd.html
2. Download:
   - 5
3. Place files in the `data/` directory:
   ```
   Anomaly_detection/
   └── data/
       ├── KDDTrain+.csv
       └── KDDTest+.csv
   ```

## Step 5: Run the Project

### Option A: Run Complete Pipeline (Recommended)
```bash
python main.py
```

This will execute all modules in sequence and generate:
- Preprocessed data
- EDA visualizations
- Model training for all three algorithms
- Comprehensive evaluation metrics
- Comparison visualizations
- Results reports

**Estimated execution time**: 15-30 minutes (depending on system)

### Option B: Run Interactive Analysis
```bash
jupyter notebook notebooks/anomaly_detection_analysis.ipynb
```

Then execute cells sequentially to see results step-by-step.

### Option C: Run Individual Components
```python
# In Python console
from src.preprocessing import DataPreprocessor
from src.eda import EDAAnalyzer
from src.isolation_forest_model import IsolationForestAnomalyDetector
# ... import other modules as needed
```

## Step 6: View Results

After running the pipeline, check the `results/` directory:

```
results/
├── class_distribution.png           # Class balance visualization
├── feature_distribution.png         # Feature statistics
├── pca_visualization.png            # PCA projection
├── tsne_visualization.png           # t-SNE projection
├── confusion_matrices.png           # Model confusion matrices
├── roc_curves.png                   # ROC curve comparison
├── anomaly_scores.png               # Anomaly score distributions
├── reconstruction_error_*.png       # Autoencoder error analysis
├── model_comparison.png             # Performance metrics
├── pca_anomalies.png               # Detected anomalies in PCA space
├── eda_report.txt                  # EDA analysis report
├── evaluation_report.txt           # Model evaluation report
└── model_comparison.csv            # Performance metrics table
```

## Configuration

To customize parameters, edit `config.py`:

```python
# Example: Change Isolation Forest contamination
ISOLATION_FOREST_CONFIG = {
    'contamination': 0.15,  # Change from 0.1 to 0.15
    'n_estimators': 100
}
```

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU Issues (TensorFlow)
```bash
# If TensorFlow GPU is not working, use CPU version
pip install tensorflow-cpu
```

### Memory Issues
- Reduce batch_size in autoencoder config
- Use smaller dataset subset for t-SNE
- Process data in chunks

### Dataset Not Found
```
ERROR: Dataset files not found!
Expected paths:
  - data/KDDTrain+.csv
  - data/KDDTest+.csv
```

Solution: Download and place files in the `data/` directory.

## File Structure

```
Anomaly_detection/
├── main.py                          # Main execution script
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
├── README.md                        # Full documentation
├── QUICKSTART.md                    # This file
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py            # Data preprocessing
│   ├── eda.py                      # Exploratory analysis
│   ├── isolation_forest_model.py   # Isolation Forest
│   ├── lof_model.py               # Local Outlier Factor
│   ├── autoencoder_model.py       # Deep Autoencoder
│   ├── evaluation.py              # Model evaluation
│   └── visualization.py           # Result visualization
│
├── data/                          # Dataset storage (create after download)
│   ├── KDDTrain+.csv
│   └── KDDTest+.csv
│
├── models/                        # Trained models (generated)
│   ├── isolation_forest_model.pkl
│   ├── lof_model.pkl
│   └── autoencoder_model.h5
│
├── results/                       # Analysis results (generated)
│   ├── *.png                      # Visualization files
│   ├── *.txt                      # Report files
│   └── *.csv                      # Results tables
│
└── notebooks/                     # Jupyter notebooks
    └── anomaly_detection_analysis.ipynb
```

## Key Parameters to Tune

### Contamination (All Models)
```python
# Adjust based on expected anomaly percentage
contamination=0.1  # 10% anomalies (default)
contamination=0.05 # 5% anomalies (less)
contamination=0.2  # 20% anomalies (more)
```

### LOF Neighbors
```python
n_neighbors=20  # Default (recommended)
n_neighbors=10  # More sensitive to local density
n_neighbors=30  # Smoother boundaries
```

### Autoencoder Threshold
```python
percentile=95   # Default (5% of normal data flagged)
percentile=90   # More aggressive (10% flagged)
percentile=99   # Conservative (1% flagged)
```

## Next Steps

1. **Analyze Results**: Open and examine the generated visualizations
2. **Fine-tune Models**: Adjust parameters in config.py and rerun
3. **Test New Data**: Use trained models for prediction on new traffic
4. **Create Report**: Use results for academic paper/presentation
5. **Extend Project**: Add ensemble methods or more algorithms

## Performance Notes

- **Isolation Forest**: Fastest, good for real-time detection
- **LOF**: Medium speed, detects local anomalies well
- **Deep Autoencoder**: Slowest, but can capture complex patterns

## Model Comparison Metrics

After running, you'll see:
- **Accuracy**: Overall correctness
- **Precision**: Reliability of positive predictions (low false alarms)
- **Recall**: Ability to find all anomalies
- **F1-Score**: Balanced metric
- **ROC-AUC**: Performance across thresholds

Choose based on your use case:
- **High precision needed**: Use Isolation Forest or tune threshold
- **High recall needed**: Reduce threshold percentile
- **Balanced**: Choose model with best F1-Score

## Support & Resources

- **NSL-KDD Dataset**: https://www.unb.ca/cic/datasets/nsl-kdd.html
- **Scikit-Learn Docs**: https://scikit-learn.org/
- **TensorFlow Docs**: https://www.tensorflow.org/
- **Matplotlib Docs**: https://matplotlib.org/

## Tips for Success

1. ✓ Start with default parameters
2. ✓ Review generated visualizations carefully
3. ✓ Compare model metrics in results/model_comparison.csv
4. ✓ Tune parameters based on your specific requirements
5. ✓ Save visualizations for your report/presentation
6. ✓ Document your findings and insights
7. ✓ Consider ensemble approaches if single model is insufficient

## Common Commands

```bash
# View configuration
python config.py

# Run with verbose output
python main.py > output.log 2>&1

# Run specific module (interactive)
python -c "from src.eda import EDAAnalyzer; help(EDAAnalyzer)"

# Check installation
pip list

# Update dependencies
pip install -r requirements.txt --upgrade
```

## Estimated Timelines

- **Data Loading**: 1-2 seconds
- **Preprocessing**: 10-20 seconds
- **EDA Analysis**: 1-5 minutes (t-SNE takes longest)
- **Model Training**: 5-10 minutes
- **Evaluation**: 1-2 minutes
- **Visualization**: 1-2 minutes
- **Total**: 15-30 minutes

## Success Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] NSL-KDD dataset downloaded and placed in data/
- [ ] main.py executed successfully
- [ ] results/ directory populated with visualizations
- [ ] models/ directory contains trained models
- [ ] evaluation_report.txt reviewed
- [ ] Visualizations look reasonable
- [ ] Ready for model comparison and analysis

---

**Happy Anomaly Detecting!** 🔍

Last Updated: December 2024
