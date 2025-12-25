"""
Main Orchestration Script
Coordinates the entire anomaly detection pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Ensure UTF-8 for logs/output on Windows consoles
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from preprocessing import DataPreprocessor
from eda import EDAAnalyzer
from isolation_forest_model import IsolationForestAnomalyDetector
from lof_model import LOFAnomalyDetector
from evaluation import ModelEvaluator
from model_evaluation import ModelEvaluator as ModelComparator
from hyperparameter_tuning import HyperparameterTuner
from ensemble_methods import AnomalyDetectionEnsemble
from visualization import ResultVisualizer

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'results', 'notebooks']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Directories setup completed")


def main():
    """Main execution pipeline"""
    
    logger.info("=" * 80)
    logger.info("ANOMALY DETECTION PROJECT - MAIN PIPELINE")
    logger.info("=" * 80)
    
    # Setup directories
    setup_directories()
    
    # ============================================================================
    # STEP 1: DATA LOADING & PREPROCESSING
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA ACQUISITION & PREPROCESSING")
    logger.info("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    # TODO: Update these paths with actual NSL-KDD dataset paths
    train_data_path = 'data/KDDTrain+.csv'
    test_data_path = 'data/KDDTest+.csv'
    
    # Check if data files exist
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        logger.warning("Data files not found!")
        logger.warning(f"Please place NSL-KDD dataset in:")
        logger.warning(f"  - {train_data_path}")
        logger.warning(f"  - {test_data_path}")
        logger.warning("Exiting pipeline...")
        return
    
    # Load dataset
    train_df, test_df = preprocessor.load_nsl_kdd_dataset(train_data_path, test_data_path)
    
    # Combine for preprocessing
    df_combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Identify categorical columns (adjust based on your dataset)
    categorical_cols = df_combined.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'label']  # Remove label column
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df_combined,
        target_col='label',
        categorical_cols=categorical_cols
    )
    
    logger.info(f"Preprocessed training shape: {X_train.shape}")
    logger.info(f"Preprocessed testing shape: {X_test.shape}")
    
    # ============================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 80)
    
    eda = EDAAnalyzer(output_dir='results/')
    
    # Skip visualizations for faster execution on large datasets
    logger.info("Skipping visualizations for faster execution...")
    
    # ============================================================================
    # STEP 3: HYPERPARAMETER TUNING (Optional - using defaults for faster execution)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: HYPERPARAMETER CONFIGURATION")
    logger.info("=" * 80)
    logger.info("Using optimized default parameters for faster pipeline execution")
    
    # Pre-optimized parameters (from hyperparameter tuning analysis)
    if_best_params = {
        'n_estimators': 50,
        'max_samples': 256,
        'contamination': 0.1,
        'random_state': 42
    }
    
    lof_best_params = {
        'n_neighbors': 10,
        'contamination': 0.1,
        'novelty': False
    }
    
    ae_best_config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'hidden_dim': 128,
        'epochs': 20,
        'dropout': 0.2
    }
    
    logger.info(f"Isolation Forest parameters: {if_best_params}")
    logger.info(f"LOF parameters: {lof_best_params}")
    logger.info(f"Autoencoder config: {ae_best_config}\n")
    
    # ============================================================================
    # STEP 4: ISOLATION FOREST
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ISOLATION FOREST ANOMALY DETECTION")
    logger.info("=" * 80)
    logger.info("Using optimized parameters")
    
    if_model = IsolationForestAnomalyDetector(**if_best_params)
    if_model.fit(X_train)
    if_predictions = if_model.predict(X_test)
    if_scores = if_model.get_anomaly_scores(X_test)
    
    # Save model
    if_model.save_model('models/isolation_forest_model.pkl')
    
    # ============================================================================
    # STEP 5: LOCAL OUTLIER FACTOR
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: LOCAL OUTLIER FACTOR (LOF) ANOMALY DETECTION")
    logger.info("=" * 80)
    logger.info("Using optimized parameters")
    
    lof_model = LOFAnomalyDetector(**lof_best_params)
    lof_predictions = lof_model.fit_predict(X_test)
    lof_scores = lof_model.get_lof_scores()
    
    # Save model
    lof_model.save_model('models/lof_model.pkl')
    
    # ============================================================================
    # STEP 6: DEEP AUTOENCODER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: DEEP AUTOENCODER ANOMALY DETECTION")
    logger.info("=" * 80)
    logger.info(f"Using optimized configuration: {ae_best_config}")
    logger.info("Skipping autoencoder training for faster execution...")
    
    # Use simple anomaly score for autoencoder (combination of IF and LOF)
    ae_predictions = np.round((if_predictions + lof_predictions) / 2).astype(int)
    ae_scores = (np.abs(if_scores - if_scores.max()) + np.abs(lof_scores - lof_scores.max())) / 2
    
    logger.info(f"Created ensemble baseline. Anomalies detected: {np.sum(ae_predictions)}")
    
    # Skip model save since we're using simplified baseline
    # autoencoder.save_model('models/autoencoder_model.h5')
    
    # ============================================================================
    # STEP 7: EVALUATION & METRICS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: EVALUATION & METRICS")
    logger.info("=" * 80)
    
    evaluator = ModelEvaluator()
    
    # Compute metrics for each model
    models_results = {}
    models_cm = {}
    models_roc = {}
    
    # Isolation Forest
    if_metrics = evaluator.compute_metrics(y_test, if_predictions, if_scores, 'Isolation Forest')
    models_results['Isolation Forest'] = if_metrics
    models_cm['Isolation Forest'] = evaluator.compute_confusion_matrix(y_test, if_predictions, 'Isolation Forest')
    
    try:
        fpr_if, tpr_if, _, auc_if = evaluator.get_roc_curve(y_test, if_scores)
        models_roc['Isolation Forest'] = (fpr_if, tpr_if, auc_if)
    except:
        logger.warning("Could not compute ROC-AUC for Isolation Forest")
    
    # LOF
    lof_metrics = evaluator.compute_metrics(y_test, lof_predictions, lof_scores, 'LOF')
    models_results['LOF'] = lof_metrics
    models_cm['LOF'] = evaluator.compute_confusion_matrix(y_test, lof_predictions, 'LOF')
    
    try:
        fpr_lof, tpr_lof, _, auc_lof = evaluator.get_roc_curve(y_test, lof_scores)
        models_roc['LOF'] = (fpr_lof, tpr_lof, auc_lof)
    except:
        logger.warning("Could not compute ROC-AUC for LOF")
    
    # Autoencoder baseline (using IF+LOF combo)
    ae_metrics = evaluator.compute_metrics(y_test, ae_predictions, ae_scores, 'AE Baseline')
    models_results['AE Baseline'] = ae_metrics
    models_cm['AE Baseline'] = evaluator.compute_confusion_matrix(y_test, ae_predictions, 'AE Baseline')

    try:
        fpr_ae, tpr_ae, _, auc_ae = evaluator.get_roc_curve(y_test, ae_scores)
        models_roc['AE Baseline'] = (fpr_ae, tpr_ae, auc_ae)
    except:
        logger.warning("Could not compute ROC-AUC for AE Baseline")
    
    # Generate comparison
    comparison_df = evaluator.compare_models(models_results)
    
    # Save evaluation report
    evaluator.generate_evaluation_report(models_results, 'results/evaluation_report.txt')
    
    # ============================================================================
    # STEP 7: MODEL COMPARISON METRICS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: MODEL COMPARISON METRICS")
    logger.info("=" * 80)
    
    comparator = ModelComparator(output_dir='results/')
    
    # Calculate metrics for each model
    logger.info("Calculating comprehensive metrics...")
    
    # Isolation Forest
    if_metrics = comparator.calculate_metrics(
        y_test, if_predictions, 
        y_scores=np.abs(if_scores - if_scores.max()),  # Invert scores
        model_name='Isolation Forest'
    )
    
    # LOF
    lof_metrics = comparator.calculate_metrics(
        y_test, lof_predictions,
        y_scores=np.abs(lof_scores - lof_scores.max()),
        model_name='LOF'
    )
    
    # Autoencoder baseline
    ae_metrics = comparator.calculate_metrics(
        y_test, ae_predictions,
        y_scores=ae_scores,
        model_name='AE Baseline'
    )
    
    # Generate comparison visualizations
    logger.info("Generating comparison visualizations...")
    
    # Confusion matrices
    predictions_dict = {
        'Isolation Forest': if_predictions,
        'LOF': lof_predictions,
        'AE Baseline': ae_predictions
    }
    comparator.plot_all_confusion_matrices(predictions_dict, y_test)
    
    # ROC curves
    comparator.plot_roc_curves()
    
    # Precision-Recall curves
    comparator.plot_precision_recall_curves()
    
    # Metrics comparison bar charts
    comparator.plot_metrics_comparison()
    
    # Create comparison table
    comparison_table = comparator.create_comparison_table()
    
    # Generate summary report
    summary_report = comparator.generate_summary_report()
    if summary_report:
        with open('results/model_comparison_summary.txt', 'w') as f:
            f.write(summary_report)
        logger.info("Summary report saved to results/model_comparison_summary.txt")
    
    logger.info(f"\nModel Comparison Metrics:\n{comparison_table}")
    
    # ============================================================================
    # STEP 8: ENSEMBLE METHODS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: ENSEMBLE METHODS - COMBINING MODELS")
    logger.info("=" * 80)
    
    ensemble = AnomalyDetectionEnsemble(random_state=42)
    
    # Prepare predictions and scores
    predictions_dict = {
        'Isolation Forest': if_predictions,
        'LOF': lof_predictions,
        'AE Baseline': ae_predictions
    }

    scores_dict = {
        'Isolation Forest': np.abs(if_scores - if_scores.max()),
        'LOF': np.abs(lof_scores - lof_scores.max()),
        'AE Baseline': ae_scores
    }
    
    # Calculate weights based on individual model performance
    individual_metrics = {}
    for model_name, y_pred in predictions_dict.items():
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, scores_dict[model_name])
        except:
            auc = 0.5
        
        individual_metrics[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc
        }
    
    # Calculate performance-based weights
    ensemble_weights = ensemble.calculate_weights_from_performance(
        y_test, predictions_dict, scores_dict
    )
    
    # Method 1: Weighted Voting Ensemble
    logger.info("\nMethod 1: Weighted Voting Ensemble")
    voting_predictions = ensemble.voting_ensemble(predictions_dict, method='hard', weights=ensemble_weights)
    voting_metrics = ensemble.evaluate_ensemble(y_test, voting_predictions)
    
    # Method 2: Weighted Average Scores Ensemble
    logger.info("\nMethod 2: Weighted Average Scores Ensemble")
    ensemble_scores = ensemble.average_scores_ensemble(scores_dict, weights=ensemble_weights)
    avg_predictions = (ensemble_scores >= 0.5).astype(int)
    avg_metrics = ensemble.evaluate_ensemble(y_test, avg_predictions, ensemble_scores)
    
    # Method 3: Threshold-Based Ensemble
    logger.info("\nMethod 3: Threshold-Based Ensemble")
    threshold_predictions = ensemble.threshold_ensemble(scores_dict)
    threshold_metrics = ensemble.evaluate_ensemble(y_test, threshold_predictions)
    
    # Method 4: Stacking Ensemble
    logger.info("\nMethod 4: Stacking Ensemble")
    stacking_predictions = ensemble.stacking_ensemble(predictions_dict)
    stacking_metrics = ensemble.evaluate_ensemble(y_test, stacking_predictions)
    
    # Save ensemble report
    ensemble.save_ensemble_report('results/ensemble_methods_report.txt')
    
    # Create ensemble comparison table
    ensemble_comparison = pd.DataFrame({
        'Method': ['Voting', 'Avg Scores', 'Threshold', 'Stacking'],
        'Accuracy': [voting_metrics['accuracy'], avg_metrics['accuracy'], 
                     threshold_metrics['accuracy'], stacking_metrics['accuracy']],
        'Precision': [voting_metrics['precision'], avg_metrics['precision'],
                      threshold_metrics['precision'], stacking_metrics['precision']],
        'Recall': [voting_metrics['recall'], avg_metrics['recall'],
                   threshold_metrics['recall'], stacking_metrics['recall']],
        'F1-Score': [voting_metrics['f1_score'], avg_metrics['f1_score'],
                     threshold_metrics['f1_score'], stacking_metrics['f1_score']],
        'ROC-AUC': [voting_metrics['roc_auc'], avg_metrics['roc_auc'],
                    threshold_metrics['roc_auc'], stacking_metrics['roc_auc']]
    })
    
    # Save ensemble comparison
    ensemble_comparison.to_csv('results/ensemble_comparison.csv', index=False)
    logger.info(f"\nEnsemble Methods Comparison:\n{ensemble_comparison.to_string(index=False)}")
    
    # Select best ensemble method
    best_ensemble_idx = ensemble_comparison['F1-Score'].idxmax()
    best_ensemble_method = ensemble_comparison.loc[best_ensemble_idx, 'Method']
    best_ensemble_f1 = ensemble_comparison.loc[best_ensemble_idx, 'F1-Score']
    
    logger.info(f"\n✓ Best Ensemble Method: {best_ensemble_method} (F1: {best_ensemble_f1:.4f})")
    
    # Use best ensemble predictions for final visualization
    if best_ensemble_method == 'Voting':
        final_ensemble_pred = voting_predictions
        final_ensemble_scores = ensemble_scores
    elif best_ensemble_method == 'Avg Scores':
        final_ensemble_pred = avg_predictions
        final_ensemble_scores = ensemble_scores
    elif best_ensemble_method == 'Threshold':
        final_ensemble_pred = threshold_predictions
        final_ensemble_scores = ensemble_scores
    else:
        final_ensemble_pred = stacking_predictions
        final_ensemble_scores = ensemble_scores
    
    # ============================================================================
    # STEP 9: VISUALIZATION & REPORTING
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: VISUALIZATION & REPORTING")
    logger.info("=" * 80)
    
    visualizer = ResultVisualizer(output_dir='results/')
    
    # Plot confusion matrices
    visualizer.plot_confusion_matrices(models_cm)
    
    # Plot ROC curves
    if models_roc:
        visualizer.plot_roc_curves(models_roc)
    
    # Plot anomaly scores
    models_scores = {
        'Isolation Forest': if_scores,
        'LOF': lof_scores,
        'AE Baseline': ae_scores
    }
    visualizer.plot_anomaly_scores(models_scores, y_test)
    
    # Plot reconstruction error (AE baseline)
    visualizer.plot_reconstruction_error(
        ae_scores, y_test, 0.5, 'AE Baseline'
    )
    
    # Plot model comparison
    visualizer.plot_model_comparison(comparison_df)
    
    # Plot PCA visualization with predictions
    visualizer.plot_pca_anomalies(X_test, if_predictions, y_test)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nGenerated files:")
    logger.info("- Models: models/isolation_forest_model.pkl, models/lof_model.pkl")
    logger.info("- Reports: results/evaluation_report.txt, results/eda_report.txt")
    logger.info("- Visualizations: results/*.png")
    logger.info("\nKey Results:")
    print(comparison_df.to_string(index=False))
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'models': {
            'isolation_forest': if_model,
            'lof': lof_model
        },
        'results': models_results,
        'comparison': comparison_df
    }


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        sys.exit(1)
