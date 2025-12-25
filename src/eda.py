"""
Exploratory Data Analysis (EDA) Module
Visualizes data distribution, correlations, and anomaly patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Performs exploratory data analysis on network traffic data"""
    
    def __init__(self, output_dir='results/'):
        """
        Initialize EDA analyzer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        sns.set_style("whitegrid")
    
    def analyze_class_distribution(self, y, title="Class Distribution"):
        """
        Visualize class distribution
        
        Args:
            y: Target labels
            title: Plot title
        """
        plt.figure(figsize=(8, 5))
        unique, counts = np.unique(y, return_counts=True)
        plt.bar(['Normal' if u == 0 else 'Anomaly' for u in unique], counts, color=['green', 'red'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}class_distribution.png', dpi=300)
        logger.info("Class distribution plot saved")
        plt.close()
    
    def analyze_feature_distribution(self, X, feature_names=None, max_features=10):
        """
        Visualize distribution of numerical features
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            max_features: Maximum number of features to plot
        """
        n_features = min(max_features, X.shape[1])
        fig, axes = plt.subplots(n_features // 2 + n_features % 2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].hist(X[:, i], bins=50, color='skyblue', edgecolor='black')
            feat_name = f"Feature {i}" if feature_names is None else feature_names[i]
            axes[i].set_title(f'Distribution of {feat_name}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        try:
            plt.tight_layout()
        except:
            pass  # Skip tight_layout if it causes issues
        plt.savefig(f'{self.output_dir}feature_distribution.png', dpi=300)
        logger.info("Feature distribution plot saved")
        plt.close()
    
    def analyze_correlation_matrix(self, df, figsize=(12, 10)):
        """
        Visualize correlation matrix heatmap
        
        Args:
            df: Input dataframe with numerical features
            figsize: Figure size
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        corr_matrix = numerical_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}correlation_matrix.png', dpi=300)
        logger.info("Correlation matrix heatmap saved")
        plt.close()
    
    def perform_pca_analysis(self, X, n_components=2, y=None):
        """
        Perform PCA and visualize
        
        Args:
            X: Feature matrix
            n_components: Number of PCA components
            y: Labels for coloring
            
        Returns:
            X_pca: Transformed data
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        logger.info(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        logger.info(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.4f}")
        
        # Visualize PCA
        plt.figure(figsize=(10, 7))
        if y is not None:
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r', 
                                 alpha=0.6, edgecolors='k', s=50)
            plt.colorbar(scatter, label='Class (0: Normal, 1: Anomaly)')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolors='k')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.title('PCA Visualization of Network Traffic', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}pca_visualization.png', dpi=300)
        logger.info("PCA visualization saved")
        plt.close()
        
        return X_pca
    
    def perform_tsne_analysis(self, X, n_components=2, y=None, perplexity=30):
        """
        Perform t-SNE and visualize
        
        Args:
            X: Feature matrix
            n_components: Number of components
            y: Labels for coloring
            perplexity: t-SNE perplexity parameter
            
        Returns:
            X_tsne: Transformed data
        """
        logger.info("Performing t-SNE analysis (this may take a while)...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Visualize t-SNE
        plt.figure(figsize=(10, 7))
        if y is not None:
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn_r', 
                                 alpha=0.6, edgecolors='k', s=50)
            plt.colorbar(scatter, label='Class (0: Normal, 1: Anomaly)')
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, edgecolors='k')
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE Visualization of Network Traffic', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}tsne_visualization.png', dpi=300)
        logger.info("t-SNE visualization saved")
        plt.close()
        
        return X_tsne
    
    def generate_eda_report(self, df, X, y, output_file='eda_report.txt'):
        """
        Generate a comprehensive EDA report
        
        Args:
            df: Original dataframe
            X: Feature matrix
            y: Target labels
            output_file: Output filename
        """
        with open(f'{self.output_dir}{output_file}', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {X.shape[0]}\n")
            f.write(f"Total features: {X.shape[1]}\n")
            f.write(f"Data shape: {X.shape}\n\n")
            
            f.write("2. CLASS DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                class_name = "Normal" if u == 0 else "Anomaly"
                percentage = (c / len(y)) * 100
                f.write(f"{class_name}: {c} samples ({percentage:.2f}%)\n")
            f.write(f"Imbalance Ratio: {max(counts) / min(counts):.2f}:1\n\n")
            
            f.write("3. FEATURE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {np.mean(X, axis=0)[:5]}\n")
            f.write(f"Std Dev: {np.std(X, axis=0)[:5]}\n")
            f.write(f"Min: {np.min(X, axis=0)[:5]}\n")
            f.write(f"Max: {np.max(X, axis=0)[:5]}\n\n")
            
            f.write("4. DATA QUALITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n")
            f.write(f"Duplicate rows: {df.duplicated().sum()}\n")
        
        logger.info(f"EDA report saved to {self.output_dir}{output_file}")
