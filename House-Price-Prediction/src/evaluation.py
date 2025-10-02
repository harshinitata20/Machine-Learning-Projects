"""
Model evaluation module for house price prediction.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    A class to handle model evaluation tasks.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics for regression.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape
        }
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            model_name (str): Name of the model
        """
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        print(f"\n📊 {model_name} Performance:")
        print("-" * 40)
        print(f"RMSE: {metrics['rmse']:,.2f}")
        print(f"MAE:  {metrics['mae']:,.2f}")
        print(f"R²:   {metrics['r2_score']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            model_name (str): Name of the model
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{model_name}: Predicted vs Actual')
        
        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """
        Plot distribution of residuals.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            model_name (str): Name of the model
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 4))
        
        # Histogram of residuals
        plt.subplot(1, 3, 1)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Residual Distribution')
        
        # Q-Q plot
        plt.subplot(1, 3, 2)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot')
        
        # Box plot of residuals
        plt.subplot(1, 3, 3)
        plt.boxplot(residuals, vert=True)
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residual Box Plot')
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        
        Args:
            model: Sklearn model object
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        results = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std()
        }
        
        return results
    
    def plot_learning_curves(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> None:
        """
        Plot learning curves to assess model performance vs training size.
        
        Args:
            model: Sklearn model object
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            model_name (str): Name of the model
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and std for training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.1, color='red')
        
        plt.xlabel('Training Size')
        plt.ylabel('R² Score')
        plt.title(f'{model_name}: Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_models(self, results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models and create a comparison table.
        
        Args:
            results_dict (Dict[str, Dict[str, float]]): Dictionary containing model results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_df = pd.DataFrame.from_dict(results_dict, orient='index')
        comparison_df = comparison_df.round(4)
        
        # Sort by R² score (descending)
        comparison_df = comparison_df.sort_values('r2_score', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict[str, float]]) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            results_dict (Dict[str, Dict[str, float]]): Dictionary containing model results
        """
        comparison_df = self.compare_models(results_dict)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # R² Score comparison
        axes[0].barh(comparison_df.index, comparison_df['r2_score'], color='skyblue')
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('Model Comparison: R² Score')
        axes[0].set_xlim(0, 1)
        
        # RMSE comparison
        axes[1].barh(comparison_df.index, comparison_df['rmse'], color='lightcoral')
        axes[1].set_xlabel('RMSE')
        axes[1].set_title('Model Comparison: RMSE')
        
        # MAE comparison
        axes[2].barh(comparison_df.index, comparison_df['mae'], color='lightgreen')
        axes[2].set_xlabel('MAE')
        axes[2].set_title('Model Comparison: MAE')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_plot(self, model, feature_names: List[str], model_name: str, top_k: int = 15) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (List[str]): List of feature names
            model_name (str): Name of the model
            top_k (int): Number of top features to show
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_k]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'{model_name}: Top {top_k} Feature Importances')
            plt.barh(range(top_k), importances[indices], align='center', color='skyblue')
            plt.yticks(range(top_k), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            # Print top features
            print(f"\nTop {top_k} Important Features for {model_name}:")
            for i, idx in enumerate(indices, 1):
                print(f"{i:2d}. {feature_names[idx]:<25} : {importances[idx]:.4f}")
    
    def calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            confidence (float): Confidence level
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bounds
        """
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        # Calculate z-score for confidence interval
        alpha = 1 - confidence
        z_score = 1.96  # For 95% confidence interval
        
        lower_bound = y_pred - z_score * residual_std
        upper_bound = y_pred + z_score * residual_std
        
        return lower_bound, upper_bound
    
    def model_diagnosis(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model diagnosis.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Diagnosis results
        """
        residuals = y_true - y_pred
        
        diagnosis = {
            'mean_residual': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': self._calculate_skewness(residuals),
            'residual_kurtosis': self._calculate_kurtosis(residuals),
            'homoscedasticity': self._test_homoscedasticity(y_pred, residuals),
            'outlier_count': self._count_outliers(residuals),
            'outlier_percentage': (self._count_outliers(residuals) / len(residuals)) * 100
        }
        
        print(f"\n🔍 Model Diagnosis for {model_name}:")
        print("-" * 40)
        print(f"Mean Residual: {diagnosis['mean_residual']:.4f}")
        print(f"Residual Std: {diagnosis['residual_std']:.2f}")
        print(f"Residual Skewness: {diagnosis['residual_skewness']:.4f}")
        print(f"Outlier Count: {diagnosis['outlier_count']} ({diagnosis['outlier_percentage']:.1f}%)")
        
        return diagnosis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _test_homoscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> bool:
        """Test for homoscedasticity (constant variance of residuals)."""
        # Simple test: correlation between predicted values and absolute residuals
        correlation = np.corrcoef(y_pred, np.abs(residuals))[0, 1]
        return abs(correlation) < 0.3  # Threshold for homoscedasticity
    
    def _count_outliers(self, residuals: np.ndarray, threshold: float = 2.5) -> int:
        """Count outliers using z-score method."""
        z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
        return np.sum(z_scores > threshold)