"""
Utility functions for the house price prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def data_info(df: pd.DataFrame) -> None:
    """
    Display comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n" + "=" * 30)
    print("COLUMN INFORMATION")
    print("=" * 30)
    print(df.info())
    
    print("\n" + "=" * 30)
    print("MISSING VALUES")
    print("=" * 30)
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found!")
    
    print("\n" + "=" * 30)
    print("STATISTICAL SUMMARY")
    print("=" * 30)
    print(df.describe())
    
    print("\n" + "=" * 30)
    print("DATA TYPES")
    print("=" * 30)
    print(df.dtypes)

def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): Dataset
        figsize (Tuple[int, int]): Figure size
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, target_col: str = 'price') -> None:
    """
    Plot distribution of numerical features.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_target_vs_features(df: pd.DataFrame, target_col: str = 'price') -> None:
    """
    Plot target variable against key features.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols
    
    n_cols = 2
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            axes[i].scatter(df[col], df[target_col], alpha=0.6, color='coral')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)
            axes[i].set_title(f'{target_col} vs {col}')
    
    # Hide unused subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Column name to check for outliers
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def print_model_results(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print model evaluation results in a formatted table.
    
    Args:
        results_dict (Dict): Dictionary containing model results
    """
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R² Score':<15}")
    print("-" * 70)
    
    for model_name, metrics in results_dict.items():
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2_score', 0)
        print(f"{model_name:<20} {rmse:<15.2f} {mae:<15.2f} {r2:<15.4f}")
    
    print("-" * 70)
    
    # Find best model based on R² score
    best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['r2_score'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   R² Score: {results_dict[best_model]['r2_score']:.4f}")
    print(f"   RMSE: {results_dict[best_model]['rmse']:.2f}")
    print(f"   MAE: {results_dict[best_model]['mae']:.2f}")

def save_model_results(results_dict: Dict[str, Dict[str, float]], filepath: str) -> None:
    """
    Save model results to a CSV file.
    
    Args:
        results_dict (Dict): Dictionary containing model results
        filepath (str): Path to save the results
    """
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df.to_csv(filepath, index_label='Model')
    print(f"Results saved to: {filepath}")

def format_price(price: float) -> str:
    """
    Format price with proper currency formatting.
    
    Args:
        price (float): Price value
        
    Returns:
        str: Formatted price string
    """
    return f"${price:,.2f}"