"""
Model training module for house price prediction.
Trains multiple regression models and compares their performance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, data_info, print_model_results
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from evaluation import ModelEvaluator

class ModelTrainer:
    """
    A class to handle model training and comparison.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with default parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.models = models
        return models
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Optimize hyperparameters for a specific model using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to optimize
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            
        Returns:
            Any: Best model after hyperparameter tuning
        """
        print(f"Optimizing hyperparameters for {model_name}...")
        
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [6, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.15]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        }
        
        if model_name in param_grids:
            model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='r2', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            return self.models[model_name]
    
    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Train a single model and evaluate its performance.
        
        Args:
            model: Model object to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(y_test, y_pred, model_name)
        
        # Print metrics
        self.evaluator.print_metrics(y_test, y_pred, model_name)
        
        # Store the trained model
        self.models[model_name] = model
        
        return metrics
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray, 
                        optimize_params: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Train all models and compare their performance.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            optimize_params (bool): Whether to optimize hyperparameters
            
        Returns:
            Dict[str, Dict[str, float]]: Results for all models
        """
        print("=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"\n🔄 Training {model_name}...")
            
            # Get the model
            if optimize_params:
                model = self.optimize_hyperparameters(model_name, X_train, y_train)
            else:
                model = self.models[model_name]
            
            # Train and evaluate
            metrics = self.train_single_model(
                model, X_train, y_train, X_test, y_test, model_name
            )
            
            results[model_name] = metrics
            
            # Perform cross-validation
            cv_results = self.evaluator.cross_validate_model(model, X_train, y_train)
            print(f"Cross-validation R² (mean ± std): {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
        
        self.results = results
        
        # Print comparison
        print_model_results(results)
        
        # Find and store best model
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🎯 Best model selected: {self.best_model_name}")
        
        return results
    
    def save_model(self, model_path: str = 'models/best_model.joblib') -> None:
        """
        Save the best trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.best_model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model with preprocessor and feature engineer
            model_package = {
                'model': self.best_model,
                'preprocessor': self.preprocessor,
                'feature_engineer': self.feature_engineer,
                'model_name': self.best_model_name,
                'results': self.results
            }
            
            joblib.dump(model_package, model_path)
            print(f"✅ Best model ({self.best_model_name}) saved to: {model_path}")
        else:
            print("❌ No trained model to save. Please train models first.")
    
    def load_model(self, model_path: str = 'models/best_model.joblib') -> Dict[str, Any]:
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Dict[str, Any]: Loaded model package
        """
        try:
            model_package = joblib.load(model_path)
            self.best_model = model_package['model']
            self.preprocessor = model_package['preprocessor']
            self.feature_engineer = model_package['feature_engineer']
            self.best_model_name = model_package['model_name']
            self.results = model_package['results']
            
            print(f"✅ Model ({self.best_model_name}) loaded from: {model_path}")
            return model_package
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best trained model.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is not None:
            return self.best_model.predict(X)
        else:
            raise ValueError("No trained model available. Please train models first.")
    
    def get_feature_importance(self, feature_names: List[str], top_k: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        
        Args:
            feature_names (List[str]): List of feature names
            top_k (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.best_model is not None and hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_k)
            
            return feature_importance_df
        else:
            print("Feature importance not available for this model.")
            return pd.DataFrame()

def main():
    """
    Main function to run the complete model training pipeline.
    """
    print("🏠 House Price Prediction - Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    data_path = '../data/house_data.csv'
    df = load_data(data_path)
    
    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return
    
    # Display data information
    data_info(df)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Initialize models
    trainer.initialize_models()
    
    # Preprocess data
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Feature engineering
    df_engineered = trainer.feature_engineer.engineer_features(df)
    
    # Data preprocessing
    X, y = trainer.preprocessor.preprocess_data(df_engineered, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize_params=False)
    
    # Visualize results
    trainer.evaluator.plot_model_comparison(results)
    
    # Feature importance for best model
    feature_names = trainer.preprocessor.get_feature_names()
    if hasattr(trainer.best_model, 'feature_importances_'):
        trainer.evaluator.feature_importance_plot(
            trainer.best_model, feature_names, trainer.best_model_name
        )
    
    # Detailed evaluation of best model
    y_pred_best = trainer.predict(X_test)
    trainer.evaluator.plot_predictions(y_test, y_pred_best, trainer.best_model_name)
    trainer.evaluator.model_diagnosis(y_test, y_pred_best, trainer.best_model_name)
    
    # Save model
    trainer.save_model('../models/best_model.joblib')
    
    print("\n🎉 Model training completed successfully!")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"📊 Best R² score: {trainer.results[trainer.best_model_name]['r2_score']:.4f}")

if __name__ == "__main__":
    main()