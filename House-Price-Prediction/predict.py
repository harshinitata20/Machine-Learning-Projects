"""
House Price Prediction Script

This script loads the trained model and provides an interface for predicting house prices.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """
    A class to handle house price predictions using the trained model.
    """
    
    def __init__(self, model_path: str = 'models/best_model.joblib'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_name = None
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model and preprocessing components.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.model_package = joblib.load(self.model_path)
            self.model = self.model_package['model']
            self.preprocessor = self.model_package['preprocessor']
            self.feature_engineer = self.model_package['feature_engineer']
            self.model_name = self.model_package['model_name']
            
            print(f"✅ Model ({self.model_name}) loaded successfully!")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            print("Please train the model first using 'python src/model_training.py'")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_price(self, features: Dict[str, Any]) -> float:
        """
        Predict house price for given features.
        
        Args:
            features (Dict[str, Any]): Dictionary containing house features
            
        Returns:
            float: Predicted house price
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please check the model file.")
        
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Apply feature engineering
            features_engineered = self.feature_engineer.engineer_features(features_df)
            
            # Preprocess features
            X_processed, _ = self.preprocessor.preprocess_data(features_engineered, fit=False)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            return float(prediction)
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def predict_multiple(self, features_list: list) -> list:
        """
        Predict house prices for multiple houses.
        
        Args:
            features_list (list): List of feature dictionaries
            
        Returns:
            list: List of predicted prices
        """
        predictions = []
        for features in features_list:
            pred = self.predict_price(features)
            predictions.append(pred)
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if self.model_package is None:
            return {}
        
        info = {
            'model_name': self.model_name,
            'model_type': str(type(self.model).__name__),
            'results': self.model_package.get('results', {}),
        }
        
        if self.model_name in info['results']:
            metrics = info['results'][self.model_name]
            info['performance'] = {
                'R² Score': f"{metrics['r2_score']:.4f}",
                'RMSE': f"{metrics['rmse']:,.2f}",
                'MAE': f"{metrics['mae']:,.2f}",
                'MAPE': f"{metrics['mape']:.2f}%"
            }
        
        return info
    
    def format_price(self, price: float) -> str:
        """
        Format price with currency formatting.
        
        Args:
            price (float): Price value
            
        Returns:
            str: Formatted price string
        """
        return f"${price:,.2f}"

def predict_house_price(features: Dict[str, Any], model_path: str = 'models/best_model.joblib') -> float:
    """
    Convenience function to predict house price.
    
    Args:
        features (Dict[str, Any]): Dictionary containing house features
        model_path (str): Path to the saved model
        
    Returns:
        float: Predicted house price
    """
    predictor = HousePricePredictor(model_path)
    return predictor.predict_price(features)

def get_sample_features() -> Dict[str, Any]:
    """
    Get sample features for testing.
    
    Returns:
        Dict[str, Any]: Sample feature dictionary
    """
    return {
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': 1,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    }

def interactive_prediction():
    """
    Interactive function to get user input and predict house price.
    """
    print("🏠 House Price Prediction Tool")
    print("=" * 40)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    if predictor.model is None:
        return
    
    # Display model information
    model_info = predictor.get_model_info()
    print(f"\nUsing model: {model_info['model_name']}")
    if 'performance' in model_info:
        print("Model Performance:")
        for metric, value in model_info['performance'].items():
            print(f"  {metric}: {value}")
    
    print("\nPlease enter the house features:")
    
    try:
        # Get user input
        area = float(input("Area (sq ft): "))
        bedrooms = int(input("Number of bedrooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        stories = int(input("Number of stories: "))
        mainroad = input("Main road access (yes/no): ").lower()
        guestroom = input("Guest room (yes/no): ").lower()
        basement = input("Basement (yes/no): ").lower()
        hotwaterheating = input("Hot water heating (yes/no): ").lower()
        airconditioning = input("Air conditioning (yes/no): ").lower()
        parking = int(input("Number of parking spaces: "))
        prefarea = input("Preferred area (yes/no): ").lower()
        furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ").lower()
        
        # Create features dictionary
        features = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'parking': parking,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus
        }
        
        # Make prediction
        predicted_price = predictor.predict_price(features)
        
        if predicted_price is not None:
            print(f"\n🎯 Predicted House Price: {predictor.format_price(predicted_price)}")
        else:
            print("❌ Could not make prediction. Please check your inputs.")
            
    except ValueError as e:
        print(f"❌ Invalid input: {str(e)}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def demo_prediction():
    """
    Demonstrate prediction with sample data.
    """
    print("🏠 House Price Prediction Demo")
    print("=" * 40)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    if predictor.model is None:
        return
    
    # Get sample features
    sample_features = get_sample_features()
    
    print("\nSample house features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    predicted_price = predictor.predict_price(sample_features)
    
    if predicted_price is not None:
        print(f"\n🎯 Predicted House Price: {predictor.format_price(predicted_price)}")
        
        # Show model information
        model_info = predictor.get_model_info()
        print(f"\nModel used: {model_info['model_name']}")
        if 'performance' in model_info:
            print(f"Model R² Score: {model_info['performance']['R² Score']}")
    else:
        print("❌ Could not make prediction.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prediction()
    else:
        demo_prediction()
        
        print("\n" + "=" * 50)
        print("To run interactive prediction, use:")
        print("python predict.py --interactive")
        print("=" * 50)