"""
Simple test script to run the house price prediction model training.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

def main():
    print("🏠 House Price Prediction - Quick Training")
    print("=" * 50)
    
    # Load data directly
    try:
        df = pd.read_csv('data/house_data.csv')
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Quick preprocessing
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"✅ Encoded {col}")
    
    # Feature engineering - basic features
    df_processed['area_per_bedroom'] = df_processed['area'] / (df_processed['bedrooms'] + 1)
    df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    df_processed['luxury_score'] = (df_processed['guestroom'] + df_processed['basement'] + 
                                   df_processed['hotwaterheating'] + df_processed['airconditioning'])
    
    print("✅ Feature engineering completed")
    
    # Prepare features and target
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"✅ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
    print("\n🔄 Training Models...")
    print("-" * 50)
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
            
            print(f"{name:<20}: R²={r2:.4f}, RMSE={rmse:,.0f}, MAE={mae:,.0f}")
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
    
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   R² Score: {best_score:.4f}")
    
    # Save best model
    try:
        os.makedirs('models', exist_ok=True)
        
        model_package = {
            'model': best_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': X.columns.tolist(),
            'model_name': best_name,
            'results': results
        }
        
        joblib.dump(model_package, 'models/best_model.joblib')
        print(f"💾 Model saved to models/best_model.joblib")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    print(f"\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()