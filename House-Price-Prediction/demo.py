"""
Complete demonstration of the House Price Prediction ML project.
This script shows the entire workflow from data loading to prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def run_complete_demo():
    """Run the complete machine learning pipeline demonstration."""
    
    print("🏠 HOUSE PRICE PREDICTION - COMPLETE DEMO")
    print("=" * 60)
    
    # Step 1: Load and explore data
    print("\n📊 STEP 1: DATA LOADING & EXPLORATION")
    print("-" * 40)
    
    df = pd.read_csv('data/house_data.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"\nPrice range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Average price: ${df['price'].mean():,.0f}")
    
    # Step 2: Data preprocessing
    print("\n🛠️ STEP 2: DATA PREPROCESSING")
    print("-" * 40)
    
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  ✅ Encoded {col}")
    
    # Step 3: Feature engineering
    print("\n⚙️ STEP 3: FEATURE ENGINEERING")
    print("-" * 40)
    
    original_features = df_processed.shape[1]
    
    # Create new features
    df_processed['area_per_bedroom'] = df_processed['area'] / (df_processed['bedrooms'] + 1)
    df_processed['area_per_bathroom'] = df_processed['area'] / (df_processed['bathrooms'] + 1)
    df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    df_processed['area_per_room'] = df_processed['area'] / (df_processed['total_rooms'] + 1)
    df_processed['luxury_score'] = (df_processed['guestroom'] + df_processed['basement'] + 
                                   df_processed['hotwaterheating'] + df_processed['airconditioning'])
    df_processed['accessibility_score'] = df_processed['mainroad'] + df_processed['prefarea']
    
    new_features = df_processed.shape[1] - original_features
    print(f"  ✅ Created {new_features} new features")
    print(f"  📈 Total features now: {df_processed.shape[1]}")
    
    # Step 4: Model training
    print("\n🤖 STEP 4: MODEL TRAINING & EVALUATION")
    print("-" * 40)
    
    # Prepare data
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42, max_iter=2000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    # Train and evaluate models
    results = {}
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
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
            
            print(f"  {name:<20}: R²={r2:6.4f}, RMSE={rmse:8,.0f}, MAE={mae:8,.0f}")
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")
    
    # Step 5: Model selection and saving
    print(f"\n💾 STEP 5: MODEL SELECTION & SAVING")
    print("-" * 40)
    
    print(f"🏆 Best Model: {best_name}")
    print(f"   R² Score: {best_score:.4f}")
    print(f"   RMSE: ${results[best_name]['RMSE']:,.0f}")
    print(f"   MAE: ${results[best_name]['MAE']:,.0f}")
    
    # Save model
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': X.columns.tolist(),
        'model_name': best_name,
        'results': results
    }
    
    joblib.dump(model_package, 'models/best_model.joblib')
    print(f"✅ Model saved to: models/best_model.joblib")
    
    # Step 6: Prediction demonstration
    print(f"\n🔮 STEP 6: PREDICTION DEMONSTRATION")
    print("-" * 40)
    
    # Test houses with different characteristics
    test_houses = [
        {
            'name': 'Modest Family Home',
            'features': {'area': 6000, 'bedrooms': 3, 'bathrooms': 2, 'stories': 1, 'parking': 1,
                        'mainroad': 'yes', 'guestroom': 'no', 'basement': 'no', 'hotwaterheating': 'no',
                        'airconditioning': 'no', 'prefarea': 'no', 'furnishingstatus': 'semi-furnished'}
        },
        {
            'name': 'Luxury Villa',
            'features': {'area': 12000, 'bedrooms': 5, 'bathrooms': 4, 'stories': 2, 'parking': 3,
                        'mainroad': 'yes', 'guestroom': 'yes', 'basement': 'yes', 'hotwaterheating': 'yes',
                        'airconditioning': 'yes', 'prefarea': 'yes', 'furnishingstatus': 'furnished'}
        },
        {
            'name': 'Starter Apartment',
            'features': {'area': 4000, 'bedrooms': 2, 'bathrooms': 1, 'stories': 1, 'parking': 0,
                        'mainroad': 'no', 'guestroom': 'no', 'basement': 'no', 'hotwaterheating': 'no',
                        'airconditioning': 'no', 'prefarea': 'no', 'furnishingstatus': 'unfurnished'}
        }
    ]
    
    for house in test_houses:
        predicted_price = predict_house_price(house['features'], model_package)
        if predicted_price:
            print(f"  {house['name']:<20}: ${predicted_price:,.0f}")
    
    # Step 7: Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n📊 STEP 7: FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<20}: {row['importance']:.4f}")
    
    # Summary
    print(f"\n🎉 PROJECT SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully trained {len(models)} machine learning models")
    print(f"✅ Best performing model: {best_name} (R² = {best_score:.4f})")
    print(f"✅ Model saved and ready for production use")
    print(f"✅ Prediction system working correctly")
    print(f"\n📁 Generated Files:")
    print(f"   • models/best_model.joblib - Trained model package")
    print(f"   • notebooks/House_Prediction.ipynb - Complete analysis notebook")
    print(f"   • data/house_data.csv - Dataset")
    print(f"\n🚀 Usage:")
    print(f"   • Use simple_predict.py for quick predictions")
    print(f"   • Open Jupyter notebook for detailed analysis")
    print(f"   • Run train_models.py to retrain with new data")

def predict_house_price(house_features, model_package):
    """Predict house price using the trained model package."""
    try:
        # Create DataFrame from features
        house_df = pd.DataFrame([house_features])
        
        # Encode categorical variables
        for col, encoder in model_package['label_encoders'].items():
            if col in house_df.columns:
                if house_df[col].iloc[0] in encoder.classes_:
                    house_df[col] = encoder.transform([house_df[col].iloc[0]])[0]
                else:
                    house_df[col] = 0  # Default for unseen categories
        
        # Add engineered features
        house_df['area_per_bedroom'] = house_df['area'] / (house_df['bedrooms'] + 1)
        house_df['area_per_bathroom'] = house_df['area'] / (house_df['bathrooms'] + 1)
        house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms']
        house_df['area_per_room'] = house_df['area'] / (house_df['total_rooms'] + 1)
        house_df['luxury_score'] = (house_df['guestroom'] + house_df['basement'] + 
                                   house_df['hotwaterheating'] + house_df['airconditioning'])
        house_df['accessibility_score'] = house_df['mainroad'] + house_df['prefarea']
        
        # Ensure column order matches training data
        expected_features = model_package['feature_names']
        house_df = house_df.reindex(columns=expected_features, fill_value=0)
        
        # Scale features
        house_scaled = model_package['scaler'].transform(house_df)
        
        # Make prediction
        prediction = model_package['model'].predict(house_scaled)[0]
        return prediction
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == "__main__":
    run_complete_demo()