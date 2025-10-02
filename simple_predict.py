"""
Simple prediction script for house prices.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def predict_price(house_features, model_path='models/best_model.joblib'):
    """
    Predict house price using the trained model.
    """
    try:
        # Load model package
        model_package = joblib.load(model_path)
        model = model_package['model']
        scaler = model_package['scaler']
        label_encoders = model_package['label_encoders']
        
        # Create DataFrame from features
        house_df = pd.DataFrame([house_features])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in house_df.columns:
                if house_df[col].iloc[0] in encoder.classes_:
                    house_df[col] = encoder.transform([house_df[col].iloc[0]])[0]
                else:
                    house_df[col] = 0  # Default for unseen categories
        
        # Add engineered features
        house_df['area_per_bedroom'] = house_df['area'] / (house_df['bedrooms'] + 1)
        house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms']
        house_df['luxury_score'] = (house_df['guestroom'] + house_df['basement'] + 
                                   house_df['hotwaterheating'] + house_df['airconditioning'])
        
        # Ensure column order matches training data
        expected_features = model_package['feature_names']
        house_df = house_df.reindex(columns=expected_features, fill_value=0)
        
        # Scale features
        house_scaled = scaler.transform(house_df)
        
        # Make prediction
        prediction = model.predict(house_scaled)[0]
        return prediction
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    print("🏠 House Price Prediction Tool")
    print("=" * 40)
    
    # Sample house features
    sample_house = {
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
    
    print("\nSample house features:")
    for key, value in sample_house.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    predicted_price = predict_price(sample_house)
    
    if predicted_price is not None:
        print(f"\n🎯 Predicted Price: ${predicted_price:,.2f}")
        
        # Test with different house types
        print(f"\n🏘️ Testing different house types:")
        
        test_houses = [
            {'area': 5000, 'bedrooms': 2, 'bathrooms': 1, 'stories': 1, 'parking': 1,
             'mainroad': 'yes', 'guestroom': 'no', 'basement': 'no', 'hotwaterheating': 'no',
             'airconditioning': 'no', 'prefarea': 'no', 'furnishingstatus': 'unfurnished'},
            
            {'area': 10000, 'bedrooms': 5, 'bathrooms': 3, 'stories': 2, 'parking': 3,
             'mainroad': 'yes', 'guestroom': 'yes', 'basement': 'yes', 'hotwaterheating': 'yes',
             'airconditioning': 'yes', 'prefarea': 'yes', 'furnishingstatus': 'furnished'}
        ]
        
        house_types = ['Basic House', 'Luxury House']
        
        for house, house_type in zip(test_houses, house_types):
            price = predict_price(house)
            if price:
                print(f"  {house_type}: ${price:,.2f}")
    else:
        print("❌ Could not make prediction")

if __name__ == "__main__":
    main()