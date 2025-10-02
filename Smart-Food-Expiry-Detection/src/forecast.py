"""
Freshness Prediction and Forecasting Models

This module implements various time-series models to predict food freshness
and estimate remaining shelf life based on historical data and environmental factors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available. Install with: pip install statsmodels")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")


class FreshnessPredictor:
    """Main class for predicting food freshness using various models."""
    
    def __init__(self):
        """Initialize the freshness predictor."""
        self.models = {}
        self.model_performance = {}
        
    def create_synthetic_data(self, 
                            food_item: str,
                            days: int = 30,
                            base_freshness: float = 100.0) -> pd.DataFrame:
        """
        Create synthetic freshness data for demonstration.
        
        Args:
            food_item: Name of the food item
            days: Number of days to generate data for
            base_freshness: Starting freshness score (0-100)
            
        Returns:
            DataFrame with date and freshness columns
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Simulate freshness decay with some randomness
        decay_rates = {
            'apple': 0.02,
            'banana': 0.05,
            'milk': 0.08,
            'bread': 0.06,
            'chicken': 0.12,
            'broccoli': 0.10,
            'default': 0.05
        }
        
        decay_rate = decay_rates.get(food_item.lower(), decay_rates['default'])
        
        freshness_scores = []
        current_freshness = base_freshness
        
        for i, date in enumerate(dates):
            # Add daily decay
            daily_decay = decay_rate * current_freshness
            
            # Add some randomness
            noise = np.random.normal(0, 2)
            
            # Add seasonal effects (weekends might have different storage conditions)
            if date.weekday() >= 5:  # Weekend
                seasonal_factor = 0.02
            else:
                seasonal_factor = 0.0
            
            current_freshness = max(0, current_freshness - daily_decay + noise - seasonal_factor)
            freshness_scores.append(current_freshness)
        
        return pd.DataFrame({
            'ds': dates,  # Prophet expects 'ds' column
            'y': freshness_scores,  # Prophet expects 'y' column
            'date': dates,
            'freshness': freshness_scores,
            'food_item': food_item
        })
    
    def predict_with_prophet(self, 
                           data: pd.DataFrame,
                           forecast_days: int = 7) -> Dict:
        """
        Predict freshness using Facebook Prophet.
        
        Args:
            data: DataFrame with 'ds' (date) and 'y' (freshness) columns
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with predictions and model info
        """
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet not available"}
        
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1
            )
            
            model.fit(data[['ds', 'y']])
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for forecast period
            future_predictions = forecast.tail(forecast_days)
            
            # Calculate when freshness drops below thresholds
            spoilage_date = None
            warning_date = None
            
            for _, row in future_predictions.iterrows():
                if row['yhat'] <= 20 and spoilage_date is None:  # 20% freshness = spoiled
                    spoilage_date = row['ds']
                if row['yhat'] <= 50 and warning_date is None:  # 50% freshness = warning
                    warning_date = row['ds']
            
            return {
                "model_type": "Prophet",
                "forecast_data": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'),
                "future_predictions": future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'),
                "spoilage_date": spoilage_date.strftime('%Y-%m-%d') if spoilage_date else None,
                "warning_date": warning_date.strftime('%Y-%m-%d') if warning_date else None,
                "current_freshness": data['y'].iloc[-1],
                "predicted_freshness_1d": future_predictions.iloc[0]['yhat'] if len(future_predictions) > 0 else None,
                "predicted_freshness_3d": future_predictions.iloc[2]['yhat'] if len(future_predictions) > 2 else None,
                "predicted_freshness_7d": future_predictions.iloc[-1]['yhat'] if len(future_predictions) > 0 else None
            }
            
        except Exception as e:
            return {"error": f"Prophet prediction failed: {str(e)}"}
    
    def predict_with_arima(self, 
                          data: pd.DataFrame,
                          forecast_days: int = 7) -> Dict:
        """
        Predict freshness using ARIMA model.
        
        Args:
            data: DataFrame with freshness time series
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with predictions and model info
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            # Prepare time series data
            ts_data = data['freshness'].values
            
            # Fit ARIMA model (auto-determine parameters)
            model = ARIMA(ts_data, order=(1, 1, 1))  # Simple ARIMA(1,1,1)
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            conf_int = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast dates
            last_date = data['date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Calculate warning and spoilage dates
            spoilage_date = None
            warning_date = None
            
            for i, pred_value in enumerate(forecast):
                if pred_value <= 20 and spoilage_date is None:
                    spoilage_date = forecast_dates[i]
                if pred_value <= 50 and warning_date is None:
                    warning_date = forecast_dates[i]
            
            return {
                "model_type": "ARIMA",
                "forecast_values": forecast.tolist(),
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "confidence_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "spoilage_date": spoilage_date.strftime('%Y-%m-%d') if spoilage_date else None,
                "warning_date": warning_date.strftime('%Y-%m-%d') if warning_date else None,
                "current_freshness": ts_data[-1],
                "predicted_freshness_1d": forecast[0] if len(forecast) > 0 else None,
                "predicted_freshness_3d": forecast[2] if len(forecast) > 2 else None,
                "predicted_freshness_7d": forecast[-1] if len(forecast) > 0 else None,
                "model_summary": str(fitted_model.summary())
            }
            
        except Exception as e:
            return {"error": f"ARIMA prediction failed: {str(e)}"}
    
    def predict_with_linear_decay(self, 
                                 data: pd.DataFrame,
                                 forecast_days: int = 7) -> Dict:
        """
        Simple linear decay prediction as fallback method.
        
        Args:
            data: DataFrame with freshness data
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Calculate average daily decay rate
            freshness_values = data['freshness'].values
            
            if len(freshness_values) < 2:
                daily_decay = 5.0  # Default 5% per day
            else:
                # Calculate decay rate from recent data
                recent_data = freshness_values[-7:]  # Last 7 days
                if len(recent_data) >= 2:
                    decay_per_day = (recent_data[0] - recent_data[-1]) / (len(recent_data) - 1)
                    daily_decay = max(0, decay_per_day)
                else:
                    daily_decay = 5.0
            
            # Make predictions
            current_freshness = freshness_values[-1]
            predictions = []
            forecast_dates = []
            
            last_date = data['date'].iloc[-1]
            
            for i in range(1, forecast_days + 1):
                predicted_freshness = max(0, current_freshness - (daily_decay * i))
                predictions.append(predicted_freshness)
                forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
            
            # Calculate warning and spoilage dates
            spoilage_date = None
            warning_date = None
            
            for i, pred_value in enumerate(predictions):
                if pred_value <= 20 and spoilage_date is None:
                    spoilage_date = forecast_dates[i]
                if pred_value <= 50 and warning_date is None:
                    warning_date = forecast_dates[i]
            
            return {
                "model_type": "Linear Decay",
                "forecast_values": predictions,
                "forecast_dates": forecast_dates,
                "daily_decay_rate": daily_decay,
                "spoilage_date": spoilage_date,
                "warning_date": warning_date,
                "current_freshness": current_freshness,
                "predicted_freshness_1d": predictions[0] if len(predictions) > 0 else None,
                "predicted_freshness_3d": predictions[2] if len(predictions) > 2 else None,
                "predicted_freshness_7d": predictions[-1] if len(predictions) > 0 else None
            }
            
        except Exception as e:
            return {"error": f"Linear decay prediction failed: {str(e)}"}
    
    def predict_freshness(self, 
                         food_item: str,
                         purchase_date: str,
                         storage_conditions: Dict = None,
                         method: str = "auto") -> Dict:
        """
        Main method to predict food freshness.
        
        Args:
            food_item: Name of the food item
            purchase_date: Purchase date in YYYY-MM-DD format
            storage_conditions: Dict with temperature, humidity, etc.
            method: Prediction method ('prophet', 'arima', 'linear', 'auto')
            
        Returns:
            Dictionary with comprehensive freshness predictions
        """
        # Create synthetic historical data (in real app, this would come from sensors)
        days_since_purchase = (datetime.now() - datetime.strptime(purchase_date, '%Y-%m-%d')).days
        
        if days_since_purchase <= 0:
            days_since_purchase = 1
        
        # Generate historical data
        historical_data = self.create_synthetic_data(
            food_item, 
            days=min(days_since_purchase + 5, 30)
        )
        
        # Try different prediction methods
        predictions = {}
        
        if method == "auto" or method == "prophet":
            if PROPHET_AVAILABLE:
                predictions["prophet"] = self.predict_with_prophet(historical_data)
            else:
                predictions["prophet"] = {"error": "Prophet not available"}
        
        if method == "auto" or method == "arima":
            if STATSMODELS_AVAILABLE:
                predictions["arima"] = self.predict_with_arima(historical_data)
            else:
                predictions["arima"] = {"error": "Statsmodels not available"}
        
        # Always include linear decay as fallback
        predictions["linear"] = self.predict_with_linear_decay(historical_data)
        
        # Select best prediction (prefer Prophet, then ARIMA, then linear)
        best_prediction = None
        if "prophet" in predictions and "error" not in predictions["prophet"]:
            best_prediction = predictions["prophet"]
        elif "arima" in predictions and "error" not in predictions["arima"]:
            best_prediction = predictions["arima"]
        else:
            best_prediction = predictions["linear"]
        
        # Add storage condition adjustments
        if storage_conditions:
            best_prediction = self._adjust_for_storage_conditions(best_prediction, storage_conditions)
        
        return {
            "food_item": food_item,
            "purchase_date": purchase_date,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "best_prediction": best_prediction,
            "all_predictions": predictions,
            "storage_conditions": storage_conditions or {},
            "recommendation": self._generate_recommendation(best_prediction)
        }
    
    def _adjust_for_storage_conditions(self, 
                                     prediction: Dict,
                                     conditions: Dict) -> Dict:
        """
        Adjust predictions based on storage conditions.
        
        Args:
            prediction: Base prediction dictionary
            conditions: Storage conditions (temperature, humidity, etc.)
            
        Returns:
            Adjusted prediction dictionary
        """
        adjustment_factor = 1.0
        
        # Temperature adjustments
        temp = conditions.get('temperature', 4)  # Default fridge temp
        if temp > 25:  # Room temperature or higher
            adjustment_factor *= 1.5  # Faster decay
        elif temp > 10:
            adjustment_factor *= 1.2
        elif temp < 0:  # Freezer
            adjustment_factor *= 0.3  # Much slower decay
        
        # Humidity adjustments
        humidity = conditions.get('humidity', 60)  # Default humidity
        if humidity > 80:
            adjustment_factor *= 1.2  # Higher decay in high humidity
        elif humidity < 30:
            adjustment_factor *= 1.1  # Faster decay in low humidity
        
        # Apply adjustments to predictions
        if 'forecast_values' in prediction:
            prediction['forecast_values'] = [
                max(0, val / adjustment_factor) for val in prediction['forecast_values']
            ]
        
        # Adjust specific predictions
        for key in ['predicted_freshness_1d', 'predicted_freshness_3d', 'predicted_freshness_7d']:
            if key in prediction and prediction[key]:
                prediction[key] = max(0, prediction[key] / adjustment_factor)
        
        prediction['storage_adjustment_factor'] = adjustment_factor
        
        return prediction
    
    def _generate_recommendation(self, prediction: Dict) -> str:
        """
        Generate human-readable recommendation based on prediction.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Recommendation string
        """
        if 'error' in prediction:
            return "Unable to generate prediction. Please check the food item manually."
        
        current_freshness = prediction.get('current_freshness', 100)
        freshness_1d = prediction.get('predicted_freshness_1d', current_freshness)
        warning_date = prediction.get('warning_date')
        spoilage_date = prediction.get('spoilage_date')
        
        if current_freshness <= 20:
            return "⛔ This item has likely spoiled. Please discard safely."
        elif current_freshness <= 50:
            return "⚠️ This item is deteriorating. Use immediately or discard."
        elif warning_date:
            try:
                days_to_warning = (datetime.strptime(warning_date, '%Y-%m-%d') - datetime.now()).days
                if days_to_warning <= 1:
                    return "🟡 Use within the next day for best quality."
                elif days_to_warning <= 3:
                    return f"🟡 Best to use within {days_to_warning} days."
                else:
                    return f"🟢 Item is fresh. Best to use within {days_to_warning} days."
            except:
                return "🟢 Item appears to be fresh."
        else:
            return "🟢 Item is fresh and safe to consume."


# Simple LSTM model class (if PyTorch is available)
class LSTMFreshnessModel(nn.Module):
    """LSTM model for freshness prediction."""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMFreshnessModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def demo_freshness_prediction():
    """Demonstration of freshness prediction functionality."""
    print("🔮 Freshness Prediction Demo")
    print("=" * 50)
    
    predictor = FreshnessPredictor()
    
    # Demo with different food items
    test_foods = [
        ("apple", "2025-09-25"),
        ("milk", "2025-09-30"), 
        ("bread", "2025-09-28"),
        ("chicken", "2025-10-01")
    ]
    
    for food_item, purchase_date in test_foods:
        print(f"\n🍎 Analyzing {food_item} (purchased: {purchase_date})")
        
        # Predict freshness
        result = predictor.predict_freshness(food_item, purchase_date)
        
        best_pred = result['best_prediction']
        
        print(f"   Current freshness: {best_pred.get('current_freshness', 'N/A'):.1f}%")
        print(f"   Model used: {best_pred.get('model_type', 'Unknown')}")
        
        if best_pred.get('warning_date'):
            print(f"   ⚠️ Warning date: {best_pred['warning_date']}")
        
        if best_pred.get('spoilage_date'):
            print(f"   ⛔ Spoilage date: {best_pred['spoilage_date']}")
        
        print(f"   📝 {result['recommendation']}")
    
    print("\n✅ Freshness prediction system ready!")
    return predictor


if __name__ == "__main__":
    # Run demo
    predictor = demo_freshness_prediction()