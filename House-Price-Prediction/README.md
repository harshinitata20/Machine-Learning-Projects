# House Price Prediction

A machine learning project to predict house prices based on various features such as area, number of rooms, location, and other property characteristics.

## 🎯 Project Objectives

- Build accurate regression models to predict house prices
- Perform comprehensive exploratory data analysis (EDA)
- Compare multiple machine learning algorithms
- Create a production-ready model with proper preprocessing pipeline
- Provide tools for making predictions on new data

## 📊 Dataset Information

The project uses house price data with features including:
- **Area**: Square footage of the house
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Stories**: Number of stories
- **Parking**: Number of parking spaces
- **Location features**: Mainroad access, guestroom, basement, hot water heating, air conditioning
- **Furnishing status**: Furnished, semi-furnished, or unfurnished

## 🏗️ Project Structure

```
Machine-Learning-Projects/
├── data/
│   └── house_data.csv          # Dataset
├── notebooks/
│   └── House_Prediction.ipynb  # Jupyter notebook for exploration and experimentation
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature engineering utilities
│   ├── model_training.py       # Model training and evaluation
│   ├── evaluation.py          # Model evaluation metrics
│   └── utils.py               # Utility functions
├── models/
│   └── best_model.joblib      # Saved trained model
├── predict.py                 # Script for making predictions
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model

1. **Run the Jupyter notebook** for exploratory analysis:
```bash
jupyter notebook notebooks/House_Prediction.ipynb
```

2. **Train models using the script**:
```bash
python src/model_training.py
```

#### Making Predictions

Use the prediction script to predict house prices:
```bash
python predict.py
```

Or import in your code:
```python
from predict import predict_house_price

# Example usage
features = {
    'area': 7420,
    'bedrooms': 4,
    'bathrooms': 1,
    'stories': 3,
    'parking': 2,
    'mainroad': 'yes',
    'guestroom': 'no',
    'basement': 'no',
    'hotwaterheating': 'no',
    'airconditioning': 'yes',
    'furnishingstatus': 'furnished'
}

predicted_price = predict_house_price(features)
print(f"Predicted price: ${predicted_price:,.2f}")
```

## 🤖 Models Implemented

- **Linear Regression**: Baseline model
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Random Forest**: Ensemble method
- **Gradient Boosting**: Boosting ensemble
- **XGBoost**: Advanced gradient boosting

## 📈 Model Performance

The models are evaluated using:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **R² Score**: Coefficient of determination

## 📋 Workflow

1. **Data Loading & Exploration**: Understanding the dataset structure and characteristics
2. **Data Preprocessing**: Handling missing values, outliers, and categorical variables
3. **Feature Engineering**: Creating new features and scaling
4. **Model Training**: Training multiple regression algorithms
5. **Model Evaluation**: Comparing performance metrics
6. **Model Selection**: Choosing the best performing model
7. **Model Saving**: Persisting the trained model for future use

## 🔧 Features

- **Robust Preprocessing Pipeline**: Handles missing values, outliers, and categorical encoding
- **Comprehensive EDA**: Correlation analysis, distribution plots, and feature relationships
- **Multiple Model Comparison**: Systematic evaluation of different algorithms
- **Production Ready**: Clean, modular code structure
- **Easy Prediction Interface**: Simple script for making new predictions

## 📊 Key Insights

- Feature importance analysis reveals the most significant predictors
- Correlation heatmap shows relationships between variables
- Model comparison helps identify the best algorithm for the dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Your Name - your.email@example.com

Project Link: https://github.com/your-username/Machine-Learning-Projects