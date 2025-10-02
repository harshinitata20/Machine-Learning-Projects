"""
Data preprocessing module for house price prediction.
Handles data cleaning, missing values, outliers, and basic transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks.
    """
    
    def __init__(self):
        """Initialize the preprocessor with encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = []
        self.numerical_columns = []
        self.target_column = 'price'
        
    def identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identify categorical and numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from numerical columns if present
        if self.target_column in self.numerical_columns:
            self.numerical_columns.remove(self.target_column)
            
        print(f"Categorical columns: {self.categorical_columns}")
        print(f"Numerical columns: {self.numerical_columns}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_copy = df.copy()
        
        # Check for missing values
        missing_values = df_copy.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found!")
            return df_copy
        
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        
        # Handle missing values for numerical columns
        for col in self.numerical_columns:
            if df_copy[col].isnull().sum() > 0:
                median_val = df_copy[col].median()
                df_copy[col].fillna(median_val, inplace=True)
                print(f"Filled {col} with median: {median_val}")
        
        # Handle missing values for categorical columns
        for col in self.categorical_columns:
            if df_copy[col].isnull().sum() > 0:
                mode_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'unknown'
                df_copy[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} with mode: {mode_val}")
        
        return df_copy
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers in numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Method to detect outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataframe with handled outliers
        """
        df_copy = df.copy()
        outlier_counts = {}
        
        for col in self.numerical_columns:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
                outlier_counts[col] = outliers.sum()
                
                # Cap outliers instead of removing them
                df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                outliers = z_scores > 3
                outlier_counts[col] = outliers.sum()
                
                # Replace outliers with median
                median_val = df_copy[col].median()
                df_copy.loc[outliers, col] = median_val
        
        print("Outliers handled:")
        for col, count in outlier_counts.items():
            if count > 0:
                print(f"  {col}: {count} outliers")
        
        return df_copy
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoder or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        for col in self.categorical_columns:
            if col in df_copy.columns:
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                else:
                    # Transform using existing encoder
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        mask = df_copy[col].isin(le.classes_)
                        df_copy.loc[mask, col] = le.transform(df_copy.loc[mask, col])
                        # Assign a default value for unseen categories
                        df_copy.loc[~mask, col] = -1
                    else:
                        print(f"Warning: No encoder found for {col}")
        
        return df_copy
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        df_copy = df.copy()
        
        if self.numerical_columns:
            if fit:
                # Fit and transform
                df_copy[self.numerical_columns] = self.scaler.fit_transform(df_copy[self.numerical_columns])
                print("Numerical features scaled using StandardScaler")
            else:
                # Transform using existing scaler
                df_copy[self.numerical_columns] = self.scaler.transform(df_copy[self.numerical_columns])
                print("Numerical features scaled using existing scaler")
        
        return df_copy
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit preprocessors or use existing ones
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        print("Starting data preprocessing...")
        print(f"Original data shape: {df.shape}")
        
        # Identify column types
        if fit:
            self.identify_column_types(df)
        
        # Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Separate features and target
        if self.target_column in df_processed.columns:
            X = df_processed.drop(columns=[self.target_column])
            y = df_processed[self.target_column]
        else:
            X = df_processed
            y = None
        
        # Update column types after dropping target
        if fit:
            self.identify_column_types(X)
        
        # Handle outliers in features only
        X_processed = self.detect_and_handle_outliers(X)
        
        # Encode categorical variables
        X_processed = self.encode_categorical_variables(X_processed, fit=fit)
        
        # Scale numerical features
        X_processed = self.scale_numerical_features(X_processed, fit=fit)
        
        print(f"Processed data shape: {X_processed.shape}")
        print("Data preprocessing completed!")
        
        return X_processed, y
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all processed features.
        
        Returns:
            List[str]: List of feature names
        """
        return self.numerical_columns + self.categorical_columns
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable if it was scaled.
        
        Args:
            y_scaled (np.ndarray): Scaled target values
            
        Returns:
            np.ndarray: Original scale target values
        """
        # If target was not scaled, return as is
        return y_scaled
    
    def preprocess_single_sample(self, sample_dict: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess a single sample for prediction.
        
        Args:
            sample_dict (Dict[str, Any]): Dictionary containing feature values
            
        Returns:
            np.ndarray: Preprocessed sample ready for prediction
        """
        # Create DataFrame from single sample
        sample_df = pd.DataFrame([sample_dict])
        
        # Preprocess using existing fitted preprocessors
        X_processed, _ = self.preprocess_data(sample_df, fit=False)
        
        return X_processed.values

def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of all features in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Feature summary
    """
    summary = pd.DataFrame({
        'Feature': df.columns,
        'Data_Type': df.dtypes,
        'Non_Null_Count': df.count(),
        'Null_Count': df.isnull().sum(),
        'Null_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Unique_Values': df.nunique()
    })
    
    return summary.reset_index(drop=True)