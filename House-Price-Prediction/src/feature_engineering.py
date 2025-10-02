"""
Feature engineering module for house price prediction.
Creates new features and performs advanced feature transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    A class to handle feature engineering tasks.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.created_features = []
        
    def create_area_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create area-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new area features
        """
        df_copy = df.copy()
        
        if 'area' in df_copy.columns:
            # Area per bedroom
            if 'bedrooms' in df_copy.columns:
                df_copy['area_per_bedroom'] = df_copy['area'] / (df_copy['bedrooms'] + 1)  # +1 to avoid division by zero
                self.created_features.append('area_per_bedroom')
            
            # Area per bathroom
            if 'bathrooms' in df_copy.columns:
                df_copy['area_per_bathroom'] = df_copy['area'] / (df_copy['bathrooms'] + 1)
                self.created_features.append('area_per_bathroom')
            
            # Total rooms
            if 'bedrooms' in df_copy.columns and 'bathrooms' in df_copy.columns:
                df_copy['total_rooms'] = df_copy['bedrooms'] + df_copy['bathrooms']
                df_copy['area_per_room'] = df_copy['area'] / (df_copy['total_rooms'] + 1)
                self.created_features.extend(['total_rooms', 'area_per_room'])
            
            # Area categories (using numeric encoding directly)
            df_copy['area_small'] = (df_copy['area'] <= 5000).astype(int)
            df_copy['area_medium'] = ((df_copy['area'] > 5000) & (df_copy['area'] <= 8000)).astype(int)
            df_copy['area_large'] = ((df_copy['area'] > 8000) & (df_copy['area'] <= 12000)).astype(int)
            df_copy['area_extra_large'] = (df_copy['area'] > 12000).astype(int)
            self.created_features.extend(['area_small', 'area_medium', 'area_large', 'area_extra_large'])
        
        return df_copy
    
    def create_room_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create room-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new room features
        """
        df_copy = df.copy()
        
        # Bedroom to bathroom ratio
        if 'bedrooms' in df_copy.columns and 'bathrooms' in df_copy.columns:
            df_copy['bedroom_bathroom_ratio'] = df_copy['bedrooms'] / (df_copy['bathrooms'] + 0.1)  # +0.1 to avoid division by zero
            self.created_features.append('bedroom_bathroom_ratio')
        
        # Room categories (using numeric encoding directly)
        if 'bedrooms' in df_copy.columns:
            df_copy['bedroom_small'] = (df_copy['bedrooms'] <= 2).astype(int)
            df_copy['bedroom_medium'] = (df_copy['bedrooms'] == 3).astype(int)
            df_copy['bedroom_large'] = (df_copy['bedrooms'] == 4).astype(int)
            df_copy['bedroom_extra_large'] = (df_copy['bedrooms'] > 4).astype(int)
            self.created_features.extend(['bedroom_small', 'bedroom_medium', 'bedroom_large', 'bedroom_extra_large'])
        
        # Stories features
        if 'stories' in df_copy.columns:
            df_copy['is_single_story'] = (df_copy['stories'] == 1).astype(int)
            df_copy['is_multi_story'] = (df_copy['stories'] > 1).astype(int)
            self.created_features.extend(['is_single_story', 'is_multi_story'])
        
        return df_copy
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location and accessibility features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new location features
        """
        df_copy = df.copy()
        
        # Accessibility score (combination of mainroad and prefarea)
        accessibility_score = 0
        if 'mainroad' in df_copy.columns:
            accessibility_score += (df_copy['mainroad'] == 'yes').astype(int)
        if 'prefarea' in df_copy.columns:
            accessibility_score += (df_copy['prefarea'] == 'yes').astype(int)
        
        df_copy['accessibility_score'] = accessibility_score
        self.created_features.append('accessibility_score')
        
        return df_copy
    
    def create_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amenity-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new amenity features
        """
        df_copy = df.copy()
        
        # Luxury score (combination of multiple amenities)
        luxury_features = ['guestroom', 'basement', 'hotwaterheating', 'airconditioning']
        luxury_score = 0
        
        for feature in luxury_features:
            if feature in df_copy.columns:
                luxury_score += (df_copy[feature] == 'yes').astype(int)
        
        df_copy['luxury_score'] = luxury_score
        self.created_features.append('luxury_score')
        
        # Climate control (heating or cooling)
        if 'hotwaterheating' in df_copy.columns and 'airconditioning' in df_copy.columns:
            df_copy['has_climate_control'] = (
                (df_copy['hotwaterheating'] == 'yes') | 
                (df_copy['airconditioning'] == 'yes')
            ).astype(int)
            self.created_features.append('has_climate_control')
        
        # Storage score (basement + parking)
        storage_score = 0
        if 'basement' in df_copy.columns:
            storage_score += (df_copy['basement'] == 'yes').astype(int)
        if 'parking' in df_copy.columns:
            storage_score += np.minimum(df_copy['parking'], 2)  # Cap at 2 for scoring
        
        df_copy['storage_score'] = storage_score
        self.created_features.append('storage_score')
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_copy = df.copy()
        
        # Area and luxury interaction
        if 'area' in df_copy.columns and 'luxury_score' in df_copy.columns:
            df_copy['area_luxury_interaction'] = df_copy['area'] * df_copy['luxury_score']
            self.created_features.append('area_luxury_interaction')
        
        # Rooms and area interaction
        if 'total_rooms' in df_copy.columns and 'area' in df_copy.columns:
            df_copy['rooms_area_interaction'] = df_copy['total_rooms'] * np.log1p(df_copy['area'])
            self.created_features.append('rooms_area_interaction')
        
        # Stories and area interaction
        if 'stories' in df_copy.columns and 'area' in df_copy.columns:
            df_copy['stories_area_interaction'] = df_copy['stories'] * df_copy['area']
            self.created_features.append('stories_area_interaction')
        
        return df_copy
    
    def create_furnished_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on furnishing status.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with furnishing features
        """
        df_copy = df.copy()
        
        if 'furnishingstatus' in df_copy.columns:
            # Binary indicators for each furnishing type
            df_copy['is_furnished'] = (df_copy['furnishingstatus'] == 'furnished').astype(int)
            df_copy['is_semi_furnished'] = (df_copy['furnishingstatus'] == 'semi-furnished').astype(int)
            df_copy['is_unfurnished'] = (df_copy['furnishingstatus'] == 'unfurnished').astype(int)
            
            self.created_features.extend(['is_furnished', 'is_semi_furnished', 'is_unfurnished'])
        
        return df_copy
    
    def create_polynomial_features(self, df: pd.DataFrame, features: List[str] = None, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (List[str]): List of features to create polynomial features for
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Dataframe with polynomial features
        """
        df_copy = df.copy()
        
        if features is None:
            # Default to area if no features specified
            features = ['area'] if 'area' in df_copy.columns else []
        
        for feature in features:
            if feature in df_copy.columns:
                for d in range(2, degree + 1):
                    new_feature_name = f"{feature}_power_{d}"
                    df_copy[new_feature_name] = df_copy[feature] ** d
                    self.created_features.append(new_feature_name)
        
        return df_copy
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering techniques.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        print("Starting feature engineering...")
        print(f"Original features: {df.shape[1]}")
        
        # Reset created features list
        self.created_features = []
        
        # Apply all feature engineering methods
        df_engineered = df.copy()
        
        # Create area-related features
        df_engineered = self.create_area_features(df_engineered)
        
        # Create room-related features
        df_engineered = self.create_room_features(df_engineered)
        
        # Create location features
        df_engineered = self.create_location_features(df_engineered)
        
        # Create amenity features
        df_engineered = self.create_amenity_features(df_engineered)
        
        # Create furnishing features
        df_engineered = self.create_furnished_features(df_engineered)
        
        # Create interaction features
        df_engineered = self.create_interaction_features(df_engineered)
        
        # Create polynomial features for area
        df_engineered = self.create_polynomial_features(df_engineered, ['area'], degree=2)
        
        print(f"Features after engineering: {df_engineered.shape[1]}")
        print(f"New features created: {len(self.created_features)}")
        if self.created_features:
            print(f"Created features: {self.created_features}")
        
        return df_engineered
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get a mapping of features to their descriptions.
        
        Returns:
            Dict[str, str]: Mapping of feature names to descriptions
        """
        feature_descriptions = {
            'area_per_bedroom': 'Area divided by number of bedrooms',
            'area_per_bathroom': 'Area divided by number of bathrooms',
            'total_rooms': 'Total bedrooms and bathrooms',
            'area_per_room': 'Area divided by total rooms',
            'area_category': 'Categorical area size (Small/Medium/Large/Extra_Large)',
            'bedroom_bathroom_ratio': 'Ratio of bedrooms to bathrooms',
            'bedroom_category': 'Categorical bedroom count',
            'is_single_story': 'Binary indicator for single story houses',
            'is_multi_story': 'Binary indicator for multi-story houses',
            'accessibility_score': 'Combined score for mainroad and preferred area access',
            'luxury_score': 'Combined score for luxury amenities',
            'has_climate_control': 'Binary indicator for heating or cooling',
            'storage_score': 'Combined score for basement and parking',
            'area_luxury_interaction': 'Interaction between area and luxury score',
            'rooms_area_interaction': 'Interaction between total rooms and log area',
            'stories_area_interaction': 'Interaction between stories and area',
            'is_furnished': 'Binary indicator for furnished status',
            'is_semi_furnished': 'Binary indicator for semi-furnished status',
            'is_unfurnished': 'Binary indicator for unfurnished status',
            'area_power_2': 'Square of area'
        }
        
        return feature_descriptions

def select_best_features(df: pd.DataFrame, target: pd.Series, method: str = 'correlation', k: int = 15) -> List[str]:
    """
    Select the best features based on correlation or other methods.
    
    Args:
        df (pd.DataFrame): Features dataframe
        target (pd.Series): Target variable
        method (str): Method for feature selection ('correlation', 'variance')
        k (int): Number of top features to select
        
    Returns:
        List[str]: List of selected feature names
    """
    if method == 'correlation':
        # Calculate correlation with target
        correlations = df.corrwith(target).abs().sort_values(ascending=False)
        selected_features = correlations.head(k).index.tolist()
        
        print(f"Top {k} features by correlation:")
        for i, (feature, corr) in enumerate(correlations.head(k).items(), 1):
            print(f"{i:2d}. {feature:<25} : {corr:.4f}")
            
    elif method == 'variance':
        # Select features with highest variance
        variances = df.var().sort_values(ascending=False)
        selected_features = variances.head(k).index.tolist()
        
        print(f"Top {k} features by variance:")
        for i, (feature, var) in enumerate(variances.head(k).items(), 1):
            print(f"{i:2d}. {feature:<25} : {var:.4f}")
    
    return selected_features