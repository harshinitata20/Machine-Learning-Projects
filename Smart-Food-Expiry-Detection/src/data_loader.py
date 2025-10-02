"""
Data Loading and Preprocessing Utilities for Food Expiry Detection

This module handles loading of various food datasets, image preprocessing,
and data augmentation for training the food detection models.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from typing import List, Tuple, Dict, Optional
import requests
import zipfile
from pathlib import Path


class FoodDataset(Dataset):
    """Custom dataset class for food images with expiry information."""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[str], 
                 expiry_dates: List[str] = None,
                 transform=None):
        """
        Initialize the food dataset.
        
        Args:
            image_paths: List of paths to food images
            labels: List of food item labels
            expiry_dates: Optional list of expiry dates
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.expiry_dates = expiry_dates
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'label': self.labels[idx],
            'image_path': self.image_paths[idx]
        }
        
        if self.expiry_dates:
            sample['expiry_date'] = self.expiry_dates[idx]
            
        return sample


class DataLoader:
    """Main data loading class for food expiry detection project."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "sample_images").mkdir(exist_ok=True)
    
    def download_sample_dataset(self):
        """Download sample food images for testing."""
        print("Setting up sample food images...")
        
        # Sample food image URLs (using placeholder service)
        sample_foods = [
            "apple", "banana", "bread", "milk", "cheese", 
            "eggs", "chicken", "broccoli", "tomato", "orange"
        ]
        
        # Create sample data CSV
        sample_data = {
            'food_name': sample_foods,
            'category': ['fruit', 'fruit', 'bakery', 'dairy', 'dairy',
                        'protein', 'protein', 'vegetable', 'vegetable', 'fruit'],
            'avg_shelf_life_days': [7, 5, 5, 7, 30, 21, 3, 10, 7, 14],
            'storage_type': ['room', 'room', 'room', 'fridge', 'fridge',
                           'fridge', 'fridge', 'fridge', 'room', 'room']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.data_dir / "food_database.csv", index=False)
        
        print(f"✅ Sample food database created at {self.data_dir / 'food_database.csv'}")
        return df
    
    def load_food_database(self) -> pd.DataFrame:
        """Load the food database with expiry information."""
        db_path = self.data_dir / "food_database.csv"
        
        if not db_path.exists():
            print("Food database not found. Creating sample database...")
            return self.download_sample_dataset()
        
        return pd.read_csv(db_path)
    
    def get_image_transforms(self, mode: str = "train") -> A.Compose:
        """
        Get image transformation pipeline for training or inference.
        
        Args:
            mode: "train" for training transforms, "val" for validation transforms
            
        Returns:
            Albumentations transformation pipeline
        """
        if mode == "train":
            return A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ])
        else:  # validation/inference
            return A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ])
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_sample_fridge_images(self):
        """Create sample fridge images with multiple food items."""
        print("Creating sample fridge layout images...")
        
        # This would typically involve creating composite images
        # For now, we'll create placeholder information
        sample_fridge_data = {
            'image_name': ['fridge_001.jpg', 'fridge_002.jpg', 'fridge_003.jpg'],
            'detected_items': [
                ['milk', 'eggs', 'cheese', 'broccoli'],
                ['apple', 'banana', 'bread', 'chicken'],
                ['tomato', 'orange', 'milk', 'cheese']
            ],
            'purchase_dates': [
                ['2025-09-28', '2025-09-25', '2025-09-20', '2025-09-30'],
                ['2025-09-29', '2025-09-27', '2025-09-26', '2025-09-24'],
                ['2025-09-30', '2025-09-28', '2025-09-28', '2025-09-22']
            ]
        }
        
        # Save sample data
        import json
        with open(self.data_dir / "sample_fridge_data.json", 'w') as f:
            json.dump(sample_fridge_data, f, indent=2)
        
        print(f"✅ Sample fridge data created at {self.data_dir / 'sample_fridge_data.json'}")
        return sample_fridge_data
    
    def get_dataloader(self, 
                      image_paths: List[str], 
                      labels: List[str],
                      batch_size: int = 16,
                      shuffle: bool = True,
                      transform=None) -> DataLoader:
        """
        Create a PyTorch DataLoader for the dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            transform: Image transformations
            
        Returns:
            PyTorch DataLoader
        """
        dataset = FoodDataset(image_paths, labels, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def validate_image(image_path: str) -> bool:
    """
    Validate if an image file is readable and has valid format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_stats(image_dir: str) -> Dict:
    """
    Get statistics about images in a directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Dictionary with image statistics
    """
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'formats': {},
        'avg_width': 0,
        'avg_height': 0
    }
    
    widths, heights = [], []
    
    for file_path in Path(image_dir).rglob('*'):
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            stats['total_images'] += 1
            
            if validate_image(str(file_path)):
                stats['valid_images'] += 1
                
                # Get image dimensions
                with Image.open(file_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    
                    # Track formats
                    fmt = img.format
                    stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1
    
    if widths:
        stats['avg_width'] = np.mean(widths)
        stats['avg_height'] = np.mean(heights)
    
    return stats


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load food database
    food_db = loader.load_food_database()
    print("Food Database:")
    print(food_db.head())
    
    # Create sample fridge data
    fridge_data = loader.create_sample_fridge_images()
    
    # Get image transformations
    train_transforms = loader.get_image_transforms("train")
    print("\n✅ Data loading utilities ready!")