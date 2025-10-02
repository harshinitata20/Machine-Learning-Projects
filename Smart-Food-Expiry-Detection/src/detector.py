"""
Food Detection using YOLOv8

This module implements food item detection using the YOLOv8 model from Ultralytics.
It can detect various food items in fridge images and return bounding boxes with confidence scores.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path
import json
from datetime import datetime


class FoodDetector:
    """YOLOv8-based food detection system."""
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize the food detector.
        
        Args:
            model_path: Path to YOLOv8 model or model name
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🔧 Initializing Food Detector on {self.device}")
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📥 Downloading YOLOv8n model...")
            self.model = YOLO("yolov8n.pt")  # This will download if not present
        
        # Food class mapping (COCO dataset has some food items)
        self.food_classes = self._get_food_classes()
        
    def _get_food_classes(self) -> Dict[int, str]:
        """
        Get mapping of COCO class IDs to food items.
        
        Returns:
            Dictionary mapping class IDs to food names
        """
        # COCO classes that are food items
        coco_food_classes = {
            47: "apple",
            48: "sandwich", 
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot_dog",
            53: "pizza",
            54: "donut",
            55: "cake"
        }
        
        return coco_food_classes
    
    def detect_food_items(self, 
                         image_path: str,
                         save_results: bool = False,
                         output_dir: str = "results") -> List[Dict]:
        """
        Detect food items in an image.
        
        Args:
            image_path: Path to the input image
            save_results: Whether to save annotated image
            output_dir: Directory to save results
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        try:
            # Run inference
            results = self.model(image_path, device=self.device)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            # Get class name
                            class_name = self.model.names.get(class_id, f"unknown_{class_id}")
                            
                            # Check if it's a food item
                            if class_id in self.food_classes:
                                food_name = self.food_classes[class_id]
                            elif any(food in class_name.lower() for food in 
                                   ["apple", "orange", "banana", "sandwich", "pizza", "cake", "donut"]):
                                food_name = class_name
                            else:
                                # Skip non-food items or treat as generic food
                                continue
                            
                            detection = {
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(confidence),
                                "class_id": class_id,
                                "class_name": class_name,
                                "food_name": food_name,
                                "detection_time": datetime.now().isoformat()
                            }
                            
                            detections.append(detection)
                
                # Save annotated image if requested
                if save_results:
                    Path(output_dir).mkdir(exist_ok=True)
                    output_path = Path(output_dir) / f"annotated_{Path(image_path).name}"
                    result.save(output_path)
                    print(f"💾 Saved annotated image: {output_path}")
            
            print(f"🔍 Detected {len(detections)} food items in {image_path}")
            return detections
            
        except Exception as e:
            print(f"❌ Error detecting food items: {e}")
            return []
    
    def detect_batch(self, 
                    image_paths: List[str],
                    save_results: bool = False) -> Dict[str, List[Dict]]:
        """
        Detect food items in multiple images.
        
        Args:
            image_paths: List of image paths
            save_results: Whether to save annotated images
            
        Returns:
            Dictionary mapping image paths to their detections
        """
        batch_results = {}
        
        print(f"🔄 Processing batch of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            detections = self.detect_food_items(image_path, save_results)
            batch_results[image_path] = detections
        
        return batch_results
    
    def analyze_fridge_contents(self, 
                               fridge_image_path: str,
                               save_analysis: bool = True) -> Dict:
        """
        Analyze contents of a fridge image and provide detailed report.
        
        Args:
            fridge_image_path: Path to fridge image
            save_analysis: Whether to save analysis results
            
        Returns:
            Detailed analysis dictionary
        """
        detections = self.detect_food_items(fridge_image_path, save_results=True)
        
        # Analyze detections
        analysis = {
            "image_path": fridge_image_path,
            "analysis_time": datetime.now().isoformat(),
            "total_items_detected": len(detections),
            "detected_foods": [],
            "food_categories": {},
            "confidence_stats": {}
        }
        
        if detections:
            confidences = [d["confidence"] for d in detections]
            food_names = [d["food_name"] for d in detections]
            
            # Confidence statistics
            analysis["confidence_stats"] = {
                "mean": float(np.mean(confidences)),
                "max": float(np.max(confidences)),
                "min": float(np.min(confidences)),
                "std": float(np.std(confidences))
            }
            
            # Food categories
            for food in food_names:
                category = self._categorize_food(food)
                analysis["food_categories"][category] = analysis["food_categories"].get(category, 0) + 1
            
            # Detailed food list
            analysis["detected_foods"] = detections
        
        # Save analysis
        if save_analysis:
            output_path = f"fridge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"💾 Saved analysis: {output_path}")
        
        return analysis
    
    def _categorize_food(self, food_name: str) -> str:
        """
        Categorize food items into groups.
        
        Args:
            food_name: Name of the food item
            
        Returns:
            Food category string
        """
        categories = {
            "fruits": ["apple", "orange", "banana", "grape", "strawberry"],
            "vegetables": ["broccoli", "carrot", "lettuce", "tomato"],
            "proteins": ["chicken", "beef", "fish", "eggs"],
            "dairy": ["milk", "cheese", "yogurt", "butter"],
            "grains": ["bread", "rice", "pasta", "cereal"],
            "snacks": ["pizza", "sandwich", "cake", "donut", "hot_dog"]
        }
        
        food_lower = food_name.lower()
        for category, items in categories.items():
            if any(item in food_lower for item in items):
                return category
        
        return "other"
    
    def get_detection_summary(self, detections: List[Dict]) -> str:
        """
        Generate a human-readable summary of detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary string
        """
        if not detections:
            return "No food items detected in the image."
        
        food_counts = {}
        total_confidence = 0
        
        for detection in detections:
            food_name = detection["food_name"]
            confidence = detection["confidence"]
            
            food_counts[food_name] = food_counts.get(food_name, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(detections)
        
        summary = f"Detected {len(detections)} food items:\n"
        for food, count in food_counts.items():
            summary += f"• {food.title()}: {count} item(s)\n"
        summary += f"\nAverage detection confidence: {avg_confidence:.2f}"
        
        return summary
    
    def visualize_detections(self, 
                           image_path: str,
                           detections: List[Dict],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image_path: Path to the original image
            detections: List of detection dictionaries
            save_path: Optional path to save the annotated image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            food_name = detection["food_name"]
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label
            label = f"{food_name}: {confidence:.2f}"
            cv2.putText(image_rgb, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved visualization: {save_path}")
        
        return image_rgb


def demo_food_detection():
    """Demonstration of food detection functionality."""
    print("🚀 Food Detection Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = FoodDetector(confidence_threshold=0.3)
    
    # Create a sample detection scenario
    print("\n📸 This demo would work with actual fridge images.")
    print("Example usage:")
    print("1. Place food items in front of camera")
    print("2. Take a photo")
    print("3. Run detection")
    
    # Sample detection output format
    sample_detection = [
        {
            "bbox": [100, 150, 200, 300],
            "confidence": 0.85,
            "class_id": 47,
            "class_name": "apple",
            "food_name": "apple",
            "detection_time": datetime.now().isoformat()
        },
        {
            "bbox": [250, 100, 350, 250],
            "confidence": 0.78,
            "class_id": 49,
            "class_name": "orange", 
            "food_name": "orange",
            "detection_time": datetime.now().isoformat()
        }
    ]
    
    print("\n📊 Sample Detection Results:")
    summary = detector.get_detection_summary(sample_detection)
    print(summary)
    
    print("\n✅ Food detection system ready!")
    return detector


if __name__ == "__main__":
    # Run demo
    detector = demo_food_detection()