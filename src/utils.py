"""
Utility Functions and Helper Classes

This module contains various utility functions and helper classes
used throughout the Smart Food Expiry Detection system.
"""

import os
import logging
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import base64
from PIL import Image
import requests


class Logger:
    """Centralized logging utility."""
    
    def __init__(self, name: str = "food_expiry", level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (optional)
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "food_expiry.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


class ImageUtils:
    """Image processing utilities."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, target_size)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    @staticmethod
    def decode_base64_image(base64_string: str, output_path: str):
        """
        Decode base64 string to image file.
        
        Args:
            base64_string: Base64 encoded image
            output_path: Output file path
        """
        image_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as img_file:
            img_file.write(image_data)
    
    @staticmethod
    def get_image_hash(image_path: str) -> str:
        """
        Generate hash for image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            SHA256 hash of the image
        """
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


class DateUtils:
    """Date and time utilities."""
    
    @staticmethod
    def format_date(date_obj: datetime, format_str: str = "%Y-%m-%d") -> str:
        """
        Format datetime object to string.
        
        Args:
            date_obj: Datetime object
            format_str: Format string
            
        Returns:
            Formatted date string
        """
        return date_obj.strftime(format_str)
    
    @staticmethod
    def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> datetime:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string
            format_str: Format string
            
        Returns:
            Datetime object
        """
        return datetime.strptime(date_str, format_str)
    
    @staticmethod
    def days_between(date1: datetime, date2: datetime) -> int:
        """
        Calculate days between two dates.
        
        Args:
            date1: First date
            date2: Second date
            
        Returns:
            Number of days between dates
        """
        return (date2 - date1).days
    
    @staticmethod
    def add_days(date_obj: datetime, days: int) -> datetime:
        """
        Add days to a date.
        
        Args:
            date_obj: Original date
            days: Number of days to add
            
        Returns:
            New date
        """
        return date_obj + timedelta(days=days)
    
    @staticmethod
    def is_date_in_range(check_date: datetime, 
                        start_date: datetime, 
                        end_date: datetime) -> bool:
        """
        Check if date is within a range.
        
        Args:
            check_date: Date to check
            start_date: Range start
            end_date: Range end
            
        Returns:
            True if date is in range
        """
        return start_date <= check_date <= end_date


class ConfigManager:
    """Configuration management utility."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4
                },
                "detection": {
                    "confidence_threshold": 0.5,
                    "model_path": "yolov8n.pt"
                },
                "database": {
                    "url": "sqlite:///./data/food_expiry.db"
                },
                "storage": {
                    "upload_dir": "./uploads",
                    "results_dir": "./results"
                },
                "notifications": {
                    "default_days_ahead": 3,
                    "email_enabled": True,
                    "sms_enabled": False
                }
            }
            
            # Save default config
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict = None):
        """Save configuration to file."""
        config_to_save = config or self.config
        
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        self.save_config()


class ValidationUtils:
    """Input validation utilities."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email string
            
        Returns:
            True if valid email format
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number string
            
        Returns:
            True if valid phone format
        """
        import re
        # Simple phone validation (international format)
        pattern = r'^\+?1?\d{9,15}$'
        clean_phone = re.sub(r'[^\d+]', '', phone)
        return re.match(pattern, clean_phone) is not None
    
    @staticmethod
    def is_valid_date(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
        """
        Validate date string format.
        
        Args:
            date_str: Date string
            format_str: Expected format
            
        Returns:
            True if valid date format
        """
        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_image_file(file_path: str) -> bool:
        """
        Validate if file is a valid image.
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if valid image file
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False


class FileUtils:
    """File system utilities."""
    
    @staticmethod
    def ensure_dir(directory: str):
        """
        Ensure directory exists, create if not.
        
        Args:
            directory: Directory path
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension (with dot)
        """
        return Path(file_path).suffix
    
    @staticmethod
    def generate_unique_filename(base_name: str, extension: str = "") -> str:
        """
        Generate unique filename with timestamp.
        
        Args:
            base_name: Base filename
            extension: File extension
            
        Returns:
            Unique filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}{extension}"
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        import re
        # Remove invalid characters
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        return cleaned.strip('_')


class APIUtils:
    """API helper utilities."""
    
    @staticmethod
    def make_request(url: str, 
                    method: str = "GET", 
                    data: Dict = None,
                    headers: Dict = None,
                    timeout: int = 30) -> Dict:
        """
        Make HTTP request with error handling.
        
        Args:
            url: Request URL
            method: HTTP method
            data: Request data
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Response dictionary
        """
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    @staticmethod
    def format_api_response(success: bool = True, 
                           data: Any = None,
                           message: str = "",
                           error: str = None) -> Dict:
        """
        Format standardized API response.
        
        Args:
            success: Whether request was successful
            data: Response data
            message: Success message
            error: Error message
            
        Returns:
            Formatted response dictionary
        """
        response = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            if data is not None:
                response["data"] = data
            if message:
                response["message"] = message
        else:
            response["error"] = error or "An error occurred"
        
        return response


class CacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.now().timestamp() - timestamp < self.ttl:
                return value
            else:
                # Remove expired item
                del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now().timestamp())
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global instances
logger = Logger()
config_manager = ConfigManager()
cache_manager = CacheManager()


def get_system_info() -> Dict:
    """Get system information for debugging."""
    import platform
    import psutil
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log2(size_bytes) / 10))
    
    if i < len(size_names):
        size = size_bytes / (1024 ** i)
        return f"{size:.1f} {size_names[i]}"
    else:
        return f"{size_bytes} B"


def truncate_string(text: str, max_length: int = 50) -> str:
    """
    Truncate string with ellipsis if too long.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


if __name__ == "__main__":
    # Demo utilities
    print("🔧 Food Expiry Detection Utilities")
    print("=" * 50)
    
    # Test logger
    logger.info("Logger initialized successfully")
    
    # Test config manager
    print(f"API host: {config_manager.get('api.host', 'localhost')}")
    
    # Test cache
    cache_manager.set("test_key", "test_value")
    cached_value = cache_manager.get("test_key")
    print(f"Cache test: {cached_value}")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System: {sys_info['platform']} {sys_info['architecture']}")
    
    print("\n✅ All utilities working correctly!")