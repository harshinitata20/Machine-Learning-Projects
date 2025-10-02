"""
Pydantic Models for API Request/Response Validation

This module defines all the data models used by the FastAPI application
for request validation and response serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum


class StorageLocation(str, Enum):
    """Enumeration for food storage locations."""
    ROOM = "room"
    FRIDGE = "fridge"
    FREEZER = "freezer"


class FoodStatus(str, Enum):
    """Enumeration for food item status."""
    FRESH = "fresh"
    WARNING = "warning"
    CRITICAL = "critical"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class NotificationType(str, Enum):
    """Enumeration for notification types."""
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"


# Detection Models
class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")


class FoodDetection(BaseModel):
    """Single food item detection result."""
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    class_id: int = Field(..., description="Class ID from the model")
    class_name: str = Field(..., description="Class name from the model")
    food_name: str = Field(..., description="Recognized food item name")
    detection_time: str = Field(..., description="ISO timestamp of detection")


class DetectionResponse(BaseModel):
    """Response model for food detection endpoint."""
    success: bool = Field(True, description="Whether detection was successful")
    image_path: str = Field(..., description="Path to processed image")
    detections: List[FoodDetection] = Field(..., description="List of detected food items")
    summary: str = Field(..., description="Human-readable summary of detections")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis data")
    total_items: int = Field(..., description="Total number of items detected")
    processing_time: str = Field(..., description="ISO timestamp of processing")


# Expiry Models
class ExpiryRequest(BaseModel):
    """Request model for expiry calculation."""
    food_name: str = Field(..., min_length=1, description="Name of the food item")
    purchase_date: str = Field(..., description="Purchase date in YYYY-MM-DD format")
    storage_location: StorageLocation = Field(default=StorageLocation.FRIDGE, description="Storage location")
    quantity: int = Field(default=1, ge=1, description="Quantity of items")
    
    @validator('purchase_date')
    def validate_purchase_date(cls, v):
        """Validate purchase date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Purchase date must be in YYYY-MM-DD format')


class ExpiryResponse(BaseModel):
    """Response model for expiry calculation."""
    food_name: str = Field(..., description="Name of the food item")
    purchase_date: str = Field(..., description="Purchase date")
    expiry_date: Optional[str] = Field(None, description="Calculated expiry date")
    days_remaining: Optional[int] = Field(None, description="Days until expiry")
    status: FoodStatus = Field(..., description="Current status of the food item")
    recommendation: str = Field(..., description="Human-readable recommendation")
    item_id: Optional[int] = Field(None, description="Database ID of the tracked item")


# Freshness Models
class StorageConditions(BaseModel):
    """Storage conditions affecting food freshness."""
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    light_exposure: Optional[str] = Field(None, description="Light exposure level")
    air_circulation: Optional[str] = Field(None, description="Air circulation quality")


class FreshnessRequest(BaseModel):
    """Request model for freshness prediction."""
    food_item: str = Field(..., min_length=1, description="Name of the food item")
    purchase_date: str = Field(..., description="Purchase date in YYYY-MM-DD format")
    storage_conditions: Optional[StorageConditions] = Field(None, description="Storage conditions")
    prediction_method: str = Field(default="auto", description="Prediction method to use")
    
    @validator('purchase_date')
    def validate_purchase_date(cls, v):
        """Validate purchase date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Purchase date must be in YYYY-MM-DD format')


class FreshnessPrediction(BaseModel):
    """Individual freshness prediction result."""
    model_type: str = Field(..., description="Type of model used for prediction")
    current_freshness: Optional[float] = Field(None, description="Current freshness percentage")
    predicted_freshness_1d: Optional[float] = Field(None, description="Predicted freshness after 1 day")
    predicted_freshness_3d: Optional[float] = Field(None, description="Predicted freshness after 3 days")
    predicted_freshness_7d: Optional[float] = Field(None, description="Predicted freshness after 7 days")
    warning_date: Optional[str] = Field(None, description="Date when freshness drops below 50%")
    spoilage_date: Optional[str] = Field(None, description="Date when freshness drops below 20%")
    forecast_values: Optional[List[float]] = Field(None, description="Full forecast timeline")
    forecast_dates: Optional[List[str]] = Field(None, description="Dates corresponding to forecast values")


class FreshnessResponse(BaseModel):
    """Response model for freshness prediction."""
    success: bool = Field(True, description="Whether prediction was successful")
    food_item: str = Field(..., description="Name of the food item")
    purchase_date: str = Field(..., description="Purchase date")
    analysis_date: str = Field(..., description="Date of analysis")
    best_prediction: FreshnessPrediction = Field(..., description="Best prediction result")
    all_predictions: Dict[str, Any] = Field(..., description="All prediction results")
    storage_conditions: Dict[str, Any] = Field(default_factory=dict, description="Storage conditions used")
    recommendation: str = Field(..., description="Human-readable recommendation")
    timestamp: str = Field(..., description="Processing timestamp")


# Notification Models
class NotificationRequest(BaseModel):
    """Request model for sending notifications."""
    user_email: Optional[str] = Field(None, description="Email address for notifications")
    phone_number: Optional[str] = Field(None, description="Phone number for SMS/WhatsApp")
    notification_type: NotificationType = Field(default=NotificationType.EMAIL, description="Type of notification")
    days_ahead: int = Field(default=3, ge=1, le=30, description="Days ahead to check for expiring items")
    
    @validator('user_email')
    def validate_email(cls, v):
        """Basic email validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class NotificationResponse(BaseModel):
    """Response model for notification sending."""
    success: bool = Field(..., description="Whether notification was sent successfully")
    message: str = Field(..., description="Status message")
    notification_type: NotificationType = Field(..., description="Type of notification sent")
    recipient: Optional[str] = Field(None, description="Recipient identifier")
    items_count: int = Field(..., description="Number of expiring items")
    notification_content: str = Field(..., description="Content that was sent")
    notifications_sent: int = Field(..., description="Number of notifications sent")


# User Item Models
class UserItem(BaseModel):
    """User's tracked food item."""
    id: Optional[int] = Field(None, description="Database ID")
    food_name: str = Field(..., description="Name of the food item")
    purchase_date: str = Field(..., description="Purchase date")
    expiry_date: Optional[str] = Field(None, description="Calculated expiry date")
    storage_location: str = Field(..., description="Storage location")
    quantity: int = Field(..., description="Quantity of items")
    status: str = Field(..., description="Current status")
    category: Optional[str] = Field(None, description="Food category")
    days_remaining: Optional[float] = Field(None, description="Days until expiry")
    notes: Optional[str] = Field(None, description="Additional notes")


class ExpiringItemsResponse(BaseModel):
    """Response model for expiring items."""
    success: bool = Field(True, description="Whether request was successful")
    days_ahead: int = Field(..., description="Days ahead that were checked")
    total_expiring: int = Field(..., description="Total number of expiring items")
    items: List[UserItem] = Field(..., description="List of expiring items")
    timestamp: str = Field(..., description="Response timestamp")


# Food Database Models
class FoodItemInfo(BaseModel):
    """Information about a food item in the database."""
    food_name: str = Field(..., description="Name of the food item")
    category: str = Field(..., description="Food category")
    shelf_life_room: Optional[int] = Field(None, description="Shelf life at room temperature (days)")
    shelf_life_fridge: Optional[int] = Field(None, description="Shelf life in refrigerator (days)")
    shelf_life_freezer: Optional[int] = Field(None, description="Shelf life in freezer (days)")
    notes: Optional[str] = Field(None, description="Additional notes about the food item")


class SearchResponse(BaseModel):
    """Response model for food search."""
    success: bool = Field(True, description="Whether search was successful")
    query: str = Field(..., description="Search query used")
    results: List[FoodItemInfo] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results found")


# Statistics Models
class APIStatus(BaseModel):
    """API status information."""
    uptime: str = Field(..., description="API uptime")
    last_updated: str = Field(..., description="Last update timestamp")
    total_api_calls: Union[int, str] = Field(..., description="Total API calls")
    avg_response_time: Union[float, str] = Field(..., description="Average response time")


class DatabaseStats(BaseModel):
    """Database statistics."""
    total_foods_in_db: int = Field(..., description="Total food items in database")
    total_user_items: int = Field(..., description="Total user tracked items")
    items_by_status: Dict[str, int] = Field(default_factory=dict, description="Items grouped by status")
    items_by_category: Dict[str, int] = Field(default_factory=dict, description="Items grouped by category")


class StatisticsResponse(BaseModel):
    """Response model for system statistics."""
    success: bool = Field(True, description="Whether request was successful")
    statistics: Dict[str, Any] = Field(..., description="System statistics")
    timestamp: str = Field(..., description="Response timestamp")


# Recipe Models (for future extension)
class Recipe(BaseModel):
    """Recipe suggestion model."""
    name: str = Field(..., description="Recipe name")
    ingredients: List[str] = Field(..., description="List of ingredients")
    prep_time: str = Field(..., description="Preparation time")
    difficulty: str = Field(..., description="Difficulty level")
    url: Optional[str] = Field(None, description="Recipe URL")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Recipe rating")


class RecipeResponse(BaseModel):
    """Response model for recipe suggestions."""
    success: bool = Field(True, description="Whether request was successful")
    input_items: List[str] = Field(..., description="Input food items")
    recipe_suggestions: List[Recipe] = Field(..., description="Recipe suggestions")
    note: Optional[str] = Field(None, description="Additional notes")


# Error Models
class ErrorDetail(BaseModel):
    """Detailed error information."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = Field(False, description="Always False for errors")
    error: ErrorDetail = Field(..., description="Error details")


# Health Check Models
class ComponentStatus(BaseModel):
    """Individual component status."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    last_check: str = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional component details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")
    version: Optional[str] = Field(None, description="API version")


# Batch Operation Models
class BatchDetectionRequest(BaseModel):
    """Request model for batch detection operations."""
    image_urls: List[str] = Field(..., description="List of image URLs to process")
    save_results: bool = Field(default=False, description="Whether to save annotated results")


class BatchExpiryRequest(BaseModel):
    """Request model for batch expiry calculations."""
    items: List[ExpiryRequest] = Field(..., description="List of items to process")


class BatchResponse(BaseModel):
    """Response model for batch operations."""
    success: bool = Field(..., description="Whether batch operation was successful")
    total_processed: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    results: List[Any] = Field(..., description="Individual results")
    errors: List[str] = Field(default_factory=list, description="Error messages for failed items")