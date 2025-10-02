"""
FastAPI Backend for Smart Food Expiry Detection System

This is the main API server that handles image uploads, food detection,
expiry predictions, and notifications.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.detector import FoodDetector
    from src.expiry_db import ExpiryDatabase, get_expiry_simple
    from src.forecast import FreshnessPredictor
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    # Create placeholder classes for development
    class FoodDetector:
        def __init__(self): pass
        def detect_food_items(self, image_path): return []
        def analyze_fridge_contents(self, image_path): return {}
    
    class ExpiryDatabase:
        def __init__(self): pass
        def get_shelf_life(self, food_name, storage): return 7
        def add_user_item(self, food_name, purchase_date, storage, quantity): return 1
        def get_expiring_items(self, days_ahead): return []
    
    class FreshnessPredictor:
        def __init__(self): pass
        def predict_freshness(self, food_item, purchase_date): return {"recommendation": "Item is fresh"}


# Initialize FastAPI app
app = FastAPI(
    title="Smart Food Expiry Detection API",
    description="AI-powered food expiry detection and freshness prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for file storage
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Initialize components
try:
    food_detector = FoodDetector()
    expiry_db = ExpiryDatabase()
    freshness_predictor = FreshnessPredictor()
    print("✅ All components initialized successfully")
except Exception as e:
    print(f"⚠️ Some components failed to initialize: {e}")
    food_detector = FoodDetector()
    expiry_db = ExpiryDatabase()
    freshness_predictor = FreshnessPredictor()


# Pydantic models for request/response
class FoodItem(BaseModel):
    name: str
    confidence: float
    bbox: List[float]
    detection_time: str


class ExpiryRequest(BaseModel):
    food_name: str
    purchase_date: str
    storage_location: str = "fridge"
    quantity: int = 1


class ExpiryResponse(BaseModel):
    food_name: str
    purchase_date: str
    expiry_date: Optional[str]
    days_remaining: Optional[int]
    status: str
    recommendation: str


class FreshnessRequest(BaseModel):
    food_item: str
    purchase_date: str
    storage_conditions: Optional[Dict[str, Any]] = None


class NotificationRequest(BaseModel):
    user_email: Optional[str] = None
    phone_number: Optional[str] = None
    notification_type: str = "email"  # email, sms, whatsapp
    days_ahead: int = 3


# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Smart Food Expiry Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "detection": "/detect",
            "expiry": "/expiry",
            "freshness": "/freshness",
            "notifications": "/notify",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "food_detector": "active",
            "expiry_database": "active", 
            "freshness_predictor": "active"
        }
    }


@app.post("/detect", response_model=Dict[str, Any])
async def detect_food(
    file: UploadFile = File(...),
    save_results: bool = Form(False)
):
    """
    Detect food items in an uploaded image.
    
    Args:
        file: Uploaded image file
        save_results: Whether to save annotated results
        
    Returns:
        Detection results with bounding boxes and confidence scores
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        temp_filename = f"upload_{timestamp}{file_extension}"
        temp_path = UPLOAD_DIR / temp_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect food items
        detections = food_detector.detect_food_items(
            str(temp_path), 
            save_results=save_results,
            output_dir=str(RESULTS_DIR)
        )
        
        # Analyze fridge contents if multiple items detected
        if len(detections) > 1:
            analysis = food_detector.analyze_fridge_contents(str(temp_path))
        else:
            analysis = {}
        
        # Generate summary
        summary = food_detector.get_detection_summary(detections)
        
        response = {
            "success": True,
            "image_path": str(temp_path),
            "detections": detections,
            "summary": summary,
            "analysis": analysis,
            "total_items": len(detections),
            "processing_time": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Clean up temporary file if not saving results
        if not save_results and temp_path.exists():
            temp_path.unlink()


@app.post("/expiry", response_model=ExpiryResponse)
async def calculate_expiry(request: ExpiryRequest):
    """
    Calculate expiry date for a food item.
    
    Args:
        request: Expiry calculation request
        
    Returns:
        Expiry information and recommendations
    """
    try:
        # Calculate expiry date using database
        expiry_date = expiry_db.calculate_expiry_date(
            request.food_name,
            request.purchase_date,
            request.storage_location
        )
        
        # Calculate days remaining
        days_remaining = None
        status = "unknown"
        
        if expiry_date:
            expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
            days_remaining = (expiry_dt - datetime.now()).days
            
            if days_remaining < 0:
                status = "expired"
            elif days_remaining <= 1:
                status = "critical"
            elif days_remaining <= 3:
                status = "warning"
            else:
                status = "fresh"
        
        # Add item to user tracking
        item_id = expiry_db.add_user_item(
            request.food_name,
            request.purchase_date,
            request.storage_location,
            request.quantity
        )
        
        # Generate recommendation
        if status == "expired":
            recommendation = "⛔ Item has expired. Discard safely."
        elif status == "critical":
            recommendation = "🔴 Use immediately or discard."
        elif status == "warning":
            recommendation = "🟡 Use within the next few days."
        else:
            recommendation = "🟢 Item is fresh and safe to consume."
        
        return ExpiryResponse(
            food_name=request.food_name,
            purchase_date=request.purchase_date,
            expiry_date=expiry_date,
            days_remaining=days_remaining,
            status=status,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expiry calculation failed: {str(e)}")


@app.post("/freshness")
async def predict_freshness(request: FreshnessRequest):
    """
    Predict food freshness using time-series models.
    
    Args:
        request: Freshness prediction request
        
    Returns:
        Detailed freshness prediction and timeline
    """
    try:
        # Predict freshness using our forecasting models
        prediction_result = freshness_predictor.predict_freshness(
            request.food_item,
            request.purchase_date,
            request.storage_conditions
        )
        
        return {
            "success": True,
            "prediction": prediction_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Freshness prediction failed: {str(e)}")


@app.get("/expiring")
async def get_expiring_items(days_ahead: int = 3):
    """
    Get list of items expiring within specified days.
    
    Args:
        days_ahead: Number of days to look ahead
        
    Returns:
        List of expiring items with details
    """
    try:
        expiring_items = expiry_db.get_expiring_items(days_ahead)
        
        return {
            "success": True,
            "days_ahead": days_ahead,
            "total_expiring": len(expiring_items),
            "items": expiring_items,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get expiring items: {str(e)}")


@app.post("/notify")
async def send_notifications(request: NotificationRequest):
    """
    Send expiry notifications to users.
    
    Args:
        request: Notification request with contact details
        
    Returns:
        Notification status
    """
    try:
        # Get expiring items
        expiring_items = expiry_db.get_expiring_items(request.days_ahead)
        
        if not expiring_items:
            return {
                "success": True,
                "message": "No items expiring soon",
                "notifications_sent": 0
            }
        
        # Generate notification message
        message_lines = [f"🚨 Food Expiry Alert - {len(expiring_items)} items expiring soon:\n"]
        
        for item in expiring_items:
            days_left = int(item.get('days_remaining', 0))
            if days_left <= 0:
                status_emoji = "⛔"
                status_text = "EXPIRED"
            elif days_left <= 1:
                status_emoji = "🔴"
                status_text = "TODAY"
            else:
                status_emoji = "🟡"
                status_text = f"{days_left} days"
            
            message_lines.append(f"{status_emoji} {item['food_name']}: {status_text}")
        
        notification_message = "\n".join(message_lines)
        
        # In a real implementation, you would integrate with:
        # - Email service (SMTP, SendGrid, etc.)
        # - SMS service (Twilio, etc.)
        # - WhatsApp Business API
        # - Telegram Bot API
        
        # For demo purposes, we'll just return the message that would be sent
        return {
            "success": True,
            "message": "Notifications prepared successfully",
            "notification_type": request.notification_type,
            "recipient": request.user_email or request.phone_number,
            "items_count": len(expiring_items),
            "notification_content": notification_message,
            "notifications_sent": 1  # Would be actual count in real implementation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notification failed: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """Get database and system statistics."""
    try:
        stats = expiry_db.get_statistics()
        
        # Add some additional metrics
        current_time = datetime.now()
        stats["api_status"] = {
            "uptime": "Active",  # Would track actual uptime in production
            "last_updated": current_time.isoformat(),
            "total_api_calls": "N/A",  # Would track in production
            "avg_response_time": "N/A"  # Would track in production
        }
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": current_time.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/foods/search")
async def search_foods(query: str):
    """
    Search for food items in the database.
    
    Args:
        query: Search query string
        
    Returns:
        List of matching food items
    """
    try:
        results = expiry_db.search_foods(query)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """
    Delete a tracked food item.
    
    Args:
        item_id: ID of the item to delete
        
    Returns:
        Deletion status
    """
    try:
        # In a real implementation, you would delete from database
        # For now, we'll just update status to 'consumed'
        expiry_db.update_item_status(item_id, "consumed")
        
        return {
            "success": True,
            "message": f"Item {item_id} marked as consumed",
            "item_id": item_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {str(e)}")


# Extension endpoints for future features

@app.get("/recipes")
async def get_recipe_suggestions(items: str = ""):
    """
    Get recipe suggestions for items nearing expiry.
    
    Args:
        items: Comma-separated list of food items
        
    Returns:
        Recipe suggestions
    """
    # Placeholder for recipe API integration
    food_items = [item.strip() for item in items.split(",") if item.strip()]
    
    sample_recipes = [
        {
            "name": "Quick Stir Fry",
            "ingredients": food_items,
            "prep_time": "15 minutes",
            "difficulty": "Easy",
            "url": "https://example.com/recipe1"
        },
        {
            "name": "Fresh Salad",
            "ingredients": food_items,
            "prep_time": "10 minutes", 
            "difficulty": "Easy",
            "url": "https://example.com/recipe2"
        }
    ]
    
    return {
        "success": True,
        "input_items": food_items,
        "recipe_suggestions": sample_recipes,
        "note": "Recipe API integration pending"
    }


if __name__ == "__main__":
    print("🚀 Starting Smart Food Expiry Detection API...")
    print("📝 API Documentation will be available at: http://localhost:8000/docs")
    print("🔄 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )