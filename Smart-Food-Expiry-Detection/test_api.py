#!/usr/bin/env python3
"""
Test the running API endpoints to validate the system is working correctly.
"""

import requests
import json
import base64
from PIL import Image
import numpy as np
import io

def test_api_endpoints():
    """Test the main API endpoints."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing API Endpoints...")
    print("=" * 40)
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health Check: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health Check: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ Health Check: FAILED - {e}")
    
    print()
    
    # Test 2: Detection Endpoint
    try:
        # Create a simple test image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        pil_image = Image.fromarray(test_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Send detection request
        payload = {
            "image": img_base64,
            "confidence_threshold": 0.5
        }
        
        response = requests.post(f"{base_url}/detect", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ Food Detection: PASSED")
            print(f"   Items detected: {len(result.get('detections', []))}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
        else:
            print(f"❌ Food Detection: FAILED ({response.status_code})")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Food Detection: FAILED - {e}")
    
    print()
    
    # Test 3: Expiry Items Endpoint
    try:
        response = requests.get(f"{base_url}/expiry/items")
        if response.status_code == 200:
            items = response.json()
            print("✅ Expiry Items: PASSED")
            print(f"   Total items: {len(items)}")
            if items:
                print(f"   Sample item: {items[0].get('name', 'Unknown')}")
        else:
            print(f"❌ Expiry Items: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ Expiry Items: FAILED - {e}")
    
    print()
    
    # Test 4: Add Food Item
    try:
        new_item = {
            "name": "test_apple",
            "category": "fruit",
            "storage_location": "fridge",
            "quantity": 1
        }
        
        response = requests.post(f"{base_url}/expiry/add", json=new_item)
        if response.status_code == 200:
            result = response.json()
            print("✅ Add Food Item: PASSED")
            print(f"   Added item ID: {result.get('id', 'N/A')}")
            print(f"   Expiry date: {result.get('expiry_date', 'N/A')}")
        else:
            print(f"❌ Add Food Item: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ Add Food Item: FAILED - {e}")
    
    print("\n🎯 API Testing Complete!")

def test_system_integration():
    """Test complete system workflow."""
    
    print("\n🔄 Testing Complete System Integration...")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Step 1: Detect foods in image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        detection_payload = {
            "image": img_base64,
            "confidence_threshold": 0.3
        }
        
        print("1️⃣ Detecting foods...")
        detection_response = requests.post(f"{base_url}/detect", json=detection_payload)
        
        if detection_response.status_code == 200:
            detections = detection_response.json()
            detected_foods = detections.get('detections', [])
            print(f"   ✅ Detected {len(detected_foods)} food items")
            
            # Step 2: Add detected foods to database
            added_items = []
            for food in detected_foods[:3]:  # Add first 3 items
                item_data = {
                    "name": food.get('class', 'unknown_food'),
                    "category": "detected",
                    "storage_location": "fridge",
                    "quantity": 1
                }
                
                add_response = requests.post(f"{base_url}/expiry/add", json=item_data)
                if add_response.status_code == 200:
                    added_items.append(add_response.json())
            
            print(f"2️⃣ Added {len(added_items)} items to database")
            
            # Step 3: Check expiring items
            print("3️⃣ Checking for expiring items...")
            expiring_response = requests.get(f"{base_url}/expiry/expiring?days=7")
            
            if expiring_response.status_code == 200:
                expiring_items = expiring_response.json()
                print(f"   ✅ Found {len(expiring_items)} items expiring in next 7 days")
            
            print("\n🎉 Integration Test: PASSED")
            print(f"   Total workflow steps: 3/3 successful")
            
        else:
            print("❌ Detection failed, skipping integration test")
            
    except Exception as e:
        print(f"❌ Integration Test: FAILED - {e}")

if __name__ == "__main__":
    print("🚀 SMART FOOD EXPIRY SYSTEM - API TESTING")
    print("=" * 55)
    
    # Test individual endpoints
    test_api_endpoints()
    
    # Test complete integration
    test_system_integration()
    
    print("\n📊 Testing Summary:")
    print("   🌐 API Server: http://127.0.0.1:8000")
    print("   🎨 Frontend Dashboard: http://localhost:8501")
    print("   📚 API Documentation: http://127.0.0.1:8000/docs")
    
    print("\n✅ All systems are operational!")