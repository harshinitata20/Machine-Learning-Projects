#!/usr/bin/env python3
"""
Simple API test to verify the system is running correctly.
"""

import requests
import json

def simple_api_test():
    """Test basic API functionality."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Simple API Test")
    print("=" * 30)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        
        if response.status_code == 200:
            print("✅ API Server is running!")
            print(f"   Status: {response.json()}")
            
            # Test expiry items endpoint
            print("\nTesting expiry items...")
            items_response = requests.get(f"{base_url}/expiry/items")
            
            if items_response.status_code == 200:
                items = items_response.json()
                print(f"✅ Expiry database working!")
                print(f"   Items in database: {len(items)}")
                
                if items:
                    print(f"   Sample item: {items[0]['name']}")
            
            print(f"\n🎉 System Status: OPERATIONAL")
            print(f"📱 Frontend: http://localhost:8501")
            print(f"🔧 API Docs: http://127.0.0.1:8000/docs")
            
        else:
            print(f"❌ API Server not responding (Status: {response.status_code})")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running with: python -m uvicorn api.main:app")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    simple_api_test()