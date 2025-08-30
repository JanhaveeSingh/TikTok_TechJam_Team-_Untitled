#!/usr/bin/env python3
"""
Test script for CLIP Parser service
"""
import requests
import os
import time
from PIL import Image, ImageDraw

def create_test_image():
    """Create a simple test image with some UI elements"""
    # Create a simple test image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some mock UI elements
    # Button
    draw.rectangle([50, 50, 150, 80], fill='lightblue', outline='black', width=2)
    draw.text((75, 60), "Button", fill='black')
    
    # Text field
    draw.rectangle([50, 100, 200, 130], fill='white', outline='gray', width=1)
    draw.text((55, 110), "Enter text here", fill='gray')
    
    # Icon (circle)
    draw.ellipse([250, 50, 280, 80], fill='green', outline='black', width=2)
    
    # Some text
    draw.text((50, 150), "Welcome to the app!", fill='black')
    draw.text((50, 170), "Please select an option:", fill='black')
    
    return img

def test_clip_parser():
    """Test the CLIP parser service"""
    print("🧪 Testing CLIP Parser Service...")
    
    # Create test image
    test_img = create_test_image()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_img_path = tmp_file.name
    test_img.save(test_img_path)
    print(f"✅ Created test image: {test_img_path}")
    
    # Test the service
    url = "http://localhost:8000/process_image/"
    
    try:
        with open(test_img_path, 'rb') as f:
            files = {'file': ('test_ui.png', f, 'image/png')}
            data = {
                'box_threshold': 0.3,
                'iou_threshold': 0.5,
                'imgsz_component': 640
            }
            
            print("📡 Sending request to CLIP Parser...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=30)
            elapsed_time = time.time() - start_time
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Processing time: {elapsed_time:.2f}s")
            print(f"🔍 Found {len(result.get('parsed_content', []))} elements")
            print(f"⏱️  Service reported time: {result.get('e_time', 0):.2f}s")
            
            # Print some details about found elements
            for i, element in enumerate(result.get('parsed_content', [])[:5]):  # Show first 5
                print(f"   Element {i}: {element.get('type', 'unknown')} - {element.get('content', 'N/A')}")
            
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the CLIP Parser service running?")
        print("💡 Try running: docker-compose up clip-parser")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

def test_health_check():
    """Test the health check endpoint"""
    print("\n🏥 Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CLIP Parser Test Suite")
    print("=" * 40)
    
    # Test health check first
    health_ok = test_health_check()
    
    if health_ok:
        # Test main functionality
        test_ok = test_clip_parser()
        
        if test_ok:
            print("\n🎉 All tests passed! CLIP Parser is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the service logs.")
    else:
        print("\n❌ Service is not responding. Make sure it's running.")
        print("💡 Start with: docker-compose up clip-parser")
