#!/usr/bin/env python3
"""
Debug script to test the complete data flow
Run this to see where the data is getting lost
"""

import requests
import json
import sys

def test_ml_service(image_path):
    """Test ML service directly"""
    print("=" * 60)
    print("STEP 1: Testing ML Service (Port 8000)")
    print("=" * 60)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/analyze-prescription', files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
        
        data = response.json()
        print(f"\n✅ ML Service Response Summary:")
        print(f"   Success: {data.get('success')}")
        print(f"   Prescription ID: {data.get('prescription_id')}")
        print(f"   Patient Name: {data.get('patient', {}).get('name')}")
        print(f"   Doctor Name: {data.get('doctor', {}).get('name')}")
        print(f"   Medicines Count: {len(data.get('medicines', []))}")
        print(f"   Confidence: {data.get('confidence_score', 0)}")
        
        return data
    except Exception as e:
        print(f"❌ ML Service Error: {e}")
        return None

def test_backend(image_path):
    """Test Backend service (Port 8080)"""
    print("\n" + "=" * 60)
    print("STEP 2: Testing Backend Service (Port 8080)")
    print("=" * 60)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8080/api/v1/upload', files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
        
        data = response.json()
        print(f"\n✅ Backend Response Summary:")
        print(f"   Success: {data.get('success')}")
        
        if data.get('success'):
            result = data.get('data', {})
            print(f"   Prescription ID: {result.get('prescription_id')}")
            print(f"   Patient Name: {result.get('patient', {}).get('name')}")
            print(f"   Doctor Name: {result.get('doctor', {}).get('name')}")
            print(f"   Medicines Count: {len(result.get('medicines', []))}")
            print(f"   Confidence: {result.get('confidence', 0)}")
        else:
            print(f"   Error: {data.get('error')}")
        
        return data
    except Exception as e:
        print(f"❌ Backend Error: {e}")
        return None

def compare_results(ml_data, backend_data):
    """Compare ML and Backend responses"""
    print("\n" + "=" * 60)
    print("STEP 3: Comparing Results")
    print("=" * 60)
    
    if not ml_data or not backend_data:
        print("❌ Cannot compare - one service failed")
        return
    
    ml_patient = ml_data.get('patient', {}).get('name', '')
    backend_patient = backend_data.get('data', {}).get('patient', {}).get('name', '')
    
    ml_doctor = ml_data.get('doctor', {}).get('name', '')
    backend_doctor = backend_data.get('data', {}).get('doctor', {}).get('name', '')
    
    ml_meds = len(ml_data.get('medicines', []))
    backend_meds = len(backend_data.get('data', {}).get('medicines', []))
    
    ml_conf = ml_data.get('confidence_score', 0)
    backend_conf = backend_data.get('data', {}).get('confidence', 0)
    
    print(f"Patient Name:")
    print(f"  ML Service:  '{ml_patient}'")
    print(f"  Backend:     '{backend_patient}'")
    print(f"  Match: {'✅' if ml_patient == backend_patient else '❌'}")
    
    print(f"\nDoctor Name:")
    print(f"  ML Service:  '{ml_doctor}'")
    print(f"  Backend:     '{backend_doctor}'")
    print(f"  Match: {'✅' if ml_doctor == backend_doctor else '❌'}")
    
    print(f"\nMedicines Count:")
    print(f"  ML Service:  {ml_meds}")
    print(f"  Backend:     {backend_meds}")
    print(f"  Match: {'✅' if ml_meds == backend_meds else '❌'}")
    
    print(f"\nConfidence Score:")
    print(f"  ML Service:  {ml_conf}")
    print(f"  Backend:     {backend_conf}")
    print(f"  Match: {'✅' if abs(ml_conf - backend_conf) < 0.01 else '❌'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_script.py <path_to_prescription_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Testing with image: {image_path}\n")
    
    # Test ML Service
    ml_data = test_ml_service(image_path)
    
    # Test Backend
    backend_data = test_backend(image_path)
    
    # Compare
    compare_results(ml_data, backend_data)
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    
    if ml_data and ml_data.get('success'):
        if backend_data and backend_data.get('success'):
            print("✅ Both services working!")
            print("   Issue is likely in FRONTEND display logic")
            print("\n   Check: frontend/src/App.js")
            print("   - Look for how it handles response.data")
            print("   - Verify field names match")
        else:
            print("❌ ML Service works but Backend fails")
            print("   Issue is in Go backend service")
            print("\n   Check: backend/internal/handlers/prescription_handler.go")
            print("   - Verify ML service URL is correct")
            print("   - Check data conversion functions")
    else:
        print("❌ ML Service is failing")
        print("   Issue is in Python ML service")
        print("\n   Check: backend/main.py or ml-service/app/main.py")
        print("   - Verify analyzer initialization")
        print("   - Check file processing logic")