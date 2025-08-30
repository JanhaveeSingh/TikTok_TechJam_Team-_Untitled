#!/usr/bin/env python3
"""
Download the pre-trained binary UI classifier from AdaT
"""

import requests
import os
from pathlib import Path

def download_model():
    model_url = "https://github.com/sidongfeng/AdaT/raw/main/GUI_classification/output/mobilenet/pytorch_model.bin.20"
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "pytorch_model.bin.20"
    
    if not model_path.exists():
        print("⬇️  Downloading pre-trained model...")
        try:
            response = requests.get(model_url, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Model downloaded: {model_path}")
            else:
                print("❌ Failed to download model (server error)")
                return None
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return None
    else:
        print(f"✅ Model already exists: {model_path}")
    
    return str(model_path)

if __name__ == "__main__":
    download_model()