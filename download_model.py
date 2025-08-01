#!/usr/bin/env python3
"""
Vaani LLM Model Downloader

This script downloads the pre-trained Vaani LLM model files.
Run this after cloning the repository to get the trained model.
"""

import os
import requests
import sys
from pathlib import Path
from tqdm import tqdm

# Model file URLs (update these with your actual URLs)
MODEL_FILES = {
    "model.pth": {
        "url": "https://github.com/AnshulBari/Vaani/releases/download/v1.1.0/model.pth",
        "size": "1.63 GB",
        "description": "Trained Vaani LLM weights (427M parameters)"
    },
    "tokenizer.pkl": {
        "url": "https://github.com/AnshulBari/Vaani/releases/download/v1.1.0/tokenizer.pkl", 
        "size": "~1 MB",
        "description": "Trained tokenizer with 262-word vocabulary"
    },
    "config.json": {
        "url": "https://github.com/AnshulBari/Vaani/releases/download/v1.1.0/config.json",
        "size": "~1 KB", 
        "description": "Model configuration parameters"
    }
}

def download_file(url, filename, description):
    """Download a file with progress bar"""
    print(f"\n📥 Downloading {filename} ({description})")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"✅ Successfully downloaded {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def check_existing_files():
    """Check which files already exist"""
    existing = []
    missing = []
    
    for filename in MODEL_FILES.keys():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024*1024)  # MB
            existing.append(f"{filename} ({size:.1f} MB)")
        else:
            missing.append(filename)
    
    return existing, missing

def main():
    print("🤖 Vaani LLM Model Downloader")
    print("=" * 50)
    
    # Check existing files
    existing, missing = check_existing_files()
    
    if existing:
        print(f"\n✅ Found existing files:")
        for file in existing:
            print(f"   - {file}")
    
    if not missing:
        print("\n🎉 All model files are already present!")
        print("You can now run: python analyze_model.py")
        return
    
    print(f"\n📋 Files to download:")
    total_size = 0
    for filename in missing:
        info = MODEL_FILES[filename]
        print(f"   - {filename} ({info['size']}) - {info['description']}")
    
    # Confirm download
    print(f"\n⚠️  Total download size: ~1.65 GB")
    response = input("Continue with download? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return
    
    # Download missing files
    success_count = 0
    for filename in missing:
        info = MODEL_FILES[filename]
        if download_file(info['url'], filename, info['description']):
            success_count += 1
        else:
            print(f"⚠️  Skipping {filename} due to download error")
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{len(missing)} files")
    
    if success_count == len(missing):
        print(f"\n🎉 All files downloaded successfully!")
        print(f"You can now run:")
        print(f"   python analyze_model.py")
        print(f"   python test_local_model.py")
    else:
        print(f"\n⚠️  Some files failed to download. Please try again or download manually from:")
        print(f"   https://github.com/AnshulBari/Vaani/releases/tag/v1.1.0")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
