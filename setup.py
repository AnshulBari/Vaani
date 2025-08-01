#!/usr/bin/env python3
"""
Vaani LLM Setup Script

Quick setup script for new users to get started with Vaani LLM.
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'numpy', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")
    
    return missing

def install_dependencies(missing):
    """Install missing dependencies"""
    if not missing:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_model_files():
    """Check if model files exist"""
    files = {
        'model.pth': 'Trained model weights (1.63 GB)',
        'tokenizer.pkl': 'Tokenizer vocabulary',
        'config.json': 'Model configuration'
    }
    
    missing = []
    for filename, description in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024*1024)
            print(f"✅ {filename} ({size:.1f} MB) - {description}")
        else:
            print(f"❌ {filename} missing - {description}")
            missing.append(filename)
    
    return missing

def main():
    print("🤖 Vaani LLM Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        install_deps = input(f"\nInstall missing dependencies? (y/N): ").strip().lower()
        if install_deps in ['y', 'yes']:
            if not install_dependencies(missing_deps):
                return False
        else:
            print("❌ Cannot proceed without dependencies")
            return False
    
    print(f"\n📁 Checking model files...")
    missing_files = check_model_files()
    
    if missing_files:
        print(f"\n⚠️  Missing model files: {', '.join(missing_files)}")
        print(f"\nTo get the pre-trained model:")
        print(f"1. Run: python download_model.py")
        print(f"2. Or download manually from: https://github.com/AnshulBari/Vaani/releases")
        print(f"3. Or train your own: python train.py")
        return False
    
    print(f"\n🎉 Setup complete! Ready to use Vaani LLM")
    print(f"\nNext steps:")
    print(f"   python analyze_model.py    # Check model specs")
    print(f"   python test_local_model.py # Test the model")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
