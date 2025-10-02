"""
Setup script for House Price Prediction project.
Run this script to set up the environment and verify installation.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def verify_installation():
    """Verify that all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'joblib', 'xgboost', 'jupyter'
    ]
    
    print("\n🔍 Verifying package installation...")
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")
            all_good = False
    
    return all_good

def check_directory_structure():
    """Check if all required directories exist"""
    required_dirs = ['data', 'notebooks', 'src', 'models']
    
    print("\n📁 Checking directory structure...")
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - Missing")
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✅ Created {dir_name}/")

def main():
    """Main setup function"""
    print("🏠 House Price Prediction - Project Setup")
    print("=" * 50)
    
    # Check directory structure
    check_directory_structure()
    
    # Install requirements
    if install_requirements():
        # Verify installation
        if verify_installation():
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run the model training: python src/model_training.py")
            print("2. Open Jupyter notebook: jupyter notebook notebooks/House_Prediction.ipynb")
            print("3. Make predictions: python predict.py")
        else:
            print("\n⚠️ Some packages failed to install. Please check the errors above.")
    else:
        print("\n❌ Setup failed. Please install the requirements manually.")

if __name__ == "__main__":
    main()