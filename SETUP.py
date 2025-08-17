#!/usr/bin/env python3
"""
🔧 SETUP SCRIPT - Run this first to set up your trading system
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements. Please run: pip install -r requirements.txt")

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("\n⚠️  No .env file found!")
        print("📝 Please create a .env file with your API keys:")
        print("   Copy env_example.txt to .env and add your keys")
        return False
    else:
        print("✅ .env file found!")
        return True

def main():
    """Main setup function"""
    print("🔧 CRYPTO AI TRADING SYSTEM SETUP")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check .env file
    print("\n🔑 Checking API configuration...")
    env_ok = check_env_file()
    
    # Install requirements
    print("\n📦 Installing packages...")
    install_requirements()
    
    print("\n" + "=" * 50)
    if env_ok:
        print("🎉 Setup complete! You can now run:")
        print("   python MAIN_TRADING_SYSTEM.py")
    else:
        print("⚠️  Setup almost complete!")
        print("   Please create your .env file first, then run:")
        print("   python MAIN_TRADING_SYSTEM.py")

if __name__ == "__main__":
    main()

