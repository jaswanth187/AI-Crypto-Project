#!/usr/bin/env python3
"""
Script to train the first ML model with live Binance data
Run this before using the trading assistant
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger

def train_first_model():
    """Train the first ML model with live data"""
    
    print("🚀 Training Your First AI Trading Model!")
    print("=" * 50)
    
    try:
        # Initialize trading assistant
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Train models with live data
        print("\n🧠 Training ML Models...")
        print("   This will collect ~6 weeks of historical data")
        print("   and train both classification and regression models")
        
        results = assistant.train_models()
        
        print("\n✅ Model Training Completed Successfully!")
        print("=" * 50)
        
        # Show training results
        print("\n📈 Training Results:")
        print(f"   🎯 Model Type: {results['classification']['model_type']}")
        print(f"   📊 Classification Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"   📊 Classification F1-Score: {results['classification']['f1_score']:.4f}")
        print(f"   📊 Regression R² Score: {results['regression']['r2_score']:.4f}")
        print(f"   📊 Regression RMSE: {results['regression']['rmse']:.6f}")
        print(f"   🔢 Features Used: {len(results['feature_names'])}")
        print(f"   📅 Training Date: {results['training_date']}")
        
        print("\n🎉 Your AI Trading Model is Ready!")
        print("   You can now run: python trading_assistant.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error training model: {e}")
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - First Model Training")
    print("=" * 60)
    
    success = train_first_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 Next Steps:")
        print("1. ✅ Model trained successfully")
        print("2. 🚀 Run: python trading_assistant.py")
        print("3. 📊 Generate your first trading signal!")
        print("4. 🔄 Set up continuous monitoring")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Troubleshooting:")
        print("1. ❌ Model training failed")
        print("2. 🔍 Check the error message above")
        print("3. 📝 Verify your API keys are working")
        print("4. 🔄 Try running again")
