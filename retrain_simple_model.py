#!/usr/bin/env python3
"""
Retrain the model using only basic technical indicators
This ensures compatibility with current data collection
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger

def retrain_simple_model():
    """Retrain model with basic features only"""
    
    print("🔄 Retraining Model with Basic Features Only!")
    print("=" * 50)
    
    try:
        # Initialize trading assistant
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Collect market data
        print("\n📈 Collecting Market Data...")
        market_data, market_info = assistant.collect_market_data(hours_back=1000)
        print(f"   ✅ Collected {len(market_data)} candles")
        
        # Check what features we actually have
        print(f"\n🔍 Available Features:")
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema_20', 'ema_50', 'sma_20', 'sma_50', 'atr']
        available_features = [f for f in basic_features if f in market_data.columns]
        
        for feature in available_features:
            print(f"   ✅ {feature}")
        
        print(f"\n📊 Total features available: {len(available_features)}")
        
        # Create simple target variables
        print("\n🎯 Creating Target Variables...")
        
        # Classification target: 0 = down, 1 = neutral, 2 = up (XGBoost compatible)
        market_data['target'] = 1  # Default to neutral
        market_data.loc[market_data['close'].pct_change(1) > 0.01, 'target'] = 2  # Up if >1% gain
        market_data.loc[market_data['close'].pct_change(1) < -0.01, 'target'] = 0  # Down if >1% loss
        
        # Regression target: actual price change percentage
        market_data['target_regression'] = market_data['close'].pct_change(1)
        
        # Remove NaN values
        market_data = market_data.dropna()
        
        print(f"   ✅ Target variables created")
        print(f"   📊 Final dataset shape: {market_data.shape}")
        
        # Train models with basic features only
        print("\n🧠 Training Models with Basic Features...")
        
        # Use only the features we have
        feature_cols = available_features + ['target', 'target_regression']
        training_data = market_data[feature_cols].dropna()
        
        print(f"   📊 Training on {len(training_data)} samples")
        print(f"   🔢 Using {len(available_features)} features")
        
        # Train the models
        results = assistant.ml_model.train_models(training_data, test_size=0.2)
        
        # Save models
        assistant.ml_model.save_models(f"{assistant.symbol.lower()}_simple_model")
        
        print("\n✅ Model Retraining Completed Successfully!")
        print("=" * 50)
        
        # Show results
        print("\n📈 Training Results:")
        print(f"   🎯 Model Type: {assistant.ml_model.model_type}")
        print(f"   📊 Classification Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"   📊 Classification F1-Score: {results['classification']['f1_score']:.4f}")
        print(f"   📊 Regression R² Score: {results['regression']['r2_score']:.4f}")
        print(f"   📊 Regression RMSE: {results['regression']['rmse']:.6f}")
        print(f"   🔢 Features Used: {len(results['feature_names'])}")
        
        print("\n🎉 Your Simple AI Model is Ready!")
        print("   You can now run: python basic_trading_signal.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error retraining model: {e}")
        logger.error(f"Retraining failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - Simple Model Retraining")
    print("=" * 60)
    
    success = retrain_simple_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 Next Steps:")
        print("1. ✅ Model retrained successfully")
        print("2. 🚀 Run: python basic_trading_signal.py")
        print("3. 📊 Generate your first trading signal!")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Troubleshooting:")
        print("1. ❌ Model retraining failed")
        print("2. 🔍 Check the error message above")
        print("3. 🔄 Try running again")
