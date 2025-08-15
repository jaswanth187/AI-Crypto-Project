#!/usr/bin/env python3
"""
Test script for the Crypto Trading Assistant
This script tests all major components without requiring API keys
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import Config
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ Feature engineering module imported successfully")
    except ImportError as e:
        print(f"❌ Feature engineering import failed: {e}")
        return False
    
    try:
        from ml_model import CryptoMLModel
        print("✅ ML model module imported successfully")
    except ImportError as e:
        print(f"❌ ML model import failed: {e}")
        return False
    
    return True

def test_feature_engineering():
    """Test feature engineering with sample data"""
    print("\nTesting feature engineering...")
    
    try:
        from feature_engineering import FeatureEngineer
        
        # Create sample data with more realistic values
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        
        # Generate more realistic price data
        base_price = 100
        price_changes = np.random.randn(200) * 0.02  # 2% volatility
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        sample_data = pd.DataFrame({
            'open': np.array(prices) * (1 + np.random.randn(200) * 0.005),
            'high': np.array(prices) * (1 + np.abs(np.random.randn(200) * 0.01)),
            'low': np.array(prices) * (1 - np.abs(np.random.randn(200) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        # Add all required technical indicators
        sample_data['rsi'] = np.random.uniform(20, 80, 200)
        sample_data['macd'] = np.random.randn(200) * 0.001
        sample_data['macd_signal'] = sample_data['macd'].rolling(9).mean()
        sample_data['macd_histogram'] = sample_data['macd'] - sample_data['macd_signal']
        
        sample_data['ema_20'] = sample_data['close'].rolling(20).mean()
        sample_data['ema_50'] = sample_data['close'].rolling(50).mean()
        sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
        sample_data['sma_50'] = sample_data['close'].rolling(50).mean()
        
        # Add Bollinger Bands
        sample_data['bollinger_upper'] = sample_data['close'].rolling(20).mean() + 2 * sample_data['close'].rolling(20).std()
        sample_data['bollinger_lower'] = sample_data['close'].rolling(20).mean() - 2 * sample_data['close'].rolling(20).std()
        sample_data['bollinger_middle'] = sample_data['close'].rolling(20).mean()
        
        # Add ATR
        sample_data['atr'] = sample_data['close'].rolling(14).std()
        
        # Add Stochastic
        sample_data['stoch_k'] = np.random.uniform(0, 100, 200)
        sample_data['stoch_d'] = sample_data['stoch_k'].rolling(3).mean()
        
        # Add Williams %R
        sample_data['williams_r'] = np.random.uniform(-100, 0, 200)
        
        # Add CCI
        sample_data['cci'] = np.random.uniform(-100, 100, 200)
        
        # Fill NaN values with reasonable defaults
        sample_data = sample_data.fillna(method='bfill').fillna(method='ffill')
        
        # Test feature engineering
        engineer = FeatureEngineer()
        # Use a shorter horizon for testing to avoid dropping all data
        engineered_data = engineer.engineer_all_features(sample_data, horizon_hours=4)
        
        print(f"✅ Feature engineering completed: {engineered_data.shape}")
        print(f"   Features created: {len(engineered_data.columns)}")
        
        if len(engineered_data) == 0:
            print("⚠️  Warning: Engineered data is empty, this may cause issues")
            return False
        
        # Test feature selection
        selected_data = engineer.select_features(engineered_data, k=30)
        print(f"✅ Feature selection completed: {selected_data.shape}")
        
        # Test feature scaling
        scaled_data = engineer.scale_features(selected_data)
        print(f"✅ Feature scaling completed: {scaled_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_ml_model():
    """Test ML model with sample data"""
    print("\nTesting ML model...")
    
    try:
        from ml_model import CryptoMLModel
        
        # Create sample data with targets
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        
        # Generate more realistic price data
        base_price = 100
        price_changes = np.random.randn(500) * 0.02
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        sample_data = pd.DataFrame({
            'open': np.array(prices) * (1 + np.random.randn(500) * 0.005),
            'high': np.array(prices) * (1 + np.abs(np.random.randn(500) * 0.01)),
            'low': np.array(prices) * (1 - np.abs(np.random.randn(500) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Add all required technical indicators
        sample_data['rsi'] = np.random.uniform(20, 80, 500)
        sample_data['macd'] = np.random.randn(500) * 0.001
        sample_data['macd_signal'] = sample_data['macd'].rolling(9).mean()
        sample_data['macd_histogram'] = sample_data['macd'] - sample_data['macd_signal']
        
        sample_data['ema_20'] = sample_data['close'].rolling(20).mean()
        sample_data['ema_50'] = sample_data['close'].rolling(50).mean()
        sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
        sample_data['sma_50'] = sample_data['close'].rolling(50).mean()
        
        # Add Bollinger Bands
        sample_data['bollinger_upper'] = sample_data['close'].rolling(20).mean() + 2 * sample_data['close'].rolling(20).std()
        sample_data['bollinger_lower'] = sample_data['close'].rolling(20).mean() - 2 * sample_data['close'].rolling(20).std()
        sample_data['bollinger_middle'] = sample_data['close'].rolling(20).mean()
        
        # Add ATR
        sample_data['atr'] = sample_data['close'].rolling(14).std()
        
        # Add Stochastic
        sample_data['stoch_k'] = np.random.uniform(0, 100, 500)
        sample_data['stoch_d'] = sample_data['stoch_k'].rolling(3).mean()
        
        # Add Williams %R
        sample_data['williams_r'] = np.random.uniform(-100, 0, 500)
        
        # Add CCI
        sample_data['cci'] = np.random.uniform(-100, 100, 500)
        
        # Fill NaN values
        sample_data = sample_data.fillna(method='bfill').fillna(method='ffill')
        
        # Create target variables
        sample_data['target'] = np.random.choice([-1, 0, 1], 500)
        sample_data['target_regression'] = np.random.randn(500) * 0.02
        
        # Test feature engineering first
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        # Use a shorter horizon for testing to avoid dropping all data
        engineered_data = engineer.engineer_all_features(sample_data, horizon_hours=4)
        
        if len(engineered_data) == 0:
            print("❌ Feature engineering produced empty dataset")
            return False
        
        # Test ML model
        model = CryptoMLModel('xgboost')
        results = model.train_models(engineered_data, test_size=0.2)
        
        print(f"✅ ML model training completed")
        print(f"   Classification accuracy: {results['classification']['accuracy']:.4f}")
        print(f"   Regression R²: {results['regression']['r2_score']:.4f}")
        
        # Test prediction
        test_features = engineered_data.iloc[-5:][model.training_history['feature_names']]
        predictions = model.predict(test_features)
        
        print(f"✅ Prediction test completed")
        print(f"   Predictions: {predictions['classification']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model test failed: {e}")
        return False

def test_config():
    """Test configuration module"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        # Test config validation (will fail without API keys, which is expected)
        validation_result = Config.validate()
        print(f"✅ Config validation: {'Passed' if validation_result else 'Failed (expected without API keys)'}")
        
        # Test config attributes
        print(f"   Model type: {Config.MODEL_TYPE}")
        print(f"   Default symbol: {Config.DEFAULT_SYMBOL}")
        print(f"   Timeframe: {Config.TIMEFRAME}")
        print(f"   Technical indicators: {len(Config.TECHNICAL_INDICATORS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def create_sample_env():
    """Create a sample .env file"""
    print("\nCreating sample .env file...")
    
    env_content = """# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=True

# CryptoPanic API (Optional - for news sentiment)
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key_here

# Twitter API (Optional - for social sentiment)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Model Configuration
MODEL_TYPE=xgboost
PREDICTION_HORIZON=24
CONFIDENCE_THRESHOLD=0.7

# Data Configuration
DEFAULT_SYMBOL=BTCUSDT
TIMEFRAME=1h
LOOKBACK_PERIOD=1000

# Sentiment Analysis
SENTIMENT_LOOKBACK_HOURS=24
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Sample .env file created")
        print("   Please edit .env with your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Crypto Trading Assistant - System Test")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Feature Engineering Test", test_feature_engineering),
        ("ML Model Test", test_ml_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    # Create sample .env file
    create_sample_env()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python trading_assistant.py")
        print("3. Or test individual components:")
        print("   python data_collector.py")
        print("   python sentiment_collector.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements_simple.txt")
    
    print("\n📚 Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
