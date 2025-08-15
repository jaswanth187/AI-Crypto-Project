#!/usr/bin/env python3
"""
Fix the model with realistic Bitcoin volatility patterns
This will create a much more sensible trading model
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger
import numpy as np

def create_realistic_model():
    """Create a realistic model with proper risk management"""
    
    print("🔧 Fixing Model with Realistic Bitcoin Patterns!")
    print("=" * 60)
    
    try:
        # Initialize trading assistant
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Collect market data
        print("\n📈 Collecting Market Data...")
        market_data, market_info = assistant.collect_market_data(hours_back=1000)
        print(f"   ✅ Collected {len(market_data)} candles")
        
        # Analyze actual Bitcoin volatility patterns
        print("\n📊 Analyzing Bitcoin Volatility Patterns...")
        
        # Calculate realistic daily movements
        market_data['daily_return'] = market_data['close'].pct_change(24)  # 24-hour returns
        market_data['hourly_return'] = market_data['close'].pct_change(1)  # 1-hour returns
        
        # Get volatility statistics
        daily_volatility = market_data['daily_return'].std()
        hourly_volatility = market_data['hourly_return'].std()
        
        print(f"   📊 Daily Volatility: {daily_volatility:.2%}")
        print(f"   📊 Hourly Volatility: {hourly_volatility:.2%}")
        
        # Calculate realistic ATR (Average True Range)
        market_data['true_range'] = np.maximum(
            market_data['high'] - market_data['low'],
            np.maximum(
                abs(market_data['high'] - market_data['close'].shift(1)),
                abs(market_data['low'] - market_data['close'].shift(1))
            )
        )
        market_data['atr_realistic'] = market_data['true_range'].rolling(window=14).mean()
        
        # Create realistic target variables
        print("\n🎯 Creating Realistic Target Variables...")
        
        # Classification: Use smaller thresholds for more realistic signals
        # 0 = Down, 1 = Neutral, 2 = Up
        market_data['target'] = 1  # Default to neutral
        
        # Use 0.5% threshold for hourly movements (more realistic)
        market_data.loc[market_data['hourly_return'] > 0.005, 'target'] = 2  # Up if >0.5%
        market_data.loc[market_data['hourly_return'] < -0.005, 'target'] = 0  # Down if >0.5%
        
        # Regression: Use actual hourly returns
        market_data['target_regression'] = market_data['hourly_return']
        
        # Remove NaN values
        market_data = market_data.dropna()
        
        print(f"   ✅ Target variables created")
        print(f"   📊 Final dataset shape: {market_data.shape}")
        
        # Check class distribution
        class_counts = market_data['target'].value_counts().sort_index()
        print(f"   📊 Class Distribution:")
        for class_val, count in class_counts.items():
            percentage = (count / len(market_data)) * 100
            print(f"      Class {class_val}: {count} ({percentage:.1f}%)")
        
        # Select only essential features
        print("\n🔍 Selecting Essential Features...")
        essential_features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'ema_20', 'ema_50', 'sma_20', 'sma_50',
            'atr_realistic', 'daily_return', 'hourly_return'
        ]
        
        available_features = [f for f in essential_features if f in market_data.columns]
        print(f"   ✅ Using {len(available_features)} essential features:")
        for feature in available_features:
            print(f"      - {feature}")
        
        # Train models with realistic features
        print("\n🧠 Training Realistic Models...")
        
        feature_cols = available_features + ['target', 'target_regression']
        training_data = market_data[feature_cols].dropna()
        
        print(f"   📊 Training on {len(training_data)} samples")
        print(f"   🔢 Using {len(available_features)} features")
        
        # Train the models
        results = assistant.ml_model.train_models(training_data, test_size=0.2)
        
        # Save models with realistic name
        assistant.ml_model.save_models(f"{assistant.symbol.lower()}_realistic_model")
        
        print("\n✅ Realistic Model Created Successfully!")
        print("=" * 60)
        
        # Show results
        print("\n📈 Training Results:")
        print(f"   🎯 Model Type: {assistant.ml_model.model_type}")
        print(f"   📊 Classification Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"   📊 Classification F1-Score: {results['classification']['f1_score']:.4f}")
        print(f"   📊 Regression R² Score: {results['regression']['r2_score']:.4f}")
        print(f"   📊 Regression RMSE: {results['regression']['rmse']:.6f}")
        print(f"   🔢 Features Used: {len(results['feature_names'])}")
        
        print("\n🎉 Your Realistic AI Model is Ready!")
        print("   You can now run: python realistic_trading_signal.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating realistic model: {e}")
        logger.error(f"Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - Realistic Model Fix")
    print("=" * 70)
    
    success = create_realistic_model()
    
    if success:
        print("\n" + "=" * 70)
        print("🎯 Next Steps:")
        print("1. ✅ Realistic model created successfully")
        print("2. 🚀 Run: python realistic_trading_signal.py")
        print("3. 📊 Get sensible trading signals!")
        print("4. 🎯 Realistic targets and stop-losses")
    else:
        print("\n" + "=" * 70)
        print("⚠️  Troubleshooting:")
        print("1. ❌ Model creation failed")
        print("2. 🔍 Check the error message above")
        print("3. 🔄 Try running again")
