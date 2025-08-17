#!/usr/bin/env python3
"""
Debug script to understand feature mismatch
"""

from trading_assistant import CryptoTradingAssistant
import pandas as pd

def debug_features():
    """Debug feature mismatch between training and prediction"""
    print("🔍 Debugging Feature Mismatch...")
    print("=" * 50)
    
    try:
        # Initialize assistant
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Check what features the model expects
        if hasattr(assistant.ml_model, 'training_history') and assistant.ml_model.training_history:
            expected_features = assistant.ml_model.training_history['feature_names']
            print(f"📊 Model expects {len(expected_features)} features:")
            print(f"   Expected features: {expected_features}")
        else:
            print("❌ No training history found")
            return
        
        # Get current data and engineer features
        print("\n🔄 Generating current features...")
        market_data, _ = assistant.collect_market_data(hours_back=100)
        
        sentiment_data = {
            'combined_sentiment_score': 0.0,
            'cryptopanic': {'sentiment_score': 0.0},
            'twitter': {'engagement_sentiment': 0.0},
            'fear_greed': {'value': 50}
        }
        
        # Engineer features
        features_df = assistant.feature_engineer.engineer_all_features(market_data, sentiment_data)
        
        print(f"📊 Current feature engineering produces {len(features_df.columns)} features:")
        print(f"   Current features: {list(features_df.columns)}")
        
        # Check for missing features
        missing_features = set(expected_features) - set(features_df.columns)
        extra_features = set(features_df.columns) - set(expected_features)
        
        print(f"\n🔍 Analysis:")
        print(f"   Missing features: {missing_features}")
        print(f"   Extra features: {extra_features}")
        
        if missing_features:
            print(f"❌ Missing {len(missing_features)} features that model expects")
        if extra_features:
            print(f"⚠️  Extra {len(extra_features)} features that model doesn't expect")
        
        # Test the exact feature selection that would be used in backtest
        print(f"\n🧪 Testing feature selection...")
        exclude_cols = ['future_price', 'price_change_future', 'target', 'target_regression']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Select only the model features plus close price for calculations
        selected_features = expected_features + ['close', 'target_regression']
        test_df = features_df[selected_features]
        
        print(f"📊 After selection: {len(test_df.columns)} features")
        print(f"   Selected features: {list(test_df.columns)}")
        
        # Check if this matches what the model expects
        model_features_only = test_df[expected_features]
        print(f"📊 Model features only: {len(model_features_only.columns)} features")
        
        if len(model_features_only.columns) == len(expected_features):
            print("✅ Feature count matches!")
        else:
            print(f"❌ Feature count mismatch: {len(model_features_only.columns)} vs {len(expected_features)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_features()
