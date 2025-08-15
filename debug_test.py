#!/usr/bin/env python3
"""
Debug script to isolate feature engineering issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample data with all required indicators"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    # Generate realistic price data
    base_price = 100
    price_changes = np.random.randn(100) * 0.02
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    sample_data = pd.DataFrame({
        'open': np.array(prices) * (1 + np.random.randn(100) * 0.005),
        'high': np.array(prices) * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': np.array(prices) * (1 - np.abs(np.random.randn(100) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add all required technical indicators
    sample_data['rsi'] = np.random.uniform(20, 80, 100)
    sample_data['macd'] = np.random.randn(100) * 0.001
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
    sample_data['stoch_k'] = np.random.uniform(0, 100, 100)
    sample_data['stoch_d'] = sample_data['stoch_k'].rolling(3).mean()
    
    # Add Williams %R
    sample_data['williams_r'] = np.random.uniform(-100, 0, 100)
    
    # Add CCI
    sample_data['cci'] = np.random.uniform(-100, 100, 100)
    
    # Fill NaN values
    sample_data = sample_data.fillna(method='bfill').fillna(method='ffill')
    
    return sample_data

def test_step_by_step():
    """Test feature engineering step by step"""
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    try:
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        
        print("\n1. Testing price features...")
        df1 = engineer.create_price_features(df)
        print(f"After price features: {df1.shape}")
        
        print("\n2. Testing volume features...")
        df2 = engineer.create_volume_features(df1)
        print(f"After volume features: {df2.shape}")
        
        print("\n3. Testing technical features...")
        df3 = engineer.create_technical_features(df2)
        print(f"After technical features: {df3.shape}")
        
        print("\n4. Testing time features...")
        df4 = engineer.create_time_features(df3)
        print(f"After time features: {df4.shape}")
        
        print("\n5. Testing target variable creation...")
        df5 = engineer.create_target_variable(df4, horizon_hours=4)
        print(f"After target variable: {df5.shape}")
        
        print("\n6. Checking for NaN values...")
        nan_counts = df5.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        if len(nan_columns) > 0:
            print(f"Columns with NaN values: {nan_columns.to_dict()}")
        else:
            print("No NaN values found")
        
        print("\n7. Final dropna...")
        df6 = df5.dropna()
        print(f"Final shape: {df6.shape}")
        
        if len(df6) > 0:
            print("✅ Feature engineering successful!")
            print(f"Final columns: {df6.columns.tolist()}")
        else:
            print("❌ Feature engineering failed - empty dataset")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_step_by_step()
