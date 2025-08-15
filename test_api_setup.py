#!/usr/bin/env python3
"""
Test script to verify API key setup
Run this after adding your API keys to .env file
"""

import os
from dotenv import load_dotenv
from config import Config

def test_api_keys():
    """Test if all required API keys are properly configured"""
    print("🔑 Testing API Key Configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Test Binance API keys (REQUIRED)
    print("\n📊 Binance API (Required):")
    if Config.BINANCE_API_KEY and Config.BINANCE_API_KEY != "your_binance_api_key_here":
        print("   ✅ API Key: Configured")
    else:
        print("   ❌ API Key: Missing or not configured")
        
    if Config.BINANCE_SECRET_KEY and Config.BINANCE_SECRET_KEY != "your_binance_secret_key_here":
        print("   ✅ Secret Key: Configured")
    else:
        print("   ❌ Secret Key: Missing or not configured")
    
    print(f"   📝 Testnet Mode: {Config.BINANCE_TESTNET}")
    
    # Test CryptoPanic API (Optional)
    print("\n📰 CryptoPanic API (Optional):")
    if Config.CRYPTOPANIC_API_KEY and Config.CRYPTOPANIC_API_KEY != "your_cryptopanic_api_key_here":
        print("   ✅ API Key: Configured")
    else:
        print("   ⚠️  API Key: Not configured (news sentiment will be simulated)")
    
    # Test Twitter API (Optional)
    print("\n🐦 Twitter API (Optional):")
    if Config.TWITTER_BEARER_TOKEN and Config.TWITTER_BEARER_TOKEN != "your_twitter_bearer_token_here":
        print("   ✅ Bearer Token: Configured")
    else:
        print("   ⚠️  Bearer Token: Not configured (Twitter sentiment will be simulated)")
    
    # Test configuration
    print("\n⚙️  Configuration Settings:")
    print(f"   🎯 Model Type: {Config.MODEL_TYPE}")
    print(f"   ⏰ Prediction Horizon: {Config.PREDICTION_HORIZON} hours")
    print(f"   🎲 Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
    print(f"   💰 Default Symbol: {Config.DEFAULT_SYMBOL}")
    print(f"   📈 Timeframe: {Config.TIMEFRAME}")
    
    # Overall status
    print("\n" + "=" * 50)
    if (Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY and 
        Config.BINANCE_API_KEY != "your_binance_api_key_here" and 
        Config.BINANCE_SECRET_KEY != "your_binance_secret_key_here"):
        print("🎉 SUCCESS: All required API keys are configured!")
        print("   You can now run the trading assistant with live data.")
    else:
        print("❌ ERROR: Required Binance API keys are missing!")
        print("   Please add your Binance API keys to the .env file.")
    
    return Config.validate()

def test_binance_connection():
    """Test if we can connect to Binance API"""
    if not Config.validate():
        print("\n⚠️  Skipping Binance connection test due to missing API keys")
        return False
    
    try:
        print("\n🔗 Testing Binance API Connection...")
        from data_collector import BinanceDataCollector
        
        collector = BinanceDataCollector()
        
        # Test getting current price
        price = collector.get_live_price('BTCUSDT')
        print(f"   ✅ BTC Current Price: ${price:,.2f}")
        
        # Test getting 24h stats
        stats = collector.get_24h_stats('BTCUSDT')
        print(f"   ✅ 24h Volume: {stats['volume']:,.0f}")
        print(f"   ✅ 24h Change: {stats['price_change_percent']:.2f}%")
        
        print("🎉 Binance API connection successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Binance API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Crypto Trading Assistant - API Setup Test")
    print("=" * 60)
    
    # Test configuration
    config_ok = test_api_keys()
    
    # Test Binance connection if keys are configured
    if config_ok:
        test_binance_connection()
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    if config_ok:
        print("1. ✅ API keys are configured")
        print("2. 🚀 Run: python trading_assistant.py")
        print("3. 📊 Or test with: python test_system.py")
    else:
        print("1. ❌ Fix missing API keys in .env file")
        print("2. 🔑 Get Binance API keys from binance.com")
        print("3. 📝 Update .env file with your keys")
        print("4. 🔄 Run this test again")
