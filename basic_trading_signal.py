#!/usr/bin/env python3
"""
Basic trading signal generator using only the features the model was trained on
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger

def generate_basic_signal():
    """Generate a trading signal using only basic features"""
    
    print("🚀 Generating Basic Trading Signal!")
    print("=" * 50)
    
    try:
        # Initialize trading assistant
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Check if models are loaded
        if not hasattr(assistant.ml_model, 'training_history') or not assistant.ml_model.training_history:
            print("❌ No trained models found!")
            return False
        
        print("✅ Models loaded successfully!")
        expected_features = assistant.ml_model.training_history['feature_names']
        print(f"   Features expected: {len(expected_features)}")
        
        # Collect market data
        print("\n📈 Collecting Market Data...")
        market_data, market_info = assistant.collect_market_data(hours_back=168)
        print(f"   ✅ Collected {len(market_data)} candles")
        print(f"   💰 Current BTC Price: ${market_info['current_price']:,.2f}")
        print(f"   📊 24h Change: {market_info['price_change_24h']:.2f}%")
        
        # Check which features are available
        available_features = [f for f in expected_features if f in market_data.columns]
        missing_features = [f for f in expected_features if f not in market_data.columns]
        
        print(f"\n🔍 Feature Analysis:")
        print(f"   ✅ Available: {len(available_features)}")
        print(f"   ❌ Missing: {len(missing_features)}")
        
        if len(missing_features) > 0:
            print(f"   Missing features: {missing_features[:5]}...")
        
        # Use only available features
        if len(available_features) < len(expected_features) * 0.8:  # Need at least 80%
            print(f"\n⚠️  Warning: Only {len(available_features)}/{len(expected_features)} features available")
            print("   This may affect prediction accuracy")
        
        # Get latest data with available features
        latest_data = market_data.iloc[-1:][available_features]
        
        # Make prediction
        print("\n🧠 Making AI Prediction...")
        prediction = assistant.ml_model.predict(latest_data)
        
        # Extract results
        direction = prediction['classification'][0]
        confidence = prediction['confidence'][0]
        price_change_pred = prediction['regression'][0]
        
        # Generate signal
        if confidence < 0.7:
            signal = "HOLD"
            reasoning = f"Low confidence ({confidence:.1%}) - insufficient signal strength"
        else:
            if direction == 1:
                signal = "BUY"
                reasoning = "AI model predicts upward price movement"
            elif direction == 0:
                signal = "HOLD"
                reasoning = "AI model predicts neutral market conditions"
            else:
                signal = "SELL"
                reasoning = "AI model predicts downward price movement"
        
        # Calculate risk levels
        current_price = market_info['current_price']
        atr = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else current_price * 0.02
        
        if direction == 1:  # BUY
            target_price = current_price + (2 * atr)
            stop_loss = current_price - atr
        elif direction == -1:  # SELL
            target_price = current_price - (2 * atr)
            stop_loss = current_price + atr
        else:  # HOLD
            target_price = current_price
            stop_loss = current_price
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 TRADING SIGNAL GENERATED!")
        print("=" * 50)
        
        print(f"📊 Symbol: BTCUSDT")
        print(f"🎯 Signal: {signal}")
        print(f"🎲 Confidence: {confidence:.1%}")
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📈 Target Price: ${target_price:,.2f}")
        print(f"🛑 Stop Loss: ${stop_loss:,.2f}")
        print(f"📊 Predicted Change: {price_change_pred:.4f}")
        print(f"🧠 Reasoning: {reasoning}")
        
        print(f"\n📊 Market Context:")
        print(f"   📈 24h Change: {market_info['price_change_24h']:.2f}%")
        print(f"   📊 Volume: {market_info['volume_24h']:,.0f}")
        
        # Show key technical indicators if available
        if 'rsi' in market_data.columns:
            print(f"   🔢 RSI: {market_data['rsi'].iloc[-1]:.1f}")
        if 'macd' in market_data.columns:
            print(f"   📊 MACD: {market_data['macd'].iloc[-1]:.4f}")
        if 'ema_20' in market_data.columns:
            print(f"   📈 EMA20: ${market_data['ema_20'].iloc[-1]:.2f}")
        
        print("\n" + "=" * 50)
        print("🎉 Signal Generation Complete!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating signal: {e}")
        logger.error(f"Signal generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - Basic Signal Generator")
    print("=" * 60)
    
    success = generate_basic_signal()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. ✅ Signal generated successfully")
        print("2. 🔄 Run again for updated signals")
        print("3. 📊 Monitor performance over time")
    else:
        print("\n⚠️  Troubleshooting:")
        print("1. ❌ Signal generation failed")
        print("2. 🔍 Check the error message above")
        print("3. 🔄 Try running again")
