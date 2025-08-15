#!/usr/bin/env python3
"""
Simple trading signal generator using trained models
This script uses the exact features the model was trained on
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger

def generate_simple_signal():
    """Generate a trading signal using the trained model"""
    
    print("🚀 Generating Trading Signal with Trained AI Model!")
    print("=" * 60)
    
    try:
        # Initialize trading assistant (will auto-load models)
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Check if models are loaded
        if not hasattr(assistant.ml_model, 'training_history') or not assistant.ml_model.training_history:
            print("❌ No trained models found!")
            print("   Please run: python train_first_model.py")
            return False
        
        print("✅ Models loaded successfully!")
        print(f"   Features expected: {len(assistant.ml_model.training_history['feature_names'])}")
        
        # Collect market data
        print("\n📈 Collecting Market Data...")
        market_data, market_info = assistant.collect_market_data(hours_back=168)
        print(f"   ✅ Collected {len(market_data)} candles")
        print(f"   💰 Current BTC Price: ${market_info['current_price']:,.2f}")
        print(f"   📊 24h Change: {market_info['price_change_24h']:.2f}%")
        
        # Collect sentiment
        print("\n😊 Collecting Sentiment Data...")
        sentiment_data = assistant.collect_sentiment_data(hours_back=24)
        print(f"   ✅ Sentiment: {sentiment_data['overall_sentiment']}")
        print(f"   📊 Score: {sentiment_data['combined_sentiment_score']:.3f}")
        
        # Prepare features using only what the model expects
        print("\n🔧 Preparing Features...")
        expected_features = assistant.ml_model.training_history['feature_names']
        
        # Get the latest data point with only expected features
        latest_data = market_data.iloc[-1:][expected_features]
        
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
        atr = market_data['atr'].iloc[-1]
        
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
        print("\n" + "=" * 60)
        print("🎯 TRADING SIGNAL GENERATED!")
        print("=" * 60)
        
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
        print(f"   😊 Sentiment: {sentiment_data['overall_sentiment']}")
        print(f"   🔢 RSI: {market_data['rsi'].iloc[-1]:.1f}")
        print(f"   📊 MACD: {market_data['macd'].iloc[-1]:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 Signal Generation Complete!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating signal: {e}")
        logger.error(f"Signal generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - Simple Signal Generator")
    print("=" * 70)
    
    success = generate_simple_signal()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. ✅ Signal generated successfully")
        print("2. 🔄 Run again for updated signals")
        print("3. 📊 Monitor performance over time")
        print("4. 🚀 Consider setting up continuous monitoring")
    else:
        print("\n⚠️  Troubleshooting:")
        print("1. ❌ Signal generation failed")
        print("2. 🔍 Check the error message above")
        print("3. 📝 Verify your API keys are working")
        print("4. 🔄 Try running again")
