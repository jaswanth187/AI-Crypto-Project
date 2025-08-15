#!/usr/bin/env python3
"""
Aggressive trading signal generator that forces BUY/SELL decisions
This will give you actual trading opportunities with realistic targets
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger
import numpy as np

def generate_aggressive_signal():
    """Generate an aggressive trading signal that forces decisions"""
    
    print("🚀 Generating Aggressive Trading Signal!")
    print("=" * 60)
    
    try:
        # Initialize trading assistant
        print("📊 Initializing Trading Assistant...")
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Check if models are loaded
        if not hasattr(assistant.ml_model, 'training_history') or not assistant.ml_model.training_history:
            print("❌ No trained models found!")
            print("   Please run: python fix_realistic_model.py first")
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
        if len(available_features) < len(expected_features) * 0.8:
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
        
        # FORCE a decision - lower confidence threshold to get more signals
        if confidence < 0.4:  # Much lower threshold for aggressive signals
            # Force a decision based on technical indicators
            latest = market_data.iloc[-1]
            
            # Use RSI and MACD to force a decision
            if latest['rsi'] < 35:  # Oversold
                signal = "BUY"
                reasoning = f"Forced BUY: RSI at {latest['rsi']:.1f} indicates oversold conditions"
                direction = 2
            elif latest['rsi'] > 65:  # Overbought
                signal = "SELL"
                reasoning = f"Forced SELL: RSI at {latest['rsi']:.1f} indicates overbought conditions"
                direction = 0
            else:
                # Use MACD to decide
                if latest['macd'] > latest.get('macd_signal', 0):
                    signal = "BUY"
                    reasoning = "Forced BUY: MACD showing bullish momentum"
                    direction = 2
                else:
                    signal = "SELL"
                    reasoning = "Forced SELL: MACD showing bearish momentum"
                    direction = 0
        else:
            # Normal signal generation
            if direction == 2:  # Up
                signal = "BUY"
                reasoning = "AI model predicts upward price movement"
            elif direction == 0:  # Down
                signal = "SELL"
                reasoning = "AI model predicts downward price movement"
            else:  # Neutral - force a decision
                if price_change_pred > 0:
                    signal = "BUY"
                    reasoning = "Forced BUY: Regression model predicts positive movement"
                    direction = 2
                else:
                    signal = "SELL"
                    reasoning = "Forced SELL: Regression model predicts negative movement"
                    direction = 0
        
        # Calculate REALISTIC risk levels based on actual Bitcoin volatility
        current_price = market_info['current_price']
        
        # Calculate realistic ATR
        high_low_range = market_data['high'] - market_data['low']
        high_close_range = abs(market_data['high'] - market_data['close'].shift(1))
        low_close_range = abs(market_data['low'] - market_data['close'].shift(1))
        
        true_range = np.maximum(high_low_range, np.maximum(high_close_range, low_close_range))
        atr_realistic = true_range.rolling(window=14).mean().iloc[-1]
        
        # Convert ATR to percentage
        atr_percentage = (atr_realistic / current_price) * 100
        
        print(f"\n📊 Volatility Analysis:")
        print(f"   📈 ATR: ${atr_realistic:,.2f} ({atr_percentage:.2f}% of price)")
        
        # Calculate realistic targets and stop-losses
        if signal == "BUY":
            # Target: 1.5x ATR above current price (realistic for Bitcoin)
            target_distance = 1.5 * atr_realistic
            target_price = current_price + target_distance
            target_percentage = (target_distance / current_price) * 100
            
            # Stop-loss: 1x ATR below current price
            stop_loss_distance = 1.0 * atr_realistic
            stop_loss = current_price - stop_loss_distance
            stop_loss_percentage = (stop_loss_distance / current_price) * 100
            
        elif signal == "SELL":
            # Target: 1.5x ATR below current price
            target_distance = 1.5 * atr_realistic
            target_price = current_price - target_distance
            target_percentage = (target_distance / current_price) * 100
            
            # Stop-loss: 1x ATR above current price
            stop_loss_distance = 1.0 * atr_realistic
            stop_loss = current_price + stop_loss_distance
            stop_loss_percentage = (stop_loss_distance / current_price) * 100
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 AGGRESSIVE TRADING SIGNAL GENERATED!")
        print("=" * 60)
        
        print(f"📊 Symbol: BTCUSDT")
        print(f"🎯 Signal: {signal}")
        print(f"🎲 Confidence: {confidence:.1%}")
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📈 Target Price: ${target_price:,.2f} ({target_percentage:+.2f}%)")
        print(f"🛑 Stop Loss: ${stop_loss:,.2f} ({stop_loss_percentage:+.2f}%)")
        print(f"📊 Predicted Change: {price_change_pred:.4f}")
        print(f"🧠 Reasoning: {reasoning}")
        
        print(f"\n📊 Market Context:")
        print(f"   📈 24h Change: {market_info['price_change_24h']:.2f}%")
        print(f"   📊 Volume: {market_info['volume_24h']:,.0f}")
        
        # Show key technical indicators if available
        if 'rsi' in market_data.columns:
            rsi_value = market_data['rsi'].iloc[-1]
            print(f"   🔢 RSI: {rsi_value:.1f} {'(Oversold)' if rsi_value < 30 else '(Overbought)' if rsi_value > 70 else '(Neutral)'}")
        
        if 'macd' in market_data.columns:
            print(f"   📊 MACD: {market_data['macd'].iloc[-1]:.4f}")
        
        if 'ema_20' in market_data.columns:
            ema_20 = market_data['ema_20'].iloc[-1]
            price_vs_ema = ((current_price - ema_20) / ema_20) * 100
            print(f"   📈 EMA20: ${ema_20:,.2f} (Price {'above' if price_vs_ema > 0 else 'below'} by {abs(price_vs_ema):.2f}%)")
        
        print(f"\n💡 Risk Management:")
        if stop_loss_percentage > 0:
            risk_reward = target_percentage / stop_loss_percentage
            print(f"   🎯 Risk-Reward Ratio: 1:{risk_reward:.1f}")
        else:
            print(f"   🎯 Risk-Reward Ratio: N/A")
        print(f"   📊 Position Size: Consider 1-2% of portfolio per trade")
        print(f"   ⚠️  Max Loss: {stop_loss_percentage:.2f}% if stop-loss is hit")
        
        print("\n" + "=" * 60)
        print("🎉 Aggressive Signal Generation Complete!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating signal: {e}")
        logger.error(f"Signal generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Trading Assistant - Aggressive Signal Generator")
    print("=" * 70)
    
    success = generate_aggressive_signal()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. ✅ Aggressive signal generated successfully")
        print("2. 🔄 Run again for updated signals")
        print("3. 📊 Monitor performance over time")
        print("4. 💰 Use proper position sizing (1-2% per trade)")
    else:
        print("\n⚠️  Troubleshooting:")
        print("1. ❌ Signal generation failed")
        print("2. 🔍 Check the error message above")
        print("3. 🔄 Try running again")
