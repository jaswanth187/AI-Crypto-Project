#!/usr/bin/env python3
"""
 MAIN CRYPTO TRADING SYSTEM
This is your ONE main file to run everything!
"""

from trading_assistant import CryptoTradingAssistant
from loguru import logger
import numpy as np
import time

def main_menu():
    """Main menu for the trading system"""
    print("\n" + "=" * 70)
    print(" CRYPTO AI TRADING ASSISTANT - MAIN SYSTEM")
    print("=" * 70)
    print("1. Generate Trading Signal")
    print("2. Retrain Model with Latest Data")
    print("3. Show Market Analysis")
    print("4. Backtest Model")
    print("5. Continuous Monitoring")
    print("6. Exit")
    print("=" * 70)


def generate_trading_signal():
    """Generate a trading signal based on the model's prediction and confidence."""
    print("\n Generating Trading Signal...")
    print("=" * 50)

    try:
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')

        if not hasattr(assistant.ml_model, 'training_history') or not assistant.ml_model.training_history:
            print(" No trained models found!")
            print("   Please choose option 2 to train models first")
            return

        market_data, market_info = assistant.collect_market_data(hours_back=168)
        sentiment_data = assistant.collect_sentiment_data(hours_back=24)

        features = assistant.prepare_features(market_data, sentiment_data)
        latest_features = features.iloc[-1:][assistant.ml_model.training_history['feature_names']]

        prediction = assistant.ml_model.predict(latest_features)

        direction = prediction['classification'][0]
        confidence = prediction['confidence'][0]
        price_change_pred = prediction['regression'][0]

        if direction == 2:
            signal = "BUY"
            reasoning = "AI model predicts upward price movement."
        elif direction == 0:
            signal = "SELL"
            reasoning = "AI model predicts downward price movement."
        else: # direction == 1 (HOLD)
            # If the model is not confident in direction, use regression to decide.
            if price_change_pred > 0.0005: # Add a small threshold to avoid noise
                signal = "BUY"
                reasoning = "Directional model is neutral, but regression predicts upward movement."
            elif price_change_pred < -0.0005:
                signal = "SELL"
                reasoning = "Directional model is neutral, but regression predicts downward movement."
            else:
                signal = "HOLD"
                reasoning = "Both directional and regression models show no clear signal."


        current_price = market_info['current_price']

        if signal != "HOLD":
            target_price = current_price * (1 + price_change_pred)
            reward = abs(target_price - current_price)
            risk = reward / 2
            if signal == "BUY":
                stop_loss = current_price - risk
            else: # SELL
                stop_loss = current_price + risk
        else:
            # For HOLD, show potential breakout levels based on ATR
            atr = market_data['atr'].iloc[-1]
            target_price = current_price + (1.0 * atr)  # Potential upside
            stop_loss = current_price - (1.0 * atr)     # Potential downside


        print(f"\n TRADING SIGNAL:")
        print(f" Symbol: BTCUSDT")
        print(f" Signal: {signal}")
        print(f" Confidence: {confidence:.1%}")
        print(f" Current Price: ${current_price:,.2f}")
        print(f" Target Price: ${target_price:,.2f}")
        print(f" Stop Loss: ${stop_loss:,.2f}")
        print(f" Reasoning: {reasoning}")
        print(f" Predicted Change: {price_change_pred:.2%}")

        print(f"\n Market Context:")
        print(f"   24h Change: {market_info['price_change_24h']:.2f}%")
        print(f"   Volume: {market_info['volume_24h']:,.0f}")
        print(f"   RSI: {market_data['rsi'].iloc[-1]:.1f}")
        print(f"   MACD: {market_data['macd'].iloc[-1]:.4f}")

    except Exception as e:
        print(f" Error: {e}")

def backtest_model():
    """Run a backtest to evaluate model performance."""
    print("\n Running Backtest...")
    print("=" * 50)
    
    try:
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        if not hasattr(assistant.ml_model, 'training_history') or not assistant.ml_model.training_history:
            print(" No trained models found!")
            print("   Please choose option 3 to train models first")
            return
            
        print("Loading historical data for backtesting...")
        # Fetch a larger dataset for backtesting
        market_data, _ = assistant.collect_market_data(hours_back=1000)
        
        # For backtesting, we need to provide sentiment data to match training
        # Since we can't get historical sentiment easily, we'll use neutral values
        sentiment_data = {
            'combined_sentiment_score': 0.0,
            'cryptopanic': {'sentiment_score': 0.0},
            'fear_greed': {'value': 50}
        }
        
        # Use the same feature engineering process as training
        features_df = assistant.feature_engineer.engineer_all_features(market_data, sentiment_data)
        
        # Align features with what the model was trained on
        model_features = assistant.ml_model.training_history['feature_names']
        
        # Check if all required features are available
        missing_features = set(model_features) - set(features_df.columns)
        if missing_features:
            print(f"❌ Missing features for backtest: {missing_features}")
            print("   This suggests the model was trained with different features.")
            print("   Please retrain the model using option 3.")
            return
            
        # Keep only the features the model was trained on, plus target and close price
        # Exclude target columns that shouldn't be features
        exclude_cols = ['future_price', 'price_change_future', 'target', 'target_regression']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Select only the model features plus target_regression (close is already in model_features)
        features_df = features_df[model_features + ['target_regression']]
        features_df = features_df.dropna()

        # We'll backtest on the last 200 periods of the dataset
        backtest_period = 200
        if len(features_df) < backtest_period:
            print(f" Not enough data for backtesting. Need at least {backtest_period} periods, but found {len(features_df)}.")
            return
            
        test_df = features_df.tail(backtest_period)
        
        results = []
        
        print(f"Backtesting over the last {backtest_period} periods...")
        for i in range(len(test_df)):
            current_features = test_df[model_features].iloc[i:i+1]
            current_price = test_df['close'].iloc[i]  # close is in model_features
            
            prediction_result = assistant.ml_model.predict(current_features)
            predicted_change = prediction_result['regression'][0]
            actual_change = test_df['target_regression'].iloc[i]
            
            predicted_price = current_price * (1 + predicted_change)
            actual_price = current_price * (1 + actual_change)
            
            results.append({
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_change": predicted_change,
                "actual_change": actual_change
            })
            
        # Calculate performance metrics
        predictions = np.array([r['predicted_change'] for r in results])
        actuals = np.array([r['actual_change'] for r in results])
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        predicted_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(predicted_direction == actual_direction) * 100
        
        print(f"Backtesting over the last {backtest_period} periods...")
        for i in range(len(test_df)):
            current_features = test_df[model_features].iloc[i:i+1]
            current_price = test_df['close'].iloc[i]  # close is in model_features
            date_time = test_df.index[i]

            prediction_result = assistant.ml_model.predict(current_features)
            predicted_change = prediction_result['regression'][0]
            actual_change = test_df['target_regression'].iloc[i]
            
            predicted_price = current_price * (1 + predicted_change)
            actual_price = current_price * (1 + actual_change)
            
            signal = "BUY" if predicted_change > 0 else "SELL"

            results.append({
                "date_time": date_time,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_change": predicted_change,
                "actual_change": actual_change,
                "signal": signal
            })
            
        # Calculate performance metrics
        predictions = np.array([r['predicted_change'] for r in results])
        actuals = np.array([r['actual_change'] for r in results])
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        predicted_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(predicted_direction == actual_direction) * 100
        
        print("\n" + "=" * 80)
        print(" Backtest Results")
        print("=" * 80)
        print(f"Mean Absolute Error (MAE): {mae:.4%}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4%}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        print("-" * 80)
        
        # Display the price prediction table for the last 20 periods
        print("\n" + "=" * 80)
        print(" Detailed Price Predictions (Last 20 Periods)")
        print("=" * 80)
        print(f"{ 'Date':<22} | { 'Signal':<8} | { 'Current Price':<18} | { 'Predicted Price':<18} | { 'Actual Price':<18}")
        print("-" * 80)
        
        for r in results[-20:]:
            date_str = r['date_time'].strftime('%Y-%m-%d %H:%M')
            print(f"{date_str:<22} | {r['signal']:<8} | ${r['current_price']:<17,.2f} | ${r['predicted_price']:<17,.2f} | ${r['actual_price']:<17,.2f}")
            
    except Exception as e:
        print(f" Error during backtest: {e}")

def retrain_model():
    """Retrain the model with latest data"""
    print("\n Retraining Model with Latest Data...")
    print("=" * 50)
    
    try:
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Collect fresh data for training
        market_data, _ = assistant.collect_market_data(hours_back=1000)
        sentiment_data = assistant.collect_sentiment_data(hours_back=168)
        
        # Prepare features
        df = assistant.prepare_features(market_data, sentiment_data)
        
        # Train models
        results = assistant.ml_model.train_models(df, test_size=0.2)
        
        # Save models
        assistant.ml_model.save_models(f"{assistant.symbol.lower()}_main_model")
        
        print(" Model retrained successfully!")
        print(f" Classification Accuracy: {results['classification']['accuracy']:.4f}")
        print(f" Regression R Score: {results['regression']['r2_score']:.4f}")
        
    except Exception as e:
        print(f" Error: {e}")

def show_market_analysis():
    """Show current market analysis"""
    print("\n Market Analysis...")
    print("=" * 50)
    
    try:
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Collect market data
        market_data, market_info = assistant.collect_market_data(hours_back=168)
        sentiment_data = assistant.collect_sentiment_data(hours_back=24)
        
        current_price = market_info['current_price']
        latest = market_data.iloc[-1]
        
        print(f" Current BTC Price: ${current_price:,.2f}")
        print(f" 24h Change: {market_info['price_change_24h']:.2f}%")
        print(f" Volume: {market_info['volume_24h']:,.0f}")
        print(f" RSI: {latest['rsi']:.1f}")
        print(f" MACD: {latest['macd']:.4f}")
        print(f" EMA20: ${latest['ema_20']:,.2f}")
        print(f" EMA50: ${latest['ema_50']:,.2f}")
        print(f" Sentiment: {sentiment_data['overall_sentiment']}")
        
    except Exception as e:
        print(f" Error: {e}")

def continuous_monitoring():
    """Run continuous monitoring"""
    print("\n Starting Continuous Monitoring...")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            print(f"\n {time.strftime('%Y-%m-%d %H:%M:%S')}")
            generate_trading_signal()
            print("\n" + "-" * 50)
            time.sleep(300)  # Wait 5 minutes
    except KeyboardInterrupt:
        print("\n Continuous monitoring stopped")

def main():
    """Main function"""
    print(" Welcome to Crypto AI Trading Assistant!")
    print("This is your ONE main system for everything!")

    while True:
        main_menu()
        choice = input("\nChoose an option (1-6): ").strip()

        if choice == '1':
            generate_trading_signal()
        elif choice == '2':
            retrain_model()
        elif choice == '3':
            show_market_analysis()
        elif choice == '4':
            backtest_model()
        elif choice == '5':
            continuous_monitoring()
        elif choice == '6':
            print(" Goodbye!")
            break
        else:
            print(" Invalid choice. Please choose 1-6.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
