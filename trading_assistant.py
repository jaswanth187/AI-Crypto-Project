import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from loguru import logger
from config import Config
from data_collector import BinanceDataCollector
from sentiment_collector import SentimentCollector
from feature_engineering import FeatureEngineer
from ml_model import CryptoMLModel

class CryptoTradingAssistant:
    def __init__(self, symbol=Config.DEFAULT_SYMBOL, model_type='xgboost'):
        """
        Initialize the Crypto Trading Assistant
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            model_type (str): ML model type ('xgboost', 'lightgbm', 'random_forest')
        """
        self.symbol = symbol
        self.model_type = model_type
        
        # Initialize components
        self.data_collector = BinanceDataCollector()
        self.sentiment_collector = SentimentCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = CryptoMLModel(model_type)
        
        # Current state
        self.current_data = None
        self.current_sentiment = None
        self.current_prediction = None
        self.last_update = None
        
        # Try to load existing models
        self._try_load_models()
        
        logger.info(f"Trading Assistant initialized for {symbol} with {model_type} model")
    
    def _try_load_models(self):
        """Try to load existing trained models"""
        try:
            model_path = Config.MODELS_DIR
            if os.path.exists(model_path):
                self.ml_model.load_models(model_path)
                logger.info("Successfully loaded existing trained models")
            else:
                logger.info("No existing models found. You'll need to train models first.")
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
            logger.info("You'll need to train models first using train_models()")
    
    def collect_market_data(self, hours_back=168):  # 1 week
        """
        Collect comprehensive market data
        
        Args:
            hours_back (int): Hours of historical data to collect
            
        Returns:
            pd.DataFrame: Market data with technical indicators
        """
        try:
            logger.info(f"Collecting market data for {self.symbol}")
            
            # Get historical OHLCV data
            df = self.data_collector.get_historical_data(
                symbol=self.symbol,
                interval=Config.TIMEFRAME,
                limit=hours_back
            )
            
            # Calculate technical indicators
            df_with_indicators = self.data_collector.calculate_technical_indicators(df)
            
            # Get current market stats
            current_price = self.data_collector.get_live_price(self.symbol)
            stats_24h = self.data_collector.get_24h_stats(self.symbol)
            
            # Add current market info
            market_info = {
                'current_price': current_price,
                'price_change_24h': stats_24h['price_change_percent'],
                'volume_24h': stats_24h['volume'],
                'high_24h': stats_24h['high_24h'],
                'low_24h': stats_24h['low_24h']
            }
            
            self.current_data = df_with_indicators
            logger.info(f"Market data collected: {len(df_with_indicators)} candles")
            
            return df_with_indicators, market_info
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            raise
    
    def collect_sentiment_data(self, hours_back=24):
        """
        Collect sentiment data from multiple sources
        
        Args:
            hours_back (int): Hours to look back for sentiment
            
        Returns:
            dict: Combined sentiment metrics
        """
        try:
            logger.info(f"Collecting sentiment data for {self.symbol}")
            
            # Extract base symbol (e.g., 'BTC' from 'BTCUSDT')
            base_symbol = self.symbol.replace('USDT', '').replace('USD', '')
            
            # Get combined sentiment
            sentiment = self.sentiment_collector.get_combined_sentiment(
                symbol=base_symbol,
                hours_back=hours_back
            )
            
            self.current_sentiment = sentiment
            logger.info(f"Sentiment collected: {sentiment['overall_sentiment']}")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
            raise
    
    def prepare_features(self, market_data, sentiment_data):
        """
        Prepare features for ML model prediction
        
        Args:
            market_data (pd.DataFrame): Market data with technical indicators
            sentiment_data (dict): Sentiment metrics
            
        Returns:
            pd.DataFrame: Engineered features ready for prediction
        """
        try:
            logger.info("Preparing features for prediction")
            
            # Engineer all features (without selection/scaling for prediction)
            engineered_data = self.feature_engineer.engineer_all_features(
                market_data, sentiment_data
            )
            
            # For prediction, we need to match the exact features the model was trained on
            # Don't apply feature selection or scaling during prediction
            # The model expects the full feature set as it was trained
            
            logger.info(f"Features prepared: {engineered_data.shape}")
            return engineered_data
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def train_models(self, data_path=None):
        """
        Train ML models on historical data
        
        Args:
            data_path (str): Optional path to pre-existing data file
            
        Returns:
            dict: Training results
        """
        try:
            logger.info("Starting model training...")
            
            if data_path:
                # Load pre-existing data
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded data from {data_path}")
            else:
                # Collect fresh data for training
                market_data, _ = self.collect_market_data(hours_back=1000)  # ~6 weeks
                sentiment_data = self.collect_sentiment_data(hours_back=168)
                
                # Prepare features (engineer all features without selection/scaling)
                df = self.prepare_features(market_data, sentiment_data)
            
            # Train models on the full feature set
            results = self.ml_model.train_models(df, test_size=0.2)
            
            # Save models
            self.ml_model.save_models(f"{self.symbol.lower()}_model")
            
            logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def generate_trading_signal(self, confidence_threshold=0.7):
        """
        Generate trading signal based on current market conditions
        
        Args:
            confidence_threshold (float): Minimum confidence for signal generation
            
        Returns:
            dict: Trading signal with reasoning
        """
        try:
            logger.info("Generating trading signal...")
            
            # Collect current data
            market_data, market_info = self.collect_market_data(hours_back=168)
            sentiment_data = self.collect_sentiment_data(hours_back=24)
            
            # Prepare features for prediction
            features = self.prepare_features(market_data, sentiment_data)
            
            # Get latest features for prediction
            expected_features = self.ml_model.training_history['feature_names']
            missing_features = set(expected_features) - set(features.columns)
            
            # Handle missing features by filling with default values
            if missing_features:
                logger.warning(f"Missing features detected: {missing_features}. Filling with default values.")
                for feature in missing_features:
                    if 'twitter' in feature:
                        # Default Twitter sentiment values
                        features[feature] = 0.0
                    elif 'sentiment' in feature:
                        # Default sentiment values
                        features[feature] = 0.0
                    else:
                        # Default value for other features
                        features[feature] = 0.0
            
            latest_features = features.iloc[-1:][expected_features]
            
            # Make prediction
            prediction = self.ml_model.predict(latest_features)
            
            # Extract prediction details
            direction = prediction['classification'][0]
            confidence = prediction['confidence'][0]
            price_change_pred = prediction['regression'][0]
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                signal = "HOLD"
                reasoning = f"Low confidence ({confidence:.2%}) - insufficient signal strength"
            else:
                if direction == 1:
                    signal = "BUY"
                    reasoning = self._generate_buy_reasoning(market_data, sentiment_data, confidence)
                elif direction == -1:
                    signal = "SELL"
                    reasoning = self._generate_sell_reasoning(market_data, sentiment_data, confidence)
                else:
                    signal = "HOLD"
                    reasoning = "Neutral market conditions - no clear directional bias"
            
            # Calculate target and stop-loss levels
            current_price = market_info['current_price']
            target_price, stop_loss = self._calculate_risk_levels(
                current_price, direction, price_change_pred, market_data
            )
            
            # Create trading signal
            trading_signal = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'predicted_change': price_change_pred,
                'reasoning': reasoning,
                'market_info': market_info,
                'sentiment': sentiment_data['overall_sentiment'],
                'technical_indicators': self._get_key_indicators(market_data.iloc[-1])
            }
            
            self.current_prediction = trading_signal
            self.last_update = datetime.now()
            
            logger.info(f"Trading signal generated: {signal} with {confidence:.2%} confidence")
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            raise
    
    def _generate_buy_reasoning(self, market_data, sentiment_data, confidence):
        """Generate reasoning for BUY signal"""
        latest = market_data.iloc[-1]
        
        reasons = []
        
        # Technical analysis
        if latest['rsi'] < 40:
            reasons.append(f"RSI at {latest['rsi']:.1f} indicates oversold conditions")
        if latest['macd'] > latest['macd_signal']:
            reasons.append("MACD showing bullish crossover")
        if latest['close'] > latest['ema_20']:
            reasons.append("Price above 20-period EMA suggesting upward momentum")
        
        # Sentiment
        if sentiment_data['combined_sentiment_score'] > 0.2:
            reasons.append(f"Positive market sentiment ({sentiment_data['overall_sentiment']})")
        
        # Volume
        if latest.get('volume_ratio', 1) > 1.2:
            reasons.append("Above-average volume supporting price action")
        
        if not reasons:
            reasons.append("Technical indicators showing bullish momentum")
        
        return " | ".join(reasons)
    
    def _generate_sell_reasoning(self, market_data, sentiment_data, confidence):
        """Generate reasoning for SELL signal"""
        latest = market_data.iloc[-1]
        
        reasons = []
        
        # Technical analysis
        if latest['rsi'] > 70:
            reasons.append(f"RSI at {latest['rsi']:.1f} indicates overbought conditions")
        if latest['macd'] < latest['macd_signal']:
            reasons.append("MACD showing bearish crossover")
        if latest['close'] < latest['ema_20']:
            reasons.append("Price below 20-period EMA suggesting downward momentum")
        
        # Sentiment
        if sentiment_data['combined_sentiment_score'] < -0.2:
            reasons.append(f"Negative market sentiment ({sentiment_data['overall_sentiment']})")
        
        # Volume
        if latest.get('volume_ratio', 1) > 1.2:
            reasons.append("High volume confirming downward move")
        
        if not reasons:
            reasons.append("Technical indicators showing bearish momentum")
        
        return " | ".join(reasons)
    
    def _calculate_risk_levels(self, current_price, direction, predicted_change, market_data):
        """Calculate target and stop-loss levels"""
        try:
            # Use ATR for volatility-based levels
            atr = market_data['atr'].iloc[-1]
            
            if direction == 1:  # BUY
                # Target: 2x ATR above current price or predicted change
                target_distance = max(2 * atr, abs(predicted_change) * current_price)
                target_price = current_price + target_distance
                
                # Stop-loss: 1x ATR below current price
                stop_loss = current_price - atr
                
            elif direction == -1:  # SELL
                # Target: 2x ATR below current price or predicted change
                target_distance = max(2 * atr, abs(predicted_change) * current_price)
                target_price = current_price - target_distance
                
                # Stop-loss: 1x ATR above current price
                stop_loss = current_price + atr
                
            else:  # HOLD
                target_price = current_price
                stop_loss = current_price
            
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating risk levels: {e}")
            # Fallback to percentage-based levels
            if direction == 1:
                return round(current_price * 1.05, 2), round(current_price * 0.97, 2)
            elif direction == -1:
                return round(current_price * 0.95, 2), round(current_price * 1.03, 2)
            else:
                return current_price, current_price
    
    def _get_key_indicators(self, latest_data):
        """Get key technical indicator values"""
        return {
            'rsi': round(latest_data['rsi'], 2),
            'macd': round(latest_data['macd'], 4),
            'ema_20': round(latest_data['ema_20'], 2),
            'ema_50': round(latest_data['ema_50'], 2),
            'bollinger_position': round(latest_data.get('price_position_bb', 0.5), 3),
            'volume_ratio': round(latest_data.get('volume_ratio', 1.0), 2)
        }
    
    def get_daily_brief(self):
        """Generate daily trading brief"""
        try:
            logger.info("Generating daily trading brief...")
            
            # Get current signal
            signal = self.generate_trading_signal()
            
            # Get market overview
            market_data, market_info = self.collect_market_data(hours_back=24)
            sentiment_data = self.collect_sentiment_data(hours_back=24)
            
            brief = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbol': self.symbol,
                'current_signal': signal,
                'market_summary': {
                    'price_change_24h': f"{market_info['price_change_24h']:.2f}%",
                    'volume_24h': f"{market_info['volume_24h']:,.0f}",
                    'high_24h': f"${market_info['high_24h']:,.2f}",
                    'low_24h': f"${market_info['low_24h']:,.2f}"
                },
                            'sentiment_summary': {
                'overall': sentiment_data['overall_sentiment'],
                'score': f"{sentiment_data['combined_sentiment_score']:.3f}",
                'news_count': sentiment_data['cryptopanic']['total_news'],
                'fear_greed_value': sentiment_data['fear_greed']['value']
            },
                'technical_summary': self._get_key_indicators(market_data.iloc[-1])
            }
            
            return brief
            
        except Exception as e:
            logger.error(f"Error generating daily brief: {e}")
            raise
    
    def run_continuous_monitoring(self, interval_minutes=60):
        """
        Run continuous monitoring and signal generation
        
        Args:
            interval_minutes (int): Minutes between signal updates
        """
        logger.info(f"Starting continuous monitoring every {interval_minutes} minutes")
        
        try:
            while True:
                try:
                    # Generate trading signal
                    signal = self.generate_trading_signal()
                    
                    # Log signal
                    logger.info(f"Signal Update - {signal['signal']}: {signal['reasoning']}")
                    
                    # Wait for next update
                    time.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    logger.info("Continuous monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in continuous monitoring: {e}")
            raise

if __name__ == "__main__":
    # Test the trading assistant
    print("Testing Crypto Trading Assistant...")
    
    try:
        # Initialize assistant
        assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')
        
        # Generate trading signal
        signal = assistant.generate_trading_signal()
        
        print("\n=== TRADING SIGNAL ===")
        print(f"Symbol: {signal['symbol']}")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Current Price: ${signal['current_price']:,.2f}")
        print(f"Target: ${signal['target_price']:,.2f}")
        print(f"Stop-Loss: ${signal['stop_loss']:,.2f}")
        print(f"Reasoning: {signal['reasoning']}")
        
        # Get daily brief
        brief = assistant.get_daily_brief()
        
        print("\n=== DAILY BRIEF ===")
        print(f"Date: {brief['date']}")
        print(f"Market Sentiment: {brief['sentiment_summary']['overall']}")
        print(f"24h Change: {brief['market_summary']['price_change_24h']}%")
        print(f"RSI: {brief['technical_summary']['rsi']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure you have valid API keys in your .env file")
