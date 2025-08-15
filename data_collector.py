import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta
import time
from datetime import datetime, timedelta
from loguru import logger
from config import Config

class BinanceDataCollector:
    def __init__(self):
        """Initialize Binance client and validate API keys"""
        if not Config.validate():
            raise ValueError("Missing required API keys. Please check your .env file.")
        
        self.client = Client(
            Config.BINANCE_API_KEY, 
            Config.BINANCE_SECRET_KEY,
            testnet=Config.BINANCE_TESTNET
        )
        
        # Test connection
        try:
            self.client.ping()
            logger.info("Successfully connected to Binance API")
        except BinanceAPIException as e:
            logger.error(f"Failed to connect to Binance API: {e}")
            raise
    
    def get_historical_data(self, symbol=Config.DEFAULT_SYMBOL, 
                           interval=Config.TIMEFRAME, 
                           limit=Config.LOOKBACK_PERIOD):
        """
        Fetch historical OHLCV data from Binance
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        try:
            logger.info(f"Fetching {limit} {interval} candles for {symbol}")
            
            # Get klines (candlestick data)
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Drop unnecessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(df)} candles")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators using TA-Lib and ta libraries
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Original data with technical indicators
        """
        logger.info("Calculating technical indicators")
        
        try:
            # Trend Indicators
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bollinger_upper'] = bollinger.bollinger_hband()
            df['bollinger_lower'] = bollinger.bollinger_lband()
            df['bollinger_middle'] = bollinger.bollinger_mavg()
            
            # Volatility Indicators
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume Indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Add realistic volatility features
            df['daily_return'] = df['close'].pct_change(24)  # 24-hour returns
            df['hourly_return'] = df['close'].pct_change(1)  # 1-hour returns
            
            # Calculate realistic ATR
            high_low_range = df['high'] - df['low']
            high_close_range = abs(df['high'] - df['close'].shift(1))
            low_close_range = abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low_range, np.maximum(high_close_range, low_close_range))
            df['atr_realistic'] = true_range.rolling(window=14).mean()
            
            # Remove rows with NaN values (from indicator calculations)
            df = df.dropna()
            
            logger.info(f"Calculated {len(df.columns)} technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def get_live_price(self, symbol=Config.DEFAULT_SYMBOL):
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting live price: {e}")
            raise
    
    def get_24h_stats(self, symbol=Config.DEFAULT_SYMBOL):
        """Get 24-hour price statistics"""
        try:
            stats = self.client.get_ticker(symbol=symbol)
            return {
                'price_change': float(stats['priceChange']),
                'price_change_percent': float(stats['priceChangePercent']),
                'volume': float(stats['volume']),
                'high_24h': float(stats['highPrice']),
                'low_24h': float(stats['lowPrice'])
            }
        except BinanceAPIException as e:
            logger.error(f"Error getting 24h stats: {e}")
            raise

if __name__ == "__main__":
    # Test the data collector
    try:
        collector = BinanceDataCollector()
        
        # Get historical data
        df = collector.get_historical_data('BTCUSDT', '1h', 100)
        print(f"Fetched {len(df)} candles")
        print(df.head())
        
        # Calculate indicators
        df_with_indicators = collector.calculate_technical_indicators(df)
        print(f"Data with indicators shape: {df_with_indicators.shape}")
        print(df_with_indicators.columns.tolist())
        
        # Get live price
        current_price = collector.get_live_price('BTCUSDT')
        print(f"Current BTC price: ${current_price:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
