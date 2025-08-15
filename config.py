import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Binance API Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
    
    # CryptoPanic API
    CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')
    
    # Twitter API (Optional - can use snscrape as fallback)
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    # Model Configuration
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'xgboost')  # xgboost, lightgbm, lstm
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '24'))  # hours
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    
    # Data Configuration
    DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'BTCUSDT')
    TIMEFRAME = os.getenv('TIMEFRAME', '1h')
    LOOKBACK_PERIOD = int(os.getenv('LOOKBACK_PERIOD', '1000'))  # candles
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        'rsi', 'macd', 'ema_20', 'ema_50', 'sma_20', 'sma_50',
        'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
        'atr', 'stoch_k', 'stoch_d', 'williams_r', 'cci'
    ]
    
    # Sentiment Analysis
    SENTIMENT_LOOKBACK_HOURS = int(os.getenv('SENTIMENT_LOOKBACK_HOURS', '24'))
    
    # File Paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    LOGS_DIR = 'logs'
    
    @classmethod
    def validate(cls):
        """Validate that required API keys are present"""
        required_keys = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            print(f"Warning: Missing required API keys: {missing_keys}")
            print("Please create a .env file with the required keys")
            return False
        return True
