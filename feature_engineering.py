import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering pipeline"""
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def create_price_features(self, df):
        """
        Create price-based features
        
        Args:
            df (pd.DataFrame): OHLCV data with technical indicators
            
        Returns:
            pd.DataFrame: Data with additional price features
        """
        try:
            # Price momentum features
            df['price_momentum_1h'] = df['close'].pct_change(1)
            df['price_momentum_4h'] = df['close'].pct_change(4)
            df['price_momentum_24h'] = df['close'].pct_change(24)
            
            # Volatility features - use more appropriate methods
            # For 1h, use the absolute price change instead of std of single value
            df['volatility_1h'] = abs(df['close'].pct_change(1))
            df['volatility_4h'] = df['close'].rolling(window=4, min_periods=1).std()
            df['volatility_24h'] = df['close'].rolling(window=24, min_periods=1).std()
            
            # Price position features - handle division by zero
            bb_range = df['bollinger_upper'] - df['bollinger_lower']
            df['price_position_bb'] = np.where(bb_range != 0, 
                (df['close'] - df['bollinger_lower']) / bb_range, 0.5)
            
            df['price_position_sma'] = np.where(df['sma_20'] != 0, 
                (df['close'] - df['sma_20']) / df['sma_20'], 0)
            
            # Support/Resistance features - use min_periods to avoid NaN
            df['support_distance'] = np.where(df['close'] != 0,
                (df['close'] - df['low'].rolling(window=20, min_periods=1).min()) / df['close'], 0)
            df['resistance_distance'] = np.where(df['close'] != 0,
                (df['high'].rolling(window=20, min_periods=1).max() - df['close']) / df['close'], 0)
            
            return df
            
        except Exception as e:
            print(f"Error creating price features: {e}")
            return df
    
    def create_volume_features(self, df):
        """
        Create volume-based features
        
        Args:
            df (pd.DataFrame): OHLCV data with technical indicators
            
        Returns:
            pd.DataFrame: Data with additional volume features
        """
        try:
            # Volume momentum
            df['volume_momentum_1h'] = df['volume'].pct_change(1)
            df['volume_momentum_4h'] = df['volume'].pct_change(4)
            
            # Volume vs price relationship
            df['volume_price_trend'] = df['volume'] * df['price_momentum_1h']
            
            # Volume relative to moving averages - use min_periods to avoid NaN
            df['volume_sma_ratio'] = np.where(df['volume'].rolling(window=20, min_periods=1).mean() != 0,
                df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean(), 1.0)
            df['volume_ema_ratio'] = np.where(df['volume'].rolling(window=20, min_periods=1).mean() != 0,
                df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean(), 1.0)
            
            # Abnormal volume detection - use min_periods to avoid NaN
            volume_std = df['volume'].rolling(window=20, min_periods=1).std()
            df['volume_z_score'] = np.where(volume_std != 0,
                (df['volume'] - df['volume'].rolling(window=20, min_periods=1).mean()) / volume_std, 0)
            
            return df
            
        except Exception as e:
            print(f"Error creating volume features: {e}")
            return df
    
    def create_technical_features(self, df):
        """
        Create technical indicator-based features
        
        Args:
            df (pd.DataFrame): OHLCV data with technical indicators
            
        Returns:
            pd.DataFrame: Data with additional technical features
        """
        try:
            # RSI-based features
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_trend'] = df['rsi'].diff()
            
            # MACD-based features
            df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
            df['macd_strength'] = np.where(df['close'] != 0, df['macd'] / df['close'], 0)
            
            # Moving average features
            df['ema_crossover'] = ((df['ema_20'] > df['ema_50']) & (df['ema_20'].shift(1) <= df['ema_50'].shift(1))).astype(int)
            df['ema_crossunder'] = ((df['ema_20'] < df['ema_50']) & (df['ema_20'].shift(1) >= df['ema_50'].shift(1))).astype(int)
            df['ma_trend'] = np.where(df['ema_50'] != 0, (df['ema_20'] - df['ema_50']) / df['ema_50'], 0)
            
            # Bollinger Bands features
            df['bb_squeeze'] = np.where(df['bollinger_middle'] != 0,
                (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle'], 0)
            df['bb_breakout_up'] = (df['close'] > df['bollinger_upper']).astype(int)
            df['bb_breakout_down'] = (df['close'] < df['bollinger_lower']).astype(int)
            
            # Stochastic features
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
            df['stoch_crossover'] = ((df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error creating technical features: {e}")
            return df
    
    def create_time_features(self, df):
        """
        Create time-based features
        
        Args:
            df (pd.DataFrame): OHLCV data with datetime index
            
        Returns:
            pd.DataFrame: Data with additional time features
        """
        try:
            # Extract time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market session features (assuming UTC)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            df['american_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error creating time features: {e}")
            return df
    
    def add_sentiment_features(self, df, sentiment_data):
        """
        Add sentiment features to the dataframe
        
        Args:
            df (pd.DataFrame): OHLCV data with technical indicators
            sentiment_data (dict): Sentiment metrics from sentiment collector
            
        Returns:
            pd.DataFrame: Data with sentiment features
        """
        try:
            # Check if sentiment_data is provided and not empty
            if not sentiment_data or sentiment_data == {}:
                # Add default sentiment features with neutral values
                df['sentiment_score'] = 0.0
                df['cryptopanic_sentiment'] = 0.0
                df['twitter_sentiment'] = 0.0
                df['fear_greed_score'] = 0.5
                df['sentiment_momentum'] = 0.0
                df['sentiment_ma'] = 0.0
                df['sentiment_std'] = 0.0
            else:
                # Add sentiment scores as features
                df['sentiment_score'] = sentiment_data.get('combined_sentiment_score', 0)
                df['cryptopanic_sentiment'] = sentiment_data.get('cryptopanic', {}).get('sentiment_score', 0)
                df['twitter_sentiment'] = sentiment_data.get('twitter', {}).get('engagement_sentiment', 0)
                df['fear_greed_score'] = sentiment_data.get('fear_greed', {}).get('value', 50) / 100
                
                # Create sentiment momentum features
                df['sentiment_momentum'] = df['sentiment_score'].diff()
                df['sentiment_ma'] = df['sentiment_score'].rolling(window=24).mean()
                df['sentiment_std'] = df['sentiment_score'].rolling(window=24).std()
            
            return df
            
        except Exception as e:
            print(f"Error adding sentiment features: {e}")
            return df
    
    def create_target_variable(self, df, horizon_hours=24, threshold=0.02):
        """
        Create target variable for prediction
        
        Args:
            df (pd.DataFrame): OHLCV data with features
            horizon_hours (int): Prediction horizon in hours
            threshold (float): Minimum price change threshold for classification
            
        Returns:
            pd.DataFrame: Data with target variable
        """
        try:
            # Calculate future price change
            df['future_price'] = df['close'].shift(-horizon_hours)
            df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
            
            # Create classification target (use 0, 1, 2 for XGBoost compatibility)
            df['target'] = 1  # Neutral (default)
            df.loc[df['price_change_future'] > threshold, 'target'] = 2  # Up
            df.loc[df['price_change_future'] < -threshold, 'target'] = 0  # Down
            
            # Create regression target (price change percentage)
            df['target_regression'] = df['price_change_future']
            
            # Remove rows where we don't have future data
            df = df.dropna(subset=['target'])
            
            return df
            
        except Exception as e:
            print(f"Error creating target variable: {e}")
            return df
    
    def select_features(self, df, target_col='target', k=50):
        """
        Select most important features using statistical tests
        
        Args:
            df (pd.DataFrame): Data with features and target
            target_col (str): Name of target column
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        try:
            # Separate features and target
            feature_cols = [col for col in df.columns if col not in ['target', 'target_regression', 'future_price', 'price_change_future']]
            X = df[feature_cols]
            y = df[target_col]
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            self.selected_features = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
            
            # Create new dataframe with selected features
            selected_df = df[self.selected_features + [target_col, 'target_regression']].copy()
            
            print(f"Selected {len(self.selected_features)} features out of {len(feature_cols)}")
            return selected_df
            
        except Exception as e:
            print(f"Error selecting features: {e}")
            return df
    
    def scale_features(self, df, target_cols=['target', 'target_regression']):
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): Data with features
            target_cols (list): Columns to exclude from scaling
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        try:
            # Separate features and targets
            feature_cols = [col for col in df.columns if col not in target_cols]
            target_data = df[target_cols].copy()
            
            # Scale features
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
            
            # Combine scaled features with targets
            final_df = pd.concat([scaled_df, target_data], axis=1)
            
            return final_df
            
        except Exception as e:
            print(f"Error scaling features: {e}")
            return df
    
    def engineer_all_features(self, df, sentiment_data=None, horizon_hours=24):
        """
        Apply all feature engineering steps
        
        Args:
            df (pd.DataFrame): Raw OHLCV data with technical indicators
            sentiment_data (dict): Optional sentiment data
            horizon_hours (int): Prediction horizon in hours
            
        Returns:
            pd.DataFrame: Fully engineered feature set
        """
        try:
            print("Starting feature engineering pipeline...")
            
            # Create all feature types
            df = self.create_price_features(df)
            df = self.create_volume_features(df)
            df = self.create_technical_features(df)
            df = self.create_time_features(df)
            
            # Add sentiment features if available
            if sentiment_data:
                df = self.add_sentiment_features(df, sentiment_data)
            
            # Create target variable
            df = self.create_target_variable(df, horizon_hours=horizon_hours)
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            print(f"Feature engineering complete. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error in feature engineering pipeline: {e}")
            return df

if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add some basic technical indicators
    sample_data['rsi'] = np.random.uniform(20, 80, 100)
    sample_data['macd'] = np.random.randn(100)
    sample_data['ema_20'] = sample_data['close'].rolling(20).mean()
    
    # Test feature engineering
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_all_features(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Engineered shape: {engineered_data.shape}")
    print(f"Features: {engineered_data.columns.tolist()}")
