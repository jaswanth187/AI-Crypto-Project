import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
from datetime import datetime
from loguru import logger
from config import Config

class CryptoMLModel:
    def __init__(self, model_type='xgboost'):
        """
        Initialize ML model for crypto price prediction
        
        Args:
            model_type (str): Type of model ('xgboost', 'lightgbm', 'random_forest')
        """
        self.model_type = model_type
        self.classification_model = None
        self.regression_model = None
        self.feature_importance = None
        self.training_history = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    def prepare_data(self, df, test_size=0.2, target_col='target'):
        """
        Prepare data for training by splitting into features and targets
        
        Args:
            df (pd.DataFrame): Data with features and targets
            test_size (float): Proportion of data for testing
            target_col (str): Name of target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        try:
            # Separate features and targets
            feature_cols = [col for col in df.columns if col not in ['target', 'target_regression', 'future_price', 'price_change_future']]
            X = df[feature_cols]
            y_classification = df['target']
            y_regression = df['target_regression']
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y_classification.isnull() | y_regression.isnull())
            X = X[mask]
            y_classification = y_classification[mask]
            y_regression = y_regression[mask]
            
            # Split data (time series aware)
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train_class = y_classification.iloc[:split_idx]
            y_test_class = y_classification.iloc[split_idx:]
            y_train_reg = y_regression.iloc[:split_idx]
            y_test_reg = y_regression.iloc[split_idx:]
            
            logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
            logger.info(f"Features: {len(feature_cols)}")
            
            return (X_train, X_test, y_train_class, y_test_class, 
                   y_train_reg, y_test_reg, feature_cols)
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_classification_model(self, X_train, y_train, X_test, y_test):
        """
        Train classification model for price direction prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Training results and metrics
        """
        try:
            logger.info(f"Training {self.model_type} classification model...")
            
            if self.model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss'
                )
            elif self.model_type == 'lightgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Store model
            self.classification_model = model
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': self.feature_importance
            }
            
            logger.info(f"Classification model trained successfully")
            logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training classification model: {e}")
            raise
    
    def train_regression_model(self, X_train, y_train, X_test, y_test):
        """
        Train regression model for price change prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Training results and metrics
        """
        try:
            logger.info(f"Training {self.model_type} regression model...")
            
            if self.model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='rmse'
                )
            elif self.model_type == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model
            self.regression_model = model
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            logger.info(f"Regression model trained successfully")
            logger.info(f"RMSE: {rmse:.6f}, R²: {r2:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training regression model: {e}")
            raise
    
    def train_models(self, df, test_size=0.2):
        """
        Train both classification and regression models
        
        Args:
            df (pd.DataFrame): Data with features and targets
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Training results for both models
        """
        try:
            logger.info("Starting model training pipeline...")
            
            # Prepare data
            (X_train, X_test, y_train_class, y_test_class, 
             y_train_reg, y_test_reg, feature_cols) = self.prepare_data(df, test_size)
            
            # Train classification model
            classification_results = self.train_classification_model(
                X_train, y_train_class, X_test, y_test_class
            )
            
            # Train regression model
            regression_results = self.train_regression_model(
                X_train, y_train_reg, X_test, y_test_reg
            )
            
            # Store training history
            self.training_history = {
                'classification': classification_results,
                'regression': regression_results,
                'feature_names': feature_cols,
                'training_date': datetime.now()
            }
            
            logger.info("Model training pipeline completed successfully")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise
    
    def predict(self, features_df, include_probabilities=True):
        """
        Make predictions using trained models
        
        Args:
            features_df (pd.DataFrame): Features for prediction
            include_probabilities (bool): Whether to include classification probabilities
            
        Returns:
            dict: Prediction results
        """
        try:
            if self.classification_model is None or self.regression_model is None:
                raise ValueError("Models not trained. Please train models first.")
            
            # Ensure features match training data
            expected_features = self.training_history['feature_names']
            missing_features = set(expected_features) - set(features_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select only the expected features in the correct order
            X = features_df[expected_features]
            
            # Make predictions
            classification_pred = self.classification_model.predict(X)
            regression_pred = self.regression_model.predict(X)
            
            # Get probabilities if requested and available
            probabilities = None
            if include_probabilities and hasattr(self.classification_model, 'predict_proba'):
                probabilities = self.classification_model.predict_proba(X)
            
            # Create prediction results
            results = {
                'classification': classification_pred,
                'regression': regression_pred,
                'probabilities': probabilities,
                'timestamp': datetime.now()
            }
            
            # Add confidence scores
            if probabilities is not None:
                max_probs = np.max(probabilities, axis=1)
                results['confidence'] = max_probs
            else:
                results['confidence'] = np.ones(len(classification_pred)) * 0.8  # Default confidence
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save_models(self, filename_prefix='crypto_model'):
        """
        Save trained models to disk
        
        Args:
            filename_prefix (str): Prefix for saved model files
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save classification model
            if self.classification_model:
                class_filename = f"{Config.MODELS_DIR}/{filename_prefix}_classification_{timestamp}.joblib"
                joblib.dump(self.classification_model, class_filename)
                logger.info(f"Classification model saved: {class_filename}")
            
            # Save regression model
            if self.regression_model:
                reg_filename = f"{Config.MODELS_DIR}/{filename_prefix}_regression_{timestamp}.joblib"
                joblib.dump(self.regression_model, reg_filename)
                logger.info(f"Regression model saved: {reg_filename}")
            
            # Save training history
            history_filename = f"{Config.MODELS_DIR}/{filename_prefix}_history_{timestamp}.joblib"
            joblib.dump(self.training_history, history_filename)
            logger.info(f"Training history saved: {history_filename}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, model_path):
        """
        Load trained models from disk
        
        Args:
            model_path (str): Path to model directory
        """
        try:
            # Find the most recent model files
            model_files = [f for f in os.listdir(model_path) if f.endswith('.joblib')]
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            # Load the most recent models
            classification_files = [f for f in model_files if 'classification' in f]
            regression_files = [f for f in model_files if 'regression' in f]
            history_files = [f for f in model_files if 'history' in f]
            
            if classification_files:
                latest_class = max(classification_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
                self.classification_model = joblib.load(os.path.join(model_path, latest_class))
                logger.info(f"Loaded classification model: {latest_class}")
            
            if regression_files:
                latest_reg = max(regression_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
                self.regression_model = joblib.load(os.path.join(model_path, latest_reg))
                logger.info(f"Loaded regression model: {latest_reg}")
            
            if history_files:
                latest_history = max(history_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
                self.training_history = joblib.load(os.path.join(model_path, latest_history))
                logger.info(f"Loaded training history: {latest_history}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model_summary(self):
        """Get summary of trained models"""
        if not self.training_history:
            return "No models trained yet"
        
        summary = {
            'model_type': self.model_type,
            'training_date': self.training_history['training_date'],
            'num_features': len(self.training_history['feature_names']),
            'classification_metrics': {
                'accuracy': self.training_history['classification']['accuracy'],
                'f1_score': self.training_history['classification']['f1_score']
            },
            'regression_metrics': {
                'rmse': self.training_history['regression']['rmse'],
                'r2_score': self.training_history['regression']['r2_score']
            }
        }
        
        return summary

if __name__ == "__main__":
    # Test the ML model
    print("Testing ML model...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Add some basic technical indicators
    sample_data['rsi'] = np.random.uniform(20, 80, 1000)
    sample_data['macd'] = np.random.randn(1000)
    sample_data['ema_20'] = sample_data['close'].rolling(20).mean()
    sample_data['ema_50'] = sample_data['close'].rolling(50).mean()
    
    # Create target variables
    sample_data['target'] = np.random.choice([-1, 0, 1], 1000)
    sample_data['target_regression'] = np.random.randn(1000) * 0.02
    
    # Test feature engineering and model training
    try:
        from feature_engineering import FeatureEngineer
        
        # Engineer features
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_all_features(sample_data)
        
        # Train model
        model = CryptoMLModel('xgboost')
        results = model.train_models(engineered_data)
        
        print("Model training completed successfully!")
        print(f"Classification accuracy: {results['classification']['accuracy']:.4f}")
        print(f"Regression R²: {results['regression']['r2_score']:.4f}")
        
        # Test prediction
        test_features = engineered_data.iloc[-10:][model.training_history['feature_names']]
        predictions = model.predict(test_features)
        print(f"Predictions: {predictions['classification']}")
        
    except Exception as e:
        print(f"Error: {e}")
