import pandas as pd
import numpy as np
from ml_model import CryptoMLModel
from feature_engineering import FeatureEngineer

def test_models():
    """
    Test and compare the performance of different ML models.
    """
    print("Starting model performance test...")

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

    # Engineer features
    try:
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_all_features(sample_data)
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return

    # --- Model Training and Evaluation ---
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    results = {}

    for model_type in model_types:
        print(f"--- Testing {model_type.upper()} ---")
        try:
            # Initialize and train the model
            model = CryptoMLModel(model_type=model_type)
            training_results = model.train_models(engineered_data)

            # Store results
            accuracy = training_results['classification']['accuracy']
            r2 = training_results['regression']['r2_score']
            results[model_type] = {'accuracy': accuracy, 'r2_score': r2}

            print(f"Classification Accuracy: {accuracy:.4f}")
            print(f"Regression R² Score: {r2:.4f}")
            print("-" * (len(model_type) + 14))

        except Exception as e:
            print(f"Error testing {model_type}: {e}")

    # --- Summary ---
    print("\n--- Model Comparison Summary ---")
    for model_type, metrics in results.items():
        print(f"{model_type.upper():<15} | Accuracy: {metrics['accuracy']:.4f} | R² Score: {metrics['r2_score']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    test_models()
