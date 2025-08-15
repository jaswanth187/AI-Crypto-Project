# 🤖 AI-Powered Crypto Trading Assistant

An intelligent cryptocurrency trading assistant that combines **Machine Learning predictions** with **Natural Language reasoning** to provide comprehensive trading signals and market analysis.

## 🎯 Project Overview

This project implements a **dual-brain architecture**:

1. **ML Model Brain** → Analyzes market data & predicts price movements
2. **LLM Brain** → Explains predictions & provides human-like reasoning

### Key Features

- 📊 **Real-time Market Data** from Binance API
- 🔍 **Technical Analysis** with 20+ indicators (RSI, MACD, Bollinger Bands, etc.)
- 📰 **Sentiment Analysis** from news, Twitter, and market sentiment
- 🤖 **ML Predictions** using XGBoost/LightGBM for price direction
- 💡 **Intelligent Reasoning** explaining why signals are generated
- 📈 **Risk Management** with automatic target/stop-loss calculation
- ⏰ **Continuous Monitoring** with configurable update intervals

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature        │    │   ML Models     │
│                 │    │  Engineering    │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Binance API   │───▶│ • Technical     │───▶│ • XGBoost       │
│ • CryptoPanic   │    │   Indicators    │    │ • LightGBM      │
│ • Twitter       │    │ • Price/Volume  │    │ • Random Forest │
│ • Fear & Greed  │    │ • Sentiment     │    │                 │
└─────────────────┘    │ • Time Features │    └─────────────────┘
                       └─────────────────┘              │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Trading       │    │   LLM           │
                       │   Signals       │◀───│   Reasoning     │
                       │                 │    │                 │
                       │ • BUY/SELL/HOLD │    │ • Market        │
                       │ • Confidence    │    │   Analysis      │
                       │ • Targets       │    │ • Daily Brief   │
                       │ • Stop-Loss     │    │ • Risk Summary  │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Crypto-Project

# Install dependencies
pip install -r requirements.txt
```

### 2. API Setup

Copy `env_example.txt` to `.env` and fill in your API keys:

```bash
cp env_example.txt .env
```

**Required APIs:**
- **Binance API** (Required) - Get from [Binance](https://www.binance.com/en/my/settings/api-management)
- **CryptoPanic API** (Optional) - Get from [CryptoPanic](https://cryptopanic.com/developers/api/)
- **Twitter API** (Optional) - Get from [Twitter Developer](https://developer.twitter.com/)

### 3. First Run

```bash
# Test the system
python trading_assistant.py

# Or run individual components
python data_collector.py
python sentiment_collector.py
python ml_model.py
```

## 📊 Usage Examples

### Basic Trading Signal

```python
from trading_assistant import CryptoTradingAssistant

# Initialize assistant
assistant = CryptoTradingAssistant('BTCUSDT', 'xgboost')

# Generate trading signal
signal = assistant.generate_trading_signal()

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Target: ${signal['target_price']:,.2f}")
print(f"Stop-Loss: ${signal['stop_loss']:,.2f}")
print(f"Reasoning: {signal['reasoning']}")
```

### Daily Market Brief

```python
# Get comprehensive daily analysis
brief = assistant.get_daily_brief()

print(f"Market Sentiment: {brief['sentiment_summary']['overall']}")
print(f"24h Change: {brief['market_summary']['price_change_24h']}%")
print(f"Technical RSI: {brief['technical_summary']['rsi']}")
```

### Continuous Monitoring

```python
# Run continuous monitoring (updates every hour)
assistant.run_continuous_monitoring(interval_minutes=60)
```

## 🔧 Configuration

### Model Types

- **XGBoost** (Default) - Best performance, good interpretability
- **LightGBM** - Fast training, good for large datasets
- **Random Forest** - Robust, less prone to overfitting

### Timeframes

- **1m, 5m, 15m** - Short-term scalping
- **1h** (Default) - Swing trading
- **4h, 1d** - Position trading

### Confidence Thresholds

- **0.8+** - High confidence signals
- **0.7** (Default) - Balanced approach
- **0.6** - More signals, lower accuracy

## 📈 Technical Indicators

The system calculates 20+ technical indicators:

- **Trend**: EMA, SMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: Volume SMA, Volume Ratio, OBV
- **Support/Resistance**: Dynamic levels based on recent highs/lows

## 🧠 ML Model Features

### Feature Engineering

- **Price Features**: Momentum, volatility, position relative to MAs
- **Volume Features**: Volume trends, abnormal volume detection
- **Technical Features**: Indicator crossovers, extreme values
- **Time Features**: Market sessions, cyclical encoding
- **Sentiment Features**: News sentiment, social media engagement

### Model Training

```python
# Train models on historical data
results = assistant.train_models()

print(f"Classification Accuracy: {results['classification']['accuracy']:.4f}")
print(f"Regression R²: {results['regression']['r2_score']:.4f}")
```

## 📊 Output Examples

### Trading Signal Output

```
=== TRADING SIGNAL ===
Symbol: BTCUSDT
Signal: BUY
Confidence: 82%
Current Price: $67,250.00
Target: $68,250.00
Stop-Loss: $66,900.00
Reasoning: RSI at 35.2 indicates oversold conditions | 
           MACD showing bullish crossover | 
           Price above 20-period EMA suggesting upward momentum
```

### Daily Brief Output

```
=== DAILY BRIEF ===
Date: 2024-01-15
Market Sentiment: Bullish
24h Change: +2.45%
RSI: 58.7
MACD: 0.0023
Volume Ratio: 1.15
```

## 🛡️ Risk Management

- **Automatic Stop-Loss**: Based on ATR (Average True Range)
- **Target Calculation**: 2x ATR or predicted price change
- **Confidence Filtering**: Only high-confidence signals
- **Position Sizing**: Recommendations based on volatility

## 🔄 Continuous Monitoring

The system can run continuously to:

- Monitor market conditions 24/7
- Generate signals at configurable intervals
- Log all predictions and outcomes
- Alert on significant market changes

## 📁 Project Structure

```
AI-Crypto-Project/
├── config.py                 # Configuration management
├── data_collector.py         # Binance data collection
├── sentiment_collector.py    # Sentiment analysis
├── feature_engineering.py    # Feature creation & selection
├── ml_model.py              # ML model training & prediction
├── trading_assistant.py     # Main trading assistant
├── requirements.txt          # Python dependencies
├── env_example.txt          # Environment variables template
├── README.md                # This file
├── data/                    # Data storage
├── models/                  # Trained ML models
└── logs/                    # Application logs
```

## 🚧 Phase 1 Status

✅ **Completed:**
- Binance API integration
- Technical indicator calculation
- Sentiment data collection
- Feature engineering pipeline
- ML model training (XGBoost/LightGBM)
- Trading signal generation
- Risk management calculations

🔄 **Next Phase (LLM Integration):**
- Integrate open-source LLM (LLaMA 3, Mistral)
- Generate human-readable market analysis
- Create comprehensive trading briefs
- Explain ML model decisions

## ⚠️ Disclaimer

**This is for educational and research purposes only.**
- Not financial advice
- Cryptocurrency trading involves significant risk
- Always do your own research
- Never invest more than you can afford to lose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the code comments and docstrings

---

**Happy Trading! 🚀📈**