# 🤖 Crypto AI Trading Assistant

An AI-powered cryptocurrency trading system that combines machine learning predictions with technical analysis and sentiment data.

## 🎯 **What This System Does**

- **ML Model**: Predicts price direction (BUY/SELL/HOLD) using XGBoost
- **Technical Analysis**: RSI, MACD, EMA, ATR, Bollinger Bands, and more
- **Sentiment Analysis**: News sentiment from CryptoPanic + Twitter data
- **Risk Management**: Automatic target and stop-loss calculation
- **Real-time Data**: Live data from Binance API

## 🚀 **Quick Start (3 Simple Steps)**

### 1. **Setup** (Run Once)
```bash
python SETUP.py
```

### 2. **Configure API Keys**
Copy `env_example.txt` to `.env` and add your API keys:
```bash
cp env_example.txt .env
# Edit .env with your actual API keys
```

### 3. **Run the Main System**
```bash
python MAIN_TRADING_SYSTEM.py
```

## 📁 **Clean Project Structure**

```
AI-Crypto-Project/
├── 🚀 MAIN_TRADING_SYSTEM.py    # ← YOUR MAIN FILE (run this!)
├── 🔧 SETUP.py                   # ← Setup script (run once)
├── 📋 requirements.txt           # Python dependencies
├── 📝 env_example.txt           # API key template
├── 📖 README.md                 # This file
│
├── 🤖 Core System Files:
│   ├── config.py                # Configuration & API keys
│   ├── data_collector.py        # Binance data collection
│   ├── sentiment_collector.py   # News & social sentiment
│   ├── feature_engineering.py   # Feature creation
│   ├── ml_model.py             # AI model training/prediction
│   └── trading_assistant.py    # Main orchestrator
│
├── 📊 Data & Models:
│   ├── data/                   # Historical data storage
│   ├── models/                 # Trained AI models
│   └── logs/                   # System logs
│
└── 🗑️ Old Files (Deleted):
    ❌ aggressive_trading_signal.py
    ❌ realistic_trading_signal.py
    ❌ fix_realistic_model.py
    ❌ retrain_simple_model.py
    ❌ basic_trading_signal.py
    ❌ simple_trading_signal.py
    ❌ train_first_model.py
    ❌ test_system.py
    ❌ debug_test.py
    ❌ rapidapi_twitter_test.py
    ❌ API_SETUP_GUIDE.md
    ❌ test_api_setup.py
    ❌ requirements_simple.txt
```

## 🎮 **How to Use the Main System**

When you run `python MAIN_TRADING_SYSTEM.py`, you get a clean menu:

```
🤖 CRYPTO AI TRADING ASSISTANT - MAIN SYSTEM
======================================================================
1. 🚀 Generate Trading Signal (Conservative)  ← High confidence only
2. ⚡ Generate Trading Signal (Aggressive)    ← Forces decisions
3. 🔄 Retrain Model with Latest Data         ← Update AI model
4. 📊 Show Market Analysis                   ← Current market status
5. 🎯 Continuous Monitoring                  ← Auto-updates every 5 min
6. ❌ Exit
======================================================================
```

## 🔑 **Required API Keys**

You need these API keys in your `.env` file:

- **Binance API**: For live market data
- **CryptoPanic API**: For news sentiment (optional)
- **Twitter API**: For social sentiment (optional)

## 📊 **What You Get**

### **Conservative Signals** (Option 1)
- High confidence threshold (70%+)
- Fewer but more reliable signals
- Good for risk-averse traders

### **Aggressive Signals** (Option 2)
- Lower confidence threshold (40%+)
- Forces BUY/SELL decisions
- More trading opportunities
- Uses RSI/MACD as backup

### **Risk Management**
- **Target**: 1.5x ATR (realistic for Bitcoin)
- **Stop-Loss**: 1x ATR (proper risk control)
- **Risk-Reward**: Typically 1:1.5

## 🎯 **Example Output**

```
🎯 TRADING SIGNAL:
📊 Symbol: BTCUSDT
🎯 Signal: BUY
🎲 Confidence: 85.2%
💰 Current Price: $117,150.12
📈 Target Price: $162,986.85 (+39.11%)
🛑 Stop Loss: $71,325.13 (-39.11%)
🧠 Reasoning: AI model predicts upward price movement

📊 Market Context:
   📈 24h Change: -0.88%
   📊 Volume: 672
   🔢 RSI: 31.1
   📊 MACD: -584.18
```

## 🚨 **Important Notes**

1. **This is for educational purposes** - Always do your own research
2. **Start with small amounts** - Never risk more than you can afford to lose
3. **Monitor performance** - Track how well the signals work for you
4. **Update regularly** - Retrain models with fresh data (Option 3)

## 🔄 **Maintenance**

- **Daily**: Run Option 4 to check market status
- **Weekly**: Run Option 3 to retrain models
- **As needed**: Use Options 1 or 2 for trading signals

## 🆘 **Troubleshooting**

- **"No trained models"**: Run Option 3 to train models first
- **API errors**: Check your `.env` file and API keys
- **Import errors**: Run `python SETUP.py` to install packages

## 🎉 **You're All Set!**

1. ✅ **Run**: `python SETUP.py`
2. ✅ **Configure**: Add API keys to `.env`
3. ✅ **Trade**: `python MAIN_TRADING_SYSTEM.py`

**That's it! One main file, clean structure, everything organized!** 🚀