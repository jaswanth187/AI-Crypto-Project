# 🔑 API Setup Guide

## 📋 **Required APIs (Must Have)**

### **1. Binance API - CRITICAL**
- **Purpose**: Live market data, price feeds, technical indicators
- **Cost**: FREE
- **Setup Time**: 5-10 minutes

#### **Step-by-Step Setup:**
1. Go to [Binance.com](https://www.binance.com)
2. Log in to your account
3. Go to **Profile** → **API Management**
4. Click **Create API**
5. **Enable these permissions:**
   - ✅ **Read Info** (Required)
   - ✅ **Spot & Margin Trading** (If you want live trading)
6. **Security Settings:**
   - Add your IP address to restrictions
   - Enable 2FA if not already enabled
7. **Copy your keys:**
   - API Key
   - Secret Key

---

## 📰 **Optional APIs (Nice to Have)**

### **2. CryptoPanic API**
- **Purpose**: News sentiment analysis
- **Cost**: FREE (100 requests/day)
- **Setup Time**: 2-3 minutes

#### **Setup:**
1. Visit [CryptoPanic.io](https://cryptopanic.io)
2. Sign up for free account
3. Go to **API** section
4. Copy your API key

### **3. Twitter API**
- **Purpose**: Social media sentiment
- **Cost**: FREE (Basic tier)
- **Setup Time**: 10-15 minutes

#### **Setup:**
1. Go to [Twitter Developer Portal](https://developer.twitter.com)
2. Apply for developer account
3. Create new app
4. Get Bearer Token

---

## ⚠️ **Security Best Practices**

### **API Key Security:**
- 🔒 **Never share your API keys**
- 🔒 **Don't commit .env to git**
- 🔒 **Use IP restrictions**
- 🔒 **Enable 2FA on Binance**
- 🔒 **Start with Testnet mode**

### **Testnet vs Live:**
- 🧪 **Testnet**: Safe testing, no real money
- 🚀 **Live**: Real trading, real money

---

## 🚀 **Quick Start Steps**

### **1. Create .env File:**
```bash
# In your project root, create .env file
touch .env  # On Windows: echo. > .env
```

### **2. Add Your Keys:**
```bash
# .env file content:
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_SECRET_KEY=your_actual_secret_key_here
BINANCE_TESTNET=True
```

### **3. Test Setup:**
```bash
python test_api_setup.py
```

### **4. Run Trading Assistant:**
```bash
python trading_assistant.py
```

---

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **"Invalid API Key" Error:**
- ✅ Check if API key is copied correctly
- ✅ Verify API key is enabled
- ✅ Check IP restrictions

#### **"Permission Denied" Error:**
- ✅ Enable "Read Info" permission
- ✅ Check if API key is active
- ✅ Verify account verification status

#### **"Rate Limit Exceeded":**
- ⏰ Wait a few minutes
- 🔄 Reduce request frequency
- 📊 Check your API usage limits

---

## 📞 **Need Help?**

### **Binance Support:**
- [Binance Support Center](https://www.binance.com/en/support)
- Live chat available 24/7

### **CryptoPanic Support:**
- [CryptoPanic Help](https://cryptopanic.io/help)

### **Twitter Developer Support:**
- [Twitter Developer Forum](https://twittercommunity.com/)

---

## 🎯 **Next Steps After Setup**

1. ✅ **Test API connection**
2. 🚀 **Train your first model**
3. 📊 **Generate trading signals**
4. 🔄 **Set up continuous monitoring**
5. 📈 **Analyze performance**

---

**Remember**: Start with Testnet mode until you're confident with the system! 🧪
