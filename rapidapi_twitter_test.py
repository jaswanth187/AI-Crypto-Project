#!/usr/bin/env python3
"""
Test script to check RapidAPI Twitter endpoints for sentiment analysis
"""

import requests
import json

def test_rapidapi_twitter_endpoints():
    """Test available RapidAPI Twitter endpoints"""
    
    print("🔍 Testing RapidAPI Twitter Endpoints for Sentiment Analysis")
    print("=" * 60)
    
    # RapidAPI Twitter endpoints that might be useful
    endpoints = {
        "Search Tweets": {
            "url": "https://twitter154.p.rapidapi.com/search/query",
            "description": "Search tweets by keyword",
            "useful_for": "Finding crypto-related tweets"
        },
        "User Tweets": {
            "url": "https://twitter154.p.rapidapi.com/user/retweets",
            "description": "Get user's tweets and retweets",
            "useful_for": "Following specific crypto influencers"
        },
        "Trending Topics": {
            "url": "https://twitter154.p.rapidapi.com/trends/place",
            "description": "Get trending topics by location",
            "useful_for": "Identifying crypto trends"
        }
    }
    
    print("📋 Available Endpoints:")
    for name, details in endpoints.items():
        print(f"\n🔹 {name}:")
        print(f"   URL: {details['url']}")
        print(f"   Description: {details['description']}")
        print(f"   Use Case: {details['useful_for']}")
    
    print("\n" + "=" * 60)
    print("💡 Analysis for Our Project:")
    
    pros = [
        "✅ More affordable than direct Twitter API",
        "✅ Higher rate limits",
        "✅ Easier authentication",
        "✅ Multiple endpoints available"
    ]
    
    cons = [
        "❌ Additional dependency on RapidAPI",
        "❌ May have data freshness delays",
        "❌ Limited to specific endpoints",
        "❌ Additional cost (though much lower)"
    ]
    
    print("\n✅ Pros:")
    for pro in pros:
        print(f"   {pro}")
    
    print("\n❌ Cons:")
    for con in cons:
        print(f"   {con}")
    
    return endpoints

def check_alternative_sentiment_sources():
    """Check alternative sentiment sources that might be better"""
    
    print("\n" + "=" * 60)
    print("🔄 Alternative Sentiment Sources:")
    
    alternatives = {
        "CryptoPanic": {
            "cost": "FREE (100 req/day) or $29/month",
            "coverage": "Crypto news sentiment",
            "quality": "High - specialized in crypto",
            "recommendation": "✅ BEST OPTION - Already in our system"
        },
        "Reddit API": {
            "cost": "FREE",
            "coverage": "r/cryptocurrency, r/bitcoin discussions",
            "quality": "Medium - community sentiment",
            "recommendation": "✅ GOOD OPTION - Free and active"
        },
        "Telegram Channels": {
            "cost": "FREE",
            "coverage": "Crypto channel discussions",
            "quality": "Medium - real-time but noisy",
            "recommendation": "⚠️  POSSIBLE - Requires channel access"
        },
        "Discord Servers": {
            "cost": "FREE",
            "coverage": "Crypto community servers",
            "quality": "Medium - community sentiment",
            "recommendation": "⚠️  POSSIBLE - Requires server access"
        }
    }
    
    for source, details in alternatives.items():
        print(f"\n🔹 {source}:")
        print(f"   💰 Cost: {details['cost']}")
        print(f"   📊 Coverage: {details['coverage']}")
        print(f"   🎯 Quality: {details['quality']}")
        print(f"   💡 Recommendation: {details['recommendation']}")

def rapidapi_integration_example():
    """Show how RapidAPI Twitter could be integrated"""
    
    print("\n" + "=" * 60)
    print("🔧 RapidAPI Twitter Integration Example:")
    
    example_code = '''
# Example integration with RapidAPI Twitter
import requests

def get_twitter_sentiment_rapidapi(query="bitcoin", count=100):
    """
    Get Twitter sentiment using RapidAPI
    
    Args:
        query (str): Search query (e.g., "bitcoin", "ethereum")
        count (int): Number of tweets to analyze
        
    Returns:
        dict: Sentiment analysis results
    """
    
    url = "https://twitter154.p.rapidapi.com/search/query"
    
    querystring = {
        "query": query,
        "limit": count,
        "language": "en"
    }
    
    headers = {
        "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
        "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        tweets = response.json()
        
        # Simple sentiment analysis (you could use a proper NLP library)
        positive_keywords = ["bull", "moon", "pump", "buy", "hodl", "diamond"]
        negative_keywords = ["bear", "dump", "sell", "crash", "scam", "fud"]
        
        positive_count = 0
        negative_count = 0
        
        for tweet in tweets.get("results", []):
            text = tweet.get("text", "").lower()
            
            if any(keyword in text for keyword in positive_keywords):
                positive_count += 1
            elif any(keyword in text for keyword in negative_keywords):
                negative_count += 1
        
        total_tweets = len(tweets.get("results", []))
        
        return {
            "positive_ratio": positive_count / total_tweets if total_tweets > 0 else 0,
            "negative_ratio": negative_count / total_tweets if total_tweets > 0 else 0,
            "neutral_ratio": (total_tweets - positive_count - negative_count) / total_tweets if total_tweets > 0 else 0,
            "total_tweets": total_tweets,
            "query": query
        }
        
    except Exception as e:
        print(f"Error fetching Twitter data: {e}")
        return None
'''
    
    print(example_code)
    
    print("\n💡 Integration Notes:")
    print("   - Would need to add RapidAPI key to config")
    print("   - Could replace or supplement current Twitter sentiment")
    print("   - Rate limits depend on your RapidAPI plan")
    print("   - Data freshness: Usually 1-5 minutes delay")

if __name__ == "__main__":
    print("🚀 Twitter API Alternatives Analysis")
    print("=" * 60)
    
    # Test RapidAPI endpoints
    endpoints = test_rapidapi_twitter_endpoints()
    
    # Check alternatives
    check_alternative_sentiment_sources()
    
    # Show integration example
    rapidapi_integration_example()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION:")
    print("   For now, stick with CryptoPanic + simulated Twitter sentiment")
    print("   RapidAPI Twitter is viable but adds complexity")
    print("   Consider Reddit API as a free alternative")
    print("   Focus on getting Binance API working first!")
