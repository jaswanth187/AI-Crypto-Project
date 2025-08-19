import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from loguru import logger
from config import Config
from transformers import pipeline

class SentimentCollector:
    def __init__(self):
        """Initialize sentiment data collectors"""
        self.cryptopanic_api_key = Config.CRYPTOPANIC_API_KEY
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="curiousily/tiny-crypto-sentiment-analysis")

    def get_cryptopanic_sentiment(self, symbol='BTC', hours_back=72):  # Increased to 72 hours
        """
        Fetch sentiment data from CryptoPanic API
        
        Args:
            symbol (str): Cryptocurrency symbol
            hours_back (int): Hours to look back for news
            
        Returns:
            dict: Sentiment metrics
        """
        try:
            logger.info(f"Fetching CryptoPanic sentiment for {symbol}")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # API endpoint
            url = "https://cryptopanic.com/api/v1/posts/"
            
            params = {
                'auth_token': self.cryptopanic_api_key,
                'currencies': symbol,
                'public': 'true',
                'filter': 'hot',
                'kind': 'news'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                logger.warning("No results found in CryptoPanic response")
                return self._default_sentiment_metrics()
            
            # Process news items
            news_items = []
            for item in data['results']:
                # Check if news is within our time range
                published_at = datetime.fromisoformat(item['published_at'].replace('Z', '+00:00'))
                # Make start_time timezone-aware for comparison
                start_time_aware = start_time.replace(tzinfo=published_at.tzinfo)
                if published_at >= start_time_aware:
                    sentiment_result = self.sentiment_pipeline(item['title'])
                    sentiment = sentiment_result[0]['label']
                    news_items.append({
                        'title': item['title'],
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'votes': item.get('votes', {}).get('positive', 0) - item.get('votes', {}).get('negative', 0)
                    })
            
            # Calculate sentiment metrics
            if not news_items:
                logger.info("No recent news found")
                return self._default_sentiment_metrics()
            
            # Count sentiment types
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_votes = 0
            
            for item in news_items:
                sentiment_counts[item['sentiment']] += 1
                total_votes += item['votes']
            
            # Calculate sentiment score (-1 to 1)
            total_news = len(news_items)
            positive_ratio = sentiment_counts['positive'] / total_news
            negative_ratio = sentiment_counts['negative'] / total_news
            sentiment_score = positive_ratio - negative_ratio
            
            # Calculate average votes per news item
            avg_votes = total_votes / total_news if total_news > 0 else 0
            
            metrics = {
                'sentiment_score': sentiment_score,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': sentiment_counts['neutral'] / total_news,
                'total_news': total_news,
                'avg_votes': avg_votes,
                'news_items': news_items[:5]  # Keep top 5 for analysis
            }
            
            logger.info(f"Processed {total_news} news items, sentiment score: {sentiment_score:.3f}")
            return metrics
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching CryptoPanic data: {e}")
            return self._default_sentiment_metrics()
        except Exception as e:
            logger.error(f"Error processing CryptoPanic data: {e}")
            return self._default_sentiment_metrics()
    
    def get_fear_greed_index(self):
        """Get Fear & Greed Index from the real API"""
        try:
            logger.info("Fetching Fear & Greed Index from API")
            
            # Fear & Greed Index API endpoint
            url = "https://api.alternative.me/fng/"
            
            # Parameters for the API request
            params = {
                'limit': 1,  # Get the latest value
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                logger.warning("No data found in Fear & Greed Index response")
                return self._default_fear_greed_metrics()
            
            # Extract the latest fear & greed data
            latest_data = data['data'][0]
            
            # Parse the fear & greed value
            fear_greed_value = int(latest_data['value'])
            timestamp_str = latest_data['timestamp']
            
            # Convert timestamp to datetime
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            
            # Determine sentiment based on value
            if fear_greed_value < 25:
                sentiment = "Extreme Fear"
            elif fear_greed_value < 45:
                sentiment = "Fear"
            elif fear_greed_value < 55:
                sentiment = "Neutral"
            elif fear_greed_value < 75:
                sentiment = "Greed"
            else:
                sentiment = "Extreme Greed"
            
            result = {
                'value': fear_greed_value,
                'sentiment': sentiment,
                'timestamp': timestamp,
                'value_classification': latest_data.get('value_classification', sentiment)
            }
            
            logger.info(f"Fear & Greed Index: {fear_greed_value} ({sentiment})")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching Fear & Greed Index: {e}")
            return self._default_fear_greed_metrics()
        except Exception as e:
            logger.error(f"Error processing Fear & Greed Index data: {e}")
            return self._default_fear_greed_metrics()
    
    def _default_fear_greed_metrics(self):
        """Default Fear & Greed metrics when API calls fail"""
        return {
            'value': 50,
            'sentiment': 'Neutral',
            'timestamp': datetime.now(),
            'value_classification': 'Neutral'
        }
    
    def get_combined_sentiment(self, symbol='BTC', hours_back=24):
        """
        Get combined sentiment from CryptoPanic and Fear & Greed Index
        
        Args:
            symbol (str): Cryptocurrency symbol
            hours_back (int): Hours to look back
            
        Returns:
            dict: Combined sentiment metrics
        """
        try:
            logger.info("Collecting combined sentiment data")
            
            # Get sentiment from different sources
            cryptopanic = self.get_cryptopanic_sentiment(symbol, hours_back)
            time.sleep(5) # Add a 5-second delay to avoid rate limiting
            fear_greed = self.get_fear_greed_index()
            
            # Combine sentiment scores (adjusted weights without Twitter)
            # CryptoPanic: 70%, Fear & Greed: 30%
            cryptopanic_weight = 0.7
            fear_greed_weight = 0.3
            
            # Normalize Fear & Greed to -1 to 1 scale
            fear_greed_normalized = (fear_greed['value'] - 50) / 50
            
            # Calculate combined sentiment score
            combined_score = (
                cryptopanic['sentiment_score'] * cryptopanic_weight +
                fear_greed_normalized * fear_greed_weight
            )
            
            # Determine overall sentiment
            if combined_score > 0.3:
                overall_sentiment = "Bullish"
            elif combined_score < -0.3:
                overall_sentiment = "Bearish"
            else:
                overall_sentiment = "Neutral"
            
            combined_metrics = {
                'combined_sentiment_score': combined_score,
                'overall_sentiment': overall_sentiment,
                'cryptopanic': cryptopanic,
                'fear_greed': fear_greed,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Combined sentiment: {overall_sentiment} (score: {combined_score:.3f})")
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error getting combined sentiment: {e}")
            return self._default_combined_metrics()
    
    def _default_sentiment_metrics(self):
        """Default sentiment metrics when API calls fail"""
        return {
            'sentiment_score': 0.0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'total_news': 0,
            'avg_votes': 0,
            'news_items': []
        }
    
    def _default_combined_metrics(self):
        """Default combined metrics when collection fails"""
        return {
            'combined_sentiment_score': 0.0,
            'overall_sentiment': 'Neutral',
            'cryptopanic': self._default_sentiment_metrics(),
            'fear_greed': self._default_fear_greed_metrics(),
            'timestamp': datetime.now()
        }

if __name__ == "__main__":
    # Test the sentiment collector
    try:
        collector = SentimentCollector()
        
        # Test combined sentiment
        print("\nTesting combined sentiment...")
        combined = collector.get_combined_sentiment('BTC', 24)
        print(f"Combined sentiment: {combined['overall_sentiment']} (score: {combined['combined_sentiment_score']:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
