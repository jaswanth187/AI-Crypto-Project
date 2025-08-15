import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import snscrape.modules.twitter as sntwitter
from loguru import logger
from config import Config

class SentimentCollector:
    def __init__(self):
        """Initialize sentiment data collectors"""
        self.cryptopanic_api_key = Config.CRYPTOPANIC_API_KEY
        self.twitter_bearer_token = Config.TWITTER_BEARER_TOKEN
        
    def get_cryptopanic_sentiment(self, symbol='BTC', hours_back=24):
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
                'filter': 'hot'
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
                if published_at >= start_time:
                    news_items.append({
                        'title': item['title'],
                        'published_at': published_at,
                        'sentiment': item.get('sentiment', 'neutral'),
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
    
    def get_twitter_sentiment(self, query='bitcoin', hours_back=24, max_tweets=100):
        """
        Fetch Twitter sentiment using snscrape (no API key required)
        
        Args:
            query (str): Search query
            hours_back (int): Hours to look back
            max_tweets (int): Maximum tweets to analyze
            
        Returns:
            dict: Twitter sentiment metrics
        """
        try:
            logger.info(f"Fetching Twitter sentiment for query: {query}")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Format query with time filter
            time_query = f"{query} since:{start_time.strftime('%Y-%m-%d')}"
            
            # Collect tweets
            tweets = []
            tweet_count = 0
            
            for tweet in sntwitter.TwitterSearchScraper(time_query).get_items():
                if tweet_count >= max_tweets:
                    break
                
                # Check if tweet is within our time range
                if tweet.date >= start_time:
                    tweets.append({
                        'text': tweet.rawContent,
                        'date': tweet.date,
                        'likes': tweet.likeCount,
                        'retweets': tweet.retweetCount,
                        'replies': tweet.replyCount
                    })
                    tweet_count += 1
                else:
                    break
            
            if not tweets:
                logger.info("No recent tweets found")
                return self._default_twitter_metrics()
            
            # Simple sentiment analysis based on engagement
            total_engagement = sum(t['likes'] + t['retweets'] + t['replies'] for t in tweets)
            avg_engagement = total_engagement / len(tweets)
            
            # Calculate engagement-based sentiment (higher engagement = more positive)
            # This is a simplified approach - in production you'd use a proper sentiment model
            max_possible_engagement = 1000  # Arbitrary baseline
            engagement_sentiment = min(avg_engagement / max_possible_engagement, 1.0)
            
            metrics = {
                'tweet_count': len(tweets),
                'total_engagement': total_engagement,
                'avg_engagement': avg_engagement,
                'engagement_sentiment': engagement_sentiment,
                'sample_tweets': tweets[:3]  # Keep sample for analysis
            }
            
            logger.info(f"Processed {len(tweets)} tweets, engagement sentiment: {engagement_sentiment:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return self._default_twitter_metrics()
    
    def get_fear_greed_index(self):
        """Get Fear & Greed Index (simulated for now)"""
        try:
            # In production, you'd use the actual Fear & Greed Index API
            # For now, we'll simulate it based on market conditions
            logger.info("Fetching Fear & Greed Index (simulated)")
            
            # Simulate fear/greed index (0-100, where 0=extreme fear, 100=extreme greed)
            # In reality, this would come from an API
            fear_greed_value = np.random.randint(20, 80)  # Random for demo
            
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
            
            return {
                'value': fear_greed_value,
                'sentiment': sentiment,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            return {'value': 50, 'sentiment': 'Neutral', 'timestamp': datetime.now()}
    
    def get_combined_sentiment(self, symbol='BTC', hours_back=24):
        """
        Get combined sentiment from all sources
        
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
            twitter = self.get_twitter_sentiment(f"{symbol.lower()}", hours_back)
            fear_greed = self.get_fear_greed_index()
            
            # Combine sentiment scores (weighted average)
            # CryptoPanic: 40%, Twitter: 30%, Fear & Greed: 30%
            cryptopanic_weight = 0.4
            twitter_weight = 0.3
            fear_greed_weight = 0.3
            
            # Normalize Fear & Greed to -1 to 1 scale
            fear_greed_normalized = (fear_greed['value'] - 50) / 50
            
            # Calculate combined sentiment score
            combined_score = (
                cryptopanic['sentiment_score'] * cryptopanic_weight +
                twitter['engagement_sentiment'] * twitter_weight +
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
                'twitter': twitter,
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
    
    def _default_twitter_metrics(self):
        """Default Twitter metrics when scraping fails"""
        return {
            'tweet_count': 0,
            'total_engagement': 0,
            'avg_engagement': 0,
            'engagement_sentiment': 0.0,
            'sample_tweets': []
        }
    
    def _default_combined_metrics(self):
        """Default combined metrics when collection fails"""
        return {
            'combined_sentiment_score': 0.0,
            'overall_sentiment': 'Neutral',
            'cryptopanic': self._default_sentiment_metrics(),
            'twitter': self._default_twitter_metrics(),
            'fear_greed': {'value': 50, 'sentiment': 'Neutral', 'timestamp': datetime.now()},
            'timestamp': datetime.now()
        }

if __name__ == "__main__":
    # Test the sentiment collector
    try:
        collector = SentimentCollector()
        
        # Test CryptoPanic sentiment
        print("Testing CryptoPanic sentiment...")
        cryptopanic = collector.get_cryptopanic_sentiment('BTC', 24)
        print(f"CryptoPanic sentiment score: {cryptopanic['sentiment_score']:.3f}")
        
        # Test Twitter sentiment
        print("\nTesting Twitter sentiment...")
        twitter = collector.get_twitter_sentiment('bitcoin', 24, 50)
        print(f"Twitter engagement sentiment: {twitter['engagement_sentiment']:.3f}")
        
        # Test combined sentiment
        print("\nTesting combined sentiment...")
        combined = collector.get_combined_sentiment('BTC', 24)
        print(f"Combined sentiment: {combined['overall_sentiment']} (score: {combined['combined_sentiment_score']:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
