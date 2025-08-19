from datetime import datetime, timezone, timedelta
import json
import requests

# CryptoPanic API fetcher
def fetch_cryptopanic_news():
    """
    Fetches crypto news from CryptoPanic API
    """
    API_KEY = "19964102c9758166f5c863559e8a8635c3c2fc04"  # CryptoPanic API key
    url = "https://cryptopanic.com/api/v1/posts/"
    
    # Calculate the timestamp for 24 hours ago in UTC
    now_utc = datetime.now(timezone.utc)
    twenty_four_hours_ago = now_utc - timedelta(hours=24)
    since_timestamp = twenty_four_hours_ago.isoformat()
    
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "filter": "all",
        "kind": "news",
        "public": True,
        "page": 1,
        "since": since_timestamp  # Add timestamp to get news from the last 24 hours
    }
    
    try:
        print("Fetching from CryptoPanic...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            posts = data.get("results", [])
            
            all_articles = []
            for post in posts:
                title = post.get('title', 'No Title')
                url = post.get('url', 'No URL available')
                published_at = post.get('published_at', 'No Date')
                
                # Get source information
                source_info = post.get('source', {})
                if isinstance(source_info, dict):
                    source = source_info.get('title', source_info.get('name', 'Unknown Source'))
                else:
                    source = 'Unknown Source'
                
                article = {
                    'title': title,
                    'url': url,
                    'published': published_at,
                    'source': f"CryptoPanic - {source}",
                }
                all_articles.append(article)
            
            print(f"  Found {len(all_articles)} Bitcoin-related articles from CryptoPanic")
            return all_articles
        else:
            print(f"  CryptoPanic API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"  Error fetching from CryptoPanic: {e}")
        return []

# RSS feed fetcher for crypto news
def fetch_crypto_rss_comprehensive():
    """
    Fetches crypto news from multiple RSS sources
    """
    from datetime import datetime, timezone
    try:
        import feedparser
        
        print("Installing feedparser if needed...")
        
    except ImportError:
        print("Installing feedparser...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "feedparser"])
        import feedparser
    
    rss_feeds = {
        'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'CoinTelegraph': 'https://cointelegraph.com/rss',
        'Bitcoin Magazine': 'https://bitcoinmagazine.com/.rss/full/',
        'Decrypt': 'https://decrypt.co/feed',
        'CryptoNews': 'https://cryptonews.com/news/feed/',
        'NewsBTC': 'https://www.newsbtc.com/feed/',
        'CryptoPotato': 'https://cryptopotato.com/feed/',
    }
    
    all_articles = []
    now = datetime.now(timezone.utc)
    
    for source_name, rss_url in rss_feeds.items():
        try:
            print(f"Fetching from {source_name}...")
            feed = feedparser.parse(rss_url)
            
            if hasattr(feed, 'entries') and feed.entries:
                for entry in feed.entries[:10]:  # Get latest 10 from each source
                    try:
                        title = entry.get('title', '').strip()
                        
                        # Filter for Bitcoin/crypto-related content
                        title_lower = title.lower()
                        crypto_keywords = ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain', 'ethereum', 'eth']
                        
                        if any(keyword in title_lower for keyword in crypto_keywords):
                            # Parse publication date
                            pub_date_str = entry.get('published', '')
                            try:
                                if pub_date_str:
                                    pub_date = feedparser._parse_date(pub_date_str)
                                    if pub_date:
                                        pub_datetime = datetime(*pub_date[:6], tzinfo=timezone.utc)
                                        time_diff = now - pub_datetime
                                        hours_ago = time_diff.total_seconds() / 3600
                                        
                                        if hours_ago < 0:
                                            time_ago_str = "Just published"
                                        elif hours_ago < 1:
                                            time_ago_str = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                                        elif hours_ago < 24:
                                            time_ago_str = f"{int(hours_ago)} hours ago"
                                        else:
                                            time_ago_str = f"{int(hours_ago / 24)} days ago"
                                    else:
                                        time_ago_str = "Unknown time"
                                        pub_datetime = None
                                else:
                                    time_ago_str = "Unknown time"
                                    pub_datetime = None
                            except:
                                time_ago_str = "Unknown time"
                                pub_datetime = None
                            
                            article = {
                                'title': title,
                                'url': entry.get('link', 'No URL'),
                                'published': pub_date_str,
                                'published_datetime': pub_datetime,
                                'time_ago': time_ago_str,
                                'source': source_name,
                                'summary': entry.get('summary', entry.get('description', 'No Summary'))
                            }
                            all_articles.append(article)
                    except Exception as e:
                        print(f"  Error processing entry: {e}")
                
                print(f"  Found {len([a for a in all_articles if a['source'] == source_name])} Bitcoin-related articles")
            else:
                print(f"  No entries found in feed")
                
        except Exception as e:
            print(f"  Error fetching {source_name}: {e}")
    
    # Sort by publication date (most recent first)
    try:
        all_articles.sort(
            key=lambda x: x['published_datetime'] if x['published_datetime'] else datetime.min.replace(tzinfo=timezone.utc), 
            reverse=True
        )
    except:
        pass
        
    return all_articles


# Main function to display crypto news titles
def main():
    print("Crypto News Headlines")
    print("=" * 70)
    
    all_articles = []
    
    # Get RSS feed articles
    try:
        rss_articles = fetch_crypto_rss_comprehensive()
        if rss_articles:
            all_articles.extend(rss_articles)
            print(f"Found {len(rss_articles)} Bitcoin-related articles from RSS feeds!")
        else:
            print("No articles from RSS feeds")
    except Exception as e:
        print(f"RSS Feed error: {e}")
        print("Try: pip install feedparser")
    
    # Get CryptoPanic articles
    try:
        cryptopanic_articles = fetch_cryptopanic_news()
        if cryptopanic_articles:
            all_articles.extend(cryptopanic_articles)
            print(f"Found {len(cryptopanic_articles)} Bitcoin-related articles from CryptoPanic!")
        else:
            print("No articles from CryptoPanic")
    except Exception as e:
        print(f"CryptoPanic error: {e}")
    
    # Display all article titles
    if all_articles:
        print(f"\nTotal: {len(all_articles)} Bitcoin-related articles")
        print("\nAll article titles:")
        
        for i, article in enumerate(all_articles, 1):
            print(f"{i:2d}. {article['title']} | {article['source']}")
    else:
        print("\nNo articles found from any source")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()