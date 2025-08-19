import requests
from datetime import datetime, timedelta, timezone
import pytz

API_KEY = "19964102c9758166f5c863559e8a8635c3c2fc04"  # Replace with your CryptoPanic API key

def fetch_latest_bitcoin_news(debug_mode=False, remove_time_filter=False):
    url = "https://cryptopanic.com/api/v1/posts/"
    
    # Calculate the timestamp for 24 hours ago in UTC
    now_utc = datetime.now(timezone.utc)
    twenty_four_hours_ago = now_utc - timedelta(hours=24)
    
    # Try without time filtering first to see latest available news
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "filter": "all",  # Get all news
        "kind": "news",
        "public": True,
        "page": 1,
        "metadata": True,
    }
    
    # Only add time filter if specifically requested
    if not remove_time_filter:
        since_timestamp = twenty_four_hours_ago.isoformat()
        params["since"] = since_timestamp
    
    if debug_mode:
        print(f"API URL: {url}")
        print(f"Parameters: {params}")
        print(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    filter_text = "WITHOUT time filtering" if remove_time_filter else "from the last 24 hours"
    print(f"Fetching Bitcoin news {filter_text}...")
    if not remove_time_filter:
        print(f"Since: {twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            posts = data.get("results", [])
            
            if not posts:
                print("No Bitcoin news found in the last 24 hours.")
                return
            
            # Sort posts by published_at timestamp (latest first)
            sorted_posts = sorted(
                posts, 
                key=lambda x: datetime.fromisoformat(x.get('published_at', '').replace('Z', '+00:00')), 
                reverse=True
            )
            
            print(f"Found {len(sorted_posts)} Bitcoin news articles {filter_text}:\n")
            
            # Show the timestamp of the most recent article
            if sorted_posts:
                latest_post = sorted_posts[0]
                latest_time = latest_post.get('published_at', '')
                if latest_time:
                    try:
                        latest_pub_date = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))
                        time_since_latest = now_utc - latest_pub_date
                        print(f"Most recent article is from: {latest_pub_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                        print(f"That's {time_since_latest.total_seconds()/3600:.1f} hours ago\n")
                    except:
                        pass
            
            for i, post in enumerate(sorted_posts, 1):
                title = post.get('title', 'No Title')
                url = post.get('url', 'No URL available')
                published_at = post.get('published_at', 'No Date')
                
                # Get source information with better handling
                source_info = post.get('source', {})
                if isinstance(source_info, dict):
                    source = source_info.get('title', source_info.get('name', 'Unknown Source'))
                else:
                    source = 'Unknown Source'
                
                # Get domain if URL is available
                domain = ""
                if url and url != 'No URL available':
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            source = f"{source} ({domain})"
                    except:
                        pass
                
                # Parse and format the published date
                if published_at and published_at != 'No Date':
                    try:
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = pub_date.strftime('%Y-%m-%d %H:%M:%S UTC')
                        
                        # Calculate time difference more accurately
                        time_diff = now_utc - pub_date
                        total_seconds = time_diff.total_seconds()
                        
                        if debug_mode:
                            print(f"   [Debug] Time diff: {total_seconds} seconds")
                        
                        if total_seconds < 60:
                            time_ago = f"{int(total_seconds)} second(s) ago"
                        elif total_seconds < 3600:  # Less than 1 hour
                            minutes = int(total_seconds // 60)
                            time_ago = f"{minutes} minute(s) ago"
                        elif total_seconds < 86400:  # Less than 24 hours
                            hours = int(total_seconds // 3600)
                            minutes = int((total_seconds % 3600) // 60)
                            if minutes > 0:
                                time_ago = f"{hours}h {minutes}m ago"
                            else:
                                time_ago = f"{hours} hour(s) ago"
                        else:
                            days = int(total_seconds // 86400)
                            hours = int((total_seconds % 86400) // 3600)
                            if hours > 0:
                                time_ago = f"{days}d {hours}h ago"
                            else:
                                time_ago = f"{days} day(s) ago"
                            
                    except Exception as e:
                        formatted_date = published_at
                        time_ago = "Unknown"
                        print(f"   [Debug] Date parsing error: {e}")
                else:
                    formatted_date = "No Date"
                    time_ago = "Unknown"
                
                print(f"{i}. {title}")
                print(f"   Source: {source}")
                print(f"   Published: {formatted_date} ({time_ago})")
                print(f"   URL: {url}")
                print("-" * 60)
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_multiple_pages(max_pages=3):
    """Fetch news from multiple pages to get more results"""
    url = "https://cryptopanic.com/api/v1/posts/"
    
    now_utc = datetime.now(timezone.utc)
    twenty_four_hours_ago = now_utc - timedelta(hours=24)
    since_timestamp = twenty_four_hours_ago.isoformat()
    
    all_posts = []
    
    for page in range(1, max_pages + 1):
        params = {
            "auth_token": API_KEY,
            "currencies": "BTC",
            "filter": "hot",
            "kind": "news",
            "public": True,
            "page": page,
            "since": since_timestamp,
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                posts = data.get("results", [])
                
                if not posts:  # No more posts available
                    break
                    
                all_posts.extend(posts)
                print(f"Fetched page {page}: {len(posts)} articles")
                
            else:
                print(f"Error fetching page {page}: {response.status_code}")
                break
                
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    # Sort all posts by published date (latest first)
    if all_posts:
        sorted_posts = sorted(
            all_posts, 
            key=lambda x: datetime.fromisoformat(x.get('published_at', '').replace('Z', '+00:00')), 
            reverse=True
        )
        
        print(f"\nTotal articles found: {len(sorted_posts)}")
        return sorted_posts
    
    return []

# Alternative function without time filtering to test API response
def fetch_recent_bitcoin_news_simple():
    """Fetch recent news without time filtering to see what we get"""
    url = "https://cryptopanic.com/api/v1/posts/"
    
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "kind": "news",
        "public": True,
        "metadata": True,
        "page": 1,
    }
    
    print("Fetching recent Bitcoin news (no time filter)...")
    print("=" * 60)
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            posts = data.get("results", [])
            
            print(f"Raw API response structure:")
            if posts:
                print(f"First post keys: {list(posts[0].keys())}")
                print(f"First post example: {posts[0]}")
                print("-" * 60)
            
            return posts
            
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

# Main execution
if __name__ == "__main__":
    if not API_KEY:
        print("Please set your CryptoPanic API key in the API_KEY variable.")
    else:
        # Test 1: Get latest news WITHOUT time filtering to see what's actually available
        print("=" * 80)
        print("TEST 1: FETCHING LATEST NEWS (NO TIME FILTER)")
        print("=" * 80)
        fetch_latest_bitcoin_news(debug_mode=True, remove_time_filter=True)
        
        # Test 2: Get news with 24-hour filter
        print("\n" + "=" * 80)
        print("TEST 2: FETCHING NEWS WITH 24-HOUR FILTER")
        print("=" * 80)
        fetch_latest_bitcoin_news(debug_mode=True, remove_time_filter=False)
        
        # Test 3: Show raw API response structure
        print("\n" + "="*80)
        print("TEST 3: RAW API RESPONSE STRUCTURE")
        print("="*80)
        raw_posts = fetch_recent_bitcoin_news_simple()
        if raw_posts:
            print(f"\nMost recent post timestamp: {raw_posts[0].get('published_at', 'N/A')}")
            print(f"Oldest post timestamp: {raw_posts[-1].get('published_at', 'N/A')}")
        
        # Test 4: Try different time ranges
        print("\n" + "="*80)
        print("TEST 4: TRYING SHORTER TIME RANGES")
        print("="*80)
        
        # Try just 1 hour ago
        now_utc = datetime.now(timezone.utc)
        one_hour_ago = now_utc - timedelta(hours=1)
        
        params_1h = {
            "auth_token": API_KEY,
            "currencies": "BTC",
            "filter": "all",
            "kind": "news",
            "public": True,
            "since": one_hour_ago.isoformat(),
            "metadata": True,
        }
        
        print(f"Trying to fetch news from last 1 hour (since {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S UTC')})...")
        
        try:
            response = requests.get("https://cryptopanic.com/api/v1/posts/", params=params_1h)
            if response.status_code == 200:
                data = response.json()
                posts_1h = data.get("results", [])
                print(f"Found {len(posts_1h)} articles from the last 1 hour")
                if posts_1h:
                    for post in posts_1h[:3]:  # Show first 3
                        print(f"  - {post.get('title', 'No Title')} ({post.get('published_at', 'No Date')})")
                else:
                    print("No articles found in the last 1 hour - this suggests API data delay")
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")