import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
import trafilatura
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Any

# --- Constants and Configuration ---
HUB_URL = "https://timesofindia.indiatimes.com/rss.cms"
TARGET_TOPICS = [
    "Top Stories", "Most Recent", "India", "World", "Business", "Cricket",
    "Sports", "Science", "Environment", "Tech", "Education", "Entertainment"
]
MAX_WORKERS = 10
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# --- Helper Functions ---
def _find_feed_urls(hub_url: str, topics: List[str]) -> Dict[str, str]:
    feeds_dict = {}
    print(f"Searching for {len(topics)} specific topics on: {hub_url}")
    try:
        response = SESSION.get(hub_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        all_links = soup.find_all('a')
        for link in all_links:
            topic_text = link.text.strip()
            if topic_text in topics:
                rss_url = link['href']
                if not rss_url.startswith('http'):
                    rss_url = "https://timesofindia.indiatimes.com" + rss_url
                feeds_dict[topic_text] = rss_url
        
        print(f"✅ Success! Found {len(feeds_dict)} matching feeds.")
        return feeds_dict
    except requests.RequestException as e:
        print(f"❌ Error fetching feed index: {e}")
        return {}

def _scrape_article_worker(entry: Any, topic: str) -> Dict[str, Any] | None:
    """Worker function to scrape a single article."""
    link = entry.get("link")
    if not link:
        return None
    try:
        response = SESSION.get(link, timeout=15)
        if response.status_code != 200:
            return None # Fail silently if blocked or page not found

        main_text = trafilatura.extract(response.text)
        if main_text:
            return {
                'topic': topic,
                'title': entry.get("title", "No Title"),
                'published_date': entry.get("published", "N/A"),
                'link': link,
                'content': main_text
            }
    except requests.RequestException:
        return None # Fail silently on network errors
    return None

# --- Main Public Function ---
def scrape_times_of_india() -> pd.DataFrame:
    main_feeds_dict = _find_feed_urls(HUB_URL, TARGET_TOPICS)
    if not main_feeds_dict:
        return pd.DataFrame()

    all_articles = []
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for topic, rss_url in tqdm(list(main_feeds_dict.items()), desc="Submitting Jobs"):
            try:
                parsed_feed = feedparser.parse(rss_url)
                for entry in parsed_feed.entries:
                    future = executor.submit(_scrape_article_worker, entry, topic)
                    futures.append(future)
            except Exception:
                continue

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scraping Articles"):
        result = future.result()
        if result:
            all_articles.append(result)

    if not all_articles:
        print("\n❌ No articles were successfully scraped from any feeds.")
        return pd.DataFrame()

    df_toi = pd.DataFrame(all_articles)
    
    # --- Post-processing and Cleaning ---
    try:
        df_toi = df_toi[df_toi['topic'] != 'Most Recent'].reset_index(drop=True)
        df_toi.loc[df_toi['topic'].isin(['Tech', 'Science']), 'topic'] = 'Tech-Science'
        df_toi['content'] = df_toi['content'].astype(str)
        df_toi['word_count'] = df_toi['content'].str.split().str.len()
        df_toi = df_toi[df_toi['word_count'] >= 100].reset_index(drop=True)
    except Exception as e:
        print(f"❌ An error occurred during DataFrame processing: {e}")
        return pd.DataFrame()
    else:
        return df_toi

if __name__ == '__main__':
    print("Running Times of India scraper as a standalone script...")
    df_final_toi = scrape_times_of_india()
    if not df_final_toi.empty:
        print("\n--- Scraper Finished ---")
        print(f"Successfully scraped and processed {len(df_final_toi)} articles.")
        print("Sample of final data:")
        print(df_final_toi.head())