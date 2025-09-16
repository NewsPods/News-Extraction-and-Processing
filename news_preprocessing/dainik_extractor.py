"""
Scraper for fetching and processing news articles from the Dainik Bhaskar (English) RSS feeds.
"""

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from typing import List, Dict, Set, Any

# --- Constants and Configuration ---
HUB_URL = "https://www.bhaskarenglish.in/rss"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en;q=0.9,hi;q=0.8",
})

# --- Helper Functions ---
def _fetch_feed_urls(hub_url: str) -> List[str]:
    """Fetches all valid RSS feed XML links from the main hub page."""
    print(f"Fetching feed index from: {hub_url}")
    try:
        resp = SESSION.get(hub_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        feed_urls: Set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if re.search(r"rss-v1--category-\d+\.xml$", href):
                abs_url = href if href.startswith("http") else urljoin(hub_url, href)
                feed_urls.add(abs_url)
        
        print(f"✅ Found {len(feed_urls)} English feeds.")
        return sorted(list(feed_urls))
    except requests.RequestException as e:
        print(f"❌ Error fetching feed index: {e}")
        return []

def _section_from_channel_link(link: str) -> str:
    """Extracts a section name (e.g., 'business') from a channel URL."""
    try:
        path = urlparse(link).path.strip("/")
        return (path.split("/", 1)[0] if path else "general") or "general"
    except Exception:
        return "general"

def _parse_items_from_feed(rss_url: str) -> List[Dict[str, Any]]:
    """Parses a single RSS feed XML and extracts all article items."""
    records = []
    try:
        resp = SESSION.get(rss_url, timeout=20)
        resp.raise_for_status()
        xml_soup = BeautifulSoup(resp.content, "xml")

        channel = xml_soup.find("channel")
        if not channel:
            return []

        section = _section_from_channel_link(getattr(channel.link, 'text', '').strip())
        
        for item in channel.find_all("item"):
            raw_html = ""
            if item.description and item.description.text:
                raw_html = item.description.text
            
            if not raw_html:
                continue

            text_content = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
            
            records.append({
                "section": section,
                "title": getattr(item.title, 'text', 'No Title').strip(),
                "published_date": getattr(item.pubDate, 'text', 'N/A').strip(),
                "link": getattr(item.link, 'text', None),
                "content": text_content,
            })
    except requests.RequestException:
        # Silently continue if a single feed fails
        pass
    return records

# --- Main Public Function ---
def scrape_dainik_bhaskar() -> pd.DataFrame:
    """
    Scrapes, processes, and cleans news articles from Dainik Bhaskar.

    Returns:
        pd.DataFrame: A DataFrame containing cleaned and formatted articles
                      with columns ['topic', 'published_date', 'title', 'link', 'content'].
                      Returns an empty DataFrame on failure.
    """
    feed_urls = _fetch_feed_urls(HUB_URL)
    if not feed_urls:
        return pd.DataFrame()

    all_records = []
    print("\nReading all feeds and extracting items...")
    for url in tqdm(feed_urls, desc="Processing Dainik Bhaskar Feeds"):
        all_records.extend(_parse_items_from_feed(url))

    if not all_records:
        print("❌ No articles were parsed from any feeds.")
        return pd.DataFrame()

    print(f"✅ Parsed {len(all_records)} total items.")
    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["link"], inplace=True, ignore_index=True)

    # --- Post-processing and Cleaning ---
    print("\nCleaning and formatting the DataFrame...")
    try:
        # 1. Filter out unwanted topics
        exclude_sections = ['local', 'originals', 'lifestyle']
        df = df[~df['section'].isin(exclude_sections)]

        # 2. Select and rename columns
        df = df[['section', 'published_date', 'title', 'link', 'content']]
        df.rename(columns={'section': 'topic'}, inplace=True)

        # 3. Map topic names for consistency
        topic_map = {
            'tech-science': 'Tech-Science',
            'national': 'India',
            'international': 'World',
            'career': 'Education',
            'sports': 'Cricket',
            'entertainment': 'Entertainment',
            'business': 'Business'
        }
        df['topic'] = df['topic'].replace(topic_map)
        
        # Ensure 'topic' column has a consistent type
        df['topic'] = df['topic'].astype(str)

    except Exception as e:
        print(f"❌ An error occurred during DataFrame processing: {e}")
        return pd.DataFrame()
    else:

        print("✅ DataFrame modified successfully.")
        return df

if __name__ == '__main__':
    # This block allows you to run the script directly for testing
    print("Running Dainik Bhaskar scraper as a standalone script...")
    df_dainik = scrape_dainik_bhaskar()
    if not df_dainik.empty:
        print("\n--- Scraper Finished ---")
        print("Sample of extracted data:")
        print(df_dainik.head())
        print("\nArticles by new topic:")
        print(df_dainik['topic'].value_counts())