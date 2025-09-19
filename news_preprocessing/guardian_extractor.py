# File: news_preprocessing/guardian_extractor.py

"""
Advanced scraper for fetching and processing news articles from The Guardian's 'World' RSS feed.
"""

import os
import re
import time
import random
import warnings
import concurrent.futures
import threading
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
import feedparser
import trafilatura
import pandas as pd
from tqdm import tqdm
from dateutil import parser as dtparser
from dateutil import tz
from dateutil.parser import UnknownTimezoneWarning
from readability import Document

# --- Constants and Configuration ---
GUARDIAN_RSS_URL = "https://www.theguardian.com/world/rss"
GLOBAL_MAX_WORKERS   = 8
MAX_WORKERS_PER_HOST = 4
TIMEOUT              = 15
MAX_RETRIES          = 3
MIN_DELAY, MAX_DELAY = 0.35, 0.9
MIN_WORDS            = 150
MAX_ARTICLES_PER_FEED = 50

warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

_TZINFOS = {
    "IST": tz.gettz("Asia/Kolkata"), "BST": tz.gettz("Europe/London"),
    "GMT": tz.gettz("Etc/GMT"), "UTC": tz.gettz("UTC"),
    "EST": tz.gettz("US/Eastern"), "EDT": tz.gettz("US/Eastern"),
}

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

_host_semaphores: Dict[str, threading.Semaphore] = {}
_host_lock = threading.Lock()

def _parse_date_safe(val: Optional[str]) -> Optional[pd.Timestamp]:
    if not val: return None
    try:
        dt = dtparser.parse(val, tzinfos=_TZINFOS)
        return pd.to_datetime(dt)
    except Exception:
        return None

def _make_headers(referer: Optional[str] = None) -> Dict[str, str]:
    h = {"User-Agent": random.choice(_UA_POOL)}
    if referer: h["Referer"] = referer
    return h

def _fetch_html(url: str, referer: Optional[str] = None) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_make_headers(referer), timeout=TIMEOUT, allow_redirects=True)
            if 200 <= resp.status_code < 300:
                return resp.text
            if resp.status_code in (403, 429):
                time.sleep(1.5 * attempt)
        except requests.RequestException:
            time.sleep(0.6 * attempt)
    return None

def _extract_main_text(html: str) -> str:
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if text and len(text.split()) >= 80:
        return re.sub(r"\s+", " ", text.strip())
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return re.sub(r"\s+", " ", (text or "").strip())
    except Exception:
        return ""

def _is_valid_article_link(url: str) -> bool:
    BAD_PATTERNS = ["/live/", "/video/", "/audio/", "/gallery/", "/picture/", "/interactive/"]
    if not url or not url.startswith("http"): return False
    return not any(p in url.lower() for p in BAD_PATTERNS)

def _entry_to_row(entry: Any, topic: str, rss_url: str) -> Optional[Dict[str, Any]]:
    link = entry.get("link")
    title = (entry.get("title") or "").strip()
    if not link or not title or not _is_valid_article_link(link):
        return None

    host = urlparse(link).netloc.lower()
    with _host_lock:
        if host not in _host_semaphores:
            _host_semaphores[host] = threading.Semaphore(MAX_WORKERS_PER_HOST)
    
    with _host_semaphores[host]:
        html = _fetch_html(link, referer=rss_url)
        if not html: return None
        content = _extract_main_text(html)
        if not content or len(content.split()) < MIN_WORDS:
            return None
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    pub_date_str = entry.get("published") or entry.get("updated") or entry.get("pubDate")
    return {
        "source": "The Guardian",
        "topic": topic,
        "title": title,
        "link": link,
        "published_date": _parse_date_safe(pub_date_str),
        "content": content
    }

def scrape_guardian_world_news() -> pd.DataFrame:
    rss_resp = requests.get(GUARDIAN_RSS_URL, headers=_make_headers(), timeout=TIMEOUT)
    rss_resp.raise_for_status()
    parsed = feedparser.parse(rss_resp.content)

    topic = (getattr(parsed, "feed", {}).get("title") or "World").replace(" | The Guardian", "").replace("World news", "World").strip()
    entries = list(getattr(parsed, "entries", []))[:MAX_ARTICLES_PER_FEED]

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=GLOBAL_MAX_WORKERS) as ex:
        futures = [ex.submit(_entry_to_row, e, topic, GUARDIAN_RSS_URL) for e in entries]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scraping Guardian"):
            if row := fut.result():
                rows.append(row)

    if not rows:
        print("No articles extracted from The Guardian.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    
    final_columns = ['source', 'topic', 'title', 'published_date', 'link', 'content']
    df = df[final_columns]
    
    print(f"✅ Scraped and processed {len(df)} articles from The Guardian.")
    return df