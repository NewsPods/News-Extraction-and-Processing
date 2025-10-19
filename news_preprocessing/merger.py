"""
Main orchestrator to fetch articles from multiple sources, merge them,
and perform final preprocessing and cleaning.
"""
import pandas as pd
# No longer need 'datetime' or 'date' as pandas handles this robustly
# from datetime import datetime, date

# --- Import scraper functions from other modules in the package ---
# The '.' tells Python to look for these files in the same directory (news_preprocessing)
from .dainik_extractor import scrape_dainik_bhaskar
from .toi_extractor import scrape_times_of_india
from .guardian_extractor import scrape_guardian_world_news

# --- Main Public Function ---
def merge_and_preprocess_articles(filter_today: bool = True) -> pd.DataFrame:
    """
    Fetches articles from all defined sources, merges them into a single
    DataFrame, and performs cleaning and standardization.

    Args:
        filter_today (bool): If True, filters the final DataFrame to only
                             include articles published within the current
                             local day (IST). Defaults to True.

    Returns:
        pd.DataFrame: A clean, merged DataFrame of news articles.
                      Returns an empty DataFrame if no articles are found.
    """
    print("--- Starting the article merging and preprocessing pipeline ---")

    # --- 1. Scrape data from all sources ---
    print("\n[Step 1/4] Scraping from all sources...")
    df_db = scrape_dainik_bhaskar()
    df_toi = scrape_times_of_india()
    df_guardian = scrape_guardian_world_news()
    
    # Store all non-empty dataframes in a list
    all_dfs = [df for df in [df_db, df_toi, df_guardian] if not df.empty]
    
    if not all_dfs:
        print("❌ No articles were scraped from any source. Exiting.")
        return pd.DataFrame()
        
    # --- 2. Harmonize and Merge ---
    print("\n[Step 2/4] Harmonizing and merging DataFrames...")
    try:
        # Note: It's best practice for each scraper to add its own 'source' column.
        # This code provides a fallback for robustness.
        if 'source' not in df_db.columns and not df_db.empty: df_db['source'] = 'Dainik Bhaskar'
        if 'source' not in df_toi.columns and not df_toi.empty: df_toi['source'] = 'Times of India'
        if 'source' not in df_guardian.columns and not df_guardian.empty: df_guardian['source'] = 'The Guardian'

        # Define the final set of columns for perfect consistency
        final_columns = ['source', 'topic', 'title', 'published_date', 'link', 'content']

        # Align all dataframes to have the same columns
        aligned_dfs = [df[final_columns] for df in all_dfs]

        # Merge the prepared DataFrames
        merged_df = pd.concat(aligned_dfs, ignore_index=True)
        print(f"✅ Merged a total of {len(merged_df)} articles.")
        
    except Exception as e:
        print(f"❌ Error during harmonization and merging: {e}")
        return pd.DataFrame()

    # --- 3. Standardize Date Column ---
    print("\n[Step 3/4] Standardizing 'published_date' column to UTC...")
    # --- MODIFICATION START ---
    # Convert all date strings to timezone-aware UTC timestamps.
    # We REMOVE .dt.normalize() to keep the precise time, which is crucial for the filter.
    merged_df['published_date'] = pd.to_datetime(
        merged_df['published_date'],
        format='mixed',
        errors='coerce',
        utc=True 
    )
    # --- MODIFICATION END ---
    
    # Drop rows where date could not be parsed
    merged_df.dropna(subset=['published_date'], inplace=True)
    
    # --- 4. Filter for Today's Articles (Optional) ---
    if filter_today:
        # --- MODIFICATION START: Timezone-aware filtering ---
        print("\n[Step 4/4] Filtering for articles in today's IST window...")
        original_count = len(merged_df)

        # Define "today" as the 24-hour period from midnight to midnight in India.
        today_in_india = pd.Timestamp.now(tz='Asia/Kolkata').normalize()
        tomorrow_in_india = today_in_india + pd.Timedelta(days=1)
        
        # Filter the UTC dates to find all articles that fall within our IST day.
        # Pandas handles the timezone comparison automatically.
        merged_df = merged_df[
            (merged_df['published_date'] >= today_in_india) &
            (merged_df['published_date'] < tomorrow_in_india)
        ].reset_index(drop=True)

        print(f"- Kept {len(merged_df)} articles out of {original_count} that were published on your 'today'.")
        # --- MODIFICATION END ---
    else:
        print("\n[Step 4/4] Skipping filter for today's articles.")

    print("\n--- Pipeline finished successfully! ---")
    return merged_df

if __name__ == '__main__':
    # This block allows you to run the merger directly for testing
    # Note: For this to work, you must run it as a module from the parent directory
    # Example command from outside the 'news_preprocessing' folder:
    # python -m news_preprocessing.merger
    
    final_df = merge_and_preprocess_articles(filter_today=True)
    
    if not final_df.empty:
        print("\n--- Final Merged DataFrame ---")
        print("Sample of final data:")
        # Use print() instead of display() for compatibility with standard Python scripts
        print(final_df.sample(min(5, len(final_df))))
        
        print("\nArticle count by source:")
        print(final_df['source'].value_counts())