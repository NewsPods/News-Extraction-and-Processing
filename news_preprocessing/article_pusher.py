"""
Connects to a CockroachDB database and pushes the final, processed
news articles DataFrame into the 'articles' table.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

# --- Import functions from your other modules ---
from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan
from .final_news_df import generate_final_news_df_langchain

# --- Configuration ---
load_dotenv()
CONN_STRING = os.environ.get("COCKROACHDB_CONN_STRING")
TARGET_TABLE = "articles"

# --- Main Public Function ---
def push_articles_to_db(df: pd.DataFrame):
    """
    Pushes a DataFrame of processed news articles to the CockroachDB table.

    Args:
        df (pd.DataFrame): The final DataFrame of articles to be inserted.
    """
    if df.empty:
        print("Input DataFrame is empty. Nothing to push to the database.")
        return

    if not CONN_STRING:
        print("❌ Error: COCKROACHDB_CONN_STRING environment variable not set.")
        return

    print(f"\n--- [Step 5/5] Preparing to push {len(df)} articles to CockroachDB ---")

    try:
        # --- 1. Prepare DataFrame to match the table schema ---
        # Create a copy to avoid modifying the original DataFrame
        df_to_push = df.copy()

        # Rename columns as per your schema mapping
        column_map = {
            'title': 'title',
            'content': 'description',
            'source': 'news_source',
            'topic': 'news_section',
            'published_date': 'created_at'
        }
        df_to_push.rename(columns=column_map, inplace=True)
        
        # Ensure 'created_at' is a proper datetime object for the database
        df_to_push['created_at'] = pd.to_datetime(df_to_push['created_at'], errors='coerce')

        # Select only the columns that exist in the target table
        final_columns = ['title', 'description', 'news_source', 'news_section', 'created_at']
        df_to_push = df_to_push[final_columns]
        
        # --- 2. Connect to the database and insert data ---
        print("Connecting to the database...")
        
        try:
            # First attempt: Use CockroachDB-specific configuration
            engine = create_engine(
                CONN_STRING,
                poolclass=NullPool,  # Disable connection pooling
                connect_args={
                    "application_name": "news_processor",
                    "options": "-c default_transaction_isolation=serializable"
                }
            )
        except Exception as e:
            print(f"First connection attempt failed: {e}")
            print("Trying alternative connection method...")
            # Fallback: Use basic PostgreSQL driver without special options
            engine = create_engine(CONN_STRING, poolclass=NullPool)
        
        with engine.connect() as connection:
            print(f"Pushing data to the '{TARGET_TABLE}' table...")
            # Use to_sql to efficiently insert the DataFrame records
            # 'if_exists="append"' adds the new articles without deleting existing ones
            df_to_push.to_sql(
                TARGET_TABLE,
                con=connection,
                if_exists="append",
                index=False,
                method='multi' # Efficient method for inserting many rows
            )
    
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")
    
    else:
        print(f"✅ Success! {len(df_to_push)} articles have been pushed to the database.")


# --- Main Execution Block ---
if __name__ == '__main__':
    # This block runs the entire pipeline from start to finish
    
    # Step 1: Fetch and merge
    todays_articles_df = merge_and_preprocess_articles(filter_today=True)
    
    if not todays_articles_df.empty:
        # Step 2: Cluster articles
        clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)
        
        # Step 3: Synthesize and create final DataFrame
        final_df, _ = generate_final_news_df_langchain(clustered_df, unique_df)
        
        # Step 4: Push the final DataFrame to the database
        push_articles_to_db(final_df)
        
    else:
        print("\n--- Pipeline finished: No articles found to process. ---")