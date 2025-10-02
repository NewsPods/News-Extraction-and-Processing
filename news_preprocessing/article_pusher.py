"""
Connects to CockroachDB and pushes processed news articles into a
normalized two-table schema: 'articles' and 'articles_sections'.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan
from .final_news_df import generate_final_news_df_langchain

# --- Configuration ---
load_dotenv()
CONN_STRING = os.environ.get("COCKROACHDB_CONN_STRING")

def push_articles_to_db(df: pd.DataFrame):
    """
    Pushes a DataFrame of articles to CockroachDB with normalized schema.
    Expects 'topic' column to contain lists of sections.
    
    Args:
        df (pd.DataFrame): DataFrame with 'topic' as list of strings
    """
    if df.empty:
        print("Input DataFrame is empty. Nothing to push.")
        return
    
    if not CONN_STRING:
        print("❌ Error: COCKROACHDB_CONN_STRING not set.")
        return

    print(f"\n--- [Step 5/5] Preparing to push {len(df)} articles to CockroachDB ---")

    try:
        # Create engine
        engine = create_engine(
            CONN_STRING,
            poolclass=NullPool,
            connect_args={
                "application_name": "news_processor"
            }
        )
        
        with engine.begin() as connection:  # Use begin() for automatic transaction
            # --- 1. Prepare articles data ---
            print("Preparing articles data...")
            articles_df = df.copy()
            
            # Rename columns to match database schema
            articles_df.rename(columns={
                'content': 'description',
                'source': 'news_source',
                'published_date': 'created_at'
            }, inplace=True)
            
            # Ensure datetime format
            articles_df['created_at'] = pd.to_datetime(
                articles_df['created_at'], 
                errors='coerce'
            )
            
            # Keep topics for later, then select only columns for articles table
            topics_data = articles_df['topic'].tolist()
            articles_df = articles_df[['title', 'description', 'news_source', 'created_at']]
            
            # --- 2. Insert articles and get their IDs ---
            print(f"Inserting {len(articles_df)} articles...")
            
            # Convert DataFrame to list of dicts for batch insert
            articles_records = articles_df.to_dict('records')
            
            # Build parameterized insert query
            insert_query = text("""
                INSERT INTO public.articles (title, description, news_source, created_at)
                VALUES (:title, :description, :news_source, :created_at)
                RETURNING article_id
            """)
            
            # Execute inserts and collect returned IDs
            article_ids = []
            for record in articles_records:
                result = connection.execute(insert_query, record)
                article_id = result.fetchone()[0]
                article_ids.append(article_id)
            
            print(f"✅ Inserted {len(article_ids)} articles successfully")
            
            # --- 3. Prepare and insert sections ---
            print("Preparing sections data...")
            sections_records = []
            
            for idx, topics in enumerate(topics_data):
                article_id = article_ids[idx]
                
                # Ensure topics is a list
                if not isinstance(topics, list):
                    topics = [topics] if pd.notna(topics) else []
                
                # Create a record for each topic
                for topic in topics:
                    topic_str = str(topic).strip()
                    if topic_str and topic_str.lower() != 'nan':
                        sections_records.append({
                            'article_id': article_id,
                            'news_section': topic_str
                        })
            
            # Insert sections (using composite primary key)
            if sections_records:
                print(f"Inserting {len(sections_records)} section mappings...")
                
                # Use INSERT with ON CONFLICT to handle any potential duplicates gracefully
                insert_sections_query = text("""
                    INSERT INTO public.articles_sections (article_id, news_section)
                    VALUES (:article_id, :news_section)
                    ON CONFLICT (article_id, news_section) DO NOTHING
                """)
                
                for section_record in sections_records:
                    connection.execute(insert_sections_query, section_record)
                
                print(f"✅ Inserted {len(sections_records)} section mappings successfully")
            else:
                print("⚠️ No valid sections to insert")
            
            # Transaction commits automatically when exiting 'with engine.begin()'
            
        print(f"\n✅ Success! Pushed {len(article_ids)} articles with {len(sections_records)} section mappings")
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        print("--- Starting News Processing Pipeline ---\n")
        
        # Step 1: Fetch and merge
        print("--- [Step 1/5] Fetching and merging articles... ---")
        todays_articles_df = merge_and_preprocess_articles(filter_today=True)
        
        if not todays_articles_df.empty:
            print(f"Found {len(todays_articles_df)} articles\n")
            
            # Step 2: Cluster
            print("--- [Step 2/5] Clustering articles... ---")
            clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)
            print(f"Clustered: {len(clustered_df)}, Unique: {len(unique_df)}\n")
            
            # Step 3: Synthesize and summarize
            print("--- [Step 3/5] Synthesizing and summarizing... ---")
            final_df, stats = generate_final_news_df_langchain(clustered_df, unique_df)
            print(f"Generated {len(final_df)} final articles\n")
            
            # Step 4: Push to database
            push_articles_to_db(final_df)
            
        else:
            print("--- Pipeline finished: No articles found to process. ---")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise