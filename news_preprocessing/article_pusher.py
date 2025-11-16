"""
Connects to CockroachDB and pushes processed news articles with embeddings.
This module ONLY handles database insertion - all data transformations 
are done in final_news_df.py.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Tuple

# --- Import functions from your other modules ---
from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan
from .final_news_df import generate_final_news_df_langchain

# --- Configuration ---
load_dotenv()
CONN_STRING = os.environ.get("COCKROACHDB_CONN_STRING")

def push_articles_to_db(df: pd.DataFrame):
    """
    Pushes the processed DataFrame (with embeddings) to the database.
    
    IMPORTANT: Expects the DataFrame to already have:
    - 'embedding' column with vector embeddings
    - Database-ready column names ('description', 'news_source', 'created_at')
    - 'topic' as a list of topics
    
    All data transformations should be done in final_news_df.py before calling this.
    """
    if df.empty:
        print("Input DataFrame is empty. Nothing to push.")
        return
    if not CONN_STRING:
        print("❌ Error: COCKROACHDB_CONN_STRING not set.")
        return

    print(f"\n--- [Step 5/5] Pushing {len(df)} articles to CockroachDB ---")
    
    # Validate that the DataFrame has the expected columns
    required_columns = ['title', 'description', 'news_source', 'created_at', 'embedding', 'topic']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: DataFrame is missing required columns: {missing_columns}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Make sure final_news_df.py has generated embeddings and renamed columns.")
        return

    try:
        # --- Connect and Insert Data ---
        engine = create_engine(CONN_STRING, poolclass=NullPool)
        
        with engine.begin() as connection:  # Transaction with automatic rollback on error
            print("Inserting articles into database...")
            
            insert_query = text("""
                INSERT INTO public.articles (title, description, news_source, created_at, embedding)
                VALUES (:title, :description, :news_source, :created_at, :embedding)
                RETURNING article_id
            """)

            # Store article_id and topics as coupled tuples to maintain data integrity
            article_topic_pairs: List[Tuple[int, any]] = []
            
            try:
                # Iterate through DataFrame rows to maintain alignment
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting Articles"):
                    # Validate row data before insertion
                    if pd.isna(row['title']) or pd.isna(row['description']):
                        print(f"⚠️ Skipping row {idx}: Missing title or description")
                        continue
                    
                    # Prepare the record for insertion
                    record = {
                        'title': row['title'],
                        'description': row['description'],
                        'news_source': row['news_source'],
                        'created_at': row['created_at'],
                        'embedding': row['embedding']
                    }
                    
                    # Insert and get article_id
                    result = connection.execute(insert_query, record)
                    article_id = result.fetchone()[0]
                    
                    # CRITICAL: Store as tuple to maintain tight coupling
                    article_topic_pairs.append((article_id, row['topic']))

                print(f"✅ Inserted {len(article_topic_pairs)} articles successfully.")
                
                # --- VALIDATION CHECK ---
                expected_count = len(df)
                actual_count = len(article_topic_pairs)
                if actual_count != expected_count:
                    print(f"⚠️ Warning: Expected {expected_count} articles, but inserted {actual_count}")
                else:
                    print(f"✓ Validation passed: All {actual_count} articles correctly paired with topics")
                
                # Debug: Show sample of inserted articles with their IDs
                if article_topic_pairs:
                    print(f"\nSample of inserted articles:")
                    for i in range(min(3, len(article_topic_pairs))):
                        article_id, topics = article_topic_pairs[i]
                        title = df.iloc[i]['title'][:50] if i < len(df) else 'N/A'
                        print(f"  {i+1}. ID={article_id}, Topics={topics}, Title='{title}...'")

                # --- 3. Prepare and insert sections using the coupled pairs ---
                sections_records = []
                for article_id, topics in article_topic_pairs:
                    # Normalize topics to list
                    if not isinstance(topics, list):
                        topics = [topics] if pd.notna(topics) else []
                    
                    # Create section mapping for each topic
                    for topic in topics:
                        if pd.notna(topic) and str(topic).strip():  # Validate topic
                            sections_records.append({
                                'article_id': article_id, 
                                'news_section': str(topic).strip()
                            })

                if sections_records:
                    print(f"\nInserting {len(sections_records)} section mappings...")
                    
                    # Debug: Show sample section mappings
                    print(f"Sample section mappings:")
                    for i in range(min(3, len(sections_records))):
                        print(f"  {i+1}. Article ID={sections_records[i]['article_id']}, Section='{sections_records[i]['news_section']}'")
                    
                    sections_df = pd.DataFrame(sections_records)
                    sections_df.to_sql('articles_sections', con=connection, if_exists='append', index=False, method='multi')
                    print(f"✅ Inserted {len(sections_records)} section mappings successfully.")
                    
                    # Final validation
                    avg_sections_per_article = len(sections_records) / len(article_topic_pairs) if article_topic_pairs else 0
                    print(f"✓ Average sections per article: {avg_sections_per_article:.2f}")
                else:
                    print("⚠️ No valid section mappings to insert")
                    
            except Exception as e:
                print(f"❌ Error during insertion, transaction will be rolled back: {e}")
                raise  # Re-raise to trigger automatic rollback

    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        raise

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        print("=" * 70)
        print("FULL PIPELINE: Scrape → Cluster → Synthesize → Push to Database")
        print("=" * 70)
        
        # Step 1: Fetch and merge articles
        print("\n--- [Step 1/5] Fetching and merging articles... ---")
        todays_articles_df = merge_and_preprocess_articles(filter_today=True)
        
        if not todays_articles_df.empty:
            print(f"Found {len(todays_articles_df)} articles to process")
            
            # Step 2: Cluster articles
            print("\n--- [Step 2/5] Clustering articles... ---")
            clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)
            print(f"Clustered articles: {len(clustered_df)}, Unique articles: {len(unique_df)}")
            
            # Step 3: Synthesize, summarize, and prepare for database
            print("\n--- [Step 3/5] Generating final news dataset with embeddings... ---")
            final_df, processing_stats = generate_final_news_df_langchain(clustered_df, unique_df)
            
            print(f"\n📊 Processing complete:")
            print(f"  • Successful syntheses: {processing_stats['successful_synthesis']}")
            print(f"  • Successful summaries: {processing_stats['successful_summary']}")
            print(f"  • Total processed: {len(final_df)} articles")
            
            # Step 4: Push to database
            push_articles_to_db(final_df)
            
            print("\n" + "=" * 70)
            print("✅ FULL PIPELINE COMPLETE!")
            print("=" * 70)
        else:
            print("\n--- Pipeline finished: No articles found to process. ---")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise