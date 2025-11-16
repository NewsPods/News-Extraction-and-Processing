"""
Takes clustered and unique articles, then uses advanced LangChain chains with
Pydantic parsing to synthesize clusters and summarize unique articles, creating
a final, consistently formatted dataset.

"""
import os
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Any, List, Optional
import concurrent.futures
import time
from dotenv import load_dotenv
from functools import wraps
import logging

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

# --- Pydantic Imports ---
from pydantic import BaseModel, Field, validator

# --- Sentence Transformers for Embeddings ---
from sentence_transformers import SentenceTransformer

# --- Import functions from your other modules ---
from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan

# --- Configuration ---
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
OUTPUT_DIR = "news_outputs"
MAX_LLM_WORKERS = 3
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_SUMMARY_WORDS = 200
MIN_CONTENT_LENGTH = 50

# --- Embedding Model for Database Storage ---
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enhanced Pydantic Models with Validation ---
class SynthesizedArticle(BaseModel):
    """Model for the synthesized article output with validation."""
    title: str = Field(description="The best, most engaging title from the source articles")
    summary: str = Field(description=f"A comprehensive news summary of max {MAX_SUMMARY_WORDS} words, synthesizing all source content", min_length=MIN_CONTENT_LENGTH, max_length=6000)

class SummarizedArticle(BaseModel):
    """Model for the summarized unique article output with validation."""
    summary: str = Field(description=f"A concise news summary of max {MAX_SUMMARY_WORDS} words, capturing the article's key facts", min_length=MIN_CONTENT_LENGTH, max_length=6000)

# --- Retry Decorator ---
def retry_on_failure(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    """Decorator to retry function calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = delay * (attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# --- MODIFIED Processing Functions ---
@retry_on_failure()
def _process_cluster(cluster_df: pd.DataFrame, chain: Runnable) -> Dict[str, Any]:
    """
    Synthesizes a single cluster of articles with retry logic.
    MODIFIED: Returns a single dict with topic as a list and cleans whitespace.
    """
    if cluster_df.empty:
        raise ValueError("Empty cluster provided")
    
    max_content_per_article = 6000
    combined_text = "\n\n".join(
        f"--- Article from {row['source']} ---\n"
        f"Title: {row['title']}\n"
        f"Content: {row['content'][:max_content_per_article]}{'...' if len(row['content']) > max_content_per_article else ''}"
        for _, row in cluster_df.iterrows()
    )
    
    response: SynthesizedArticle = chain.invoke({"articles_text": combined_text})
    
    # ADDED: Clean up excessive whitespace and newlines
    cleaned_summary = ' '.join(response.summary.split())
    
    # MODIFICATION: Store all unique topics in a list
    unique_topics = cluster_df['topic'].unique().tolist()
    published_date = cluster_df['published_date'].min()
    
    # MODIFICATION: Return single dict with topics as list
    return {
        'source': 'Multiple', 
        'topic': unique_topics,  # List of topics
        'title': response.title,
        'published_date': published_date, 
        'link': cluster_df.iloc[0]['link'],
        'content': cleaned_summary,  # Use cleaned version
        'word_count': len(cleaned_summary.split()),
        'article_count': len(cluster_df)
    }

@retry_on_failure()
def _process_unique_article(article_row: pd.Series, chain: Runnable) -> Dict[str, Any]:
    """
    Summarizes a single unique article with retry logic.
    MODIFIED: Ensures topic is returned as a list and cleans whitespace.
    """
    if pd.isna(article_row['content']) or len(article_row['content'].strip()) < MIN_CONTENT_LENGTH:
        raise ValueError("Article content is too short or empty")
    
    max_content = 4000
    content = article_row['content'][:max_content]
    if len(article_row['content']) > max_content:
        content += "..."
    
    response: SummarizedArticle = chain.invoke({"article_text": content})
    
    # ADDED: Clean up excessive whitespace and newlines
    cleaned_summary = ' '.join(response.summary.split())
    
    result = article_row.to_dict()
    result['content'] = cleaned_summary  # Use cleaned version
    result['word_count'] = len(cleaned_summary.split())
    result['original_length'] = len(article_row['content'])
    
    # MODIFICATION: Ensure topic is a list
    if 'topic' in result:
        if not isinstance(result['topic'], list):
            result['topic'] = [result['topic']]
    
    return result

def _handle_processing_failure(item_info: str, error: Exception, original_data: Any) -> Any:
    """Centralized error handling for processing failures."""
    logger.error(f"Failed to process {item_info}: {error}")
    if isinstance(original_data, pd.DataFrame):
        # For clusters, aggregate topics
        unique_topics = original_data['topic'].unique().tolist()
        result = original_data.iloc[0].to_dict()
        result['topic'] = unique_topics
        result['source'] = 'Original (Failed Processing)'
        return result
    else:
        result = original_data.to_dict() if hasattr(original_data, 'to_dict') else original_data
        if 'topic' in result and not isinstance(result['topic'], list):
            result['topic'] = [result['topic']]
        result['source'] = 'Original (Failed Processing)'
        return result

# --- Enhanced Main Function ---
def generate_final_news_df_langchain(clustered_df: pd.DataFrame, unique_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a final DataFrame by synthesizing clusters and summarizing unique articles.
    MODIFIED: Returns DataFrame with 'topic' column containing lists.
    """
    
    if clustered_df.empty and unique_df.empty:
        logger.warning("No articles to process")
        return pd.DataFrame(), {}
    
    # --- 1. Set up LangChain Components ---
    synthesis_parser = PydanticOutputParser(pydantic_object=SynthesizedArticle)
    summary_parser = PydanticOutputParser(pydantic_object=SummarizedArticle)

    synthesis_prompt = ChatPromptTemplate.from_template(
    """
    You are a professional news editor. Synthesize a single, high-quality news article from the following collection of articles about the same event.

    **CRITICAL INSTRUCTION:**
    The title you select MUST accurately represent the content you synthesize. DO NOT pick a title from one article and then write content about a different article. The title and content must match and be about the SAME topic.

    **Instructions:**
    1. Identify the MAIN topic/event that all these articles are about.
    2. Select the best, most engaging title that represents THIS MAIN TOPIC.
    3. Write a comprehensive narrative integrating all crucial facts, quotes, and perspectives from all sources about THIS SAME TOPIC.
    4. Maintain a professional, objective, and engaging journalistic style.
    5. **The final summary MUST be exactly {max_words} words or fewer.**
    6. Ensure proper flow and readability - no abrupt sentences or formatting artifacts.
    
    **VALIDATION CHECK:**
    Before finalizing, ask yourself: "Does my chosen title accurately describe the content I wrote?" If not, either change the title or rewrite the content.

    **Key Requirements:**
    - Title and content MUST be about the same topic
    - Professional journalism style
    - Maximum {max_words} words
    - Include key facts, quotes, and context
    - No meta-commentary about synthesis

    {format_instructions}

    **Source Articles:**
    {articles_text}
    """
)
    
    summary_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional news editor. Rewrite the following news article into a concise, engaging summary.

        **Instructions:**
        1. Capture all the crucial facts, details, and context. Do not leave out any facts, details, context or quotes.
        2. Maintain a professional, objective, and journalistic style.
        3. **The final summary MUST be exactly {max_words} words or fewer.**
        4. Preserve important quotes and specific details.
        5. Ensure clarity and readability.

        **Key Requirements:**
        - Professional journalism style
        - Maximum {max_words} words
        - Preserve essential information
        - Clear and engaging narrative

        {format_instructions}

        **Original Article:**
        {article_text}
        """
    )
    
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=API_KEY,
        temperature=0.3,
    )
    
    synthesis_chain = synthesis_prompt.partial(
        format_instructions=synthesis_parser.get_format_instructions(),
        max_words=MAX_SUMMARY_WORDS
    ) | model | synthesis_parser
    
    summary_chain = summary_prompt.partial(
        format_instructions=summary_parser.get_format_instructions(),
        max_words=MAX_SUMMARY_WORDS
    ) | model | summary_parser

    # --- 2. Process all articles ---
    final_article_list = []
    failed_items_to_retry = []  # Store failed items for retry
    
    stats = {
        'total_clusters': len(clustered_df.groupby('cluster_id')) if not clustered_df.empty else 0,
        'total_unique': len(unique_df) if not unique_df.empty else 0,
        'successful_synthesis': 0,
        'failed_synthesis': 0,
        'successful_summary': 0,
        'failed_summary': 0,
        'retry_successful': 0,
        'permanently_failed': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
        futures = []
        
        # Submit cluster processing tasks
        if not clustered_df.empty:
            cluster_groups = [group for _, group in clustered_df.groupby('cluster_id')]
            for group in cluster_groups:
                future = executor.submit(_process_cluster, group, synthesis_chain)
                futures.append(('cluster', future, group))
        
        # Submit unique article processing tasks
        if not unique_df.empty:
            for _, row in unique_df.iterrows():
                future = executor.submit(_process_unique_article, row, summary_chain)
                futures.append(('unique', future, row))

        print(f"\n--- [Step 3/4] Processing {len(futures)} items with Llama 3.1-8B via Groq... ---")
        
        for item_type, future, original_data in tqdm(futures, desc="Processing Articles"):
            try:
                result = future.result()
                final_article_list.append(result)
                
                if item_type == 'cluster':
                    stats['successful_synthesis'] += 1
                else:
                    stats['successful_summary'] += 1
                        
            except Exception as e:
                # Store failed items for retry instead of using fallback immediately
                logger.warning(f"Initial processing failed for {item_type}: {e}")
                failed_items_to_retry.append((item_type, original_data))
                
                if item_type == 'cluster':
                    stats['failed_synthesis'] += 1
                else:
                    stats['failed_summary'] += 1

    # --- 2.5. RETRY LOGIC: Attempt to process failed articles again (2 attempts) ---
    if failed_items_to_retry:
        print(f"\n⚠️ Retrying {len(failed_items_to_retry)} failed items (2 attempts per item)...")
        
        for item_type, original_data in tqdm(failed_items_to_retry, desc="Retrying Failed Items"):
            success = False
            last_error = None
            
            # Determine which chain to use
            chain = synthesis_chain if item_type == 'cluster' else summary_chain
            process_func = _process_cluster if item_type == 'cluster' else _process_unique_article
            
            # Try 2 more times
            for retry_attempt in range(2):
                try:
                    time.sleep(1)  # Small delay between retries
                    result = process_func(original_data, chain)
                    final_article_list.append(result)
                    success = True
                    stats['retry_successful'] += 1
                    logger.info(f"✓ Retry attempt {retry_attempt + 1} succeeded for {item_type}")
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Retry attempt {retry_attempt + 1} failed: {e}")
                    time.sleep(2)  # Longer delay before next retry
            
            # If still failed after retries, exclude it
            if not success:
                stats['permanently_failed'] += 1
                if item_type == 'cluster':
                    cluster_id = original_data['cluster_id'].iloc[0] if isinstance(original_data, pd.DataFrame) else 'Unknown'
                    logger.error(f"❌ Permanently excluding Cluster ID {cluster_id} after 2 retry attempts. Last error: {last_error}")
                else:
                    title = original_data.get('title', 'Unknown') if isinstance(original_data, dict) else original_data['title']
                    logger.error(f"❌ Permanently excluding article '{title[:50]}...' after 2 retry attempts. Last error: {last_error}")


    stats['processing_time'] = time.time() - start_time

    # --- 3. Create and validate final DataFrame ---
    if not final_article_list:
        logger.warning("No articles were successfully processed")
        return pd.DataFrame(), stats
    
    final_df = pd.DataFrame(final_article_list)
    final_columns = ['source', 'topic', 'title', 'published_date', 'link', 'content', 'word_count']
    
    optional_columns = ['article_count', 'original_length']
    for col in optional_columns:
        if col in final_df.columns:
            final_columns.append(col)
    
    final_df = final_df.reindex(columns=final_columns)
    final_df = final_df.dropna(subset=['title', 'content'])
    
    # --- 4. Generate embeddings for database storage ---
    if not final_df.empty:
        logger.info("Generating embeddings for database storage...")
        texts_for_embedding = (final_df['title'].fillna('') + ". " + final_df['content'].fillna('')).tolist()
        
        embeddings = EMBEDDING_MODEL.encode(
            texts_for_embedding,
            show_progress_bar=True,
            batch_size=32
        )
        final_df['embedding'] = [emb.tolist() for emb in embeddings]
        logger.info(f"✅ Generated embeddings for {len(final_df)} articles")
    
    # --- 5. Rename columns to match database schema ---
    final_df.rename(columns={
        'content': 'description',
        'source': 'news_source',
        'published_date': 'created_at'
    }, inplace=True)
    
    # Ensure datetime format for database
    final_df['created_at'] = pd.to_datetime(final_df['created_at'], errors='coerce')
    
    return final_df, stats

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        print("--- [Step 1/4] Fetching and merging articles... ---")
        todays_articles_df = merge_and_preprocess_articles(filter_today=True)
        
        if not todays_articles_df.empty:
            print(f"Found {len(todays_articles_df)} articles to process")
            
            print("--- [Step 2/4] Clustering articles... ---")
            clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)
            
            print(f"Clustered articles: {len(clustered_df)}, Unique articles: {len(unique_df)}")
            
            final_df, processing_stats = generate_final_news_df_langchain(clustered_df, unique_df)
            
            print("\n--- [Step 4/4] Saving final database-ready file... ---")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Add timestamp to filename
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_filename = os.path.join(OUTPUT_DIR, f'final_news_database_ready_{timestamp}.csv')
            
            final_df.to_csv(final_filename, index=False, encoding='utf-8')
            
            print(f"\n✅ Pipeline complete! Database-ready news saved to: {final_filename}")
            print(f"\n📊 Processing Statistics:")
            print(f"  • Total articles processed: {len(final_df)}")
            print(f"  • Successful syntheses: {processing_stats['successful_synthesis']}")
            print(f"  • Failed syntheses (initial): {processing_stats['failed_synthesis']}")
            print(f"  • Successful summaries: {processing_stats['successful_summary']}")
            print(f"  • Failed summaries (initial): {processing_stats['failed_summary']}")
            print(f"  • Retry successful: {processing_stats['retry_successful']}")
            print(f"  • Permanently excluded: {processing_stats['permanently_failed']}")
            print(f"  • Processing time: {processing_stats['processing_time']:.2f} seconds")
            
            if len(final_df) > 0:
                avg_words = final_df['word_count'].mean() if 'word_count' in final_df.columns else 0
                print(f"  • Average summary length: {avg_words:.1f} words")
                print(f"  • Embeddings generated: {'Yes' if 'embedding' in final_df.columns else 'No'}")
            
            if processing_stats['permanently_failed'] > 0:
                print(f"\n⚠️ Note: {processing_stats['permanently_failed']} article(s) were excluded after failing 2 retry attempts.")
                print(f"   These articles could not be processed by the LLM and have been removed from the dataset.")
                
            print(f"\n💾 CSV columns: {list(final_df.columns)}")
            print(f"   Ready for direct database insertion!")
        else:
            print("\n--- Pipeline finished: No articles found to process. ---")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise