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

# --- Import functions from your other modules ---
from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan

# --- Configuration ---
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
OUTPUT_DIR = "news_outputs"
MAX_LLM_WORKERS = 5  # Reduced further for Groq rate limits
MAX_RETRIES = 3
RETRY_DELAY = 2  # Increased delay for Groq API
MAX_SUMMARY_WORDS = 200
MIN_CONTENT_LENGTH = 50

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

# --- Enhanced Processing Functions ---
@retry_on_failure()
def _process_cluster(cluster_df: pd.DataFrame, chain: Runnable) -> List[Dict[str, Any]]:
    """Synthesizes a single cluster of articles with retry logic."""
    # Validate input
    if cluster_df.empty:
        raise ValueError("Empty cluster provided")
    
    # Process full content length to handle longer articles
    max_content_per_article = 6000  # Increased to handle longer articles
    combined_text = "\n\n".join(
        f"--- Article from {row['source']} ---\n"
        f"Title: {row['title']}\n"
        f"Content: {row['content'][:max_content_per_article]}{'...' if len(row['content']) > max_content_per_article else ''}"
        for _, row in cluster_df.iterrows()
    )
    
    response: SynthesizedArticle = chain.invoke({"articles_text": combined_text})
    
    unique_topics = cluster_df['topic'].unique().tolist()
    published_date = cluster_df['published_date'].min()
    
    return [{
        'source': 'Multiple', 
        'topic': topic, 
        'title': response.title,
        'published_date': published_date, 
        'link': cluster_df.iloc[0]['link'],
        'content': response.summary,
        'word_count': len(response.summary.split()),
        'article_count': len(cluster_df)
    } for topic in unique_topics]

@retry_on_failure()
def _process_unique_article(article_row: pd.Series, chain: Runnable) -> Dict[str, Any]:
    """Summarizes a single unique article with retry logic."""
    # Validate input
    if pd.isna(article_row['content']) or len(article_row['content'].strip()) < MIN_CONTENT_LENGTH:
        raise ValueError("Article content is too short or empty")
    
    # Handle full content length for longer articles
    max_content = 4000  # Increased to handle longer articles
    content = article_row['content'][:max_content]
    if len(article_row['content']) > max_content:
        content += "..."
    
    response: SummarizedArticle = chain.invoke({"article_text": content})
    
    result = article_row.to_dict()
    result['content'] = response.summary
    # Keep the original source unchanged
    result['word_count'] = len(response.summary.split())
    result['original_length'] = len(article_row['content'])
    
    return result

def _handle_processing_failure(item_info: str, error: Exception, original_data: Any) -> Any:
    """Centralized error handling for processing failures."""
    logger.error(f"Failed to process {item_info}: {error}")
    if isinstance(original_data, pd.DataFrame):
        result = original_data.to_dict('records')
        for record in result:
            record['source'] = 'Original (Failed Processing)'
        return result
    else:
        result = original_data.to_dict() if hasattr(original_data, 'to_dict') else original_data
        result['source'] = 'Original (Failed Processing)'
        return result

# --- Enhanced Main Function ---
def generate_final_news_df_langchain(clustered_df: pd.DataFrame, unique_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a final DataFrame by synthesizing clusters and summarizing unique articles.
    Returns both the DataFrame and processing statistics.
    """
    
    # Validate inputs
    if clustered_df.empty and unique_df.empty:
        logger.warning("No articles to process")
        return pd.DataFrame(), {}
    
    # --- 1. Set up LangChain Components ---
    synthesis_parser = PydanticOutputParser(pydantic_object=SynthesizedArticle)
    summary_parser = PydanticOutputParser(pydantic_object=SummarizedArticle)

    synthesis_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional news editor. Synthesize a single, high-quality news article from the following collection of articles about the same event.

        **Instructions:**
        1. Select the best, most engaging title from the provided articles.
        2. Write a comprehensive narrative integrating all crucial facts, quotes, and perspectives from all sources.
        3. Maintain a professional, objective, and engaging journalistic style.
        4. **The final summary MUST be exactly {max_words} words or fewer.**
        5. Ensure proper flow and readability - no abrupt sentences or formatting artifacts.
        6. Focus on the most newsworthy and important information.

        **Key Requirements:**
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
        1. Capture the most crucial facts, details, and context.
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
    
    # Initialize Groq model with Llama 3.1-8B-Instant
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=API_KEY,
        temperature=0.3,  # Lower temperature for more consistent output
    )
    
    synthesis_chain = synthesis_prompt.partial(
        format_instructions=synthesis_parser.get_format_instructions(),
        max_words=MAX_SUMMARY_WORDS
    ) | model | synthesis_parser
    
    summary_chain = summary_prompt.partial(
        format_instructions=summary_parser.get_format_instructions(),
        max_words=MAX_SUMMARY_WORDS
    ) | model | summary_parser

    # --- 2. Process all articles with comprehensive tracking ---
    final_article_list = []
    stats = {
        'total_clusters': len(clustered_df.groupby('cluster_id')) if not clustered_df.empty else 0,
        'total_unique': len(unique_df) if not unique_df.empty else 0,
        'successful_synthesis': 0,
        'failed_synthesis': 0,
        'successful_summary': 0,
        'failed_summary': 0,
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
                if isinstance(result, list):
                    final_article_list.extend(result)
                    if item_type == 'cluster':
                        stats['successful_synthesis'] += 1
                else:
                    final_article_list.append(result)
                    if item_type == 'unique':
                        stats['successful_summary'] += 1
                        
            except Exception as e:
                # Handle failures gracefully
                if item_type == 'cluster':
                    cluster_id = original_data['cluster_id'].iloc[0]
                    fallback = _handle_processing_failure(f"Cluster ID {cluster_id}", e, original_data)
                    final_article_list.extend(fallback)
                    stats['failed_synthesis'] += 1
                else:
                    title = original_data.get('title', 'Unknown')
                    fallback = _handle_processing_failure(f"Article: {title}", e, original_data)
                    final_article_list.append(fallback)
                    stats['failed_summary'] += 1

    stats['processing_time'] = time.time() - start_time

    # --- 3. Create and validate final DataFrame ---
    if not final_article_list:
        logger.warning("No articles were successfully processed")
        return pd.DataFrame(), stats
    
    final_df = pd.DataFrame(final_article_list)
    final_columns = ['source', 'topic', 'title', 'published_date', 'link', 'content', 'word_count']
    
    # Add optional columns if they exist
    optional_columns = ['article_count', 'original_length']
    for col in optional_columns:
        if col in final_df.columns:
            final_columns.append(col)
    
    final_df = final_df.reindex(columns=final_columns)
    
    # Remove any completely empty rows
    final_df = final_df.dropna(subset=['title', 'content'])
    
    return final_df, stats

# --- Enhanced Main Execution Block ---
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
            
            print("\n--- [Step 4/4] Saving final consolidated news file... ---")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            final_filename = os.path.join(OUTPUT_DIR, 'final_summarized_news.csv')
            final_df.to_csv(final_filename, index=False, encoding='utf-8')
            
            # Print comprehensive statistics
            print(f"\n✅ Pipeline complete! Final summarized news saved to: {final_filename}")
            print(f"\n📊 Processing Statistics:")
            print(f"  • Total articles processed: {len(final_df)}")
            print(f"  • Successful syntheses: {processing_stats['successful_synthesis']}")
            print(f"  • Failed syntheses: {processing_stats['failed_synthesis']}")
            print(f"  • Successful summaries: {processing_stats['successful_summary']}")
            print(f"  • Failed summaries: {processing_stats['failed_summary']}")
            print(f"  • Processing time: {processing_stats['processing_time']:.2f} seconds")
            
            if len(final_df) > 0:
                avg_words = final_df['word_count'].mean() if 'word_count' in final_df.columns else 0
                print(f"  • Average summary length: {avg_words:.1f} words")
        else:
            print("\n--- Pipeline finished: No articles found to process. ---")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise