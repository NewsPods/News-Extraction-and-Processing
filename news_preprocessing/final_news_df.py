"""
Takes clustered and unique articles, then uses an advanced LangChain chain 
with a generative AI model to synthesize and de-duplicate the clustered articles
into a high-quality, journalistic format.
"""
import os
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Any, List
import concurrent.futures
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# --- Import functions from your other modules ---
from .merger import merge_and_preprocess_articles
from .dbscan import cluster_articles_with_dbscan

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_DIR = "news_outputs"

# --- Main LangChain Processing Function ---
def _process_cluster_langchain(cluster_df: pd.DataFrame, chain: Runnable) -> List[Dict[str, Any]]:
    """Processes a single cluster using the advanced LangChain chain."""
    try:
        combined_text = ""
        for _, row in cluster_df.iterrows():
            combined_text += f"--- Article from {row['source']} ---\n"
            combined_text += f"Title: {row['title']}\n"
            combined_text += f"Content: {row['content']}\n\n"

        response_text = chain.invoke({"articles_text": combined_text})
        
        new_title = response_text.split("TITLE:")[1].split("SUMMARY:")[0].strip()
        new_summary = response_text.split("SUMMARY:")[1].strip()

    except Exception as e:
        cluster_id = cluster_df['cluster_id'].iloc[0]
        print(f"\n⚠️ Warning: Could not process Cluster ID {cluster_id}. Error: {e}. Using original articles.")
        return cluster_df.to_dict('records')
        
    else:
        unique_topics = cluster_df['topic'].unique().tolist()
        published_date = cluster_df['published_date'].min()
        
        synthesized_rows = []
        for topic in unique_topics:
            synthesized_rows.append({
                'source': 'Synthesized', 'topic': topic, 'title': new_title,
                'published_date': published_date, 'link': cluster_df.iloc[0]['link'],
                'content': new_summary
            })
        return synthesized_rows

# --- Main Public Function ---
def generate_final_news_df_langchain(clustered_df: pd.DataFrame, unique_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a final, de-duplicated DataFrame by synthesizing clustered articles using LangChain."""
    if clustered_df.empty:
        return unique_df

    # --- Set up the NEW, Advanced LangChain Prompt ---
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a professional news editor for a major international news agency.
        Your task is to synthesize a single, high-quality news article from the following collection of articles covering the same event.

        **Your Instructions:**
        1.  **Select the Best Title:** From all the titles provided, choose the one that is most clear, descriptive, and engaging.
        2.  **Synthesize a Comprehensive Narrative:** Write a new article that seamlessly integrates all the crucial facts, details, quotes, and perspectives from all the provided source articles. Ensure the final text flows logically and is easy to read.
        3.  **Maintain a Journalistic Tone:** The style should be professional, objective, and authoritative. The language should be clear and engaging, similar to the source articles, but structured in your own sentences.
        4.  **Formatting Rules:**
            - Ensure proper grammar, spelling, and punctuation.
            - Do not include any strange characters, artifacts (like 'â'), or abrupt sentences.
            - Do not include any meta-commentary like "This article combines..."
            - The output must be a single, clean piece of journalism.

        **Output Format:**
        Format your response EXACTLY as follows, with no extra text before or after:
        TITLE: [The best title you selected]
        SUMMARY: [Your new, comprehensive summary article]

        **Source Articles to Synthesize:**
        {articles_text}
        """
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser

    # --- Process Clusters in Parallel ---
    print("\n--- [Step 3/4] Synthesizing clustered articles with Advanced Prompt... ---")
    final_article_list = []
    cluster_groups = [group for _, group in clustered_df.groupby('cluster_id')]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process_cluster_langchain, group, chain) for group in cluster_groups]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Synthesizing Clusters"):
            final_article_list.extend(future.result())

    # --- Combine and Finalize ---
    synthesized_df = pd.DataFrame(final_article_list)
    final_columns = ['source', 'topic', 'title', 'published_date', 'link', 'content']
    
    final_df = pd.concat([
        unique_df.reindex(columns=final_columns), 
        synthesized_df.reindex(columns=final_columns)
    ], ignore_index=True)
    
    return final_df

# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- [Step 1/4] Fetching and merging articles... ---")
    todays_articles_df = merge_and_preprocess_articles(filter_today=True)
    
    if not todays_articles_df.empty:
        print("--- [Step 2/4] Clustering articles... ---")
        clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)
        
        final_df = generate_final_news_df_langchain(clustered_df, unique_df)
        
        print("\n--- [Step 4/4] Saving final consolidated news file... ---")
        final_filename = os.path.join(OUTPUT_DIR, 'final_consolidated_news_advanced.csv')
        final_df.to_csv(final_filename, index=False, encoding='utf-8')
        
        print(f"\n✅ Pipeline complete! Final de-duplicated news saved to: {final_filename}")
        print(f"Total articles in final dataset: {len(final_df)}")
    else:
        print("\n--- Pipeline finished: No articles found to process. ---")