"""
Main pipeline script. Fetches articles from all sources, merges them,
and then performs DBSCAN clustering to find and group similar stories.
Finally, it saves the clustered and unique articles to CSV files.
"""
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from typing import Tuple

# --- Import the main function from your merger script ---
from .merger import merge_and_preprocess_articles

# --- Constants and Configuration ---
EMBEDDING_MODEL = 'all-mpnet-base-v2'
OUTPUT_DIR = "news_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

# --- Helper Function to Determine Best Device ---
def _get_best_device() -> str:
    """Checks for CUDA (Nvidia GPU) or MPS (Apple Silicon) availability."""
    if torch.cuda.is_available():
        print("-> Using CUDA (Nvidia GPU) for embeddings.")
        return 'cuda'
    if torch.backends.mps.is_available():
        print("-> Using MPS (Apple Silicon GPU) for embeddings.")
        return 'mps'
    print("-> No GPU found, using CPU for embeddings. This will be slower.")
    return 'cpu'

# --- Main Clustering Function ---
def cluster_articles_with_dbscan(
    df: pd.DataFrame,
    eps: float = 0.325,
    min_samples: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clusters articles using sentence embeddings and the DBSCAN algorithm.
    (This function remains the same as before)
    """
    if df.empty:
        print("Input DataFrame is empty. Cannot perform clustering.")
        return pd.DataFrame(), pd.DataFrame()

    print("\n--- [Step 2/3] Starting DBSCAN Clustering ---")
    
    device = _get_best_device()
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("-> Creating combined text field and generating embeddings...")
    df['title_and_content'] = df['title'].astype(str) + ". " + df['content'].astype(str)
    
    embeddings = model.encode(
        df['title_and_content'].tolist(),
        show_progress_bar=True,
        convert_to_tensor=True
    )

    print("-> Calculating distance matrix and running DBSCAN...")
    distance_matrix = 1 - util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    distance_matrix = np.clip(distance_matrix, 0, 1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distance_matrix)
    df['cluster_id'] = labels
    
    valid_clusters = []
    for cluster_id in sorted(df['cluster_id'].unique()):
        if cluster_id == -1: continue
        cluster_df = df[df['cluster_id'] == cluster_id]
        if cluster_df['source'].nunique() > 1:
            valid_clusters.append(cluster_id)
            
    clustered_articles_df = df[df['cluster_id'].isin(valid_clusters)].copy()
    unique_articles_df = df[~df['cluster_id'].isin(valid_clusters)].copy()
    
    clustered_articles_df['cluster_id'] = clustered_articles_df['cluster_id'].map(
        {old_id: new_id for new_id, old_id in enumerate(valid_clusters, 1)}
    )

    print(f"✅ Clustering complete! Found {len(valid_clusters)} valid multi-source clusters.")
    return clustered_articles_df, unique_articles_df

# --- Main Execution Block ---
if __name__ == '__main__':
    # This is the main pipeline orchestrator
    
    # 1. Call the merger to get today's preprocessed articles
    print("--- [Step 1/3] Fetching and merging articles from all sources... ---")
    todays_articles_df = merge_and_preprocess_articles(filter_today=True)
    
    if not todays_articles_df.empty:
        # 2. Run the clustering function on the merged data
        clustered_df, unique_df = cluster_articles_with_dbscan(todays_articles_df)

        # 3. Save the final DataFrames to CSV files with timestamp
        print("\n--- [Step 3/3] Saving final CSV files... ---")
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        clustered_filename = os.path.join(OUTPUT_DIR, f'clustered_articles_{timestamp}.csv')
        unique_filename = os.path.join(OUTPUT_DIR, f'unique_articles_{timestamp}.csv')

        # Group clustered articles by cluster_id before saving
        clustered_df_sorted = clustered_df.sort_values(by=["cluster_id"])
        clustered_df_sorted.to_csv(clustered_filename, index=False, encoding='utf-8')
        unique_df.to_csv(unique_filename, index=False, encoding='utf-8')

        print(f"✅ Clustered articles saved to: {clustered_filename}")
        print(f"✅ Unique articles saved to: {unique_filename}")

    else:
        print("\n--- Pipeline finished: No articles found to process. ---")