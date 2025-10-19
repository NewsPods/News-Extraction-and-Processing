"""
FastAPI-based search service for news articles using hybrid search (BM25 + Semantic).
MODIFIED: Enhanced data integrity with strict alignment validation and better error handling.
"""
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# --- Configuration ---
load_dotenv()
CONN_STRING = os.environ.get("COCKROACHDB_CONN_STRING")
# Using the powerful mpnet model as you requested
SEARCH_MODEL = SentenceTransformer('all-mpnet-base-v2') 

class ArticleSearchService:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        # MODIFIED: Store as list of tuples to maintain tight coupling
        self.article_data: List[Tuple[int, str, np.ndarray, List[str]]] = []
        self.bm25 = None
        self._load_articles()

    def _load_articles(self):
        """
        Load today's articles with embeddings from the database using SQLAlchemy.
        MODIFIED: Uses tuple-based storage to prevent misalignment.
        
        Note: article_id is INT8 (BIGINT) in CockroachDB. We carefully preserve
        these 64-bit integers throughout to avoid precision loss from float conversion.
        """
        print("Loading today's articles from database...")
        try:
            engine = create_engine(
                self.conn_string, 
                poolclass=NullPool,
                connect_args={"application_name": "news_search_api"}
            )
            
            with engine.connect() as connection:
                # Query is timezone-aware and more robust
                query = text("""
                    SELECT a.article_id::BIGINT, a.title, a.description, a.embedding
                    FROM public.articles a
                    WHERE a.created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                    ORDER BY a.created_at DESC;
                """)
                result = connection.execute(query)
                results = result.fetchall()
            
            if not results:
                print("⚠️ No articles found for today.")
                return

            # CRITICAL FIX: Store all data as coupled tuples
            article_tuples: List[Tuple[int, str, np.ndarray, List[str]]] = []
            
            for row in results:
                # Validate embedding exists
                if row[3] is None:
                    print(f"⚠️ Skipping article '{row[1][:30]}...' - No embedding found")
                    continue
                
                # article_id is INT8 (64-bit integer) from CockroachDB
                # Convert directly to Python int, which supports arbitrary precision
                raw_id = row[0]
                
                # Handle different possible types from SQLAlchemy/psycopg2
                if isinstance(raw_id, (int, np.integer)):
                    article_id = int(raw_id)
                elif isinstance(raw_id, float):
                    # If it came as float, convert carefully to avoid precision loss
                    article_id = int(np.int64(raw_id))
                else:
                    # Fallback: convert via string
                    article_id = int(str(raw_id).split('.')[0])
                
                title = row[1]
                description = row[2] or ""
                embedding = np.array(row[3], dtype=np.float32)
                
                # Tokenize for BM25
                bm25_tokens = f"{title} {description}".lower().split()
                
                # CRITICAL: Store as tuple to maintain atomic coupling
                article_tuples.append((article_id, title, embedding, bm25_tokens))
            
            # Validate data integrity
            if not article_tuples:
                print("⚠️ No valid articles with embeddings found.")
                return
            
            self.article_data = article_tuples
            
            # Extract components for processing (maintaining order)
            embeddings_list = [tup[2] for tup in article_tuples]
            texts_for_bm25 = [tup[3] for tup in article_tuples]
            
            # Build search indexes
            self.bm25 = BM25Okapi(texts_for_bm25)
            
            print(f"✅ Loaded {len(self.article_data)} articles and built search indexes.")
            
            # Debug: Print first few articles to verify alignment
            if article_tuples:
                print(f"\nSample loaded articles:")
                for i in range(min(3, len(article_tuples))):
                    article_id, title, _, _ = article_tuples[i]
                    print(f"  Index {i}: ID={article_id}, Title='{title[:50]}...'")
            
            # Validation check
            assert len(self.article_data) == len(embeddings_list) == len(texts_for_bm25), \
                "Data alignment mismatch detected!"
            print(f"✓ Validation passed: All {len(self.article_data)} articles properly aligned")
            
        except Exception as e:
            print(f"❌ Failed to load articles: {e}")
            raise

    def search(self, query: str, top_k: int = 10, k1: float = 60.0) -> List[int]:
        """
        Performs hybrid search using Reciprocal Rank Fusion (RRF).
        MODIFIED: Uses tuple-based data structure for guaranteed alignment.
        """
        if not self.article_data:
            print("⚠️ No articles available for search")
            return []

        # Tokenize query for BM25
        tokenized_query = query.lower().split()
        
        # Extract BM25 tokens from article tuples (maintaining order)
        texts_for_bm25 = [tup[3] for tup in self.article_data]
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get sorted indices (highest scores first)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1]
        
        # Debug: Verify BM25 ranking
        print(f"\n[BM25 Debug] Query: '{query[:50]}'")
        print(f"Top 3 BM25 results:")
        for rank, idx in enumerate(bm25_sorted_indices[:3]):
            if idx < len(self.article_data):
                article_id, title, _, _ = self.article_data[idx]
                score = bm25_scores[idx]
                print(f"  Rank {rank+1}: Index={idx}, ID={article_id}, Score={score:.4f}, Title='{title[:40]}...'")
        
        # Map article IDs to their BM25 ranks
        bm25_ranks = {
            int(self.article_data[int(idx)][0]): rank + 1 
            for rank, idx in enumerate(bm25_sorted_indices)
        }

        # Compute semantic similarity scores
        query_embedding = SEARCH_MODEL.encode([query])
        embeddings_array = np.array([tup[2] for tup in self.article_data])
        cosine_scores = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Get sorted indices (highest scores first)
        semantic_sorted_indices = np.argsort(cosine_scores)[::-1]
        
        # Debug: Verify semantic ranking
        print(f"\nTop 3 Semantic results:")
        for rank, idx in enumerate(semantic_sorted_indices[:3]):
            if idx < len(self.article_data):
                article_id, title, _, _ = self.article_data[idx]
                score = cosine_scores[idx]
                print(f"  Rank {rank+1}: Index={idx}, ID={article_id}, Score={score:.4f}, Title='{title[:40]}...'")
        
        # Map article IDs to their semantic ranks
        semantic_ranks = {
            int(self.article_data[int(idx)][0]): rank + 1 
            for rank, idx in enumerate(semantic_sorted_indices)
        }
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

        for doc_id in all_doc_ids:
            rrf_score = 0.0
            if doc_id in bm25_ranks:
                rrf_score += 1.0 / (k1 + bm25_ranks[doc_id])
            if doc_id in semantic_ranks:
                rrf_score += 1.0 / (k1 + semantic_ranks[doc_id])
            rrf_scores[doc_id] = rrf_score
            
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Ensure IDs are returned as standard Python integers (preserving INT8 precision)
        result_ids = [int(doc_id) for doc_id, score in sorted_docs[:top_k]]
        
        # Debug: Print the final result IDs with their titles to verify correctness
        print(f"\n[Final RRF Results]")
        print(f"Top {min(3, len(result_ids))} articles:")
        for i, article_id in enumerate(result_ids[:3]):
            # Find the article with this ID using tuple structure
            matching_article = next((tup for tup in self.article_data if tup[0] == article_id), None)
            if matching_article:
                _, title, _, _ = matching_article
                rrf_score = rrf_scores[article_id]
                print(f"  {i+1}. ID={article_id}, RRF={rrf_score:.6f}, Title='{title[:50]}...'")
            else:
                print(f"  {i+1}. ID={article_id}, ⚠️ WARNING: No matching article found!")
        
        return result_ids

    def get_article_info(self, article_id: int) -> Dict[str, str]:
        """
        Retrieve article information by ID for verification.
        """
        matching_article = next((tup for tup in self.article_data if tup[0] == article_id), None)
        if matching_article:
            article_id, title, _, _ = matching_article
            return {"id": str(article_id), "title": title}
        return {}

# --- FastAPI Application ---
app = FastAPI(title="News Article Search API")
search_service = None

class SearchRequest(BaseModel):
    keywords: List[str]

@app.on_event("startup")
def startup_event():
    """Initialize the search service when the API starts."""
    global search_service
    if not CONN_STRING:
        raise RuntimeError("COCKROACHDB_CONN_STRING not found in environment variables.")
    search_service = ArticleSearchService(CONN_STRING)

@app.post("/search", response_model=Dict[str, List[str]])
def search_articles(request: SearchRequest):
    """
    Accepts a list of keywords and returns relevant article IDs for each.
    
    Note: Article IDs are returned as strings to preserve INT8 precision.
    JavaScript/JSON cannot safely represent integers larger than 2^53 - 1.
    """
    if search_service is None or not search_service.article_data:
        raise HTTPException(status_code=503, detail="Search service is not ready or has no articles.")
        
    results = {}
    for keyword in request.keywords:
        article_ids = search_service.search(keyword)
        # Convert INT8 IDs to strings to prevent precision loss in JSON serialization
        results[keyword] = [str(id) for id in article_ids]
    return results

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if search_service is None:
        return {"status": "initializing", "articles_loaded": 0}
    return {
        "status": "healthy",
        "articles_loaded": len(search_service.article_data)
    }

@app.get("/article/{article_id}")
def get_article(article_id: int):
    """Get article information by ID (for debugging)."""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service not ready")
    
    article_info = search_service.get_article_info(article_id)
    if not article_info:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return article_info