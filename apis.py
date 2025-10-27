"""
FastAPI-based search service for news articles using hybrid search.
✅ Modified Logic:
An article is considered relevant if ANY of the following is true:
    1. BM25 > 3 AND Semantic > 0.27
    2. BM25 > 5
    3. Semantic > 0.4
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
SEARCH_MODEL = SentenceTransformer('all-mpnet-base-v2')

# --- Thresholds ---
BM25_BASE_THRESHOLD = 3
SEMANTIC_BASE_THRESHOLD = 0.27
BM25_HIGH_THRESHOLD = 5.0
SEMANTIC_HIGH_THRESHOLD = 0.4
TOP_K = 10


class ArticleSearchService:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.article_data: List[Tuple[int, str, np.ndarray, List[str]]] = []
        self.bm25 = None
        self._load_articles()

    def _load_articles(self):
        """Load all articles with embeddings from the database."""
        print("📥 Loading all articles from database...")
        try:
            engine = create_engine(
                self.conn_string,
                poolclass=NullPool,
                connect_args={"application_name": "news_search_api"}
            )

            with engine.connect() as connection:
                query = text("""
                    SELECT a.article_id::BIGINT, a.title, a.description, a.embedding
                    FROM public.articles a
                    WHERE a.embedding IS NOT NULL and a.created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                    
                """)
                result = connection.execute(query)
                results = result.fetchall()

            if not results:
                print("⚠️ No articles found in database.")
                return

            article_tuples = []
            for row in results:
                if row[3] is None:
                    continue

                raw_id = row[0]
                if isinstance(raw_id, (int, np.integer)):
                    article_id = int(raw_id)
                elif isinstance(raw_id, float):
                    article_id = int(np.int64(raw_id))
                else:
                    article_id = int(str(raw_id).split('.')[0])

                title = row[1]
                description = row[2] or ""
                embedding = np.array(row[3], dtype=np.float32)
                bm25_tokens = f"{title} {description}".lower().split()
                article_tuples.append((article_id, title, embedding, bm25_tokens))

            if not article_tuples:
                print("⚠️ No valid articles with embeddings found.")
                return

            self.article_data = article_tuples
            texts_for_bm25 = [tup[3] for tup in article_tuples]
            self.bm25 = BM25Okapi(texts_for_bm25)

            print(f"✅ Loaded {len(self.article_data)} articles.")
            print(f"📊 Thresholds set to:")
            print(f"   1️⃣ BM25 > {BM25_BASE_THRESHOLD} AND Semantic > {SEMANTIC_BASE_THRESHOLD}")
            print(f"   2️⃣ BM25 > {BM25_HIGH_THRESHOLD}")
            print(f"   3️⃣ Semantic > {SEMANTIC_HIGH_THRESHOLD}")

        except Exception as e:
            print(f"❌ Failed to load articles: {e}")
            raise

    def search(self, query: str) -> List[int]:
        """
        Performs hybrid search using three relevance conditions:
        1. BM25 > 3 and Semantic > 0.27
        2. BM25 > 5
        3. Semantic > 0.4
        """
        if not self.article_data:
            print("⚠️ No articles available for search")
            return []

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        query_embedding = SEARCH_MODEL.encode([query])
        embeddings_array = np.array([tup[2] for tup in self.article_data])
        semantic_scores = cosine_similarity(query_embedding, embeddings_array)[0]

        qualified_articles = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0  # Prevent division by zero

        for idx, (article_id, title, _, _) in enumerate(self.article_data):
            bm25_score = bm25_scores[idx]
            semantic_score = semantic_scores[idx]

            # ✅ Apply new 3-condition logic
            if (
                (bm25_score > BM25_BASE_THRESHOLD and semantic_score > SEMANTIC_BASE_THRESHOLD)
                or bm25_score > BM25_HIGH_THRESHOLD
                or semantic_score > SEMANTIC_HIGH_THRESHOLD
            ):
                bm25_normalized = bm25_score / max_bm25
                combined_score = (bm25_normalized + semantic_score) / 2
                
                qualified_articles.append({
                    'id': int(article_id),
                    'title': title,
                    'bm25_score': float(bm25_score),
                    'semantic_score': float(semantic_score),
                    'combined_score': float(combined_score)
                })

        # Sort by a heuristic combined score
        qualified_articles.sort(key=lambda x: x['combined_score'], reverse=True)

        print(f"\n🔍 Query: '{query}'")
        print(f"✅ Found {len(qualified_articles)} qualifying articles.")
        print(f"Conditions used: (BM25 > 3 and Semantic > 0.27) or BM25 > 5 or Semantic > 0.4")

        if qualified_articles:
            for i, art in enumerate(qualified_articles[:5]):
                print(f"  {i+1}. ID={art['id']}, BM25={art['bm25_score']:.3f}, "
                      f"Semantic={art['semantic_score']:.3f}, Combined={art['combined_score']:.3f}")
                print(f"      Title: '{art['title'][:70]}...'")
        else:
            print("⚠️ No articles met relevance criteria.")

        return [a['id'] for a in qualified_articles[:TOP_K]]

    def get_article_info(self, article_id: int) -> Dict[str, str]:
        """Retrieve article information by ID."""
        match = next((tup for tup in self.article_data if tup[0] == article_id), None)
        if match:
            article_id, title, _, _ = match
            return {"id": str(article_id), "title": title}
        return {}


# --- FastAPI Application ---
app = FastAPI(title="News Article Search API - Custom Hybrid Conditions")
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
    """Accepts keywords and returns article IDs that meet relevance conditions."""
    if search_service is None or not search_service.article_data:
        raise HTTPException(status_code=503, detail="Search service is not ready or has no articles.")

    results = {}
    for keyword in request.keywords:
        article_ids = search_service.search(keyword)
        results[keyword] = [str(id) for id in article_ids]

    return results


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if search_service is None:
        return {"status": "initializing", "articles_loaded": 0}
    return {
        "status": "healthy",
        "articles_loaded": len(search_service.article_data),
        "conditions": {
            "1": f"BM25 > {BM25_BASE_THRESHOLD} and Semantic > {SEMANTIC_BASE_THRESHOLD}",
            "2": f"BM25 > {BM25_HIGH_THRESHOLD}",
            "3": f"Semantic > {SEMANTIC_HIGH_THRESHOLD}"
        }
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