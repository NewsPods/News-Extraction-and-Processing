"""
FastAPI-based hybrid search service for news articles.
This service supports keyword-based hybrid search as well as filtering articles
by multiple news sections and newspapers.
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

# Load credentials from .env file
load_dotenv()
CONN_STRING = os.environ.get("COCKROACHDB_CONN_STRING")
SEARCH_MODEL = SentenceTransformer('all-mpnet-base-v2')

# These thresholds decide how strict our hybrid search algorithm will be
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
        # Pull today's articles (with embeddings) from the DB when the API boots up
        print("📥 Pulling today's articles from the database...")
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
                    WHERE a.embedding IS NOT NULL
                    AND a.created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                """)
                results = connection.execute(query).fetchall()

            if not results:
                print("⚠️ No articles found in DB for today.")
                return

            article_tuples = []
            for row in results:
                article_id = int(str(row[0]).split('.')[0])
                title = row[1]
                description = row[2] or ""
                embedding = np.array(row[3], dtype=np.float32)

                # BM25 works on bag-of-words tokens, so combine title + description and tokenize
                bm25_tokens = f"{title} {description}".lower().split()

                article_tuples.append((article_id, title, embedding, bm25_tokens))

            self.article_data = article_tuples

            # Build the BM25 index once, instead of scoring DB records repeatedly
            self.bm25 = BM25Okapi([t[3] for t in article_tuples])

            print(f"✅ Ready — {len(self.article_data)} articles loaded into memory.")

        except Exception as e:
            print(f"❌ Couldn't load articles: {e}")
            raise

    def search(self, query: str) -> List[int]:
        # If nothing was loaded, return gracefully
        if not self.article_data:
            return []

        # BM25 scoring (term frequency style matching)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Semantic similarity using embeddings
        query_embedding = SEARCH_MODEL.encode([query])
        embeddings_array = np.array([tup[2] for tup in self.article_data])
        semantic_scores = cosine_similarity(query_embedding, embeddings_array)[0]

        qualified_articles = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # Apply our scoring rules to decide relevant articles
        for idx, (article_id, title, _, _) in enumerate(self.article_data):
            bm25_score = bm25_scores[idx]
            semantic_score = semantic_scores[idx]

            if (
                (bm25_score > BM25_BASE_THRESHOLD and semantic_score > SEMANTIC_BASE_THRESHOLD)
                or bm25_score > BM25_HIGH_THRESHOLD
                or semantic_score > SEMANTIC_HIGH_THRESHOLD
            ):
                combined_score = (bm25_score / max_bm25 + semantic_score) / 2
                qualified_articles.append({'id': int(article_id), 'combined_score': combined_score})

        # Sort most relevant first
        qualified_articles.sort(key=lambda x: x['combined_score'], reverse=True)

        return [a['id'] for a in qualified_articles[:TOP_K]]

    def get_article_info(self, article_id: int) -> Dict[str, str]:
        # Just return lightweight info for debugging/inspection
        for tup in self.article_data:
            if tup[0] == article_id:
                return {"id": str(article_id), "title": tup[1]}
        return {}


# ---------------- FASTAPI BOOTSTRAP ----------------
app = FastAPI(title="News Article Search API")
search_service = None


class SearchRequest(BaseModel):
    keywords: List[str]


class SectionNewspaperRequest(BaseModel):
    news_sections: List[str]
    newspapers: List[str]


@app.on_event("startup")
def startup_event():
    # Warm up the search pipeline on startup
    global search_service
    if not CONN_STRING:
        raise RuntimeError("Database connection string missing")
    search_service = ArticleSearchService(CONN_STRING)


@app.post("/search", response_model=Dict[str, List[str]])
def search_articles(request: SearchRequest):
    # Search across multiple keywords and return top matches
    if not search_service or not search_service.article_data:
        raise HTTPException(status_code=503, detail="Search service is still loading")
    return {kw: [str(i) for i in search_service.search(kw)] for kw in request.keywords}


# -------- Multi-section + multi-newspaper filter endpoint --------
@app.post("/sections")
def get_articles_by_section_and_newspaper(request: SectionNewspaperRequest):
    # DB query for all combinations of (section × newspaper)
    if not CONN_STRING:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        engine = create_engine(
            CONN_STRING,
            poolclass=NullPool,
            connect_args={"application_name": "news_search_api"}
        )

        # Final response format → {section: {newspaper: {count, article_ids}}}
        results = {section: {} for section in request.news_sections}

        with engine.connect() as connection:
            for section in request.news_sections:
                for paper in request.newspapers:
                    query = text("""
                        SELECT DISTINCT a.article_id::BIGINT
                        FROM public.articles_sections s
                        JOIN public.articles a
                            ON s.article_id = a.article_id
                        WHERE s.news_section = :section
                        AND (
                                a.news_source = :newspaper
                                OR a.news_source = 'Multiple'
                            )
                        AND a.created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                        ORDER BY a.article_id
                    """)

                    rows = connection.execute(query, {
                        "section": section,
                        "newspaper": paper
                    }).fetchall()

                    # Convert to clean int-strings for consistency
                    article_ids = [str(int(r[0])) for r in rows]
                    results[section][paper] = {
                        "count": len(article_ids),
                        "article_ids": article_ids
                    }

                    print(f"📌 Section='{section}', Newspaper='{paper}' → {len(article_ids)} articles")

        return results

    except Exception as e:
        print(f"❌ Issue while fetching articles: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/health")
def health_check():
    # Quick health/status endpoint
    if not search_service:
        return {"status": "initializing"}
    return {"status": "healthy", "articles_loaded": len(search_service.article_data)}


@app.get("/article/{article_id}")
def get_article(article_id: int):
    # Fetch minimal info about a specific article (mainly for debugging UI/logs)
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not ready yet")
    info = search_service.get_article_info(article_id)
    if not info:
        raise HTTPException(status_code=404, detail="Article not found")
    return info