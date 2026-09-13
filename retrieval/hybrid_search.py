"""
Combines BM25 (keyword/lexical) retrieval with sentence-embedding
(semantic) retrieval over the corpus documents, then merges both into a
single ranked result. Keyword search catches exact terms (names, specific
words like "cyanide"); semantic search catches conceptually related text
even when the wording differs from the query.

"""

import os
import re
import sys
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import RetrievedChunk
from ingestion.preprocess import load_documents, CORPUS_DIR


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


class HybridRetriever:
    """Builds keyword + semantic indexes once, then serves fast repeated
    queries against them. Each document is treated as a single chunk."""

    def __init__(self, corpus_dir: str = CORPUS_DIR, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.documents = load_documents(corpus_dir)
        self.doc_ids = [d["document_id"] for d in self.documents]
        self.texts = [d["text"] for d in self.documents]

        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.embedder = TextEmbedding(model_name=embedding_model)
        self.embeddings = np.array(list(self.embedder.embed(self.texts)))

    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        scores = np.array(self.bm25.get_scores(_tokenize(query)))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                document_id=self.doc_ids[i],
                text=self.texts[i],
                keyword_score=float(scores[i]),
            )
            for i in top_indices
        ]

    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_embedding = list(self.embedder.embed([query]))[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                document_id=self.doc_ids[i],
                text=self.texts[i],
                semantic_score=float(scores[i]),
            )
            for i in top_indices
        ]

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[RetrievedChunk]:
        """alpha controls the semantic vs keyword weighting: 1.0 = pure
        semantic, 0.0 = pure keyword, 0.5 = balanced (default)."""
        keyword_scores = np.array(self.bm25.get_scores(_tokenize(query)))

        query_embedding = list(self.embedder.embed([query]))[0]
        semantic_scores = self.embeddings @ query_embedding

        keyword_norm = _min_max_normalize(keyword_scores)
        semantic_norm = _min_max_normalize(semantic_scores)

        combined = alpha * semantic_norm + (1 - alpha) * keyword_norm
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [
            RetrievedChunk(
                document_id=self.doc_ids[i],
                text=self.texts[i],
                keyword_score=float(keyword_scores[i]),
                semantic_score=float(semantic_scores[i]),
                combined_score=float(combined[i]),
            )
            for i in top_indices
        ]


_retriever_instance = None


def get_retriever() -> HybridRetriever:
    """Singleton accessor so the embedding model and indexes are only built
    once, then reused across every API call."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


if __name__ == "__main__":
    retriever = get_retriever()

    test_query = "who was near the bar during the poisoning window"

    print(f"Query: '{test_query}'\n")

    print("--- Keyword results ---")
    for r in retriever.keyword_search(test_query, top_k=3):
        print(f"  {r.document_id} (score={r.keyword_score:.2f})")

    print("\n--- Semantic results ---")
    for r in retriever.semantic_search(test_query, top_k=3):
        print(f"  {r.document_id} (score={r.semantic_score:.3f})")

    print("\n--- Hybrid results ---")
    for r in retriever.hybrid_search(test_query, top_k=3):
        print(f"  {r.document_id} (combined={r.combined_score:.3f}, kw={r.keyword_score:.2f}, sem={r.semantic_score:.3f})")