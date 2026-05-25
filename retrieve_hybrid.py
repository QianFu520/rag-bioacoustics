"""
Hybrid retrieval: BM25 (lexical) + semantic (dense embeddings), fused via
Reciprocal Rank Fusion (RRF).

Exposes the same retrieve(question, k) signature as retrieve.py, so it's a
drop-in replacement for the eval framework and generate.py.

Architecture:
  1. Run semantic retrieval (ChromaDB + MiniLM) — fetch K_FETCH candidates
  2. Run BM25 retrieval (rank_bm25 over same chunks) — fetch K_FETCH candidates
  3. Fuse the two ranked lists with RRF
  4. Return top-k from the fused ranking
"""
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from build_store import tokenize  

# --- paths and constants ---
_HERE = Path(__file__).parent
_CHROMA_PATH = _HERE / "chroma_db"
_BM25_INDEX_PATH = _HERE / "bm25_index.pkl"
_COLLECTION_NAME = "bioacoustics_papers"

# How many candidates each retriever fetches before fusion.

K_FETCH = 20

# RRF constant. The standard value from the 2009 paper; rarely tuned.
RRF_K = 60

# --- load both indexes once at import time ---
_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
_collection = _client.get_collection(name=_COLLECTION_NAME)
_model = SentenceTransformer("all-MiniLM-L6-v2")

with open(_BM25_INDEX_PATH, "rb") as f:
    _bm25_data = pickle.load(f)
_bm25 = _bm25_data["bm25"]
_bm25_chunk_ids = _bm25_data["chunk_ids"]


def _retrieve_semantic(question, k):
    """Return list of (chunk_id, rank) tuples from semantic retrieval.
    Rank starts at 1 (best).
    """
    query_vector = _model.encode([question])
    results = _collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=k,
    )
    chunk_ids = results["ids"][0]  # Chroma's batch-nested format
    return [(cid, rank + 1) for rank, cid in enumerate(chunk_ids)]


def _retrieve_bm25(question, k):
    """Return list of (chunk_id, rank) tuples from BM25 retrieval.
    Rank starts at 1 (best).
    """
    tokenized_query = tokenize(question)
    scores = _bm25.get_scores(tokenized_query)
    # Sort indices by score descending, take top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(_bm25_chunk_ids[idx], rank + 1) for rank, idx in enumerate(top_indices)]


def _reciprocal_rank_fusion(ranked_lists, rrf_k=RRF_K):
    """Fuse multiple ranked lists into a single ranking.

    Each ranked_list is a list of (chunk_id, rank) tuples. Each chunk's
    final score is the sum of 1/(rrf_k + rank) across all lists it appears in.
    Chunks that appear in only one list still get a score; chunks in
    multiple lists get boosted.
    """
    fused_scores = {}
    for ranked in ranked_lists:
        for chunk_id, rank in ranked:
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1.0 / (rrf_k + rank)
    # Return chunk_ids sorted by fused score descending
    return sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)


def retrieve(question, k=3):
   
    semantic_results = _retrieve_semantic(question, K_FETCH)
    bm25_results = _retrieve_bm25(question, K_FETCH)
    fused_ids = _reciprocal_rank_fusion([semantic_results, bm25_results])
    top_ids = fused_ids[:k]

    # Fetch chunk content and metadata for the top-k from ChromaDB
    
    fetched = _collection.get(ids=top_ids)
    
    id_to_idx = {cid: i for i, cid in enumerate(fetched["ids"])}
    ordered_docs = [fetched["documents"][id_to_idx[cid]] for cid in top_ids]
    ordered_meta = [fetched["metadatas"][id_to_idx[cid]] for cid in top_ids]

    
    return {
        "ids": [top_ids],
        "documents": [ordered_docs],
        "metadatas": [ordered_meta],
    }


if __name__ == "__main__":
    # Smoke test
    question = "What is the detection range of the AudioMoth gunshot algorithm?"
    results = retrieve(question, k=5)
    print(f"question: {question}\n")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        cid = results["ids"][0][i]
        source = results["metadatas"][0][i]["source"]
        print(f"--- result {i+1} (id: {cid.rsplit(':', 1)[-1].strip()}, source: {source}) ---")
        print(doc[:300])
        print()