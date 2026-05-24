import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from loader import load_docx
from chunking import chunk_by_sentences


PAPERS = [
    "papers/neuroethology_review.docx",
    "papers/opensoundscape_methods.docx",
    "papers/audiomoth_deployment.docx",
]

BM25_INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"


def tokenize(text):
    """Simple whitespace+lowercase tokenizer for BM25.

    Splits on non-word characters, lowercases, drops empties. Same tokenizer
    must be used for indexing (here) and for querying (in retrieve_hybrid.py).
    """
    return [t for t in re.split(r"\W+", text.lower()) if t]


def build():
    """Build ChromaDB collection AND BM25 index from the papers."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "bioacoustics_papers"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"deleted existing collection '{collection_name}'")
    collection = client.create_collection(name=collection_name)
    print(f"created collection '{collection_name}'\n")

    print("loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    all_sources = []
    all_ids = []

    for path in PAPERS:
        text = load_docx(path)
        chunks = chunk_by_sentences(text, target_size=1200, overlap=0)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_sources.append(path)
            all_ids.append(f"{path}: :chunk_{i}")
        print(f"{path}: {len(chunks)} chunks")

    print(f"\ntotal chunks: {len(all_chunks)}")
    print("embedding...")
    vectors = model.encode(all_chunks, show_progress_bar=True)

    collection.add(
        ids=all_ids,
        embeddings=vectors.tolist(),
        documents=all_chunks,
        metadatas=[{"source": s} for s in all_sources],
    )
    print(f"\nstored {collection.count()} vectors in the collection.")

    print("building BM25 index...")
    tokenized_chunks = [tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": all_ids,
        }, f)
    print(f"BM25 index built over {len(tokenized_chunks)} chunks, "
          f"saved to {BM25_INDEX_PATH.name}")
    print("done. ChromaDB vectors persisted to ./chroma_db/, "
          "BM25 index to bm25_index.pkl")


if __name__ == "__main__":
    build()