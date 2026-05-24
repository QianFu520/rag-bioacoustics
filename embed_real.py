from sentence_transformers import SentenceTransformer
from loader import load_docx
from chunking import chunk_by_sentences

PAPERS = [
    "papers/neuroethology_review.docx",
    "papers/opensoundscape_methods.docx",
    "papers/audiomoth_deployment.docx",
]

print("loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded.\n")

## step one: load and chunk every paper and keeping track of which paper each chunk came from.

all_chunks = []
all_sources = []

for path in PAPERS:
    text = load_docx(path)
    chunks = chunk_by_sentences(text, target_size=800)
    all_chunks.extend(chunks)
    all_sources.extend([path] * len(chunks))
    print(f"{path}: {len(chunks)} chunks")

print(f"\ntotal chunks across all papers: {len(all_chunks)}")

## step two: embed all chunks at once, the model handle batching internally

print("\nembedding all chunks...")
vectors = model.encode(all_chunks, show_progress_bar = True)
print(f"\nshape of vector matrix: {vectors.shape}")
print(f"that's {vectors.shape[0]} chunks, each as a {vectors.shape[1]}-dim vector.")
