from sentence_transformers import SentenceTransformer

## load the embedding model.
print("loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded")

## three short sentences: two relaed, one unrelated

sentences = [
    "The bird sang at dawn from the top of the oak tree.",
    "At sunrise, a songbird called out from a high branch.",
    "The committee approved the quarterly budget.",
]

## embed all three at once. returns a numpy array of shape (3, 384)
vectors = model.encode(sentences)
print(f"\nshape of vectors: {vectors.shape}")
print(f"first vector (first 10 dims): {vectors[0][:10]}")

## now: how similar is each sentence to the first one?
## we use cosine similarity, which is the standard for embeddings 

from sentence_transformers.util import cos_sim

for i, s in enumerate(sentences):
    similarity = cos_sim(vectors[0], vectors[i]).item()
    print(f"similarity(sentence 0, sentence {i}) = {similarity:.3f}  |  {s}")