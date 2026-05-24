import chromadb
from sentence_transformers import SentenceTransformer

from pathlib import Path
_HERE = Path(__file__).parent

client = chromadb.PersistentClient(path=str(_HERE / "chroma_db"))
collection = client.get_collection(name = "bioacoustics_papers")

# use the same embed model
model = SentenceTransformer("all-MiniLM-L6-v2")

## create the retrieve function
def retrieve(question, k=3):
    "return the top k most relevant chunks for a question"
    query_vector = model.encode([question])
    results = collection.query(
        query_embeddings = query_vector.tolist(),
        n_results = k,
    )
    return results

if __name__ == "__main__":
    question = "What is the detection range of the AudioMoth gunshot algorithm?"
    results = retrieve(question, k=3)

    print(f"question: {question}\n")

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        source = results['metadatas'][0][i]['source']
        distance = results['distances'][0][i]

        print(f"--- result {i+1} (distance: {distance:.3f}, source: {source}) ---")
        print(doc[:400])  # truncate to first 400 chars for readability
        print()
