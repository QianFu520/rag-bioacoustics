from loader import load_docx
from chunking import chunk_by_sentences

PAPERS = [
    "papers/neuroethology_review.docx",
    "papers/opensoundscape_methods.docx",
    "papers/audiomoth_deployment.docx",
]

TARGET_SIZE = 1200 ## we can tune this 

for path in PAPERS:
    text = load_docx(path)
    chunks = chunk_by_sentences(text, target_size=TARGET_SIZE)
    print(f"\n============ {path} =========")
    print(f"total chars: {len(text)}, total chunks: {len(chunks)}")
    print(f"\n--- first chunk ---\n{chunks[0]}")
    print(f"\n--- middle chunk (#{len(chunks)//2}) ---\n{chunks[len(chunks)//2]}")
    print(f"\n--- last chunk ---\n{chunks[-1]}")

