import os
from anthropic import Anthropic
from dotenv import load_dotenv
from retrieve import retrieve
from pathlib import Path

_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")

client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"

def build_prompt(question, retrieved):
    chunks = retrieved['documents'][0]
    sources = retrieved['metadatas'][0]

    context = ""

    for i, (chunk, meta) in enumerate(zip(chunks, sources), start=1):
        context += f"[{i}] (source: {meta['source']})\n{chunk}\n\n"
    prompt = f"""Answer the question using ONLY the passages below. \
if the passages don't contain enough information, say so explicitly.
Cite the passage number(s) you used, like [1] or [1, 2].

PASSAGES:
{context}
QUESTION: {question}

ANSWER:"""
    return prompt


def answer(question, k=3, return_chunks=False):
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)

    response = client.messages.create(
        model = MODEL,
        max_tokens=500, 
        messages=[{"role": "user", "content": prompt}], 

    )
    answer_text = response.content[0].text

    if return_chunks:
        return answer_text, retrieved
    return answer_text
    

if __name__ == "__main__":
    question = "What is the detection range of the AudioMoth gunshot algorithm, and what affects it?"
    print(f"Q: {question}\n")
    print("A:", answer(question, k=3))