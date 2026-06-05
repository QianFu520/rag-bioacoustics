from contextlib import asynccontextmanager

from anthropic import Anthropic
from fastapi import FastAPI, Request
from pydantic import BaseModel

from generate import build_prompt
from retrieve_hybrid import retrieve


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # retrieve_hybrid loads the embedding model, ChromaDB client, and BM25 index
    # at import time above. Anthropic client is cheap but we create it here
    # to keep all startup work in one place.
    app.state.anthropic = Anthropic()
    yield
    # nothing to clean up


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    retrieved = retrieve(body.question, k=body.k)
    prompt = build_prompt(body.question, retrieved)

    response = request.app.state.anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = response.content[0].text

    citations = [meta["source"] for meta in retrieved["metadatas"][0]]
    chunks = retrieved["documents"][0]

    return QueryResponse(answer=answer_text, citations=citations, chunks=chunks)
