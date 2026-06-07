from contextlib import asynccontextmanager

from anthropic import Anthropic
from fastapi import FastAPI, Request
from pydantic import BaseModel

from generate import build_prompt
import retrieve_hybrid
from rag_graph import run as graph_run


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks: list[str]


class AgenticQueryResponse(BaseModel):
    answer: str
    route: str
    sub_questions: list[str]
    citations: list[str]
    chunks: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # If the ChromaDB collection is empty (e.g. fresh Docker container),
    # build the index from the papers before accepting any requests.
    if retrieve_hybrid._collection.count() == 0:
        print("Collection is empty — building index from papers...")
        from build_store import build
        build()
        retrieve_hybrid.reload()
        print("Index build complete.")

    app.state.anthropic = Anthropic()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    retrieved = retrieve_hybrid.retrieve(body.question, k=body.k)
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


@app.post("/query/agentic", response_model=AgenticQueryResponse)
def query_agentic(body: QueryRequest):
    result = graph_run(body.question)
    return AgenticQueryResponse(**result)
