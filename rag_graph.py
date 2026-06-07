"""
Agentic RAG via LangGraph.

Two paths through the graph:
  simple  → retrieve → generate
  complex → decompose → retrieve_multi → generate

The router (route_query) uses Claude Haiku to classify the question.
Complex questions are decomposed into sub-questions, each retrieved
independently, then merged before generation.
"""
from typing import TypedDict

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

import retrieve_hybrid
from generate import build_prompt, MODEL

client = Anthropic()

MAX_MERGED_CHUNKS = 8


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    route: str                   # "simple" | "complex"
    sub_questions: list[str]
    retrieved_chunks: dict
    answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def route_query(state: RAGState) -> RAGState:
    prompt = f"""Classify this question as either "simple" or "complex".

- simple: asks about one topic or one tool, even if multi-sentence or detailed
  Examples: "What is AudioMoth?", "How does OpenSoundscape handle CNN training?",
  "What methods does OpenSoundscape provide for signal processing?"
- complex: explicitly compares two or more tools/papers, or asks for differences between them
  Examples: "Compare AudioMoth and OpenSoundscape", "How do AudioMoth and OpenSoundscape differ?",
  "What are the differences between X and Y?"

Only classify as complex if the question explicitly asks for a comparison across multiple sources.
When in doubt, classify as simple.

Question: {state["question"]}

Reply with a single word: simple or complex"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip().lower()
    route = "complex" if "complex" in raw else "simple"

    return {"route": route}


def decompose(state: RAGState) -> RAGState:
    prompt = f"""Break the following question into 2-3 focused sub-questions that together cover the original.
Each sub-question should be answerable from a single topic or paper.

Write one sub-question per line, numbered. Example:
1. What is X?
2. What is Y?

Question: {state["question"]}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    lines = [
        line.lstrip("0123456789. ").strip()
        for line in raw.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    sub_questions = lines if lines else [state["question"]]

    return {"sub_questions": sub_questions}


def retrieve_node(state: RAGState) -> RAGState:
    retrieved = retrieve_hybrid.retrieve(state["question"], k=5)
    return {"retrieved_chunks": retrieved}


def retrieve_multi(state: RAGState) -> RAGState:
    seen_ids = set()
    merged_ids = []
    merged_docs = []
    merged_metas = []

    for sub_q in state["sub_questions"]:
        results = retrieve_hybrid.retrieve(sub_q, k=5)
        for cid, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged_ids.append(cid)
                merged_docs.append(doc)
                merged_metas.append(meta)

    # cap to avoid overwhelming the generator
    merged_ids = merged_ids[:MAX_MERGED_CHUNKS]
    merged_docs = merged_docs[:MAX_MERGED_CHUNKS]
    merged_metas = merged_metas[:MAX_MERGED_CHUNKS]

    return {
        "retrieved_chunks": {
            "ids": [merged_ids],
            "documents": [merged_docs],
            "metadatas": [merged_metas],
        }
    }


def generate_node(state: RAGState) -> RAGState:
    prompt = build_prompt(state["question"], state["retrieved_chunks"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"answer": response.content[0].text}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def decide_path(state: RAGState) -> str:
    return state["route"]


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("route_query", route_query)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retrieve_multi", retrieve_multi)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("route_query")

    graph.add_conditional_edges(
        "route_query",
        decide_path,
        {"simple": "retrieve", "complex": "decompose"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("decompose", "retrieve_multi")
    graph.add_edge("retrieve_multi", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(question: str) -> dict:
    """Run a question through the agentic graph. Returns answer + metadata."""
    result = rag_graph.invoke({"question": question})
    return {
        "answer": result["answer"],
        "route": result["route"],
        "sub_questions": result.get("sub_questions", []),
        "citations": [m["source"] for m in result["retrieved_chunks"]["metadatas"][0]],
        "chunks": result["retrieved_chunks"]["documents"][0],
    }


def retrieve_agentic(question: str, k: int = 10) -> dict:
    """Retrieval-only path through the graph — for eval measurement.

    Routes the question, decomposes if complex, retrieves per sub-question,
    and merges. Returns the same dict format as retrieve_hybrid.retrieve().
    """
    route_result = route_query({"question": question})
    route = route_result["route"]

    if route == "simple":
        return retrieve_hybrid.retrieve(question, k=k)

    decompose_result = decompose({"question": question})
    sub_questions = decompose_result["sub_questions"]

    seen_ids = set()
    merged_ids, merged_docs, merged_metas = [], [], []
    for sub_q in sub_questions:
        results = retrieve_hybrid.retrieve(sub_q, k=k)
        for cid, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged_ids.append(cid)
                merged_docs.append(doc)
                merged_metas.append(meta)

    # Re-rank merged pool by hybrid score on the original question.
    # Appearance order (sub-q 1 first, sub-q 2 second) is arbitrary;
    # this restores relevance ordering so scoring at small k is meaningful.
    ranking = retrieve_hybrid.retrieve(question, k=20)
    rank_order = {cid: rank for rank, cid in enumerate(ranking["ids"][0])}
    merged = sorted(
        zip(merged_ids, merged_docs, merged_metas),
        key=lambda x: rank_order.get(x[0], len(rank_order)),
    )
    merged_ids = [m[0] for m in merged]
    merged_docs = [m[1] for m in merged]
    merged_metas = [m[2] for m in merged]

    return {
        "ids": [merged_ids],
        "documents": [merged_docs],
        "metadatas": [merged_metas],
    }


if __name__ == "__main__":
    simple_q = "What is AudioMoth?"
    complex_q = "Compare the field deployment approaches of AudioMoth and OpenSoundscape."

    for q in [simple_q, complex_q]:
        print(f"\nQ: {q}")
        result = run(q)
        print(f"Route: {result['route']}")
        if result["sub_questions"]:
            print(f"Sub-questions: {result['sub_questions']}")
        print(f"Answer: {result['answer'][:300]}...")
