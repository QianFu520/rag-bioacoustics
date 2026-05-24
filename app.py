"""
Streamlit demo for the bioacoustics RAG project.

Three tabs:
  1. "Ask a question" — interactive chat-style retrieval over the three papers
  2. "Eval results" — iteration comparison table from the project's eval framework
  3. "Try the eval framework" — run any of the 24 eval questions live and see
     retrieval + generation + faithfulness judgment

The heavy lifting (loading models, building/loading indexes) is done once at
app startup via @st.cache_resource. Subsequent user interactions are fast.
"""
import os
from pathlib import Path

import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
BM25_PATH = PROJECT_ROOT / "bm25_index.pkl"


# --- Heavy initialization (cached across reruns and sessions) ---

@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline. Builds indexes if they don't exist.

    Cached with @st.cache_resource so this runs ONCE per app lifetime,
    regardless of how many users hit the app or how many reruns happen.
    """
    # Check if both indexes already exist
    indexes_exist = CHROMA_PATH.exists() and BM25_PATH.exists()

    if not indexes_exist:
        # Streamlit Cloud first boot — build from scratch
        st.info("First-time setup: building indexes from source papers. This takes ~30 seconds.")
        from build_store import build
        build()

    # Now load and return the retriever module — its module-level setup
    # connects to ChromaDB and loads BM25 from disk
    from retrieve_hybrid import retrieve
    return retrieve


# --- Page config (must be the first Streamlit call) ---
st.set_page_config(
    page_title="Bioacoustics RAG Demo",
    page_icon="🔬",
    layout="wide",
)

# --- Title and intro ---
st.title("Bioacoustics RAG — Project Demo")
st.markdown(
    "An end-to-end RAG system over three bioacoustics research papers, with "
    "a documented evaluation framework and three measured iterations of "
    "retrieval improvements."
)

# Initialize the pipeline (cached — runs once)
retrieve = init_pipeline()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Ask a question",
    "Eval results",
    "Try the eval framework",
])

with tab1:
    st.header("Ask a question")
    st.markdown(
        "Ask anything about the three bioacoustics papers in the corpus "
        "(neuroethology review, OpenSoundscape methods, AudioMoth deployment). "
        "The system retrieves the 5 most relevant passages using hybrid "
        "retrieval (BM25 + dense embeddings, fused with RRF), then answers "
        "using only those passages — refusing if the passages don't contain "
        "enough information."
    )

    # The form pattern: widgets inside `with st.form` only trigger
    # a rerun when the submit button is clicked, not on every keystroke.
    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the detection range of the AudioMoth gunshot algorithm?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        # User submitted a non-empty question — run the pipeline
        from generate import answer

        with st.spinner("Retrieving passages and generating answer..."):
            answer_text, retrieved = answer(question, k=5, return_chunks=True)

        # Display the answer
        st.markdown("### Answer")
        st.markdown(answer_text)

        # Show retrieved passages in a collapsible expander
        with st.expander("See retrieved passages"):
            chunks = retrieved["documents"][0]
            sources = retrieved["metadatas"][0]
            for i, (chunk, meta) in enumerate(zip(chunks, sources), start=1):
                source_name = meta["source"].split("/")[-1]
                st.markdown(f"**[{i}]** _{source_name}_")
                st.markdown(chunk)
                st.markdown("---")  # horizontal divider between chunks

    elif submitted and not question.strip():
        st.warning("Please enter a question.")

with tab2:
    st.header("Evaluation results")
    st.markdown(
        "I built an evaluation framework alongside this RAG system to "
        "measure retrieval quality (recall@k with hit and coverage metrics) "
        "and generation quality (faithfulness via LLM-as-judge). The "
        "framework runs on a hand-curated 24-question eval set with "
        "verified ground truth chunks. Below: how three iterations of "
        "retrieval improvements moved the metrics."
    )

    # --- Headline metrics: recall@5 coverage across four configs ---
    st.markdown("### Overall recall@5 coverage across iterations")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Baseline",
            "0.60",
            help="Chunk size 800, semantic retrieval only",
        )
    with col2:
        st.metric(
            "Iteration 1",
            "0.77",
            delta="+0.17",
            help="Chunk size 1200, semantic retrieval",
        )
    with col3:
        st.metric(
            "Iteration 2",
            "0.65",
            delta="-0.12",
            help="Chunk size 1200 + overlap 200 (reverted; see ITERATION.md)",
        )
    with col4:
        st.metric(
            "Iteration 3",
            "0.88",
            delta="+0.11",
            help="Chunk size 1200 + hybrid retrieval (BM25 + semantic with RRF)",
        )

    st.markdown("&nbsp;")  # vertical spacing

    # --- Full comparison table ---
    st.markdown("### Recall@k and faithfulness by configuration")

    import pandas as pd

    df = pd.DataFrame(
        {
            "Configuration": [
                "Baseline (chunk size 800)",
                "Iter 1 (chunk size 1200)",
                "Iter 2 (1200 + overlap 200)",
                "Iter 3 (1200 + hybrid)",
            ],
            "any@3": [0.63, 0.83, 0.67, 0.87],
            "cov@3": [0.53, 0.78, 0.60, 0.83],
            "any@5": [0.67, 0.80, 0.67, 0.90],
            "cov@5": [0.60, 0.77, 0.65, 0.88],
            "any@10": [0.83, 0.93, 0.87, 0.90],
            "cov@10": [0.80, 0.93, 0.85, 0.90],
            "Faithfulness": [1.00, 0.98, 0.96, 0.98],
        }
    )

    st.table(df.style.format({col: "{:.2f}" for col in df.columns if col != "Configuration"}))

    # --- Brief reading guide ---
    st.markdown("### Reading the table")
    st.markdown(
        "- **any@k**: did retrieval get at least one ground-truth chunk into "
        "the top-k? (binary, averaged across 30 anchors)\n"
        "- **cov@k**: what fraction of ground-truth chunks did retrieval find? "
        "(0.0 to 1.0, averaged across 30 anchors)\n"
        "- **Faithfulness**: LLM-as-judge score of whether the generated "
        "answer is supported by the retrieved chunks (1.0 = all supported)\n"
        "\n"
        "**Iter 2 regressed** and was reverted — bigger chunks with overlap "
        "diluted the embedding distinctiveness that drove Iter 1's gains. "
        "**Iter 3 added lexical retrieval (BM25)** in addition to semantic, "
        "fused with reciprocal rank fusion (RRF). Cross-doc questions, where "
        "lexical signal complements semantic similarity, benefited most.\n"
        "\n"
        "Full iteration writeups in [`ITERATION.md`](https://github.com/QianFu520/rag-bioacoustics/blob/main/evals/ITERATION.md). "
        "Documented limitations in [`NORMALIZER_NOTES.md`](https://github.com/QianFu520/rag-bioacoustics/blob/main/evals/NORMALIZER_NOTES.md)."
    )

with tab3:
    st.header("Try the eval framework")
    st.write("Tab 3 content coming in the next step.")