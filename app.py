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
    st.header("Eval results")
    st.write("Tab 2 content coming in the next step.")

with tab3:
    st.header("Try the eval framework")
    st.write("Tab 3 content coming in the next step.")