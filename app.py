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


import re

def demote_markdown_headings(text):
    """Demote markdown headings by 2 levels for display.

    The generator (Haiku) tends to use # and ## for answer structure, which
    renders larger than the page's own headings in Streamlit. This shifts
    every heading down 2 levels: # → ###, ## → ####, etc. Preserves the
    structural intent of the answer while making it sit visually below
    page-level headings.
    """
    def demote(match):
        hashes = match.group(1)
        # Add 2 hashes, cap at 6 (the max heading level)
        new_hashes = "#" * min(len(hashes) + 2, 6)
        return new_hashes
    return re.sub(r"(?m)^(#{1,6})(?=\s)", demote, text)
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
        st.markdown(demote_markdown_headings(answer_text))

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
    st.markdown(
        "Pick any of the 24 questions from the eval set. The system will "
        "run hybrid retrieval, generate an answer, and have an LLM-as-judge "
        "evaluate whether the answer is supported by the retrieved passages. "
        "You'll see the full eval pipeline in one view: ground truth chunks, "
        "retrieved chunks, recall@5 score, the generated answer, and the "
        "faithfulness verdict."
    )

    # --- Load eval set + ground truth (cached at app startup) ---

    @st.cache_resource
    def load_eval_data():
        """Load the eval set questions and the ground truth file once."""
        import json
        import re

        eval_set_path = PROJECT_ROOT / "evals" / "eval_set.md"
        ground_truth_path = PROJECT_ROOT / "evals" / "ground_truth_1200_hybrid.json"

        # Parse eval_set.md the same way our scripts do
        text = eval_set_path.read_text(encoding="utf-8")
        questions = []
        blocks = re.split(r"\n(?=## Q\d+ — Category)", text)
        for block in blocks:
            m = re.match(r"## (Q\d+) — Category (\d+)", block)
            if not m:
                continue
            qid, cat = m.group(1), int(m.group(2))
            q_match = re.search(
                r"\*\*Question:\*\*\s*(.+?)(?=\n\n|\n\*\*)",
                block, re.DOTALL,
            )
            if q_match:
                questions.append({
                    "qid": qid,
                    "category": cat,
                    "question": q_match.group(1).strip(),
                })

        ground_truth = json.loads(ground_truth_path.read_text())
        return questions, ground_truth

    questions, ground_truth = load_eval_data()

    CATEGORY_NAMES = {
        1: "Single-fact",
        2: "Synthesis",
        3: "Cross-doc",
        4: "Numerical",
        5: "Refusal",
    }

    # --- Question picker ---
    question_options = [
        f"{q['qid']} ({CATEGORY_NAMES[q['category']]}): {q['question'][:90]}..."
        for q in questions
    ]
    selected_idx = st.selectbox(
        "Choose a question:",
        range(len(question_options)),
        format_func=lambda i: question_options[i],
    )
    selected = questions[selected_idx]

    run_button = st.button("Run evaluation")

    # --- The cached eval function ---

    @st.cache_data(show_spinner=False)
    def run_one_eval(qid, question_text, category):
        """Run retrieve + generate + judge for one question. Cached per question.

        Cache key is (qid, question_text, category). Same question → same cache hit,
        no repeated API calls.
        """
        # Make evals/ importable so we can import the judge
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "evals"))
        from faithfulness import judge_faithfulness
        from generate import answer

        answer_text, retrieved = answer(question_text, k=5, return_chunks=True)
        judgment = judge_faithfulness(question_text, retrieved, answer_text)

        return {
            "answer": answer_text,
            "retrieved_ids": retrieved["ids"][0],
            "retrieved_docs": retrieved["documents"][0],
            "retrieved_sources": [m["source"] for m in retrieved["metadatas"][0]],
            "judgment": judgment["judgment"],
            "rationale": judgment["rationale"],
            "unsupported_claims": judgment["unsupported_claims"],
        }

    if run_button:
        with st.spinner("Running evaluation: retrieve → generate → judge..."):
            result = run_one_eval(
                selected["qid"],
                selected["question"],
                selected["category"],
            )

        # --- Display the question ---
        st.markdown("---")
        st.markdown(f"### {selected['qid']} — {CATEGORY_NAMES[selected['category']]}")
        st.markdown(f"**Question:** {selected['question']}")

        # --- Determine ground truth for this question ---
        gt_entry = ground_truth.get(selected["qid"], {})
        gt_anchors = gt_entry.get("anchors", [])

        # Refusal questions (cat 5) have no anchors — handle separately
        is_refusal = selected["category"] == 5

        # --- Ground truth display ---
        st.markdown("### Ground truth chunks")
        if is_refusal:
            st.info(
                "This is a refusal question. No ground truth chunks exist — "
                "the correct behavior is for the model to refuse to answer."
            )
            gt_chunk_ids = set()
        else:
            gt_chunk_ids = set()
            for anchor in gt_anchors:
                gt_chunk_ids.update(anchor.get("ground_truth_chunks", []))
            gt_short_ids = [cid.rsplit(":", 1)[-1].strip() for cid in gt_chunk_ids]
            st.markdown(
                f"This question needs **{len(gt_chunk_ids)} chunk(s)** to be "
                f"fully answerable: `{', '.join(gt_short_ids)}`"
            )

        # --- Retrieved chunks ---
        st.markdown("### Retrieved chunks (top 5)")
        retrieved_id_set = set(result["retrieved_ids"])
        for i, (cid, doc, src) in enumerate(zip(
            result["retrieved_ids"],
            result["retrieved_docs"],
            result["retrieved_sources"],
        ), start=1):
            short_id = cid.rsplit(":", 1)[-1].strip()
            src_name = src.split("/")[-1]
            is_match = cid in gt_chunk_ids

            # Mark matching chunks with a checkmark
            match_marker = "✅" if is_match else ""
            st.markdown(f"**[{i}] {short_id}** _{src_name}_ {match_marker}")
            with st.expander("See chunk text"):
                st.markdown(doc)

        # --- Scorecard ---
        st.markdown("### Scorecard")

        if is_refusal:
            # Only show faithfulness for refusals — custom HTML for font sizing
            col1 = st.columns(1)[0]
            verdict_emoji = {
                "supported": "✓",
                "partially_supported": "◐",
                "not_supported": "✗",
            }.get(result["judgment"], "?")
            verdict_label = result["judgment"].replace("_", " ")
            with col1:
                st.markdown(
                    f"""
                    <div style="padding-top: 0.3rem;">
                        <div style="color: rgba(250, 250, 250, 0.6); font-size: 0.875rem;">Faithfulness</div>
                        <div style="font-size: 1.5rem; font-weight: 400; padding-top: 0.4rem;">{verdict_emoji} {verdict_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
          
        else:
            col1, col2, col3 = st.columns(3)

            # Recall@5 hit — custom HTML to match the smaller font size used in col3
            any_hit = len(retrieved_id_set & gt_chunk_ids) > 0
            recall_label = "Yes" if any_hit else "No"
            with col1:
                st.markdown(
                    f"""
                    <div style="padding-top: 0.3rem;">
                        <div style="color: rgba(250, 250, 250, 0.6); font-size: 0.875rem;">Recall@5 hit</div>
                        <div style="font-size: 1.5rem; font-weight: 400; padding-top: 0.4rem;">{recall_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Coverage — custom HTML for consistent font sizing
            n_hit = len(retrieved_id_set & gt_chunk_ids)
            n_gt = len(gt_chunk_ids)
            with col2:
                st.markdown(
                    f"""
                    <div style="padding-top: 0.3rem;">
                        <div style="color: rgba(250, 250, 250, 0.6); font-size: 0.875rem;">Coverage@5</div>
                        <div style="font-size: 1.5rem; font-weight: 400; padding-top: 0.4rem;">{n_hit}/{n_gt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Faithfulness
            # Faithfulness — custom HTML so we control the font size
            # (st.metric doesn't expose font size, and "partially supported" at
            # st.metric's default ~36pt looks too loud)
            verdict_emoji = {
                "supported": "✓",
                "partially_supported": "◐",
                "not_supported": "✗",
            }.get(result["judgment"], "?")
            verdict_label = result["judgment"].replace("_", " ")
            with col3:
                st.markdown(
                    f"""
                    <div style="padding-top: 0.3rem;">
                        <div style="color: rgba(250, 250, 250, 0.6); font-size: 0.875rem;">Faithfulness</div>
                        <div style="font-size: 1.5rem; font-weight: 400; padding-top: 0.4rem;">{verdict_emoji} {verdict_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --- Generated answer ---
        st.markdown("### Generated answer")
        st.markdown(demote_markdown_headings(result["answer"]))

        # --- Judge rationale ---
        st.markdown("### Judge's reasoning")
        st.markdown(f"_{result['rationale']}_")
        if result["unsupported_claims"]:
            st.markdown("**Specific claims the judge flagged as unsupported:**")
            for claim in result["unsupported_claims"]:
                st.markdown(f"- {claim}")