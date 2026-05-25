"""
Recall@k for the bioacoustics RAG pipeline.

For each fact-based anchor in ground_truth.json:
  1. Get the question text and the ground truth chunk IDs from the eval set
  2. Call retrieve(question, k=10) — the actual production retrieval function
  3. Compute, at k=3, 5, 10:
       - any_hit: at least one ground truth chunk in top-k  (binary 0/1)
       - coverage: fraction of ground truth chunks found in top-k  (0.0–1.0)
  4. Aggregate per category and overall
  5. Save per-anchor results to recall_at_k.json for Day 6 comparison

Refusal questions (Category 5) are skipped — they have no anchors and are
tested separately on Day 5 by the faithfulness check.

"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
GROUND_TRUTH_PATH = HERE / "ground_truth.json"
EVAL_SET_PATH = HERE / "eval_set.md"
RESULTS_PATH = HERE / "recall_at_k.json"


sys.path.insert(0, str(PROJECT_ROOT))
from retrieve_hybrid import retrieve

# k values to measure. We retrieve at the largest k once and score at all
# three from the same result, so this is essentially free.
K_VALUES = [3, 5, 10]
MAX_K = max(K_VALUES)


def load_question_texts(eval_set_path):
  
    text = eval_set_path.read_text(encoding="utf-8")
    questions = {}
    blocks = re.split(r"\n(?=## Q\d+ — Category)", text)
    for block in blocks:
        id_match = re.match(r"## (Q\d+) — Category", block)
        if not id_match:
            continue
        qid = id_match.group(1)
        q_match = re.search(r"\*\*Question:\*\*\s*(.+?)(?=\n\n|\n\*\*)", block, re.DOTALL)
        if q_match:
            questions[qid] = q_match.group(1).strip()
    return questions


def score_anchor(retrieved_chunk_ids, ground_truth_chunk_ids, k):
  
    topk = set(retrieved_chunk_ids[:k])
    gt = set(ground_truth_chunk_ids)
    hits = topk & gt
    return {
        "any_hit": 1 if hits else 0,
        "coverage": len(hits) / len(gt) if gt else 0.0,
        "n_hit": len(hits),
        "n_gt": len(gt),
    }


def run():
    print("=" * 75)
    print("RECALL@K EVALUATION")
    print("=" * 75)

    # Load inputs
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    question_texts = load_question_texts(EVAL_SET_PATH)
    print(f"\nLoaded ground truth for {len(ground_truth)} questions")
    print(f"Loaded question texts for {len(question_texts)} questions")
    print(f"Running retrieval at k={MAX_K}, scoring at k={K_VALUES}")
    print()

    # Per-anchor results
    all_results = []  # list of dicts, one per anchor

    print("=" * 75)
    print("PER-ANCHOR RESULTS")
    print("=" * 75)
    header = f"{'Anchor':14} | {'cat':3} | "
    for k in K_VALUES:
        header += f"{'any@'+str(k):>5} {'cov@'+str(k):>6} | "
    print(header)
    print("-" * len(header))

    for qid, q_data in ground_truth.items():
        category = q_data["category"]
        anchors = q_data["anchors"]

        if not anchors:
            continue  # refusal question

        question_text = question_texts.get(qid)
        if not question_text:
            print(f"⚠ No question text found for {qid} — skipping")
            continue

        # One retrieval call per question — same question text retrieves the
        # same chunks regardless of which anchor we're scoring against.
        results = retrieve(question_text, k=MAX_K)
        retrieved_ids = results["ids"][0]  

        for anchor in anchors:
            anchor_idx = anchor["anchor_index"]
            gt_chunks = anchor["ground_truth_chunks"]

            # Score at each k
            scores_by_k = {}
            for k in K_VALUES:
                scores_by_k[k] = score_anchor(retrieved_ids, gt_chunks, k)

            # Print one row per anchor
            label = f"{qid} a{anchor_idx}" if len(anchors) > 1 else qid
            row = f"{label:14} | {category:3} | "
            for k in K_VALUES:
                s = scores_by_k[k]
                cov_str = f"{s['n_hit']}/{s['n_gt']}"
                row += f"{s['any_hit']:>5} {cov_str:>6} | "
            print(row)

            all_results.append({
                "qid": qid,
                "category": category,
                "anchor_index": anchor_idx,
                "n_ground_truth": len(gt_chunks),
                "ground_truth_chunks": gt_chunks,
                "retrieved_top_k": retrieved_ids,  # full top-10, for Day 6 forensics
                "scores": {str(k): scores_by_k[k] for k in K_VALUES},
            })

    # Aggregate: by category, then overall
    print()
    print("=" * 75)
    print("AGGREGATE — BY CATEGORY")
    print("=" * 75)
    print(f"{'Category':<12} {'n':>4} | "
          + " | ".join(f"{'any@'+str(k):>6} {'cov@'+str(k):>6}" for k in K_VALUES))
    print("-" * 70)

    category_groups = defaultdict(list)
    for r in all_results:
        category_groups[r["category"]].append(r)

    category_names = {
        1: "Single-fact",
        2: "Synthesis",
        3: "Cross-doc",
        4: "Numerical",
    }

    for cat in sorted(category_groups):
        rows = category_groups[cat]
        line = f"{category_names.get(cat, str(cat)):<12} {len(rows):>4} | "
        for k in K_VALUES:
            any_hits = [r["scores"][str(k)]["any_hit"] for r in rows]
            coverages = [r["scores"][str(k)]["coverage"] for r in rows]
            line += f"{mean(any_hits):>6.2f} {mean(coverages):>6.2f} | "
        print(line)

    # Overall
    print("-" * 70)
    line = f"{'OVERALL':<12} {len(all_results):>4} | "
    for k in K_VALUES:
        any_hits = [r["scores"][str(k)]["any_hit"] for r in all_results]
        coverages = [r["scores"][str(k)]["coverage"] for r in all_results]
        line += f"{mean(any_hits):>6.2f} {mean(coverages):>6.2f} | "
    print(line)

    
    artifact = {
        "config": {
            "k_values": K_VALUES,
            "embedding_model": "all-MiniLM-L6-v2",
            "collection": "bioacoustics_papers",
        },
        "per_anchor": all_results,
    }
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2))
    print()
    print(f"  Per-anchor results saved to: {RESULTS_PATH.name}")
    print()


if __name__ == "__main__":
    run()