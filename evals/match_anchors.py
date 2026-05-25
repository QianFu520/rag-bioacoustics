"""
Anchor-to-chunk matcher.

For each eval question's anchor, the matcher splits the anchor into sentences
using the same regex the chunker uses, then searches for each sentence
independently in the normalized chunks. The anchor's ground truth chunks are
the UNION of chunks containing any sentence — reflecting the reality that
multi-sentence anchors may span chunk boundaries.

Output:
  - Console diagnostic table (one row per anchor)
  - ground_truth.json: machine-readable artifact for recall@k
"""
import json
import re
from pathlib import Path
from collections import Counter

import chromadb

from normalize import normalize


# --- paths ---
HERE = Path(__file__).parent
EVAL_SET_PATH = HERE / "eval_set.md"
CHROMA_PATH = HERE.parent / "chroma_db"
GROUND_TRUTH_PATH = HERE / "ground_truth.json"
COLLECTION_NAME = "bioacoustics_papers"


def parse_eval_set(path):
    """Parse eval_set.md into a list of question dicts.

    Returns: list of {"id": "Q1", "category": 1, "anchors": [str, ...]}
    Refusal questions (Category 5) have anchors=[].
    """
    text = path.read_text(encoding="utf-8")
    question_blocks = re.split(r"\n(?=## Q\d+ — Category)", text)

    questions = []
    for block in question_blocks:
        id_match = re.match(r"## (Q\d+) — Category (\d+)", block)
        if not id_match:
            continue
        qid = id_match.group(1)
        category = int(id_match.group(2))

        anchor_pattern = re.compile(
            r'\*\*Anchor(?:\s+\d+)?(?:\s*\([^)]*\))?:\*\*\s*"([^"]+)"',
            re.DOTALL,
        )
        anchors = anchor_pattern.findall(block)

        questions.append({
            "id": qid,
            "category": category,
            "anchors": anchors,
        })

    return questions


def split_into_sentences(text):
  
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def load_chunks_normalized(chroma_path, collection_name):
   
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_collection(collection_name)
    data = coll.get()
    return [(cid, normalize(doc)) for cid, doc in zip(data["ids"], data["documents"])]


def find_chunks_containing(normalized_text, normalized_chunks):
    """Return list of chunk IDs whose normalized text contains the input."""
    return [cid for cid, norm in normalized_chunks if normalized_text in norm]


def match_anchor(anchor_text, normalized_chunks):
   
    sentences = split_into_sentences(anchor_text)
    sentence_results = []
    all_chunks = set()
    missing = 0

    for s in sentences:
        matches = find_chunks_containing(normalize(s), normalized_chunks)
        sentence_results.append({"text": s, "chunks": matches})
        if matches:
            all_chunks.update(matches)
        else:
            missing += 1

    return {
        "sentences": sentence_results,
        "ground_truth_chunks": sorted(all_chunks),
        "missing_sentences": missing,
    }


def short_id(chunk_id):
  
    if ":" in chunk_id:
        return chunk_id.rsplit(":", 1)[-1].strip()
    return chunk_id


def run():
    print("=" * 75)
    print("ANCHOR-TO-CHUNK MATCHER (Option A: per-sentence)")
    print("=" * 75)

    questions = parse_eval_set(EVAL_SET_PATH)
    print(f"\nParsed {len(questions)} questions from {EVAL_SET_PATH.name}")

    normalized_chunks = load_chunks_normalized(CHROMA_PATH, COLLECTION_NAME)
    print(f"Loaded and normalized {len(normalized_chunks)} chunks\n")

    # Build the ground truth structure and print diagnostic table
    print("=" * 75)
    print("PER-ANCHOR RESULTS")
    print("=" * 75)

    ground_truth = {}      
    total_anchors = 0
    fully_matched = 0      # anchors where every sentence found a chunk
    partial_matched = 0    # anchors where SOME sentences matched, some didn't
    empty_anchors = 0      # anchors where NO sentences matched (shouldn't happen now)

    for q in questions:
        qid = q["id"]
        cat = q["category"]
        ground_truth[qid] = {"category": cat, "anchors": []}

        if not q["anchors"]:
            print(f"{qid:5} | cat {cat} | (refusal — no anchors)")
            continue

        for anchor_idx, anchor_text in enumerate(q["anchors"], start=1):
            total_anchors += 1
            result = match_anchor(anchor_text, normalized_chunks)

            n_sentences = len(result["sentences"])
            n_missing = result["missing_sentences"]
            chunks = result["ground_truth_chunks"]

            # Classify the anchor
            if n_missing == 0:
                fully_matched += 1
                status_marker = "✓"
            elif chunks:  # some matched, some didn't
                partial_matched += 1
                status_marker = "◐"  # partial
            else:
                empty_anchors += 1
                status_marker = "✗"

            short_chunks = ", ".join(short_id(c) for c in chunks) if chunks else "(none)"

            label = f"anchor {anchor_idx}" if len(q["anchors"]) > 1 else "anchor"
            sentence_summary = (
                f"{n_sentences} sentence{'s' if n_sentences > 1 else ''}"
                + (f", {n_missing} missing" if n_missing else "")
            )
            print(
                f"{qid:5} | cat {cat} | {label:9} | {status_marker} "
                f"{sentence_summary:18} | gt: {short_chunks}"
            )

            # Store in ground truth structure (full chunk IDs, not short)
            ground_truth[qid]["anchors"].append({
                "anchor_index": anchor_idx,
                "n_sentences": n_sentences,
                "missing_sentences": n_missing,
                "ground_truth_chunks": chunks,
            })

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Total fact-based anchors: {total_anchors}")
    print(f"  ✓ Fully matched (every sentence found a chunk): {fully_matched}")
    print(f"  ◐ Partially matched (some sentences missing):   {partial_matched}")
    print(f"  ✗ Empty (no sentence found any chunk):          {empty_anchors}")

    if partial_matched:
        print("\n  Partial matches (documented limitations):")
        for qid, q_data in ground_truth.items():
            for a in q_data["anchors"]:
                if a["missing_sentences"] > 0 and a["ground_truth_chunks"]:
                    print(
                        f"    {qid} anchor {a['anchor_index']}: "
                        f"{a['missing_sentences']} of {a['n_sentences']} "
                        f"sentence(s) missing — see NORMALIZER_NOTES.md"
                    )

    # Coverage stat
    anchors_with_ground_truth = fully_matched + partial_matched
    pct = 100 * anchors_with_ground_truth / total_anchors if total_anchors else 0
    print(f"\n  Coverage: {anchors_with_ground_truth}/{total_anchors} "
          f"anchors have ground truth chunks ({pct:.1f}%)")

    # Save ground truth artifact
    GROUND_TRUTH_PATH.write_text(json.dumps(ground_truth, indent=2))
    print(f"\n  Ground truth written to: {GROUND_TRUTH_PATH.name}")
    print()


if __name__ == "__main__":
    run()