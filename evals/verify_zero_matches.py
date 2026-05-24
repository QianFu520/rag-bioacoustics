"""
Verify the hypothesis: all 11 zero-match anchors fail because they span chunk
boundaries (multi-sentence anchors whose sentences live in different chunks).

For each zero-match anchor:
  1. Split it into its constituent sentences
  2. For each sentence, find which chunk(s) contain it
  3. Report the result

Expected pattern (if the hypothesis holds): every sentence finds at least one
chunk, and the chunks for different sentences are adjacent (chunk N and N+1).

Any anchor whose sentence has ZERO matches is a different problem (anchor was
edited, or a character we missed) — that's what we're looking for.
"""
import re
from pathlib import Path

import chromadb

from normalize import normalize
from match_anchors import parse_eval_set, load_chunks_normalized, short_id


HERE = Path(__file__).parent
EVAL_SET_PATH = HERE / "eval_set.md"
CHROMA_PATH = HERE.parent / "chroma_db"
COLLECTION_NAME = "bioacoustics_papers"


def split_into_sentences(text):
    """Same regex as the chunker — split after .!? followed by whitespace.

    Using the same logic as the chunker ensures that if the chunker considered
    these as separate sentences during chunking, we split them the same way now.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def find_matching_chunks(normalized_text, normalized_chunks):
    return [cid for cid, _orig, norm in normalized_chunks if normalized_text in norm]


def run():
    print("=" * 75)
    print("ZERO-MATCH ANCHOR VERIFICATION")
    print("=" * 75)

    questions = parse_eval_set(EVAL_SET_PATH)
    normalized_chunks = load_chunks_normalized(CHROMA_PATH, COLLECTION_NAME)
    print(f"\nLoaded {len(normalized_chunks)} chunks\n")

    # The 11 zero-match anchors from the previous run, as (qid, anchor_index)
    # We re-derive them rather than hardcoding, so this script stays accurate
    # if the eval set changes.
    zero_match = []
    for q in questions:
        for i, anchor in enumerate(q["anchors"], start=1):
            norm_anchor = normalize(anchor)
            if not find_matching_chunks(norm_anchor, normalized_chunks):
                zero_match.append((q["id"], i, anchor))

    print(f"Found {len(zero_match)} zero-match anchors to investigate.\n")

    boundary_cases = 0       # anchor splits cleanly, every sentence finds a chunk
    truly_missing = []       # at least one sentence finds NO chunk — needs investigation

    for qid, anchor_idx, anchor_text in zero_match:
        print("=" * 75)
        print(f"{qid} anchor {anchor_idx}")
        print("=" * 75)

        sentences = split_into_sentences(anchor_text)
        print(f"Anchor splits into {len(sentences)} sentence(s):\n")

        all_sentences_found = True
        sentence_chunks = []  # list of (sentence_index, [chunk_ids]) for the summary

        for s_idx, sentence in enumerate(sentences, start=1):
            preview = sentence[:90].replace("\n", " ")
            print(f"  Sentence {s_idx}: \"{preview}{'...' if len(sentence) > 90 else ''}\"")
            matches = find_matching_chunks(normalize(sentence), normalized_chunks)
            if matches:
                short_matches = ", ".join(short_id(m) for m in matches)
                print(f"    → matches: {short_matches}")
                sentence_chunks.append((s_idx, [short_id(m) for m in matches]))
            else:
                print(f"    → ⚠ NO MATCHES — this sentence is not in any chunk")
                all_sentences_found = False
                sentence_chunks.append((s_idx, []))
            print()

        # Verdict for this anchor
        if all_sentences_found:
            # Check if the chunks are adjacent (boundary case) or scattered
            chunk_nums = []
            for _, chunks in sentence_chunks:
                for c in chunks:
                    # Extract numeric part from "chunk_42"
                    m = re.match(r"chunk_(\d+)", c)
                    if m:
                        chunk_nums.append(int(m.group(1)))
            if chunk_nums:
                span = max(chunk_nums) - min(chunk_nums)
                if span <= 2:
                    print(f"  ✓ Boundary case — sentences span chunks {min(chunk_nums)}–{max(chunk_nums)} (span={span})")
                else:
                    print(f"  ⚠ Sentences scattered across chunks {min(chunk_nums)}–{max(chunk_nums)} (span={span}) — wider than expected")
            boundary_cases += 1
        else:
            truly_missing.append((qid, anchor_idx))
            print(f"  ⚠ TRULY MISSING — at least one sentence is not in any chunk. Needs investigation.")
        print()

    # Summary
    print("=" * 75)
    print("VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"  Total zero-match anchors: {len(zero_match)}")
    print(f"  Boundary cases (every sentence found, just split across chunks): {boundary_cases}")
    print(f"  Truly missing (at least one sentence not in any chunk): {len(truly_missing)}")

    if truly_missing:
        print(f"\n  ⚠ These need investigation beyond Option A:")
        for qid, idx in truly_missing:
            print(f"    {qid} anchor {idx}")
    else:
        print(f"\n  ✓ All 11 zero-match anchors are boundary cases.")
        print(f"  ✓ Option A (per-sentence matching) will recover all of them.")
    print()


if __name__ == "__main__":
    run()