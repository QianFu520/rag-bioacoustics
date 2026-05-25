
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from anthropic import Anthropic

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
EVAL_SET_PATH = HERE / "eval_set.md"
RESULTS_PATH = HERE / "faithfulness.json"


sys.path.insert(0, str(PROJECT_ROOT))
from generate import answer


JUDGE_CLIENT = Anthropic()
JUDGE_MODEL = "claude-haiku-4-5-20251001"
RAG_K = 5


JUDGE_SYSTEM_PROMPT = """You are evaluating whether an AI assistant's answer is faithful to a set of source passages.

Your task is to judge SUPPORTEDNESS, not correctness. You must base your judgment ONLY on the passages provided. Do NOT use your own knowledge to verify the answer's factual accuracy.

Three possible judgments:

1. "supported" — every claim in the answer is directly supported by the passages. This includes:
   - Substantive answers where each claim can be traced to a passage
   - Faithful refusals: the answer correctly says the passages don't contain the requested information, and the passages indeed don't contain it
   - Partial answers that correctly note what the passages do and don't cover

2. "partially_supported" — some claims in the answer are supported by the passages, but at least one specific claim is NOT supported. The answer contains a mix of grounded and unsupported information.

3. "not_supported" — major claims in the answer cannot be traced to the passages. This includes:
   - Answers that state specific facts (numbers, names, dates) not in the passages
   - Refusals that fabricate details about what the passages "almost said"
   - Answers that contradict the passages

Important edge cases:
- If the answer says "the passages don't mention X" and the passages indeed don't mention X, that is SUPPORTED (a faithful refusal).
- If the answer says "the passages don't mention X" but the passages DO mention X, that is NOT_SUPPORTED.
- General background framing (e.g., "AudioMoth is a recording device") counts as supported if the passages establish it, even if not stated in exactly those words.
- Citations like [1] or [2] in the answer are formatting, not claims — don't judge their correctness, judge the surrounding text.

Output your judgment as a JSON object with exactly these keys:
{
  "judgment": "supported" | "partially_supported" | "not_supported",
  "rationale": "<one or two sentences explaining the judgment>",
  "unsupported_claims": ["<claim 1>", "<claim 2>"]  // empty list if judgment is "supported"
}

Output ONLY the JSON object. No preamble, no markdown fences, no extra text."""


def build_judge_prompt(question, retrieved, answer_text):
    """Format the judge input."""
    chunks = retrieved['documents'][0]
    sources = retrieved['metadatas'][0]

    passages = ""
    for i, (chunk, meta) in enumerate(zip(chunks, sources), start=1):
        passages += f"[{i}] (source: {meta['source']})\n{chunk}\n\n"

    return f"""QUESTION:
{question}

PASSAGES PROVIDED TO THE AI:
{passages}
AI'S ANSWER:
{answer_text}

Judge the answer's faithfulness to the passages. Output JSON only."""


def parse_judge_response(response_text):
    """Extract the JSON judgment from the judge's response.

    Defensive: even with strict prompt, models sometimes add stray characters.
    """
    # Try direct JSON parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: structured "couldn't parse" sentinel
    return {
        "judgment": "parse_error",
        "rationale": f"Failed to parse judge response: {response_text[:200]}",
        "unsupported_claims": [],
    }


def judge_faithfulness(question, retrieved, answer_text):
    """Run the judge on one (question, chunks, answer) triple."""
    judge_prompt = build_judge_prompt(question, retrieved, answer_text)
    response = JUDGE_CLIENT.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return parse_judge_response(response.content[0].text)


def parse_eval_set(path):
    """Pull (qid, category, question_text) for every question in eval_set.md."""
    text = path.read_text(encoding="utf-8")
    questions = []
    blocks = re.split(r"\n(?=## Q\d+ — Category)", text)
    for block in blocks:
        m = re.match(r"## (Q\d+) — Category (\d+)", block)
        if not m:
            continue
        qid, category = m.group(1), int(m.group(2))
        q_match = re.search(r"\*\*Question:\*\*\s*(.+?)(?=\n\n|\n\*\*)", block, re.DOTALL)
        if q_match:
            questions.append({
                "qid": qid,
                "category": category,
                "question": q_match.group(1).strip(),
            })
    return questions


def run():
    print("=" * 75)
    print("FAITHFULNESS EVALUATION")
    print("=" * 75)

    questions = parse_eval_set(EVAL_SET_PATH)
    print(f"\nLoaded {len(questions)} questions from {EVAL_SET_PATH.name}")
    print(f"Pipeline config: k={RAG_K}, generator+judge model={JUDGE_MODEL}")
    print(f"\nRunning {len(questions)} (retrieve → generate → judge) triples...\n")

    all_results = []

    for q in questions:
        qid = q["qid"]
        category = q["category"]
        question = q["question"]

        # production pipeline (retrieve + generate) — what the user would see
        answer_text, retrieved = answer(question, k=RAG_K, return_chunks=True)

        #judge
        judgment = judge_faithfulness(question, retrieved, answer_text)

        # Print one line per question
        verdict_marker = {
            "supported": "✓",
            "partially_supported": "◐",
            "not_supported": "✗",
            "parse_error": "?",
        }.get(judgment["judgment"], "?")

        print(f"  {qid:5} | cat {category} | {verdict_marker} {judgment['judgment']}")

        
        all_results.append({
            "qid": qid,
            "category": category,
            "question": question,
            "retrieved_chunk_ids": retrieved["ids"][0],
            "answer": answer_text,
            "judgment": judgment["judgment"],
            "rationale": judgment["rationale"],
            "unsupported_claims": judgment["unsupported_claims"],
        })

    # Aggregate
    print()
    print("=" * 75)
    print("AGGREGATE — BY CATEGORY")
    print("=" * 75)
    print(f"{'Category':<14} {'n':>3} | {'supported':>10} {'partial':>9} {'unsupported':>12} {'error':>6}")
    print("-" * 70)

    by_category = defaultdict(list)
    for r in all_results:
        by_category[r["category"]].append(r)

    category_names = {
        1: "Single-fact",
        2: "Synthesis",
        3: "Cross-doc",
        4: "Numerical",
        5: "Refusal",
    }

    for cat in sorted(by_category):
        rows = by_category[cat]
        counts = defaultdict(int)
        for r in rows:
            counts[r["judgment"]] += 1
        n = len(rows)
        print(
            f"{category_names.get(cat, str(cat)):<14} {n:>3} | "
            f"{counts['supported']:>10} {counts['partially_supported']:>9} "
            f"{counts['not_supported']:>12} {counts['parse_error']:>6}"
        )

    print("-" * 70)
    counts = defaultdict(int)
    for r in all_results:
        counts[r["judgment"]] += 1
    n = len(all_results)
    print(
        f"{'OVERALL':<14} {n:>3} | "
        f"{counts['supported']:>10} {counts['partially_supported']:>9} "
        f"{counts['not_supported']:>12} {counts['parse_error']:>6}"
    )

    # A faithfulness "score" for the overall pipeline:
    # supported = 1.0, partial = 0.5, unsupported = 0.0
    
    scored = [r for r in all_results if r["judgment"] != "parse_error"]
    if scored:
        score_map = {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0}
        score = mean(score_map[r["judgment"]] for r in scored)
        print(f"\n  Faithfulness score: {score:.2f} (1.0 = all supported, 0.0 = all unsupported)")
        print(f"    Computed over {len(scored)} judged answers; "
              f"{n - len(scored)} parse_error excluded")

    # Save artifact
    artifact = {
        "config": {
            "rag_k": RAG_K,
            "generator_model": JUDGE_MODEL,
            "judge_model": JUDGE_MODEL,
        },
        "per_question": all_results,
    }
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Full audit trail saved to: {RESULTS_PATH.name}")
    print()


if __name__ == "__main__":
    run()