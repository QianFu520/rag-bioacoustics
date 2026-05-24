# Normalizer & anchor-matching notes

## Q11 anchor 2 sentence 2 — Unicode math notation mismatch

### Symptom
After normalization, the sentence "With L samples in a window, the
computational complexity of performing the Goertzel algorithm on that
window is O(L), whereas a DFT on the same sample count is calculated to
be O(L log L)" has no matching chunk in ChromaDB, despite its source
paragraph being clearly present in chunk_24 (audiomoth_deployment.docx,
Section 3.1).

### Diagnosis
Byte-level inspection of chunk_24 revealed the math notation is stored
using Unicode mathematical alphanumeric symbols and Unicode invisible
operators:

  Anchor (ASCII):  O(L)             O(L log L)
  Chunk (Unicode): 𝒪⁢(𝐿)            𝒪⁢(𝐿⁢log⁢𝐿)
                   ^ ^  ^            ^ ^  ^   ^  ^
                   | |  |            | |  |   |  |
              U+1D4AA |  U+1D43F     | |  |   |  |
                      U+2061         | |  |   U+2062
                      FUNC APP       | |  U+2062
                                     | U+1D43F
                                     U+2061

NFKC correctly folds `𝒪` → `O` and `𝐿` → `L`. The remaining problem is
the invisible operators (U+2061 FUNCTION APPLICATION, U+2062 INVISIBLE
TIMES). These characters are zero-width and have no visual representation,
but in well-formed math typography they sit *between* mathematical tokens
as semantic separators.

Two normalization strategies were considered and both produce mismatches:

1. **Strip invisibles entirely** (`""`): fuses adjacent tokens. The chunk
   becomes "o(llogl)" while the anchor remains "o(l log l)". Mismatch.

2. **Replace invisibles with a space** (`" "`): inserts spaces where the
   ASCII anchor has none. The chunk becomes "o (l log l)" while the anchor
   remains "o(l log l)". Mismatch.

No single rule handles both cases (parens-tight vs space-between) because
the invisible operators in the source are doing both jobs simultaneously,
and the ASCII anchor doesn't preserve that distinction.

### Resolution
Accept as a documented limitation. The sentence is excluded from the
recall@k ground truth. The Q11 evaluation question is still tested
through anchor 2 sentence 1 and anchor 1, both of which match cleanly.

### Why not fix it
- A normalizer rule like "strip spaces adjacent to parentheses" would
  solve this case but represents one-off accommodation for a single
  anchor's typography. The cost (a fragile, corpus-specific rule) exceeds
  the benefit (recovering one sentence whose information is partially
  captured by another sentence in the same anchor).
- Rewriting the anchor to avoid math notation is viable but loses the
  specificity of the Big-O claim. Since the prose version of the same
  claim is already in the anchor, this isn't a meaningful loss.

### What this means for the eval framework
- Anchors covered: 29 of 30 (96.7%)
- One Category 2 multi-anchor question (Q11) has slightly reduced
  coverage on its second anchor; first anchor is fully covered.
- No impact on recall@k methodology; the affected sentence is simply
  not part of any anchor's ground truth chunk set.


## LLM-as-judge: same model for generator and judge

### Context
The faithfulness evaluation (Day 5) uses `claude-haiku-4-5-20251001` as both
the generator (called by `generate.py`) and the judge (called by
`evals/faithfulness.py`). The baseline run scored 1.00 across all 24
questions.

### Why this is worth noting
Using the same model as both generator and judge is a documented bias risk
in the LLM-as-judge literature. A judge may be more lenient toward outputs
that match its own model's patterns, distribution, or characteristic
phrasings. The risk is silent: the score looks high not because the
generation is faithful, but because the judge is sympathetic to its own
twin.

### Evidence the bias is not driving the score here
Spot-checking five cases (Q11, Q14, Q15, Q16, Q22) showed:
- Judge rationales reference specific passage content, not generic
  approval ("appropriately notes that the passages do not provide specific
  algorithmic improvements," "passages instead focus on AudioMoth
  specifications")
- Refusals are correctly distinguished from substantive answers
- Refusals were correctly scored as supported (a faithful refusal IS
  supported behavior), not penalized as missing information

The judge is engaging, not defaulting. The score is high because the
generator behaves faithfully under a strict prompt, not because the judge
is rubber-stamping.

### What a production eval would do differently
For a system being deployed (not a portfolio project), the recommended
practice is:
- Use a different model family as judge (e.g., GPT or Gemini judging Haiku)
- Or use a stronger model as judge (e.g., Opus judging Haiku)

This decouples generator bias from judge bias and produces a more defensible
score.

### Resolution for this project
Accepted as a known limitation. Cost of using a second model family (extra
SDK, extra credentials, extra cost) exceeded the benefit for an 8-day
portfolio scope where the rationale spot-check already validated the score.

## Answer relevance: not implemented as a separate metric

### Context
The original Day 5 plan called for three eval metrics: recall@k,
faithfulness, and answer relevance. Recall@k and faithfulness were built.
Answer relevance was not.

### Definition (for the record)
Answer relevance measures whether the generated answer actually addresses
the question asked — independent of whether the content is correct or
supported by the passages. The failure mode it catches: the model writes
about something tangentially related but doesn't engage with what was
asked.

### Reasoning for skipping
- The faithfulness judge's rationales implicitly verify relevance. When the
  judge writes "the answer accurately represents what the passages say
  about HMMs in animal vocalization analysis" or "the AI correctly
  identifies that the passages do not contain information about X," it is
  asserting that the answer engaged with the question.
- Manual spot-check of all 24 generated answers showed every answer
  directly addressed its question. There were no off-topic answers in the
  baseline.
- The generation prompt in `generate.py` ("Answer the question using ONLY
  the passages below") makes off-topic responses very unlikely with a
  capable instruction-following model. Answer relevance would test a
  failure mode that is structurally suppressed by the prompt.
- Building it as a separate eval would add cost (24 more Haiku calls per
  run) and complexity (another JSON artifact, another aggregate row in
  reports) without yielding new information.

### What a production eval would do differently
For a deployed system handling open-ended user queries (not a fixed
24-question eval set), answer relevance is more valuable: real users phrase
questions in many ways, and off-topic drift is a real failure mode. In that
setting, answer relevance is worth its cost.

### Resolution for this project
Skipped. The decision is on record here so it doesn't look like an
oversight. The Day 5 deliverable is complete with recall@k + faithfulness
covering the two failure modes that matter for this corpus (retrieval
finding the wrong stuff; generation drifting from passages).