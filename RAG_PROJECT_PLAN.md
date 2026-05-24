# RAG Project — 8-Day Plan (May 19–27, 2026)

**Goal:** End-to-end RAG over the NYT capstone data, with a real evaluation
framework as the centerpiece, deployed, on public GitHub with a serious README.

**Cadence:** 5–6 focused hours/day. May 25 is a protected rest day.
**Method:** One concept taught → drilled → only then move on. Report back every
evening with where you actually landed (not where the plan says you should be).

**The one rule that makes or breaks this:** if something has to compress,
compress polish — NEVER the eval framework. A modest pipeline with a serious
eval beats a polished pipeline with a weak one, every time, in an interview.

---

## How to use this file

Each evening, fill in the "Actual" line for the day. Be honest — "didn't finish
embeddings, retrieval confused me" is the input I need to re-sequence. A plan
adjusted daily hits May 27; a plan checked only on May 27 does not.

Status key: [ ] not started · [~] in progress · [x] done · [!] blocked

---

## PHASE 1 — CORE PIPELINE (Days 1–3)

By end of Day 3 you have a working RAG system answering questions over real
NYT articles. You're learning each concept as you build, so this is
deliberately not faster.

### Day 1 — Tue May 19 — Chunking
- [~] Environment set up (venv, 4 libs, Python path verified) — DONE
- [~] Concept 1: chunking — naive character splitter built & run — DONE
- [~] Drill: run tiny / huge / medium chunk sizes, articulate the tradeoff
- [~] Concept 2: upgrade to a sentence-aware splitter
- [~] Load the real NYT capstone data; inspect what the documents look like
- [~] Chunk the real data; eyeball chunk quality on actual articles

**Planned deliverable:** clean chunking function that produces sensible chunks
on real NYT text.
**Actual (fill in tonight):** ___all done________________________________________

### Day 2 — Wed May 20 — Embeddings + Vector Store
- [~] Concept 3: embeddings (connect to your BirdNET embedding intuition)
- [~] Embed all chunks with sentence-transformers; inspect vector shapes
- [~] Concept 4: vector store — what ChromaDB is and why
- [~] Load all chunk vectors into ChromaDB; persist to disk
- [~] Sanity check: query the store with a test string, eyeball results

**Planned deliverable:** all NYT chunks embedded and stored in a persistent
ChromaDB collection, returning plausible nearest chunks.
**Actual:** ____________________________________________

### Day 3 — Thu May 21 — Retrieval + Generation (END TO END)
- [~] Concept 5: retrieval — embed the query the SAME way, get top-k
- [~] Set up Anthropic account + `.env` + CLAIM the $5 credit today (14-day
      clock starts now, intentionally — you'll use it within the window)
- [~] Concept 6: generation — stuff retrieved chunks into a grounded prompt
- [~] Wire it all: question → retrieve → generate → cited answer
- [~] Ask 5 real questions over the NYT data, read the answers critically

**Planned deliverable:** working end-to-end RAG. Ask a question, get an answer
grounded in retrieved NYT chunks with sources. THE PIPELINE IS ALIVE.
**Actual:** ____________________________________________

---

## PHASE 2 — EVALUATION FRAMEWORK (Days 4–6) — THE RESUME CENTERPIECE

This is the part that makes the project worth more than a tutorial. When you're
tired around Day 5, this is what you'll be tempted to rush. Don't.

### Day 4 — Fri May 22 — Test Set + Retrieval Metrics
- [~] Concept 7: why eval is hard for RAG (two failure points: retrieval, then
      generation — measure them separately)
- [~] Build a small gold test set: ~15–25 questions over the NYT data, each
      with the known correct source chunk(s)
- [~] Implement recall@k: did the right chunk make it into top-k?
- [~] Run it; get a real recall@k number for your current config

**Planned deliverable:** a test set + a recall@k score for the baseline.
**Actual:** ____________________________________________

### Day 5 — Sat May 23 — Faithfulness + Answer Relevance
- [~] Concept 8: faithfulness — is the answer actually supported by the
      retrieved chunks, or did the model drift / hallucinate?
- [~] Implement a faithfulness check (LLM-as-judge on Haiku to keep cost ~$0)
- [~] Concept 9: answer relevance — does the answer actually address the
      question asked?
- [~] Run the full eval suite; record baseline numbers for all 3 metrics

**Planned deliverable:** full eval harness producing recall@k, faithfulness,
and answer-relevance scores in one run.
**Actual:** ____________________________________________

### Day 6 — Sun May 24 — The Iteration Story
- [~] Change ONE thing (chunk size OR top-k OR overlap) — re-run eval
- [~] Change another — re-run eval
- [~] Build a small comparison table: config → metrics
- [~] Write up the analysis: "config A gave X, B gave Y, here's why" — this
      paragraph IS your interview answer; write it deliberately

**Planned deliverable:** a documented experiment showing you measured,
changed something, measured again, and can explain the result.
**Actual:** ____________________________________________

---

## REST DAY — Mon May 25

No project work. This is load-bearing, not optional. A rested brain on Day 7
produces a better README than an exhausted one on Day 6.5.

---

## PHASE 3 — POLISH + GITHUB (Days 7–8)

Integration friction lives here and always expands. Budgeted, not pessimistic.

### Day 7 — Tue May 26 — Hardening + Deploy
- [ ] Fix anything that breaks when run clean from scratch
- [ ] `requirements.txt` (the reason we used a clean venv — it pays off here)
- [ ] Tune final chunk/retrieval config based on Day 6 eval results
- [ ] Deploy a simple demo (Streamlit / HF Spaces / FastAPI — decide Day 6)
- [ ] Make sure NO api key is anywhere in the code or git history

**Planned deliverable:** project runs clean from a fresh clone; demo is live.
**Actual:** ____________________________________________

### Day 8 — Wed May 27 — README + Portfolio Polish
- [ ] README: problem, architecture diagram, the eval results, the iteration
      story, how to run it. Recruiter-grade, not a stub.
- [ ] GitHub repo public under QianFu520, About description filled in
- [ ] Final read-through as if you're an interviewer seeing it cold
- [ ] Draft the resume bullet for this project (strong verb, real impact,
      no inflation — your usual standard)

**Planned deliverable:** DONE. Public, documented, defensible in a screen.
**Actual:** ____________________________________________

---

## Daily report template (paste this to me each evening)

```
Day __ (date):
- Got through: ...
- Got stuck on: ...
- Did NOT finish: ...
- Energy level today (1-5): ...
- Question for tomorrow: ...
```

I use this to re-sequence the remaining days against reality. Falling behind is
fine and expected — hiding it is the only thing that breaks the May 27 date.

---

## Risk log (the honest failure modes)

1. **Eval gets rushed when tired (Days 5–6).** Mitigation: compress polish,
   never eval. The eval is the differentiator.
2. **Silent drift.** Mitigation: daily report, non-negotiable.
3. **8h-of-focus fantasy.** Plan assumes 5–6 REAL hours. If you only got 4 good
   ones, say so — I'd rather move a task than have you fake the checkmark.
4. **Deploy rabbit-hole on Day 7.** If deploy eats the day, deploy is the
   compressible part — a flawless local repo + great README still interviews
   well. Eval and README are not compressible.
