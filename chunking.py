import re

def split_into_sentences(text):
    
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _tail_sentences_for_overlap(chunk_text, overlap_chars):
    
    if overlap_chars <= 0:
        return []
    sentences = split_into_sentences(chunk_text)
    tail = []
    tail_size = 0
    # Walk from the last sentence backward
    for s in reversed(sentences):
        # +1 for the space between sentences when we rejoin
        added = len(s) + (1 if tail else 0)
        if tail_size + added > overlap_chars:
            break
        tail.insert(0, s)
        tail_size += added
    return tail


def chunk_by_sentences(text, target_size, overlap=0):
   
    sentences = split_into_sentences(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= target_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
                # Seed the next chunk with overlap from the just-finished one
                if overlap > 0:
                    tail = _tail_sentences_for_overlap(current, overlap)
                    current = " ".join(tail)
                else:
                    current = ""
            # Now add the sentence that triggered the boundary
            if len(current) + len(s) <= target_size:
                current = (current + " " + s).strip()
            else:
                # Edge case: the seeded overlap plus s exceeds target_size.
                # Drop the overlap for this transition rather than splitting
                # a sentence. The next chunk loses the overlap but the
                # whole-sentence invariant holds.
                current = s
    if current:
        chunks.append(current)
    return chunks

if __name__ == "__main__":
    sample = (
        "The city council approved the new budget on Tuesday. "
        "The vote was seven to two. The mayor praised the decision. "
        "Critics argued the spending was excessive. "
        "A public hearing is scheduled for next month."
    )

    chunks = chunk_by_sentences(sample, target_size=120)

    for i, c in enumerate(chunks):
        print(f"--- chunk {i} (len {len(c)}) ---")
        print(c)
     
   

