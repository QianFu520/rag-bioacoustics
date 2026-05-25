import re
import unicodedata


INVISIBLE_CHARS = [
    "\u2061",  # FUNCTION APPLICATION       — found in audiomoth §3.1 "𝒪(L)"
    "\u2062",  # INVISIBLE TIMES            — found in audiomoth §3.1 equations
    "\u2063",  # INVISIBLE SEPARATOR        — defensive (same Unicode block)
    "\u2064",  # INVISIBLE PLUS             — defensive (same Unicode block)
    "\u200B",  # ZERO WIDTH SPACE           — defensive
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE  — defensive (BOM)
]



SMART_QUOTE_MAP = {
    "\u2018": "'",  
    "\u2019": "'",  
    "\u201C": '"',  
    "\u201D": '"',  
}

def normalize(text: str) -> str:

    text = unicodedata.normalize("NFKC", text)

    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, " ")

    for smart, plain in SMART_QUOTE_MAP.items():
        text = text.replace(smart, plain)

    text = re.sub(r"\s+", " ", text).strip()

    text = text.lower()

    return text


if __name__ == "__main__":
   

    print("=" * 70)
    print("TEST 1: math salad from audiomoth chunk_23")
    print("=" * 70)
    raw = "the desired bandwidth B:\n\n𝐿=4\u2062\n\n𝑓𝑠\n\n𝐵\n\n(1)\n\nA temporary sequence"
    print(f"BEFORE: {repr(raw)}")
    print(f"AFTER:  {repr(normalize(raw))}")
    print()

    print("=" * 70)
    print("TEST 2: smart quotes")
    print("=" * 70)
    raw = "Pook\u2019s Hill Reserve and the \u201Cnoise state\u201D"
    print(f"BEFORE: {repr(raw)}")
    print(f"AFTER:  {repr(normalize(raw))}")
    print()

    print("=" * 70)
    print("TEST 3: messy whitespace (tabs, double spaces, non-breaking space)")
    print("=" * 70)
    raw = "Section\t2.\u00a0Signal  Processing\n\nand Denoising"
    print(f"BEFORE: {repr(raw)}")
    print(f"AFTER:  {repr(normalize(raw))}")
    print()

    print("=" * 70)
    print("TEST 4: substring check after normalization (the real point)")
    print("=" * 70)
    anchor = "the algorithm achieved a true-positive rate of 0.84"
    chunk = "...iteration of the model was tested in a wider array of terrain, measuring the detection accuracy with hills and valleys between the source gunshot and the test devices. For gunshots fired within 500 m of the AudioMoth, the algorithm achieved a true-positive rate of 0.84."
    print(f"naive 'anchor in chunk': {anchor in chunk}")
    print(f"normalized substring check: {normalize(anchor) in normalize(chunk)}")