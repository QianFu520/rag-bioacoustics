from docx import Document

def load_docx(path):
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

if __name__ == "__main__":
    text = load_docx("papers/audiomoth_deployment.docx")
    print(f"loaded {len(text)} characters")
    print(f"first 500 chars:\n{text[:500]}")
    print("...")
    print(f"last 500 chars:\n{text[-500:]}")

