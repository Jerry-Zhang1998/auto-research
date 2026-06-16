#!/usr/bin/env python3
"""
Extract structured text from a PDF academic paper.
Usage: python3 scripts/parse_pdf.py <pdf_path> [--include-full-text]
Output: JSON with {title, abstract, sections, figures, tables, references_count}

By default `full_text` is omitted from the JSON: it duplicates the concatenation
of `sections` (~10K redundant tokens for a typical paper) and the skill only needs
`sections` to write raw.md. Pass --include-full-text to restore it if a downstream
consumer needs the unsegmented text.

Requires: pip install pdfplumber
Falls back to basic text extraction if pdfplumber unavailable.
"""
import sys, os, re, json

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None
INCLUDE_FULL_TEXT = "--include-full-text" in sys.argv[1:]

# Section heading patterns for academic papers
SECTION_PATTERNS = [
    r"^(\d+\.?\s+[A-Z][A-Za-z\s\-:]+)$",           # "1. Introduction" or "1 Introduction"
    r"^([A-Z][A-Z\s]+)$",                             # "INTRODUCTION" (all caps)
    r"^(Abstract|Introduction|Related Work|Background|"
    r"Method(?:ology)?|(?:Our\s+)?(?:Proposed\s+)?(?:Approach|Model|Framework|Architecture)|"
    r"Experiment(?:s|al(?:\s+Setup)?)?|Results?|"
    r"Ablation(?:\s+Stud(?:y|ies))?|Discussion|"
    r"Conclusion(?:s)?|Appendix|References?)$",
]

SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)


def extract_with_pdfplumber(pdf_path: str) -> dict:
    import pdfplumber
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages_text.append(text)

    full_text = "\n".join(pages_text)
    return parse_text(full_text)


def extract_fallback(pdf_path: str) -> dict:
    """Fallback: use system pdftotext if available."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(["pdftotext", "-layout", pdf_path, tmp], check=True, capture_output=True)
        with open(tmp) as f:
            full_text = f.read()
        return parse_text(full_text)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "error": "Install pdfplumber (pip install pdfplumber) or pdftotext for PDF extraction",
            "title": "",
            "abstract": "",
            "sections": [],
            "figures": [],
            "tables": [],
            "references_count": 0,
            "full_text": "",
        }
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_text(full_text: str) -> dict:
    lines = full_text.split("\n")

    # Extract title: usually the largest text on page 1 — approximate as first non-empty lines
    title_lines = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped and len(stripped) > 10 and not stripped.startswith("arXiv"):
            title_lines.append(stripped)
            if len(title_lines) >= 2:
                break
    title = " ".join(title_lines)

    # Split into sections
    sections = []
    current_heading = "Preamble"
    current_text = []

    for line in lines:
        stripped = line.strip()
        if SECTION_RE.match(stripped) and len(stripped) < 80:
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_text).strip()
                })
            current_heading = stripped
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections.append({"heading": current_heading, "text": "\n".join(current_text).strip()})

    # Extract abstract
    abstract = ""
    for sec in sections:
        if re.match(r"^Abstract$", sec["heading"], re.IGNORECASE):
            abstract = sec["text"]
            break
    if not abstract:
        # Try to find abstract in preamble
        preamble = next((s["text"] for s in sections if s["heading"] == "Preamble"), "")
        m = re.search(r"(?:Abstract|ABSTRACT)[:\s]+(.*?)(?=\n\n|\Z)", preamble, re.DOTALL | re.IGNORECASE)
        if m:
            abstract = m.group(1).strip()

    # Figure and table captions
    figures = re.findall(r"(?:Figure|Fig\.?)\s+(\d+)[:\.]?\s+(.{10,200})", full_text, re.IGNORECASE)
    tables = re.findall(r"Table\s+(\d+)[:\.]?\s+(.{10,200})", full_text, re.IGNORECASE)

    # Reference count
    ref_section = next((s for s in sections if re.match(r"References?", s["heading"], re.I)), None)
    references_count = len(re.findall(r"^\[\d+\]", ref_section["text"], re.MULTILINE)) if ref_section else 0

    result = {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "figures": [{"number": f[0], "caption": f[1]} for f in figures[:20]],
        "tables": [{"number": t[0], "caption": t[1]} for t in tables[:20]],
        "references_count": references_count,
    }
    if INCLUDE_FULL_TEXT:
        result["full_text"] = full_text
    return result


def main():
    if not PDF_PATH:
        print(json.dumps({"error": "Usage: parse_pdf.py <pdf_path>"}))
        sys.exit(1)

    if not os.path.exists(PDF_PATH):
        print(json.dumps({"error": f"File not found: {PDF_PATH}"}))
        sys.exit(1)

    try:
        result = extract_with_pdfplumber(PDF_PATH)
    except ImportError:
        result = extract_fallback(PDF_PATH)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
