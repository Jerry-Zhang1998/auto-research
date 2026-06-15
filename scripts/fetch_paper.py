#!/usr/bin/env python3
"""
Fetch a paper from arxiv or copy a local PDF into papers/.
Usage: python3 scripts/fetch_paper.py <arxiv_url_or_id_or_pdf_path> [name]
Output: prints JSON with {name, pdf_path, title, authors, year, abstract}
"""
import sys, os, re, json, shutil, urllib.request

PAPERS_DIR = "papers"
os.makedirs(PAPERS_DIR, exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:60]


def parse_arxiv_id(source: str) -> str | None:
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    ]
    for p in patterns:
        m = re.search(p, source)
        if m:
            return m.group(1)
    return None


def fetch_arxiv(arxiv_id: str, name: str | None) -> dict:
    # Fetch metadata via arxiv API
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=15) as r:
            xml = r.read().decode()
    except Exception as e:
        return {"error": f"Failed to fetch arxiv metadata: {e}"}

    title = re.search(r"<title>(?!ArXiv)(.*?)</title>", xml, re.DOTALL)
    title = title.group(1).strip().replace("\n", " ") if title else f"arxiv-{arxiv_id}"

    authors_raw = re.findall(r"<name>(.*?)</name>", xml)
    authors = ", ".join(authors_raw[:5]) + (" et al." if len(authors_raw) > 5 else "")

    year_m = re.search(r"<published>(\d{4})", xml)
    year = year_m.group(1) if year_m else "unknown"

    abstract_m = re.search(r"<summary>(.*?)</summary>", xml, re.DOTALL)
    abstract = abstract_m.group(1).strip().replace("\n", " ") if abstract_m else ""

    resolved_name = name or slugify(title)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = os.path.join(PAPERS_DIR, f"{resolved_name}.pdf")

    if not os.path.exists(pdf_path):
        print(f"Downloading {pdf_url} ...", file=sys.stderr)
        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "auto-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(pdf_path, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as e:
            return {"error": f"Failed to download PDF: {e}"}
    else:
        print(f"PDF already exists: {pdf_path}", file=sys.stderr)

    return {
        "name": resolved_name,
        "pdf_path": pdf_path,
        "title": title,
        "authors": authors,
        "year": year,
        "arxiv": arxiv_id,
        "abstract": abstract,
    }


def fetch_local(pdf_path: str, name: str | None) -> dict:
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    resolved_name = name or slugify(basename)
    dest = os.path.join(PAPERS_DIR, f"{resolved_name}.pdf")

    if os.path.abspath(pdf_path) != os.path.abspath(dest):
        shutil.copy2(pdf_path, dest)

    return {
        "name": resolved_name,
        "pdf_path": dest,
        "title": resolved_name.replace("-", " ").title(),
        "authors": "unknown",
        "year": "unknown",
        "arxiv": "local",
        "abstract": "",
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: fetch_paper.py <url_or_path> [name]"}))
        sys.exit(1)

    source = sys.argv[1].strip()
    name = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() else None

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        result = fetch_arxiv(arxiv_id, name)
    else:
        result = fetch_local(source, name)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
