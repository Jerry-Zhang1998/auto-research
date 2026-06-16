#!/usr/bin/env python3
"""
Extract figures from a PDF paper and identify the architecture diagram.

Strategy:
  1. Extract embedded raster images (xref) from each PDF page.
  2. For pages that have figure captions but only vector graphics, render the
     full page as a PNG (so architecture diagrams drawn as PDF paths are captured).
  3. Match figures to captions using co-occurrence on the same page.
  4. Score figures by caption keywords + figure number to find the architecture one.

Requirements:
    pip install pymupdf

Usage:
    python3 scripts/extract_figures.py papers/my-paper.pdf analyses/my-paper/

Outputs:
    analyses/{name}/figures/fig_001.png  fig_002.png ...
    analyses/{name}/figures/manifest.json
Prints JSON to stdout:
    {total, manifest_path, figures: [...], arch_figure: {..., b64: "..."}}
"""
import sys, os, json, re, base64
from typing import Optional

# ── Tuning constants ──────────────────────────────────────────────────────────
MIN_W, MIN_H       = 150, 120   # ignore tiny images (logos, icons)
RENDER_DPI_FACTOR  = 1.8        # zoom factor when rendering vector pages
ARCH_KEYWORDS      = [
    "architecture", "overview", "framework", "pipeline",
    "model", "structure", "network", "proposed", "method", "system",
]


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_raster_images(doc, page_num: int, figures_dir: str, counter: list) -> list[dict]:
    """Extract embedded raster images from one PDF page."""
    import fitz
    page    = doc[page_num]
    results = []

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            # CMYK → RGB
            if pix.colorspace and pix.colorspace.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < MIN_W or pix.height < MIN_H:
                continue

            idx   = counter[0] + 1
            fname = f"fig_{idx:03d}.png"
            fpath = os.path.join(figures_dir, fname)
            pix.save(fpath)
            counter[0] = idx

            results.append({
                "index":    idx,
                "file":     fpath,
                "page":     page_num + 1,
                "width":    pix.width,
                "height":   pix.height,
                "rendered": False,
                "caption":  "",
            })
        except Exception:
            continue

    return results


def render_page_as_image(doc, page_num: int, figures_dir: str, counter: list) -> Optional[dict]:
    """Render a whole PDF page to PNG (for vector-graphic architecture diagrams)."""
    import fitz
    page = doc[page_num]

    # Only render pages that contain meaningful drawn paths (vector graphics)
    if len(page.get_drawings()) < 8:
        return None

    mat = fitz.Matrix(RENDER_DPI_FACTOR, RENDER_DPI_FACTOR)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    if pix.width < 300 or pix.height < 300:
        return None

    idx   = counter[0] + 1
    fname = f"fig_{idx:03d}_page{page_num+1}.png"
    fpath = os.path.join(figures_dir, fname)
    pix.save(fpath)
    counter[0] = idx

    return {
        "index":    idx,
        "file":     fpath,
        "page":     page_num + 1,
        "width":    pix.width,
        "height":   pix.height,
        "rendered": True,
        "caption":  "",
    }


def match_captions(doc, figures: list) -> None:
    """Assign paper figure captions to extracted images by page co-occurrence."""
    # Collect all captions and their page locations
    cap_re = re.compile(
        r"((?:Figure|Fig\.?)\s*(\d+))[:\.\s]+([^\n]{15,250})",
        re.IGNORECASE,
    )

    # page → list of (fig_number, caption_text)
    page_captions: dict[int, list[tuple[int, str]]] = {}
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        for m in cap_re.finditer(text):
            fig_num  = int(m.group(2))
            cap_text = m.group(3).strip().rstrip(".")
            page_captions.setdefault(page_num + 1, []).append((fig_num, cap_text))

    # First pass: assign by page match
    for fig in figures:
        caps = page_captions.get(fig["page"], [])
        if len(caps) == 1:
            num, txt = caps[0]
            fig["caption"] = f"Figure {num}: {txt}"
            fig["fig_num"] = num
        elif caps:
            # Multiple captions on same page — pick closest by figure number order
            num, txt = caps[0]
            fig["caption"] = f"Figure {num}: {txt}"
            fig["fig_num"] = num

    # Second pass: for uncaptioned figures, search nearby pages ±1
    for fig in figures:
        if fig["caption"]:
            continue
        for delta in (0, -1, 1, -2, 2):
            caps = page_captions.get(fig["page"] + delta, [])
            if caps:
                num, txt = caps[0]
                fig["caption"] = f"Figure {num}: {txt}"
                fig["fig_num"] = num
                break


# ── Architecture figure selection ─────────────────────────────────────────────

def score_figure(fig: dict) -> float:
    cap   = fig.get("caption", "").lower()
    score = 0.0

    # Caption keyword match (highest weight)
    for kw in ARCH_KEYWORDS:
        if kw in cap:
            score += 3.0

    # Earlier figure numbers are more likely to be the architecture diagram
    fig_num = fig.get("fig_num", fig["index"])
    score  += max(0.0, 6.0 - fig_num * 0.8)

    # Wide figures are more likely to be architecture diagrams (landscape)
    ratio = fig["width"] / max(fig["height"], 1)
    if ratio > 1.4:
        score += 1.5
    elif ratio > 1.0:
        score += 0.5

    # Prefer raster-extracted over full-page renders
    if fig.get("rendered"):
        score -= 0.5

    # Reasonably large figures score higher
    area = fig["width"] * fig["height"]
    if area > 300_000:
        score += 1.0
    elif area < 60_000:
        score -= 1.0

    return score


def find_architecture_figure(figures: list) -> Optional[dict]:
    if not figures:
        return None
    scored = sorted(figures, key=score_figure, reverse=True)
    return scored[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_all_figures(pdf_path: str, output_dir: str) -> list[dict]:
    try:
        import fitz
    except ImportError:
        print(
            "PyMuPDF not installed. Run:\n  pip install pymupdf",
            file=sys.stderr,
        )
        return []

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    doc     = fitz.open(pdf_path)
    counter = [0]
    figures: list = []

    # Collect pages that have figure captions (for vector render fallback)
    cap_pages: set[int] = set()
    for pn in range(len(doc)):
        if re.search(r"(Figure|Fig\.?)\s*\d+", doc[pn].get_text(), re.IGNORECASE):
            cap_pages.add(pn)

    # Extract raster images
    raster_pages: set[int] = set()
    for pn in range(len(doc)):
        imgs = extract_raster_images(doc, pn, figures_dir, counter)
        if imgs:
            raster_pages.add(pn)
            figures.extend(imgs)

    # Render vector-only pages
    for pn in sorted(cap_pages - raster_pages):
        rendered = render_page_as_image(doc, pn, figures_dir, counter)
        if rendered:
            figures.append(rendered)

    match_captions(doc, figures)
    return figures


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_figures.py <pdf_path> <analysis_dir>")
        sys.exit(1)

    pdf_path    = sys.argv[1]
    output_dir  = sys.argv[2]

    if not os.path.exists(pdf_path):
        print(json.dumps({"error": f"PDF not found: {pdf_path}"}))
        sys.exit(1)

    print(f"Extracting figures from {pdf_path} ...", file=sys.stderr)
    figures = extract_all_figures(pdf_path, output_dir)

    # Build manifest (without base64 for size)
    manifest = [{k: v for k, v in f.items()} for f in figures]
    manifest_path = os.path.join(output_dir, "figures", "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    # Identify architecture figure. Write its base64 to a SIDECAR FILE — never to
    # stdout. The b64 string is ~150K+ tokens; dumping it to stdout floods the
    # caller's context. /generate-report reads the PNG (or this sidecar) directly
    # at HTML-build time. stdout carries only the path, caption, and a boolean.
    arch_fig = find_architecture_figure(figures)
    arch_out = None
    if arch_fig:
        b64_path = os.path.join(output_dir, "figures", "arch_b64.txt")
        try:
            with open(arch_fig["file"], "rb") as fp:
                b64 = base64.b64encode(fp.read()).decode()
            with open(b64_path, "w") as fp:
                fp.write(b64)
            arch_out = {
                "file":     arch_fig["file"],
                "caption":  arch_fig.get("caption", ""),
                "b64_file": b64_path,
                "has_b64":  True,
            }
        except Exception as e:
            arch_out = {
                "file":    arch_fig["file"],
                "caption": arch_fig.get("caption", ""),
                "has_b64": False,
                "error":   str(e),
            }

    print(json.dumps({
        "total":        len(figures),
        "manifest":     manifest_path,
        "figures":      manifest,
        "arch_figure":  arch_out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
