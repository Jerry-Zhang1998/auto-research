#!/usr/bin/env python3
"""Shared utilities for auto-research scripts."""
import os, re, json
from datetime import date


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:max_len]


def today() -> str:
    return date.today().isoformat()


def ensure_paper_dir(name: str) -> tuple[str, str]:
    """Create analyses/{name}/ and reproductions/{name}/, return paths."""
    analysis_dir = os.path.join("analyses", name)
    repro_dir = os.path.join("reproductions", name)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(repro_dir, exist_ok=True)
    return analysis_dir, repro_dir


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote: {path} ({len(content.splitlines())} lines)")


def list_papers() -> list[str]:
    """List all paper names that have been parsed."""
    if not os.path.exists("analyses"):
        return []
    return [d for d in os.listdir("analyses")
            if os.path.isdir(os.path.join("analyses", d)) and not d.startswith("_")]


def pipeline_status(name: str) -> dict:
    """Return which pipeline stages are complete for a given paper."""
    return {
        "pdf":        os.path.exists(f"papers/{name}.pdf"),
        "raw":        os.path.exists(f"analyses/{name}/raw.md"),
        "innovations": os.path.exists(f"analyses/{name}/innovations.md"),
        "code":       os.path.exists(f"reproductions/{name}/train.py"),
    }


if __name__ == "__main__":
    print("Papers in pipeline:")
    for name in list_papers():
        status = pipeline_status(name)
        stages = [k for k, v in status.items() if v]
        print(f"  {name}: {' → '.join(stages)}")
