#!/usr/bin/env python3
"""
Clone an official GitHub repository and analyze its structure for code reproduction.

Clones with --depth=1 (shallow). Identifies key ML source files by name.
If already cloned, runs git pull to update.

Usage:
    python3 scripts/fetch_repo.py https://github.com/author/repo analyses/my-paper/

Clones to: analyses/my-paper/_official_repo/
Output: JSON printed to stdout.

JSON schema:
    success       bool
    github_url    str
    repo_dir      str
    action        "cloned" | "updated"
    py_files      int   total .py files found
    requirements  list  contents of requirements.txt (up to 20 lines)
    key_files     dict  category → [path, ...]  (model/loss/train/dataset/config)
    file_summaries dict  category → [{path, lines, preview}, ...]
    error         str   only present on failure
"""
import sys
import os
import json
import subprocess
from pathlib import Path

KEY_FILE_PATTERNS: dict[str, list[str]] = {
    "model":   ["model.py", "models.py", "network.py", "net.py", "arch.py", "architecture.py"],
    "loss":    ["loss.py", "losses.py", "criterion.py", "objectives.py"],
    "train":   ["train.py", "trainer.py", "main.py", "run.py"],
    "dataset": ["dataset.py", "datasets.py", "data.py", "dataloader.py", "data_utils.py"],
    "config":  ["config.py", "configs.py", "args.py", "options.py", "params.py", "hparams.py"],
}

# Directories to skip during traversal (speed + relevance)
SKIP_DIRS = {
    ".git", "__pycache__", ".github", "docs", "doc",
    "tests", "test", "demo", "examples", "notebooks", ".eggs",
}

PREVIEW_LINES = 80


def find_key_files(repo_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {k: [] for k in KEY_FILE_PATTERNS}

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            for category, patterns in KEY_FILE_PATTERNS.items():
                if fname in patterns:
                    result[category].append(os.path.join(root, fname))
                    break

    # Sort each category by path depth (shallow = more likely to be the main file)
    for cat in result:
        result[cat].sort(key=lambda p: p.count(os.sep))

    return result


def read_file_preview(fpath: str) -> dict:
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {
            "path":    fpath,
            "lines":   len(lines),
            "preview": "".join(lines[:PREVIEW_LINES]),
        }
    except Exception as e:
        return {"path": fpath, "lines": 0, "preview": "", "error": str(e)}


def get_requirements(repo_dir: str) -> list[str]:
    for fname in ("requirements.txt", "requirements-dev.txt", "setup.py", "pyproject.toml"):
        fpath = os.path.join(repo_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    return [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ][:20]
            except Exception:
                pass
    return []


def clone_or_update(github_url: str, dest: str) -> dict:
    if os.path.exists(os.path.join(dest, ".git")):
        r = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=dest, capture_output=True, text=True,
        )
        action = "updated"
    else:
        os.makedirs(dest, exist_ok=True)
        r = subprocess.run(
            ["git", "clone", "--depth=1", github_url, dest],
            capture_output=True, text=True,
        )
        action = "cloned"

    if r.returncode != 0:
        return {"success": False, "action": action, "error": r.stderr.strip()}
    return {"success": True, "action": action}


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: fetch_repo.py <github_url> <analysis_dir>", file=sys.stderr)
        sys.exit(1)

    github_url   = sys.argv[1].rstrip("/")
    analysis_dir = sys.argv[2]
    repo_dir     = os.path.join(analysis_dir, "_official_repo")

    print(f"→ {github_url}", file=sys.stderr)
    print(f"  dest: {repo_dir}", file=sys.stderr)

    clone_res = clone_or_update(github_url, repo_dir)
    if not clone_res["success"]:
        print(json.dumps({
            "success":    False,
            "github_url": github_url,
            "repo_dir":   repo_dir,
            "error":      clone_res["error"],
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    key_files    = find_key_files(repo_dir)
    requirements = get_requirements(repo_dir)
    py_count     = sum(1 for _ in Path(repo_dir).rglob("*.py"))

    file_summaries: dict[str, list[dict]] = {}
    for cat, paths in key_files.items():
        file_summaries[cat] = [read_file_preview(p) for p in paths[:3]]

    print(json.dumps({
        "success":       True,
        "github_url":    github_url,
        "repo_dir":      repo_dir,
        "action":        clone_res["action"],
        "py_files":      py_count,
        "requirements":  requirements,
        "key_files":     {k: v for k, v in key_files.items() if v},
        "file_summaries": {k: v for k, v in file_summaries.items() if v},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
