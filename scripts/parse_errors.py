#!/usr/bin/env python3
"""
Parse Python tracebacks from training log files to extract structured error info.

Usage:
    python3 scripts/parse_errors.py logs/my-paper/run_0/train.log
    python3 scripts/parse_errors.py -        # read from stdin

Output: JSON printed to stdout with fields:
    found          bool
    error_type     str   e.g. "RuntimeError"
    error_message  str   full error line
    all_frames     list  all traceback frames
    user_frames    list  frames from reproductions/ or src/ only
    primary_frame  dict  best frame to fix (last user frame)
    raw_traceback  str   the original traceback block
"""
import sys
import re
import json

# Only flag frames from our own code (not site-packages / torch internals)
USER_CODE_DIRS = ["reproductions/", "src/"]


def parse_log(text: str) -> dict:
    """Find the LAST Python traceback in the text and extract structured info."""
    tb_pattern = re.compile(
        r"Traceback \(most recent call last\):\n"
        r"((?:(?:  File [^\n]+\n)|(?:    [^\n]*\n))*)"
        r"(\w[\w.]*(?:Error|Exception|Warning)[^\n]*)",
        re.MULTILINE,
    )
    matches = list(tb_pattern.finditer(text))

    if not matches:
        # Fallback: look for bare error line without traceback
        bare = re.search(r"(\w[\w.]*(?:Error|Exception)): (.+)", text)
        if bare:
            return {
                "found":         True,
                "error_type":    bare.group(1),
                "error_message": bare.group(0),
                "all_frames":    [],
                "user_frames":   [],
                "primary_frame": None,
                "raw_traceback": bare.group(0),
            }
        return {"found": False, "error_message": "No traceback found in log."}

    # Use the last (most recent) traceback
    m = matches[-1]
    tb_body   = m.group(1)
    error_msg = m.group(2).strip()

    # Parse frame lines
    frame_re = re.compile(
        r'  File "([^"]+)", line (\d+), in (\S+)\n(?:    ([^\n]+))?'
    )
    all_frames = []
    for fm in frame_re.finditer(tb_body):
        all_frames.append({
            "file": fm.group(1),
            "line": int(fm.group(2)),
            "func": fm.group(3),
            "code": fm.group(4).strip() if fm.group(4) else "",
        })

    # Filter to user code
    user_frames = [
        f for f in all_frames
        if any(d in f["file"] for d in USER_CODE_DIRS)
    ]

    # Primary frame: last user frame, or last frame overall
    primary = (
        user_frames[-1] if user_frames
        else (all_frames[-1] if all_frames else None)
    )

    err_type_m = re.match(r"(\w+Error|\w+Exception|\w+Warning)", error_msg)

    return {
        "found":         True,
        "error_type":    err_type_m.group(1) if err_type_m else "Error",
        "error_message": error_msg,
        "all_frames":    all_frames,
        "user_frames":   user_frames,
        "primary_frame": primary,
        "raw_traceback": m.group(0),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_errors.py <log_file_or_->", file=sys.stderr)
        sys.exit(1)

    src = sys.argv[1]
    if src == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(src, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except FileNotFoundError:
            print(json.dumps({
                "found": False,
                "error_message": f"File not found: {src}",
            }))
            sys.exit(1)

    result = parse_log(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
