#!/usr/bin/env python3
"""Fail when a relative Markdown link points at a missing local path."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    ".git",
    "archive",
    "__pycache__",
    ".venv",
    "node_modules",
    "lerobot",
    "lerobot_0.3.3_pzy",
    "lerobot_0.3.3_ycy",
    "lerobot_train",
    "lerobot_ycy",
    "XLeRobot",
    "vector-os-nano",
}
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def _target(link: str) -> str | None:
    link = link.strip()
    if not link or link.startswith(("#", "http://", "https://", "mailto:", "tel:")):
        return None
    if link.startswith("<") and link.endswith(">"):
        link = link[1:-1]
    link = link.split("#", 1)[0].split("?", 1)[0]
    return link or None


def main() -> int:
    missing: list[tuple[Path, str]] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw in LINK.findall(text):
            target = _target(raw)
            if target is None or target.startswith("/"):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            if not resolved.exists():
                missing.append((path.relative_to(ROOT), raw.strip()))
    if missing:
        for source, link in missing:
            print(f"{source}: missing link target {link!r}")
        return 1
    print(f"checked {len(_markdown_files())} Markdown files: no broken relative links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
