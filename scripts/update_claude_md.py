"""
scripts/update_claude_md.py
============================
Regenerates the File Map section in CLAUDE.md from the actual src/ directory
contents. Called automatically by the PostToolUse hook in settings.local.json
after every Write or Edit so CLAUDE.md stays current without manual effort.

The script only rewrites the "## File Map" code block — all other sections
(gotchas, patterns, env vars, etc.) are preserved exactly as written.

Safe to run repeatedly; it is idempotent.
"""

import re
import os
from pathlib import Path

# Resolve paths relative to project root (this script lives in scripts/)
ROOT = Path(__file__).parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
SRC_DIR   = ROOT / "src"


def build_file_map() -> str:
    """
    Walk src/ and scripts/ to build a fresh file map block.
    Only lists .py files (ignores __pycache__, .pyc, etc.).
    """
    lines = ["```"]

    # Fixed top-level files
    top_level = [
        ("app.py",               "entry point; auth gate, sidebar UI, graph wiring, streaming"),
        ("scripts/update_claude_md.py", "auto-regenerates CLAUDE.md File Map section"),
    ]
    for fname, desc in top_level:
        if (ROOT / fname.replace("/", os.sep)).exists():
            lines.append(f"{fname:<45} — {desc}")

    # src/ files sorted alphabetically
    lines.append("")
    if SRC_DIR.exists():
        for py_file in sorted(SRC_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            # Try to extract the module docstring's first line as the description
            desc = _extract_description(py_file)
            lines.append(f"src/{py_file.name:<40} — {desc}")

    lines.append("```")
    return "\n".join(lines)


def _extract_description(path: Path) -> str:
    """Read the first non-empty line of a module docstring as a one-liner description."""
    try:
        text = path.read_text(encoding="utf-8")
        # Match triple-quoted docstring at top of file (after optional shebang/encoding)
        m = re.search(r'^\s*"""(.+?)"""', text, re.DOTALL)
        if not m:
            m = re.search(r"^\s*'''(.+?)'''", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip()
                # Skip the filename header line (e.g. "src/models.py")
                if line and not line.startswith("src/") and not line.startswith("==="):
                    return line
    except Exception:
        pass
    return path.stem


def update_file_map_in_claude_md() -> None:
    if not CLAUDE_MD.exists():
        print(f"[update_claude_md] CLAUDE.md not found at {CLAUDE_MD} — skipping")
        return

    content = CLAUDE_MD.read_text(encoding="utf-8")

    # Replace the entire ## File Map section's code block
    # Pattern: ## File Map\n```\n...anything...\n```
    new_block = build_file_map()
    new_content = re.sub(
        r"(## File Map\s*\n)```.*?```",
        lambda m: m.group(1) + new_block,
        content,
        flags=re.DOTALL,
    )

    if new_content != content:
        CLAUDE_MD.write_text(new_content, encoding="utf-8")
        print("[update_claude_md] CLAUDE.md File Map updated")
    else:
        print("[update_claude_md] CLAUDE.md already up to date")


if __name__ == "__main__":
    update_file_map_in_claude_md()
