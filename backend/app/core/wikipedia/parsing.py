"""Turning a plain-text extract into sections.

``exsectionformat=wiki`` leaves ``== Heading ==`` markers in the extract, which
is what lets chunking follow real section boundaries and lets citations deep
link to the section an answer came from.
"""

from __future__ import annotations

from .models import WikiSection


def lead_paragraph(extract: str) -> str:
    """First paragraph before any ``== Heading ==``."""
    head = extract.split("\n==", 1)[0].strip()
    for para in head.split("\n"):
        para = para.strip()
        if len(para) > 80:
            return para
    return head[:600]


def parse_sections(extract: str) -> list[WikiSection]:
    """Split a plain-text extract into sections using its ``==`` headings.

    The text before the first heading becomes an implicit "Introduction"
    section, which is where most short factual answers live.
    """
    sections: list[WikiSection] = []
    current = {"title": "Introduction", "level": 1, "ancestors": []}
    current_lines: list[str] = []
    # Tracks the heading hierarchy so each section knows its parent path.
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append(
            WikiSection(
                title=str(current["title"]),
                level=int(current["level"]),
                text=text,
                path=[*current["ancestors"], str(current["title"])],
            )
        )

    for raw_line in extract.splitlines():
        line = raw_line.strip()
        if line.startswith("==") and line.endswith("==") and len(line) > 4:
            flush()
            equals = len(line) - len(line.lstrip("="))
            level = max(equals - 1, 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            current = {
                "title": line.strip("=").strip(),
                "level": level,
                "ancestors": [t for _, t in stack],
            }
            current_lines = []
            stack.append((level, str(current["title"])))
        else:
            current_lines.append(raw_line)

    flush()

    # Drop boilerplate tail sections that carry no answerable content. The check
    # walks the whole heading path, not just the leaf: "Further reading >
    # Articles" is still a bibliography, and left in it wastes retrieval slots
    # on citation lists.
    noise = {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "works cited",
    }
    return [
        s
        for s in sections
        if not any(part.strip().lower() in noise for part in s.path)
    ]
