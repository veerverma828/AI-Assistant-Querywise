import textwrap
from pathlib import Path

from pypdf import PdfReader

CHUNK_SIZE = 1000  # max characters per chunk, heading included
OVERLAP_LINES = 1  # last line of a chunk is repeated at the start of the next


def extract_pages(path: Path) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...] for a PDF or TXT file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [
            {"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)
        ]
    return [{"page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]


def is_heading(line: str) -> bool:
    """ALL-CAPS short line such as 'TECHNICAL PROJECTS' or 'EDUCATION & CERTIFICATIONS'."""
    return line.isupper() and 4 <= len(line) <= 60 and not line.endswith(".")


def split_sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Group lines under the heading that precedes them: [(heading, [lines...]), ...]."""
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if is_heading(line):
            if lines:
                sections.append((heading, lines))
            heading, lines = line, []
        else:
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    return sections


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap_lines: int = OVERLAP_LINES) -> list[str]:
    """
    Pack whole lines into chunks of at most `size` characters.
    Cuts happen only at line ends (or at spaces inside a line longer than the limit),
    and every chunk starts with its section heading so it keeps its context.
    """
    chunks: list[str] = []
    for heading, lines in split_sections(text):
        prefix = f"{heading}\n" if heading else ""
        budget = size - len(prefix)
        current: list[str] = []
        length = 0
        new_lines = 0  # lines added since the last flush (ignores the repeated overlap line)

        for line in lines:
            for piece in textwrap.wrap(line, budget) or [line]:
                if current and new_lines and length + len(piece) + 1 > budget:
                    chunks.append(prefix + "\n".join(current))
                    current = current[-overlap_lines:] if overlap_lines else []
                    length = sum(len(l) + 1 for l in current)
                    new_lines = 0
                current.append(piece)
                length += len(piece) + 1
                new_lines += 1

        if current and new_lines:
            chunks.append(prefix + "\n".join(current))
    return chunks


def build_chunks(path: Path) -> list[dict]:
    """Chunk every page, keep filename + page number as metadata for citations."""
    chunks = []
    for page in extract_pages(path):
        for i, piece in enumerate(chunk_text(page["text"])):
            chunks.append(
                {
                    "source": path.name,
                    "page": page["page"],
                    "chunk_index": i,
                    "text": piece,
                }
            )
    return chunks
