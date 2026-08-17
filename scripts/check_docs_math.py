#!/usr/bin/env python3
"""Check that LaTeX in the Markdown docs survives GitHub's renderer.

GitHub applies CommonMark inline processing to math content *before* handing it
to MathJax. Several perfectly valid TeX constructs are silently mangled by that
pass, so the page renders as a MathJax error (typically "Extra close brace or
missing open brace") rather than as an equation. The failure is invisible
locally — most Markdown previewers render the source correctly — so it is worth
a lint.

The five mangling modes, all verified against GitHub's /markdown API:

1. Backslash-escapes. ``\\`` followed by any ASCII punctuation has its backslash
   stripped (CommonMark escape). So ``\\,`` -> ``,``, ``\\;`` -> ``;``,
   ``\\!`` -> ``!``, ``\\{`` -> ``{``, ``\\\\`` -> ``\\``. Only ``\\`` + LETTERS
   and ``\\`` + SPACE survive. Use ``\\ ``, ``\\quad``, ``\\lbrace``,
   ``\\rbrace``, ``\\cr``.
2. Emphasis. A pair of ``*`` (or of ``_`` preceded by punctuation) inside one
   paragraph is consumed as emphasis, destroying both math spans. Use ``\\ast``,
   and prefer ``u_n'`` over ``u'_n`` so the underscore follows a letter
   (intraword ``_`` cannot open emphasis).
3. Table cells. A literal ``|`` inside math splits the cell. Use ``\\vert``.
4. Adjacency. An inline ``$`` opener must be preceded by whitespace or
   start-of-line. ``$a$/$b$`` and ``$a$-$b$`` leave the second span as literal
   text.
5. Line breaks. An inline ``$...$`` span cannot straddle a source newline.

Usage:  python scripts/check_docs_math.py [FILE ...]     (default: all repo .md)
Exit 0 if clean, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SAFE_AFTER_BACKSLASH = re.compile(r"[A-Za-z ]")
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _iter_blocks(text: str):
    """Yield (line_no, line, kind) with kind in {prose, display}; skips code fences."""
    in_fence = in_disp = False
    for n, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped == "$$":
            in_disp = not in_disp
            continue
        yield n, line, "display" if in_disp else "prose"


def _mask_code(line: str) -> str:
    return re.sub(r"`[^`]*`", lambda m: "\x00" * len(m.group()), line)


def _inline_spans(line: str):
    """Yield (start, end) index pairs of inline math content on one line."""
    masked = _mask_code(line)
    for m in re.finditer(r"(?<!\\)\$([^$]+)\$", masked):
        yield m.span(1)


def check_file(path: Path) -> list[str]:
    text = path.read_text()
    problems: list[str] = []

    def report(n: int, msg: str) -> None:
        problems.append(f"{path}:{n}: {msg}")

    for n, line, kind in _iter_blocks(text):
        if kind == "display":
            maths = [(0, len(line))]
            in_table = False
        else:
            maths = list(_inline_spans(line))
            in_table = line.lstrip().startswith("|")
            masked = _mask_code(line)
            # 5. straddling span: an odd number of unescaped $ on a prose line
            if len(re.findall(r"(?<!\\)\$", masked)) % 2:
                report(
                    n,
                    "inline math span is not closed on this line "
                    "(GitHub inline $...$ cannot span a newline)",
                )
            # 4. adjacency: an opening `$` is only recognised after whitespace,
            # start-of-line, or '(' — measured against GitHub's /markdown API.
            # Anything else ('/', '-', '–', ',', a letter, ...) leaves the span
            # as literal text, which is how `$a$/$b$` and `$a$–$b$` break.
            for a, _b in maths:
                if a >= 2 and not (line[a - 2].isspace() or line[a - 2] == "("):
                    report(
                        n,
                        f"inline math opener preceded by {line[a - 2]!r} — GitHub "
                        "only opens math after whitespace, line start or '('; "
                        "add a space or merge the spans",
                    )

        for a, b in maths:
            m = line[a:b]
            for mo in re.finditer(r"\\(.)", m):
                c = mo.group(1)
                if c in PUNCT:
                    report(
                        n,
                        rf"'\{c}' inside math — GitHub strips the backslash; "
                        r"use '\ ' / \quad / \lbrace / \rbrace / \cr",
                    )
            if "*" in m:
                report(n, r"'*' inside math — markdown emphasis eats it; use \ast")
            if in_table and "|" in m:
                report(n, r"'|' inside math in a table cell — splits the cell; use \vert")

    # 2. emphasis pairing: two punctuation-flanked underscores in one paragraph
    para: list[tuple[int, str]] = []
    for n, line, kind in _iter_blocks(text):
        if kind == "display":
            continue
        if not line.strip():
            _check_underscores(para, report)
            para = []
        else:
            para.append((n, line))
    _check_underscores(para, report)
    return problems


def _check_underscores(para, report) -> None:
    hits = []
    for n, line in para:
        for a, b in _inline_spans(line):
            m = line[a:b]
            for mo in re.finditer(r"(?<=[^\w\\])_", m):
                hits.append((n, m[max(0, mo.start() - 6) : mo.start() + 4]))
    if len(hits) >= 2:
        report(
            hits[0][0],
            "two or more underscores preceded by punctuation in one "
            f"paragraph's inline math ({[h[1] for h in hits[:3]]}) — "
            "these pair as markdown emphasis and destroy both spans; "
            "put the subscript before the prime (u_n' not u'_n)",
        )


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    if argv:
        files = [Path(a) for a in argv]
    else:
        files = sorted(
            p
            for p in root.rglob("*.md")
            if not any(part in {".venv", ".pytest_cache", "node_modules"} for part in p.parts)
        )
    problems: list[str] = []
    for f in files:
        problems.extend(check_file(f))
    for p in problems:
        print(p)
    n_files = len(files)
    if problems:
        print(f"\n{len(problems)} problem(s) in {n_files} file(s).")
        return 1
    print(f"docs math OK ({n_files} file(s) checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
