"""HTML -> clean text.

Keeps the main content, drops boilerplate (scripts, menus, footers, cookie banners),
preserves code blocks exactly, and keeps math as LaTeX where the page provides it.
Uses only the standard library so it runs anywhere.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

SKIP_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header", "aside", "form",
    "iframe", "svg", "button", "select", "template", "canvas", "object",
}
BLOCK_TAGS = {
    "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
    "article", "pre", "blockquote", "table", "ul", "ol", "dd", "dt", "figcaption",
    "main", "hr", "details", "summary",
}
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "wbr"}
# class/id hints that mark boilerplate blocks
BOILERPLATE_HINT = re.compile(
    r"(^|[\s_-])(cookie|consent|banner|advert|ads?|sidebar|menu|breadcrumbs?|share|social|"
    r"newsletter|popup|modal|subscribe|related|promo)([\s_-]|$)",
    re.I,
)
TEX_TYPES = {"math/tex", "math/tex; mode=display"}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_tag: str | None = None
        self.skip_depth = 0
        self.pre_depth = 0
        self.in_title = False
        self.math_depth = 0
        self.math_tex: list[str] = []
        self.math_text: list[str] = []
        self.math_display = False
        self.in_tex_annotation = False
        self.tex_script: str | None = None  # "inline" / "display" while inside a MathJax script

    # -- helpers
    def _newline(self) -> None:
        self.parts.append("\n")

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.skip_tag is not None:
            if tag == self.skip_tag and tag not in VOID_TAGS:
                self.skip_depth += 1
            return
        if tag == "title":
            self.in_title = True
            return
        if tag == "script" and (a.get("type") or "").lower() in TEX_TYPES:
            self.tex_script = "display" if "display" in a["type"] else "inline"
            self.math_tex = []
            return
        hint = " ".join(filter(None, [a.get("class"), a.get("id"), a.get("role")]))
        if tag in SKIP_TAGS or (hint and BOILERPLATE_HINT.search(hint)) or "hidden" in a:
            if tag not in VOID_TAGS:
                self.skip_tag, self.skip_depth = tag, 1
            return
        if tag == "math":
            self.math_depth += 1
            if self.math_depth == 1:
                self.math_tex, self.math_text = [], []
                self.math_display = a.get("display") == "block"
            return
        if self.math_depth and tag == "annotation" and "tex" in (a.get("encoding") or "").lower():
            self.in_tex_annotation = True
            return
        if tag == "pre":
            self.pre_depth += 1
            self._newline()
        elif tag in BLOCK_TAGS:
            self._newline()
        if tag == "li":
            self.parts.append("- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag == "img" and a.get("alt") and self.math_depth == 0:
            # Many sites render formulas as images with LaTeX alt text.
            alt = a["alt"].strip()
            if "\\" in alt or (alt.startswith("$") and alt.endswith("$")):
                self.parts.append(f" {alt} ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_depth -= 1
                if self.skip_depth == 0:
                    self.skip_tag = None
            return
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.tex_script:
            tex = "".join(self.math_tex).strip()
            self.parts.append(f" $${tex}$$ " if self.tex_script == "display" else f" ${tex}$ ")
            self.tex_script = None
        elif tag == "annotation":
            self.in_tex_annotation = False
        elif tag == "math" and self.math_depth:
            self.math_depth -= 1
            if self.math_depth == 0:
                tex = "".join(self.math_tex).strip() or " ".join("".join(self.math_text).split())
                if tex:
                    d = "$$" if self.math_display else "$"
                    self.parts.append(f" {d}{tex}{d} ")
        elif tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
            self._newline()
        elif tag in BLOCK_TAGS:
            self._newline()

    def handle_data(self, data):
        if self.skip_tag is not None:
            return
        if self.in_title:
            self.title_parts.append(data)
        elif self.tex_script:
            self.math_tex.append(data)
        elif self.math_depth:
            (self.math_tex if self.in_tex_annotation else self.math_text).append(data)
        elif self.pre_depth:
            self.parts.append(data.replace("\n", "\x00"))  # protect code newlines
        else:
            self.parts.append(data)


def _clean(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        if "\x00" in line:  # code block: keep exactly
            lines.extend(line.split("\x00"))
            continue
        line = " ".join(line.split())
        if line == "-" or line == "|":
            line = ""
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, main_text) for an HTML page."""
    p = _Extractor()
    try:
        p.feed(html)
        p.close()
    except Exception:  # malformed HTML: keep whatever was parsed
        pass
    title = " ".join("".join(p.title_parts).split())
    return title, _clean("".join(p.parts))
