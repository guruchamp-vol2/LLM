"""Synthetic tool-use conversations for teaching the calculator.

Each conversation: a user asks an arithmetic question in one of several phrasings, the
assistant calls the calculator with the expression, the tool answers, and the assistant
states the result. Mixed with ordinary chat data, this teaches the model *when* to use
the tool as well as how.
"""

from __future__ import annotations

import random

from .tools import calculator

_OPS = [("+", "plus", "add"), ("-", "minus", "subtract"), ("*", "times", "multiply")]

_TEMPLATES = [
    "What is {a} {w} {b}?",
    "Calculate {a} {s} {b}.",
    "What's {a} {s} {b}?",
    "How much is {a} {w} {b}?",
    "Compute {a} {s} {b} for me.",
    "Can you work out {a} {s} {b}?",
]

_ANSWERS = [
    "{a} {s} {b} = {r}.",
    "The answer is {r}.",
    "{a} {w} {b} is {r}.",
]


def calc_conversation(rng: random.Random, max_digits: int = 4) -> list[dict]:
    sym, word, _ = rng.choice(_OPS)
    a = rng.randint(1, 10 ** rng.randint(1, max_digits) - 1)
    b = rng.randint(1, 10 ** rng.randint(1, max_digits) - 1)
    expr = f"{a}{sym}{b}"
    r = calculator(expr)
    q = rng.choice(_TEMPLATES).format(a=a, b=b, s=sym, w=word)
    ans = rng.choice(_ANSWERS).format(a=a, b=b, s=sym, w=word, r=r)
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "tool_call": {"name": "calculator", "arguments": {"expression": expr}}},
        {"role": "tool", "content": r},
        {"role": "assistant", "content": ans},
    ]


def calc_conversations(n: int, seed: int = 0, max_digits: int = 4) -> list[list[dict]]:
    rng = random.Random(seed)
    return [calc_conversation(rng, max_digits) for _ in range(n)]


# ---------------------------------------------------------------------- search
_SEARCH_Q = [
    "What is {t}?",
    "Tell me about {t}.",
    "What's the latest on {t}?",
    "Can you look up {t}?",
    "Find information about {t}.",
]


def _first_sentence(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    end = text.find(". ")
    s = text[: end + 1] if 0 < end < limit else text[:limit]
    return s.strip()


def _short_title(title: str, limit: int = 80) -> str:
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    return title[:limit].rsplit(" ", 1)[0]  # cut at a word boundary


def keywords(title: str, n: int = 5) -> str:
    """The first n meaningful words of a title: a short query that is easy to copy and
    still finds the page."""
    from ..retrieval.index import _STOP, _WORD

    words = [w for w in _WORD.findall(title) if w.lower() not in _STOP]
    return " ".join(words[:n])


def in_split(doc_id: str, split: str, test_frac: float = 0.1) -> bool:
    """Stable train/test split of pages by id, so test questions are about pages the model
    never saw in training (it has to really search and read)."""
    is_test = int(doc_id[:8], 16) / 0xFFFFFFFF < test_frac
    return is_test if split == "test" else not is_test


def search_conversations(index, n: int, seed: int = 0, k: int = 2, width: int = 200,
                         split: str | None = None) -> list[list[dict]]:
    """Questions about pages in the index, answered by calling the search tool.

    The tool result is the real output of searching the index, so the model learns to
    read results as they will look at answer time. The answer is the opening sentence
    of the page that was searched for.
    """
    from ..retrieval.index import search_tool

    rng = random.Random(seed)
    from ..retrieval.collect import _clean_summary

    tool = search_tool(index, k=k, width=width)
    docs = [d for d in index.docs if d.get("title") and len(d["text"]) > 40
            and (split is None or in_split(d["id"], split))]
    out = []
    if not docs:
        return out
    for _ in range(n * 3):  # pages repeat with different phrasings; some tries are skipped
        d = rng.choice(docs)
        title = _short_title(d["title"])
        query = keywords(title)
        if not query:
            continue
        result = tool(query)
        if d["title"][:40] not in result:
            continue  # the page isn't in the top results for its keywords: skip
        answer = _first_sentence(_clean_summary(d["text"]))
        src = d.get("url") or d.get("source", "")
        out.append([
            {"role": "user", "content": rng.choice(_SEARCH_Q).format(t=title)},
            {"role": "assistant", "tool_call": {"name": "search", "arguments": {"query": query}}},
            {"role": "tool", "content": result},
            {"role": "assistant", "content": answer + (f" (Source: {src})" if src else "")},
        ])
        if len(out) >= n:
            break
    return out
