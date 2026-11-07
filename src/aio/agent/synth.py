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
