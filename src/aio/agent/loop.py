"""Let a chat model use tools: generate, run any tool call, feed the result back, repeat."""

from __future__ import annotations

import json
from typing import Callable

import jax

from ..model.generate import generate
from ..model.transformer import ModelConfig
from ..post.chat import END, TOOL_CALL, render
from ..tokenizer import BPETokenizer


def run(params, cfg: ModelConfig, tok: BPETokenizer, messages: list[dict], tools: dict[str, Callable[..., str]],
        key: jax.Array, max_new: int = 100, max_tool_calls: int = 4, temperature: float = 0.0) -> list[dict]:
    """Continue a conversation until the model answers in text.

    Returns the new messages: any tool calls with their results, then the final answer.
    Bad calls (invalid JSON, unknown tool, wrong arguments) are answered with an error
    message as the tool result, so the model can recover.
    """
    end_id, call_id = tok.special_tokens[END], tok.special_tokens[TOOL_CALL]
    history = list(messages)
    new: list[dict] = []
    for _ in range(max_tool_calls + 1):
        ids, _ = render(tok, history, add_generation_prompt=True)
        ids = ids[-(cfg.max_seq_len - max_new):]
        key, sub = jax.random.split(key)
        out = generate(params, cfg, ids, max_new, sub, temperature=temperature, stop_id=end_id)
        out = [t for t in out if t != end_id]
        if out and out[0] == call_id:
            text = tok.decode(out[1:])
            name, args = "?", {}
            try:
                call = json.loads(text)
                name, args = str(call["name"]), call.get("arguments", {})
                if name not in tools:
                    result = f"error: unknown tool {name!r}"
                elif not isinstance(args, dict):
                    result = "error: arguments must be an object"
                else:
                    result = tools[name](**args)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                result = f"error: bad tool call ({e})"
            msgs = [{"role": "assistant", "tool_call": {"name": name, "arguments": args if isinstance(args, dict) else {}}},
                    {"role": "tool", "content": str(result)}]
            history += msgs
            new += msgs
            continue
        new.append({"role": "assistant", "content": tok.decode(out).strip()})
        return new
    new.append({"role": "assistant", "content": "(stopped: too many tool calls)"})
    return new
