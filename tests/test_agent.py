import random
import re
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent import DEFAULT_TOOLS, calculator, run  # noqa: E402
from aio.agent.synth import calc_conversation, calc_conversations  # noqa: E402
from aio.model import ModelConfig  # noqa: E402
from aio.post import render, save_conversations  # noqa: E402
from aio.post.chat import tool_call_text  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import OptimConfig, TrainConfig, train  # noqa: E402


def test_calculator_arithmetic():
    assert calculator("347*29") == "10063"
    assert calculator("2^10") == "1024"
    assert calculator("(1+2)*3 - 4/2") == "7"
    assert calculator("10/4") == "2.5"
    assert calculator("sqrt(16) + abs(-2)") == "6"
    assert calculator("1/0") == "error: division by zero"


def test_calculator_refuses_code():
    for bad in ["__import__('os').system('echo hi')", "open('x')", "().__class__", "x + 1",
                "[1, 2]", "lambda: 1", "2 ** 1000000", "a" * 300]:
        assert calculator(bad).startswith("error"), bad


def test_render_tool_messages():
    tok = BPETokenizer.train(["the answer is 42"], vocab_size=280)
    conv = calc_conversation(random.Random(1))
    ids, mask = render(tok, conv)
    sp = tok.special_tokens
    call_at = ids.index(sp["<|tool_call|>"])
    res_at = ids.index(sp["<|tool_result|>"])
    assert mask[call_at] == 1                                # the model learns to call...
    res_end = ids.index(sp["<|end|>"], res_at)
    assert sum(mask[res_at : res_end + 1]) == 0              # ...but not to invent tool results
    assert '"expression"' in tok.decode(ids[call_at + 1 :])


def test_agent_loop_handles_bad_calls():
    tok = BPETokenizer.train(["x"], vocab_size=270)

    class FakeModel:
        """Stands in for generate(): replays scripted outputs."""

    outputs = iter([
        [tok.special_tokens["<|tool_call|>"]] + tok.encode_ordinary('{"name":"nope","arguments":{}}'),
        [tok.special_tokens["<|tool_call|>"]] + tok.encode_ordinary("not json"),
        tok.encode_ordinary("done"),
    ])
    import aio.agent.loop as loop
    real = loop.generate
    loop.generate = lambda *a, **k: next(outputs)
    try:
        cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=256)
        new = run(None, cfg, tok, [{"role": "user", "content": "hi"}], DEFAULT_TOOLS, jax.random.PRNGKey(0))
    finally:
        loop.generate = real
    assert new[1]["content"].startswith("error: unknown tool")
    assert new[3]["content"].startswith("error: bad tool call")
    assert new[-1] == {"role": "assistant", "content": "done"}


def test_search_conversations(tmp_path):
    from aio.agent.synth import in_split, search_conversations
    from aio.retrieval import SearchIndex

    idx = SearchIndex(tmp_path)
    idx.add([{"url": f"https://e.org/{i}", "title": f"Topic number {w}",
              "text": f"Topic {w} is a subject about {w}. More detail follows here."}
             for i, w in enumerate(["apples", "rivers", "comets", "violins", "glaciers", "tigers", "kites", "lasers"])],
            day="2026-11-06")
    convs = search_conversations(idx, 20, seed=0)
    assert convs
    for c in convs:
        q = c[1]["tool_call"]["arguments"]["query"]
        assert c[1]["tool_call"]["name"] == "search" and q.startswith("Topic number")
        assert q in c[2]["content"]                     # the result lists the page searched for
        assert c[3]["content"].endswith(")") and "Source: https://e.org/" in c[3]["content"]
    train_ids = {d["id"] for d in idx.docs if in_split(d["id"], "train")}
    test_ids = {d["id"] for d in idx.docs if in_split(d["id"], "test")}
    assert not train_ids & test_ids and len(train_ids | test_ids) == len(idx)


def test_tiny_model_learns_to_use_calculator(tmp_path):
    """After training on synthetic calculator chats, a tiny model answers unseen sums
    exactly: something it could never do by itself."""
    train_convs = calc_conversations(3000, seed=0, max_digits=3)
    # The tokenizer must have seen the tool-call JSON, or every call is ~50 single-character
    # tokens and the model has too much to get exactly right.
    texts = [m["content"] if "content" in m else tool_call_text(m["tool_call"]) for c in train_convs[:500] for m in c]
    tok = BPETokenizer.train(texts, vocab_size=400)
    save_conversations(tmp_path / "tools.npz", tok, train_convs)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=96)
    tc = TrainConfig(data_dir=str(tmp_path / "tools.npz"), out_dir=str(tmp_path / "run"), seq_len=96, batch_size=16,
                     steps=1000, log_every=100, ckpt_every=1000, model=cfg,
                     optim=OptimConfig(lr=3e-3, warmup_steps=30, total_steps=1000))
    params = train(tc, log=lambda s: None)["state"]["params"]

    rng = random.Random(123)  # different seed: unseen questions
    solved = 0
    for _ in range(10):
        conv = calc_conversation(rng, max_digits=3)
        a, sym, b = re.match(r"(\d+)([+\-*])(\d+)", conv[1]["tool_call"]["arguments"]["expression"]).groups()
        expected = calculator(f"{a}{sym}{b}")
        new = run(params, cfg, tok, conv[:1], DEFAULT_TOOLS, jax.random.PRNGKey(0), max_new=40)
        called = [m for m in new if m.get("tool_call")]
        if called and called[0]["tool_call"]["arguments"].get("expression") == f"{a}{sym}{b}" \
                and expected in new[-1]["content"]:
            solved += 1
    assert solved >= 8, solved
