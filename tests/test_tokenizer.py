import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.tokenizer import BPETokenizer  # noqa: E402

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "def add(a, b):\n    return a + b\n" * 10,
    "Numbers like 12345 and 3.14159 appear often. " * 10,
    "Hindi: नमस्ते दुनिया. Emoji: 🚀🔥. Chinese: 你好世界. " * 5,
]

SAMPLES = [
    "",
    "Hello, world!",
    "def add(a, b):\n    return a + b\n",
    "नमस्ते 🚀 你好 — mixed text\t\twith tabs\r\n",
    "   leading and trailing spaces   ",
    "unseen words: xylophone zygote",
]


def train(vocab_size=400):
    return BPETokenizer.train(CORPUS, vocab_size=vocab_size)


def test_roundtrip():
    tok = train()
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress():
    tok = train()
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_vocab_size_exact():
    tok = train(320)
    assert tok.vocab_size == 320


def test_deterministic():
    assert train().merges == train().merges


def test_special_tokens():
    tok = train()
    text = "<|user|>hi<|end|><|assistant|>hello<|end|>"
    ids = tok.encode(text)
    assert tok.special_tokens["<|user|>"] in ids
    assert tok.decode(ids) == text
    # With specials disabled, the string is encoded as plain text instead.
    plain = tok.encode(text, allow_special=False)
    assert tok.special_tokens["<|user|>"] not in plain
    assert tok.decode(plain) == text


def test_save_load(tmp_path):
    tok = train()
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    for s in SAMPLES:
        assert tok.encode(s) == tok2.encode(s)


def test_matches_naive_training():
    """The fast incremental trainer must pick the same merges as a slow, obvious one."""
    import regex as re
    from collections import Counter

    from aio.tokenizer.bpe import SPLIT_PATTERN, _merge, _pairs

    vocab_size, n_special = 350, len(BPETokenizer().special_tokens)
    words = Counter(
        c.encode() for t in CORPUS for c in re.findall(SPLIT_PATTERN, t)
    )
    seqs = {w: list(w) for w in words}
    naive = []
    for _ in range(vocab_size - 256 - n_special):
        counts = Counter()
        for w, ids in seqs.items():
            for p in _pairs(ids):
                counts[p] += words[w]
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        naive.append(best)
        seqs = {w: _merge(ids, best, 255 + len(naive)) for w, ids in seqs.items()}

    assert BPETokenizer.train(CORPUS, vocab_size=vocab_size).merges == naive
