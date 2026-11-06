import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data import Pipeline, PipelineConfig  # noqa: E402
from aio.data.quality import QualityScorer, SparseRows, evaluate, features, pick_threshold  # noqa: E402

N_BUCKETS = 1 << 14
EDU = ["theorem", "photosynthesis", "equation", "molecule", "history", "explain", "chapter", "experiment"]
SPAM = ["buy", "cheap", "discount", "click", "casino", "login", "offer", "deal"]
FILLER = ["the", "and", "of", "to", "in", "is", "a", "with", "for", "on"]


def _docs(n: int, seed: int = 0):
    """Synthetic pages: grade rises with the share of educational words."""
    rng = np.random.default_rng(seed)
    texts, ys = [], []
    for _ in range(n):
        grade = int(rng.integers(0, 6))
        k = 40
        n_edu = grade * 6
        words = list(rng.choice(EDU, n_edu)) + list(rng.choice(SPAM, 30 - n_edu)) + list(rng.choice(FILLER, k))
        rng.shuffle(words)
        texts.append(" ".join(words))
        ys.append(grade)
    return texts, np.array(ys, dtype=np.float32)


def _trained() -> QualityScorer:
    texts, y = _docs(1500)
    s = QualityScorer(N_BUCKETS, max_words=200)
    s.fit(SparseRows.build(texts, N_BUCKETS, 200), y, epochs=6, lr=0.05, log=lambda r: None)
    return s


def test_features_stable_and_include_pairs():
    a = features("The cat sat", N_BUCKETS, 100)
    assert np.array_equal(a, features("the  CAT, sat!", N_BUCKETS, 100))  # case and punctuation ignored
    assert len(a) == 5  # 3 words + 2 word pairs (barring a hash collision)
    assert not np.array_equal(features("cat sat", N_BUCKETS, 100), features("sat cat", N_BUCKETS, 100))
    assert len(features("", N_BUCKETS, 100)) == 0
    assert len(features("one two three four", N_BUCKETS, 2)) == 3  # only the first max_words


def test_scorer_learns_and_ranks():
    s = _trained()
    texts, y = _docs(300, seed=1)
    pred = s.score_many(texts)
    assert evaluate(pred, y)["pearson"] > 0.9
    assert s.score(" ".join(EDU * 5)) > 4 > 1 > s.score(" ".join(SPAM * 5))
    assert 0.0 <= s.score("") <= 5.0
    # score() and the batched path agree
    assert abs(s.score(texts[0]) - float(pred[0])) < 1e-4


def test_save_load_roundtrip(tmp_path):
    s = _trained()
    s.keep_threshold = 2.35
    s.save(tmp_path / "m.npz")
    t = QualityScorer.load(tmp_path / "m.npz")
    assert t.keep_threshold == 2.35 and t.max_words == 200 and t.n_buckets == N_BUCKETS
    doc = " ".join(EDU[:4] + SPAM[:4])
    assert abs(s.score(doc) - t.score(doc)) < 1e-6


def test_pick_threshold():
    y = np.array([0, 1, 2, 3, 4, 3, 1, 4], dtype=np.float32)
    pred = np.array([0.5, 1.0, 1.4, 2.2, 2.6, 2.0, 1.2, 2.9])  # good pages score >= 2.0
    th = pick_threshold(pred, y)
    assert 1.4 < th <= 2.0
    assert evaluate(pred, y, th)["f1"] == 1.0


def test_pipeline_scores_and_filters(tmp_path):
    s = _trained()
    model = tmp_path / "q.npz"
    s.save(model)
    rng = np.random.default_rng(5)
    edu = " ".join(rng.choice(EDU + FILLER, 120))
    spam = " ".join(rng.choice(SPAM + FILLER, 120))

    def cfg(**kw):
        # Synthetic pages would trip the repetition heuristics, so switch those off here.
        c = PipelineConfig(near_dedup=False, quality_model=str(model), **kw)
        q = c.quality
        q.min_words, q.require_stop_words = 1, False
        q.max_dup_line_frac = q.max_dup_para_frac = q.max_dup_line_char_frac = 1.0
        q.top_ngram_char_frac = {n: 1.0 for n in q.top_ngram_char_frac}
        q.dup_ngram_char_frac = {n: 1.0 for n in q.dup_ngram_char_frac}
        return c

    kept = list(Pipeline(cfg()).process([{"text": edu}, {"text": spam}]))
    assert len(kept) == 2
    assert kept[0]["quality"] > 3 > kept[1]["quality"]

    p = Pipeline(cfg(min_quality=2.5))
    kept = list(p.process([{"text": edu}, {"text": spam}]))
    assert [d["text"] for d in kept] == [edu]
    assert p.stats["removed:low_quality_score"] == 1
