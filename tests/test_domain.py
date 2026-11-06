import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data import DomainClassifier, Pipeline, PipelineConfig  # noqa: E402
from aio.data.domain import evaluate  # noqa: E402
from aio.data.quality import SparseRows  # noqa: E402

N_BUCKETS = 1 << 12
VOCAB = {
    "science": ["atom", "molecule", "energy", "physics", "cell", "experiment"],
    "sports": ["goal", "match", "team", "league", "coach", "score"],
    "food": ["recipe", "oven", "flour", "sauce", "bake", "dinner"],
}
FILLER = ["the", "and", "of", "to", "in", "is", "a", "with"]
LABELS = list(VOCAB)


def _docs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts, targets = [], []
    for _ in range(n):
        c = int(rng.integers(0, len(LABELS)))
        words = list(rng.choice(VOCAB[LABELS[c]], 20)) + list(rng.choice(FILLER, 20))
        rng.shuffle(words)
        texts.append(" ".join(words))
        p = np.full(len(LABELS), 0.05, dtype=np.float32)
        p[c] = 0.9
        targets.append(p)
    return texts, np.array(targets)


def _trained() -> DomainClassifier:
    texts, P = _docs(600)
    vt, vP = _docs(100, seed=9)
    c = DomainClassifier(LABELS, N_BUCKETS, max_words=100)
    c.fit(SparseRows.build(texts, N_BUCKETS, 100), P, SparseRows.build(vt, N_BUCKETS, 100), vP,
          epochs=4, lr=0.05, log=lambda r: None)
    return c


def test_classifier_learns_topics():
    c = _trained()
    texts, P = _docs(200, seed=1)
    m = evaluate(c.proba_rows(SparseRows.build(texts, N_BUCKETS, 100)), P)
    assert m["accuracy"] > 0.95 and m["macro_f1"] > 0.95
    label, prob = c.predict("we bake the flour in the oven and make a sauce for dinner")
    assert label == "food" and prob > 0.5
    assert abs(c.proba("").sum() - 1.0) < 1e-5  # empty text still gives a valid distribution
    # single and batched paths agree
    assert c.predict_many([texts[0]])[0][0] == c.predict(texts[0])[0]


def test_save_load_roundtrip(tmp_path):
    c = _trained()
    c.save(tmp_path / "d.npz")
    d = DomainClassifier.load(tmp_path / "d.npz")
    assert d.labels == LABELS
    doc = "the team scored a goal in the league match"
    assert d.predict(doc)[0] == c.predict(doc)[0] == "sports"
    assert np.allclose(d.proba(doc), c.proba(doc), atol=1e-2)  # weights are stored as float16


def test_pipeline_tags_domain(tmp_path):
    c = _trained()
    c.save(tmp_path / "d.npz")
    cfg = PipelineConfig(near_dedup=False, domain_model=str(tmp_path / "d.npz"))
    text = (
        "Physics explains how energy moves through every atom and molecule in a cell. In this "
        "experiment we measure the energy of each molecule, and we compare the results with what "
        "physics predicts. The cell is a good place to start, because the atom and the molecule "
        "inside it follow the same rules as everything else in the world around us today."
    )
    p = Pipeline(cfg)
    [out] = list(p.process([{"text": text}]))
    assert out["domain"] == "science" and 0 < out["domain_prob"] <= 1
    assert p.stats["domain:science"] == 1
