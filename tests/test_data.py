import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data import (  # noqa: E402
    MinHashDeduper,
    Pipeline,
    PipelineConfig,
    QualityConfig,
    SafetyConfig,
    ShardWriter,
    html_to_text,
    normalize_url,
    quality_filter,
    read_jsonl,
    redact_pii,
    url_filter,
)

GOOD = (
    "The history of the printing press is one of the most important stories in the "
    "development of human knowledge. Before it existed, books had to be copied by hand, "
    "which made them rare and expensive. When Johannes Gutenberg introduced movable type "
    "in the fifteenth century, the cost of producing books fell sharply. Ideas could "
    "travel faster than ever, and literacy began to spread across Europe. Scholars have "
    "argued that the press played a central role in the Reformation, the Scientific "
    "Revolution, and the rise of modern newspapers that shaped public opinion."
)


# ------------------------------------------------------------------ extraction
def test_extract_main_content():
    html = """<html><head><title>My Page</title><style>body{}</style></head><body>
    <nav><a>Home</a><a>About</a></nav>
    <div class="cookie-banner">We use cookies</div>
    <article><h1>Heading</h1><p>First   paragraph.</p><p>Second &amp; last.</p></article>
    <script>var x = 1;</script>
    <footer>Copyright 2024</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "My Page"
    assert "First paragraph." in text and "Second & last." in text
    for junk in ("Home", "cookies", "var x", "Copyright", "body{}"):
        assert junk not in text


def test_extract_keeps_code_and_math():
    html = """<p>Code:</p><pre>def f(x):
    return x  *  2</pre>
    <p>Euler: <math><semantics><mi>e</mi><annotation encoding="application/x-tex">e^{i\\pi}+1=0</annotation></semantics></math></p>
    <script type="math/tex">a^2+b^2=c^2</script>"""
    _, text = html_to_text(html)
    assert "def f(x):\n    return x  *  2" in text
    assert "$e^{i\\pi}+1=0$" in text
    assert "$a^2+b^2=c^2$" in text


def test_extract_nested_skip():
    html = "<div class='sidebar'><div>inner</div><div>more</div></div><p>kept</p>"
    _, text = html_to_text(html)
    assert text == "kept"


# ------------------------------------------------------------------ filters
def test_quality_keeps_good_text():
    assert quality_filter(GOOD, QualityConfig()) is None


def test_quality_rejects_bad_text():
    cfg = QualityConfig()
    assert quality_filter("too short to keep", cfg) == "too_short"
    assert quality_filter("buy now cheap " * 50, cfg) is not None
    assert quality_filter("\n".join(["- item and the of"] * 60), cfg) is not None
    assert quality_filter(GOOD + " lorem ipsum dolor sit amet", cfg) == "lorem_ipsum"


def test_multilingual_mode():
    hindi = "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति बहुत समृद्ध है। " * 3
    hindi += "यहाँ के लोग त्योहार मनाते हैं, इतिहास पढ़ते हैं और विज्ञान में योगदान देते हैं। हर राज्य की अपनी परंपरा, भोजन, संगीत और कला है जो दुनिया भर में प्रसिद्ध है।"
    assert quality_filter(hindi, QualityConfig()) == "no_stop_words"
    # with the English-only check off it passes, unless the repetition rule catches it
    r = quality_filter(hindi, QualityConfig(require_stop_words=False))
    assert r != "no_stop_words"


def test_url_filter():
    cfg = SafetyConfig(blocked_domains={"badsite.com"})
    assert url_filter("https://cdn.badsite.com/page", cfg) == "blocked_domain"
    assert url_filter("https://example.com/xxx-videos", cfg) == "url_banned_word"
    assert url_filter("https://en.wikipedia.org/wiki/Printing", cfg) is None


# ------------------------------------------------------------------ PII
def test_pii():
    text = (
        "Mail me at john.doe@example.com or call +1 415-555-0132. "
        "Server 93.184.216.34, router 192.168.1.1. Card 4111 1111 1111 1111. "
        "The year 2024 and pi 3.14159 stay. Version 1.2.300.4 stays."
    )
    out, n = redact_pii(text)
    assert "[EMAIL]" in out and "john.doe" not in out
    assert "[PHONE]" in out
    assert "[IP]" in out and "192.168.1.1" in out
    assert "[CARD]" in out
    assert "2024" in out and "3.14159" in out
    assert n == 4
    assert redact_pii("Tel: 020 7946 0958")[0] == "Tel: [PHONE]"
    assert redact_pii("Amex 3782 822463 10005, Visa 4111111111111111")[1] == 2


def test_pii_leaves_maths_alone():
    # All of these were redacted by the old rules (found in FineMath pages).
    maths = ["xi 60 61 62 63 64 65 66 67 68 fi 1 2 3 4", "Divisors: 1 5 11 37 55 185 407 2035",
             "Enter A Number : 113 197 199", "Example 1.2.3.4 shows", "Section 2.1.4.3 (Contagion)",
             "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", "2^50 = 1125899906842624", "the optimum is 100.4111111111111111",
             "x = 0.3333333333333333"]
    for t in maths:
        assert redact_pii(t) == (t, 0), t


# ------------------------------------------------------------------ dedup
def test_minhash_near_duplicates():
    d = MinHashDeduper()
    assert not d.is_dup(GOOD)
    assert d.is_dup(GOOD.replace("sharply", "dramatically"))  # one word changed
    assert not d.is_dup("A completely different article about rivers, mountains and the climate of the alpine valleys.")


def test_normalize_url():
    assert normalize_url("https://www.Example.com/a/") == normalize_url("http://example.com/a#top")


# ------------------------------------------------------------------ end to end
def test_pipeline_end_to_end(tmp_path):
    footer = "Subscribe to our weekly newsletter for more stories like this one"
    docs = [
        {"url": "https://a.com/1", "text": GOOD + "\n" + footer},
        {"url": "https://a.com/1/", "text": "different text"},               # dup URL
        {"url": "https://b.com/2", "text": GOOD + "\n" + footer},             # exact dup
        {"url": "https://c.com/3", "text": GOOD.replace("sharply", "a lot") + "\n" + footer},  # near dup
        {"url": "https://porn.com/x", "text": GOOD},                          # unsafe URL
        {"url": "https://d.com/4", "text": "short"},                          # too short
    ]
    # Unique articles that share the same footer line -> footer is boilerplate.
    topics = ["rivers", "volcanoes", "glaciers", "deserts", "forests"]
    for i, t in enumerate(topics):
        body = (
            f"This article explains how {t} form and change over long periods of time. "
            f"Scientists study {t} with satellites, field work and computer models, and "
            f"they have found that {t} respond to climate in ways that are complex. "
            f"Understanding {t} helps us plan for the future, protect wildlife and keep "
            f"communities safe. Contact the author at writer{i}@example.org for details "
            f"about the research and the data that was used to build these models of {t}."
        )
        docs.append({"url": f"https://e.com/{t}", "text": body + "\n" + footer})

    raw = tmp_path / "raw"
    with ShardWriter(raw, "raw", docs_per_shard=4) as w:
        w.write_all(docs)

    stats = Pipeline(PipelineConfig()).run(sorted(raw.glob("*.jsonl.gz")), tmp_path / "out")
    final = [d for p in sorted((tmp_path / "out" / "final").glob("*.jsonl.gz")) for d in read_jsonl(p)]

    assert stats["removed:dup_url"] == 1
    assert stats["removed:dup_exact"] == 1
    assert stats["removed:dup_near"] == 1
    assert stats["removed:url_banned_word"] == 1
    assert stats["removed:too_short"] == 1
    assert len(final) == 6
    assert all(footer not in d["text"] for d in final)
    assert all("@example.org" not in d["text"] for d in final)
    assert not list((tmp_path / "out").rglob("*.tmp"))
