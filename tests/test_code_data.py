import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.code import clean_code, code_filter, redact_secrets  # noqa: E402
from aio.data.decontam import Decontaminator  # noqa: E402

GOOD = '''import math


def area(r):
    """Area of a circle with radius r."""
    return math.pi * r * r


if __name__ == "__main__":
    print(area(2.0))
'''


def test_filters():
    assert code_filter(GOOD, "Python", "mit") is None
    assert code_filter(GOOD, "Python", "gpl-3.0") == "license"
    assert code_filter("x = 1\n", "Python", "mit") == "too_short"
    assert code_filter("print 'old python 2'\n" * 10, "Python", "mit") == "python_syntax"
    assert code_filter("print 'fine in another language'\n" * 10, "Ruby", "mit") is None
    assert code_filter("var a=" + "1," * 2000 + "\n" * 3, "JavaScript", "mit") == "long_line"
    assert code_filter("# Auto-generated file. Do not edit.\n" + GOOD, "Python", "mit") == "generated"
    assert code_filter("\n".join("0x%04x, 0x%04x, 1234, 5678" % (i, i) for i in range(50)), "C", "mit") == "data_file"


def test_secrets_and_cleaning():
    code = ('AWS = "AKIAABCDEFGHIJKLMNOP"\napi_key = "sk_live_abcdefghijklmnop123456"\n'
            'tok = "ghp_' + "a" * 36 + '"\nname = "not a secret"\n')
    out, n = redact_secrets(code)
    assert n == 3 and "AKIA" not in out and "<SECRET>" in out and "<GITHUB_TOKEN>" in out
    assert 'name = "not a secret"' in out
    cleaned, stats = clean_code("a = 1   \r\nb = 2\r\n\r\n\r\n")
    assert cleaned == "a = 1\nb = 2\n" and stats == {"secrets": 0, "pii": 0}


def test_decontamination():
    prompt = 'def has_close_elements(numbers, threshold):\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold."""\n'
    d = Decontaminator([prompt])
    copy = "# found on github\n" + prompt.replace("    ", "\t") + "    return False\n"
    assert d.contaminated(copy)                      # reformatted copy is still caught
    assert not d.contaminated(GOOD)
    # Number/letter runs from test data must not flag ordinary code.
    tests = Decontaminator(["assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) == 0"])
    assert not tests.contaminated("table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]")
    assert not tests.contaminated(" ".join("abcdefghijklmnop"))
