import math
import sys
from pathlib import Path

import jax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent.sandbox import python_tool, run_python  # noqa: E402
from aio.eval.code import CodeProblem, evaluate, extract_code, pass_at_k, truncate  # noqa: E402
from aio.model import ModelConfig, init_params  # noqa: E402
from aio.model.generate import generate, generate_many  # noqa: E402

PROBLEM = CodeProblem("toy/add", 'def add(a, b):\n    """Return a + b."""\n',
                      "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n", "add", "    return a + b\n")


def test_sandbox_runs_and_isolates():
    r = run_python("print(6 * 7)")
    assert r.ok and r.stdout.strip() == "42"
    r = run_python("raise ValueError('boom')")
    assert not r.ok and "ValueError: boom" in r.stderr
    r = run_python("import os; print(sorted(k for k in os.environ if 'TOKEN' in k or 'KEY' in k))")
    assert r.stdout.strip() == "[]"                       # no secrets passed to model code
    r = run_python("import os; print(len(os.listdir('.')))")
    assert r.stdout.strip() == "1"                         # fresh empty folder (just main.py)


def test_sandbox_timeout():
    r = run_python("while True: pass", timeout=1.5)
    assert r.timed_out and not r.ok


def test_python_tool():
    py = python_tool(timeout=5)
    assert py("print(sum(range(10)))") == "45"
    assert "ZeroDivisionError" in py("1/0")
    assert py("x = 1") == "(no output)"
    assert py("") == "error: no code"


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert math.isclose(pass_at_k(10, 3, 1), 0.3)
    assert math.isclose(pass_at_k(5, 2, 2), 1 - math.comb(3, 2) / math.comb(5, 2))
    assert pass_at_k(5, 4, 2) == 1.0                       # can't pick 2 wrong out of 1


def test_truncate_and_extract():
    assert truncate("    return a + b\n\ndef other():\n    pass") == "    return a + b"
    assert truncate("    x = 1\n    # inner comment kept\n    return x\n# top-level") == \
        "    x = 1\n    # inner comment kept\n    return x"
    assert extract_code("Sure:\n```python\ndef f():\n    return 1\n```\nDone") == "def f():\n    return 1\n"
    assert extract_code("no fence") == "no fence"


def test_evaluate_grades_by_running():
    answers = {"good": "    return a + b", "bad": "    return a - b", "hang": "    while True: pass"}
    res = evaluate([PROBLEM], lambda p, n: [answers["good"], answers["bad"], answers["hang"]], n=3, ks=(1, 2),
                   timeout=2)
    assert res["problems"][0]["passed"] == 1
    assert math.isclose(res["pass@1"], 1 / 3) and math.isclose(res["pass@2"], 2 / 3)


@pytest.mark.parametrize("task", ["humaneval", "mbpp"])
def test_reference_solutions_pass(task):
    path = Path("C:/llm-data/evals") / f"{task}.parquet"
    if not path.exists():
        pytest.skip("benchmark file not downloaded")
    from aio.eval.code import check, load

    probs = load(task, str(path))[:15]
    full = task == "mbpp"
    assert all(check(p, p.solution, 10, full_program=full) for p in probs)
    assert not any(check(p, "    pass" if not full else "pass", 10, full_program=full) for p in probs[:5])


def test_generate_many_matches_single_greedy():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=32)
    params = init_params(cfg, jax.random.PRNGKey(1))
    one = generate(params, cfg, [1, 2, 3], 8, jax.random.PRNGKey(0), temperature=0)
    many = generate_many(params, cfg, [1, 2, 3], 3, 8, jax.random.PRNGKey(0), temperature=0)
    assert many == [one] * 3
    stopped = generate_many(params, cfg, [1, 2, 3], 2, 8, jax.random.PRNGKey(0), temperature=0,
                            stop_check=lambda t: len(t) >= 2)
    assert stopped == [one[:2]] * 2
