"""Coding benchmarks graded by running the code: HumanEval and MBPP, scored as pass@k.

pass@k is the chance that at least one of k sampled programs passes all of a problem's
hidden tests. We use the unbiased estimator from the Codex paper (Chen et al. 2021):
sample n >= k programs per problem, count the c that pass, and average
1 - C(n-c, k) / C(n, k) over problems.

| Task      | Dataset                                   | Licence   | Problems |
|-----------|-------------------------------------------|-----------|----------|
| humaneval | openai/openai_humaneval (test)            | MIT       | 164      |
| mbpp      | google-research-datasets/mbpp full (test) | CC-BY-4.0 | 500      |

Prompts are for base (pretrained) models: HumanEval gives the function signature and
docstring, and the model writes the body. MBPP gives the task and one example test in
a docstring (the format of the BigCode evaluation harness), and the model writes the
function. A chat model can be graded the same way by putting the prompt in a user
message and taking the first code block of the answer (`extract_code`).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..agent.sandbox import run_python

URLS = {
    "humaneval": "https://huggingface.co/api/datasets/openai/openai_humaneval/parquet/openai_humaneval/test/0.parquet",
    "mbpp": "https://huggingface.co/api/datasets/google-research-datasets/mbpp/parquet/full/test/0.parquet",
}

# A completion ends where a new top-level block starts (Codex paper stop sequences).
STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ", '\n"""', "\n```"]


@dataclass
class CodeProblem:
    task_id: str
    prompt: str          # what the model continues
    test: str            # code that raises if the program is wrong
    entry_point: str = ""
    solution: str = ""   # reference solution (for checking the grader itself)
    meta: dict = field(default_factory=dict)

    def program(self, completion: str) -> str:
        """The full program that is run: prompt + completion + tests."""
        return self.prompt + completion + "\n\n" + self.test


def load(task: str, path: str) -> list[CodeProblem]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if task == "humaneval":
        return [CodeProblem(r["task_id"], r["prompt"], r["test"] + f"\n\ncheck({r['entry_point']})\n",
                            r["entry_point"], r["canonical_solution"]) for r in rows]
    if task == "mbpp":
        out = []
        for r in rows:
            tests = list(r["test_list"]) if not isinstance(r["test_list"], str) else json.loads(r["test_list"])
            prompt = f'"""\n{r["text"].strip()}\n{tests[0]}\n"""\n'
            setup = (r.get("test_setup_code") or "").strip()
            test = (setup + "\n" if setup else "") + "\n".join(tests) + "\n"
            # MBPP's reference code is a whole program, not a continuation.
            out.append(CodeProblem(f"mbpp/{r['task_id']}", prompt, test, solution=r["code"].replace("\r\n", "\n")))
        return out
    raise ValueError(f"unknown task {task!r}; choose from {sorted(URLS)}")


def truncate(completion: str, stops: list[str] = STOP) -> str:
    cut = len(completion)
    for s in stops:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut].rstrip()


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|$)", re.S)


def extract_code(answer: str) -> str:
    """First fenced code block of a chat answer, or the whole answer if there is none."""
    m = _FENCE.search(answer)
    return m.group(1) if m else answer


def check(problem: CodeProblem, completion: str, timeout: float = 10.0, full_program: bool = False) -> bool:
    """Does the completion pass the problem's tests? With full_program the completion is
    a whole solution (e.g. a chat answer) and the prompt is not prepended."""
    code = (completion + "\n\n" + problem.test) if full_program else problem.program(completion)
    return run_python(code, timeout=timeout).ok


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem with c of n samples correct."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def evaluate(problems: list[CodeProblem], sample: Callable[[CodeProblem, int], list[str]], n: int = 1,
             ks: tuple[int, ...] = (1,), timeout: float = 10.0, workers: int = 4, full_program: bool = False,
             log=None) -> dict:
    """sample(problem, n) returns n completions. Runs the tests in parallel processes.
    Returns pass@k for each k <= n, plus per-problem pass counts."""
    per = []
    with ThreadPoolExecutor(workers) as pool:
        for i, p in enumerate(problems):
            comps = sample(p, n)
            ok = list(pool.map(lambda c: check(p, c, timeout, full_program), comps))
            per.append({"task_id": p.task_id, "passed": sum(ok), "n": len(ok), "first": comps[0] if comps else ""})
            if log and (i + 1) % 10 == 0:
                done = sum(r["passed"] > 0 for r in per)
                log(f"{i + 1}/{len(problems)} problems, {done} solved at least once")
    res = {f"pass@{k}": float(np.mean([pass_at_k(r["n"], r["passed"], k) for r in per])) for k in ks if k <= n}
    res.update({"n_problems": len(per), "samples": n, "problems": per})
    return res
