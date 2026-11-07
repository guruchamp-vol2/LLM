"""Run model-written Python in a separate process, with a time limit.

Used three ways: grading code benchmarks (HumanEval, MBPP), the `python` tool the
model can call to test its own code, and (later) rewards for training on code that
passes tests.

Protection, in layers:
- a fresh process started with `python -I` (ignores PYTHON* variables, user
  site-packages and the current folder), in a new empty temporary folder that is
  deleted afterwards, with a minimal environment (no API keys or tokens);
- a wall-clock timeout, after which the process is killed;
- output capped, so a print loop can't fill memory;
- on Linux, limits on memory, CPU time, file size and number of processes.

This is NOT a full sandbox: the code can still read files the user can read and use
the network. Only run model code on machines where that's acceptable (Kaggle/TPU VMs,
throwaway containers), or behind a real container/VM boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class RunResult:
    ok: bool          # exit code 0 and no timeout
    stdout: str
    stderr: str
    timed_out: bool
    returncode: int | None


def _limits(memory_mb: int, cpu_s: int):
    def apply():  # runs in the child, Linux/macOS only
        import resource

        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        os.setsid()  # own process group, so the whole group can be killed

    return apply


def run_python(code: str, timeout: float = 10.0, stdin: str = "", memory_mb: int = 1024,
               max_output: int = 20_000) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="aio-sandbox-") as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
               "HOME": tmp, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp}
        if os.name == "nt":  # Windows needs these to start Python at all
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits(memory_mb, int(timeout) + 1)
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-I", path], cwd=tmp, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        try:
            out, err = proc.communicate(stdin.encode("utf-8"), timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill(proc)
            out, err = proc.communicate()
            timed_out = True
        dec = lambda b: b[:max_output].decode("utf-8", errors="replace")  # noqa: E731
        return RunResult(ok=proc.returncode == 0 and not timed_out, stdout=dec(out), stderr=dec(err),
                         timed_out=timed_out, returncode=proc.returncode)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            import signal

            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def python_tool(timeout: float = 10.0, max_chars: int = 2000):
    """A tool for the agent loop: python(code) -> what the program printed (or the error)."""

    def python(code: str) -> str:
        if not isinstance(code, str) or not code.strip():
            return "error: no code"
        r = run_python(code, timeout=timeout)
        if r.timed_out:
            return f"error: timed out after {timeout:g}s"
        text = r.stdout
        if not r.ok:
            # Keep the end of the traceback: the actual error is on its last lines.
            text += ("\n" if text else "") + r.stderr[-max_chars:]
        text = text.strip() or "(no output)"
        return text if len(text) <= max_chars else text[: max_chars - 20] + "\n...(truncated)"

    return python
