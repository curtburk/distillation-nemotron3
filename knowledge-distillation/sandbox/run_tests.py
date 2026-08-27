#!/usr/bin/env python3
"""
run_tests.py — Executes inside the Docker sandbox container.

Reads a solution file and a test file, executes them, and reports results
as JSON to stdout.

Usage (called by the host executor):
    python run_tests.py /path/to/solution.py /path/to/tests.py

Exit codes:
    0 = all tests passed
    1 = some/all tests failed
    2 = execution error (syntax, import, timeout)
"""

import json
import sys
import traceback
import io
import signal
from contextlib import redirect_stdout, redirect_stderr


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_solution_with_tests(solution_code: str, test_code: str) -> dict:
    """Execute solution code, then run test code against it."""

    result = {
        "passed": False,
        "num_tests": 0,
        "num_passed": 0,
        "num_failed": 0,
        "errors": [],
        "stdout": "",
        "stderr": "",
    }

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set up timeout (30 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    try:
        # Create a shared namespace for solution + tests
        namespace = {"__builtins__": __builtins__}

        # Execute the solution code first
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(solution_code, "<solution>", "exec"), namespace)

        # Now execute the test code in the same namespace
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(test_code, "<tests>", "exec"), namespace)

        # If we get here without exception, all assertions passed
        result["passed"] = True

    except AssertionError as e:
        result["errors"].append({
            "type": "AssertionError",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
    except TimeoutError:
        result["errors"].append({
            "type": "TimeoutError",
            "message": "Execution exceeded 30 second timeout",
        })
    except SyntaxError as e:
        result["errors"].append({
            "type": "SyntaxError",
            "message": str(e),
            "line": e.lineno,
            "offset": e.offset,
        })
    except Exception as e:
        result["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        signal.alarm(0)  # Cancel timeout
        result["stdout"] = stdout_capture.getvalue()[:5000]  # Truncate
        result["stderr"] = stderr_capture.getvalue()[:5000]

    return result


def run_solution_with_io(solution_path: str, io_tests: list) -> dict:
    """
    I/O mode (APPS-style): run the solution as a script per test case,
    feed input on stdin, compare stripped stdout to expected output.
    """
    import subprocess

    result = {
        "passed": False,
        "num_tests": len(io_tests),
        "num_passed": 0,
        "num_failed": 0,
        "errors": [],
        "stdout": "",
        "stderr": "",
    }

    for i, case in enumerate(io_tests):
        try:
            proc = subprocess.run(
                [sys.executable, solution_path],
                input=case["input"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            actual = proc.stdout.strip()
            expected = case["output"].strip()

            # Normalize: compare line-by-line with stripped whitespace
            actual_lines = [l.strip() for l in actual.splitlines()]
            expected_lines = [l.strip() for l in expected.splitlines()]

            if actual_lines == expected_lines:
                result["num_passed"] += 1
            else:
                result["num_failed"] += 1
                if len(result["errors"]) < 3:
                    result["errors"].append({
                        "type": "OutputMismatch",
                        "message": f"case {i}: expected {expected[:200]!r}, got {actual[:200]!r}",
                    })
        except subprocess.TimeoutExpired:
            result["num_failed"] += 1
            result["errors"].append({"type": "TimeoutError", "message": f"case {i} exceeded 15s"})
        except Exception as e:
            result["num_failed"] += 1
            result["errors"].append({"type": type(e).__name__, "message": str(e)})

    result["passed"] = result["num_failed"] == 0 and result["num_passed"] > 0
    return result


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: run_tests.py <solution.py> <tests.py|io_tests.json>"}))
        sys.exit(2)

    solution_path = sys.argv[1]
    test_path = sys.argv[2]

    # I/O mode: test file is a JSON list of {"input", "output"} pairs
    if test_path.endswith(".json"):
        try:
            with open(test_path) as f:
                io_tests = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(json.dumps({"error": f"Bad io_tests file: {e}"}))
            sys.exit(2)
        result = run_solution_with_io(solution_path, io_tests)
        print(json.dumps(result))
        sys.exit(0 if result["passed"] else 1)

    # Assertion mode: test file is Python executed in shared namespace
    try:
        with open(solution_path) as f:
            solution_code = f.read()
        with open(test_path) as f:
            test_code = f.read()
    except FileNotFoundError as e:
        print(json.dumps({"error": f"File not found: {e}"}))
        sys.exit(2)

    result = run_solution_with_tests(solution_code, test_code)
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
