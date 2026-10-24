"""Command-line and isolated-worker contracts for the encoding experiment."""

# pattern: Imperative Shell
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_encodings.py"


@pytest.mark.parametrize(
    "arguments", [["--sizes", "1"], ["--warm-repeats", "0"], ["--accuracies", "nan"]]
)
def test_encoding_benchmark_rejects_invalid_settings(arguments):
    """Fail at the CLI boundary before importing numerical dependencies."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert not result.stdout


def test_encoding_benchmark_worker_protocol():
    """Emit first-result timing before one complete diagnostic record."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "filtered",
            "--sizes",
            "16",
            "--accuracies",
            "1e-6",
            "--warm-repeats",
            "1",
            "--operation",
            "advance",
            "--diagnostics",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    first, report = map(json.loads, result.stdout.splitlines())
    assert first["event"] == "first_result_ready"
    assert first["compile_seconds"] > 0
    assert len(report["warm_seconds"]) == 1
    assert report["persistent_compilation_cache"] is False
    assert report["diagnostics"]["row_l1_max"] < 1e-10
    assert report["diagnostics"]["powers"]
