"""Benchmark command-line contracts, exercised through the actual script."""

# Imperative Shell: subprocess and temporary-file integration tests.
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--selections", "nan"],
        ["--accuracy", "0"],
        ["--accuracy", "nan"],
        ["--batch-size", "0"],
        ["--source-gene-copies", "0"],
        ["--repeats", "0"],
        ["--seed", "-1"],
    ],
)
def test_invalid_benchmark_arguments(arguments):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sizes",
            "16",
            "--repeats",
            "1",
            "--solve-max-m",
            "0",
            *arguments,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_benchmark_help_and_json_output():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--device" in result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sizes",
            "16",
            "--selections",
            "0",
            "--repeats",
            "1",
            "--solve-max-m",
            "0",
            "--device",
            "cpu",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    report = json.loads(result.stdout)
    assert report["runs"][0]["M"] == 16
    assert report["config"]["device"] == "cpu"
