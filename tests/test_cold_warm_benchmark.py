"""Independent-process timing and numerical contracts for the startup benchmark."""

# Imperative Shell: subprocess integration tests with optional fastDTWF.
import argparse
import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_startup.py"


def test_measure_process_preserves_records_already_in_pipe(monkeypatch):
    """Retain both timing records when a worker finishes before the first read."""
    measure_process = runpy.run_path(str(SCRIPT))["measure_process"]
    first = {
        "event": "first_result_ready",
        "imports_seconds": 0.01,
        "setup_seconds": 0.02,
        "first_call_seconds": 0.03,
    }
    result = {"warm_seconds": [0.004], "mass": 1.0}
    output = json.dumps(first) + "\n" + json.dumps(result) + "\n"
    real_popen = subprocess.Popen

    def completed_worker(command, **kwargs):
        """Make both subprocess records available before the parent reads."""
        process = real_popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write(sys.argv[1]); "
                "sys.stderr.write('worker diagnostic\\n')",
                output,
            ],
            **kwargs,
        )
        # Both small records fit in the pipe; force them to be available before
        # the parent reads, exposing text-buffer read-ahead deterministically.
        process.wait(timeout=10)
        return process

    monkeypatch.setattr(subprocess, "Popen", completed_worker)
    args = argparse.Namespace(
        gene_copies=16,
        row_eps=1e-8,
        tail_tol=1e-12,
        warm_repeats=1,
        threads=None,
    )
    measured = measure_process(args, "nufft", 0.1, 0.1)
    assert measured["first_call_seconds"] == first["first_call_seconds"]
    assert measured["warm_seconds"] == result["warm_seconds"]
    assert measured["mass"] == result["mass"]
    assert measured["stderr"] == "worker diagnostic\n"
    assert measured["process_to_first_result_seconds"] > 0


@pytest.mark.parametrize("backend", ["nufft", "fastdtwf", "fft"])
def test_isolated_worker_times_first_result_and_matches_reference(backend, tmp_path):
    """Verify backend isolation, completed timings, and dense one-step accuracy."""
    if backend == "fastdtwf" and importlib.util.find_spec("fastDTWF") is None:
        pytest.skip("Install the optional fastdtwf dependency group")
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            backend,
            "--gene-copies",
            "16",
            "--selections",
            "0.1",
            "--warm-repeats",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    first, result = [json.loads(line) for line in process.stdout.splitlines()]
    assert first["event"] == "first_result_ready"
    assert all(
        first[key] > 0
        for key in ("imports_seconds", "setup_seconds", "first_call_seconds")
    )
    assert len(result["warm_seconds"]) == 2
    assert all(duration > 0 for duration in result["warm_seconds"])
    assert result["loaded_backends"] == {
        "jax": backend != "fastdtwf",
        "torch": backend == "fastdtwf",
    }
    frequency = np.arange(17) / 16
    probabilities = (frequency - 0.1 * frequency * (1 - frequency)) / (
        1 - 0.2 * frequency * (1 - frequency)
    )
    reference = binom.pmf(np.arange(17)[None, :], 16, probabilities[:, None]).mean(
        axis=0
    )
    np.testing.assert_allclose(
        result["sample_values"], reference[result["sample_indices"]], atol=1e-10
    )
    assert abs(result["mass"] - 1) < 1e-12


@pytest.mark.skipif(
    importlib.util.find_spec("fastDTWF") is None,
    reason="Requires optional fastdtwf group",
)
def test_parent_reports_process_startup_and_amortized_estimates():
    """Include interpreter startup in cold timing and amortize subsequent calls."""
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gene-copies",
            "16",
            "--selections",
            "0",
            "--tv-sds",
            "0.1",
            "--cold-repeats",
            "1",
            "--warm-repeats",
            "1",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert len(report["runs"]) == 3
    for run in report["runs"]:
        sample = run["samples"][0]
        cold = sample["process_to_first_result_seconds"]
        assert cold >= sum(
            sample[key]
            for key in ("imports_seconds", "setup_seconds", "first_call_seconds")
        )
        assert run["estimated_total_seconds"]["1"] == cold
        assert run["estimated_total_seconds"]["10"] > cold


def test_invalid_repetitions_fail_before_loading_backend():
    """Reject zero repetitions with an argparse error rather than a worker failure."""
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--cold-repeats", "0"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "error:" in process.stderr
    assert "Traceback" not in process.stderr
