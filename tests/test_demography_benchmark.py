"""Exercise the demographic benchmark through its supported command line."""

# pattern: Imperative Shell
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark


def test_demography_benchmark_runs_isolated_cold_and_warm_passes(tmp_path):
    """Compare both backends on a small changing-size history and retain provenance."""
    history = tmp_path / "history.tsv"
    history.write_text("Ne\tT_gen\n12\t8\n10\t5\n14\t2\n20\t0\n")
    output = tmp_path / "result.json"
    script = Path(__file__).resolve().parents[1] / "benchmarks/compare_demography.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(history),
            "--cutoff",
            "2",
            "--krylov-dimension",
            "16",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["history"]["ne"] == [12, 10, 14]
    assert report["history"]["generations"] == [3, 3]
    assert report["initialization"]["converged"]
    assert report["relative_l1_difference"] < 1e-7
    assert report["provenance"]["input_sha256"]
    assert "src/drifft/_backend/fft.py" in report["provenance"]["source_sha256"]
    for name in ["fft", "krylov"]:
        worker = report["backends"][name]
        assert worker["setup_seconds"] > 0
        assert worker["peak_rss_bytes"] > 0
        assert worker["versions"]["jax"]
        assert [p["label"] for p in worker["passes"]] == ["cold", "warm"]
        for timed_pass in worker["passes"]:
            assert timed_pass["seconds"] > 0
            assert timed_pass["failed_intervals"] == 0
            assert len(timed_pass["intervals"]) == 2
        assert worker["passes"][0]["final"] == worker["passes"][1]["final"]


def test_demography_benchmark_rejects_invalid_time_order(tmp_path):
    """Reject duplicate times before launching any numerical workers."""
    history = tmp_path / "history.tsv"
    history.write_text("Ne\tT_gen\n12\t8\n10\t8\n")
    output = tmp_path / "result.json"
    script = Path(__file__).resolve().parents[1] / "benchmarks/compare_demography.py"
    result = subprocess.run(
        [sys.executable, str(script), str(history), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "strictly decreasing" in result.stderr
    assert result.stdout == ""
    assert not output.exists()


def test_demography_benchmark_help():
    """Expose the input, cutoff, and solver controls without importing JAX."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/compare_demography.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--cutoff" in result.stdout
    assert "--krylov-dimension" in result.stdout
    assert result.stderr == ""


def test_demography_benchmark_rejects_missing_cells(tmp_path):
    """Explain a short TSV row as an input error, without a Python traceback."""
    history = tmp_path / "history.tsv"
    history.write_text("Ne\tT_gen\n12\t8\n10\n")
    script = Path(__file__).resolve().parents[1] / "benchmarks/compare_demography.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(history),
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "row 3" in result.stderr
    assert "Traceback" not in result.stderr
