"""CPU comparison contracts and independent checks of matched DTWF workloads."""

# Imperative Shell: optional-dependency subprocess integration tests.
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_fastdtwf.py"


@pytest.mark.parametrize(
    "arguments", [["--tv-sds", "0"], ["--row-eps", "nan"], ["--threads", "0"]]
)
def test_invalid_fastdtwf_arguments(arguments):
    """Reject malformed benchmark options before loading optional dependencies."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("fastDTWF") is None,
    reason="Install the optional fastdtwf dependency group",
)
@pytest.mark.parametrize("gene_copies,tv_sd", [(16, 0.01), (1024, 0.1)])
def test_fastdtwf_matches_transition_and_expected_visits(gene_copies, tv_sd):
    """Compare isolated upstream results with independent transition references."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sizes",
            str(gene_copies),
            "--selections",
            "0",
            "0.5",
            "--tv-sds",
            str(tv_sd),
            "--repeats",
            "1",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["device"] == "cpu"
    assert report["versions"]["fastdtwf"]
    assert report["versions"]["jax-finufft"]
    assert report["drifft_backend"] == "chebyshev_nufft"
    assert report["worker_backends"] == {
        "drifft": {"drifft": True, "jax": True, "torch": False, "fastDTWF": False},
        "fastdtwf": {"drifft": False, "jax": False, "torch": True, "fastDTWF": True},
    }
    assert len(report["runs"]) == 2
    for run in report["runs"]:
        for transition in run["transitions"]:
            for backend in ["drifft", "fastdtwf"]:
                metrics = transition[backend]
                assert metrics["median_seconds"] > 0
                tolerance = 1e-9 if backend == "drifft" or gene_copies == 16 else 1e-3
                assert metrics["exact_l1_error"] < tolerance
                assert abs(metrics["mass"] - 1) < 1e-10
        if gene_copies > 256:
            assert run["fastdtwf_group_count"] < gene_copies + 1
            assert "expected_visits" not in run
            continue
        solves = run["expected_visits"]
        assert solves["drifft"]["converged"]
        for backend in ["drifft", "fastdtwf"]:
            assert solves[backend]["exact_residual_inf"] < 1e-8
            assert solves[backend]["exact_max_error"] < 1e-7
