"""Check evolving benchmark trajectories against an independent dense chain."""

# pattern: Imperative Shell
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_epochs.py"


@pytest.mark.parametrize(
    "backend", ["nufft", "fft", "fastdtwf", "condensed", "refreshed"]
)
@pytest.mark.parametrize("source", [16, 12])
def test_epoch_worker_matches_dense_chain(backend, source):
    """Compare successive generations and size changes with a dense chain."""
    if backend not in {"nufft", "fft"} and importlib.util.find_spec("fastDTWF") is None:
        pytest.skip("Requires optional fastdtwf dependency group")
    generations = [1, 7, 60] if source == 16 else [1]
    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        backend,
        "--gene-copies",
        "16",
        "--source-gene-copies",
        str(source),
        "--selections",
        "0.1",
        "--generations",
        *map(str, generations),
        "--tv-sd",
        "0.001",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    p = np.arange(source + 1) / source
    q = (p - 0.1 * p * (1 - p)) / (1 - 0.2 * p * (1 - p))
    transition = binom.pmf(np.arange(17)[None, :], 16, q[:, None])
    first = np.full(source + 1, 1 / (source + 1)) @ transition
    p = np.arange(17) / 16
    q = (p - 0.1 * p * (1 - p)) / (1 - 0.2 * p * (1 - p))
    square = binom.pmf(np.arange(17)[None, :], 16, q[:, None])
    for row in report["epochs"]:
        expected = first @ np.linalg.matrix_power(square, row["generations"] - 1)
        np.testing.assert_allclose(row["distribution"], expected, atol=2e-8, rtol=0)
        assert row["seconds"] > 0
        assert row["total_seconds"] >= row["seconds"]


def test_rectangular_epoch_rejects_multiple_generations():
    """Require one generation when source and destination sizes differ."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gene-copies",
            "16",
            "--source-gene-copies",
            "12",
            "--generations",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "single generation" in result.stderr
