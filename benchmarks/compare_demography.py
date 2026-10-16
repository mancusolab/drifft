"""Compare neutral demographic evaluation with Sequential FFT and RationalKrylov."""

# pattern: Imperative Shell
# A standard-library parent isolates compilation and peak RSS in serial workers.
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def run_worker(config: dict[str, Any], mode: str, directory: Path) -> dict[str, Any]:
    """Initialize once or time synchronized public epoch calls in a fresh process."""
    import jax
    import jax.numpy as jnp

    import drifft

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_compilation_cache", False)
    sizes = config["history"]["ne"]
    durations = config["history"]["generations"]
    settings = config["settings"]
    mutation_rate = settings["mutation_rate"]
    frequency_map = drifft.Neutral()
    start = time.perf_counter()
    if mode == "initialize":
        if settings["initialization"] == "equilibrium":
            transition = drifft.Transition(drifft.FFT(2 * sizes[0]), frequency_map)
            solution = transition.expected_visits(
                rtol=settings["equilibrium_rtol"],
                atol=settings["equilibrium_atol"],
                max_steps=settings["equilibrium_max_steps"],
            )
            jax.block_until_ready(solution)
            if not bool(solution.converged):
                raise RuntimeError("equilibrium initialization did not converge")
            initial = solution.visits * (2 * sizes[0] * mutation_rate)
            residual = float(solution.residual_norm)
        else:
            initial = jnp.zeros(2 * sizes[0] - 1, dtype=jnp.float64)
            residual = None
        initial.block_until_ready()
        seconds = time.perf_counter() - start
        (directory / "initial.json").write_text(json.dumps(initial.tolist()))
        return dict(seconds=seconds, converged=True, residual_norm=residual)

    initial = jnp.asarray(json.loads((directory / "initial.json").read_text()))
    fft = {size: drifft.FFT(2 * size) for size in dict.fromkeys(sizes)}
    epochs = {
        size: (
            drifft.Sequential(backend)
            if mode == "fft"
            else drifft.RationalKrylov(
                2 * size,
                krylov_dimension=settings["krylov_dimension"],
                rank=settings["rank"],
                rtol=settings["rtol"],
                atol=settings["atol"],
            )
        )
        for size, backend in fft.items()
    }
    jax.block_until_ready((initial, fft, epochs))
    report: dict[str, Any] = {
        "setup_seconds": time.perf_counter() - start,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("drifft", "jax", "jaxlib", "equinox", "lineax", "scipy")
        },
        "devices": [str(device) for device in jax.devices()],
        "passes": [],
    }
    for label in ("cold", "warm"):
        distribution = initial
        intervals = []
        start = time.perf_counter()
        for source, destination, duration in zip(
            sizes[:-1], sizes[1:], durations, strict=True
        ):
            tick = time.perf_counter()
            immigration = (
                jnp.zeros(2 * source - 1).at[0].set(2 * source * mutation_rate)
            )
            result = epochs[source].advance(
                distribution,
                frequency_map=frequency_map,
                generations=duration - 1,
                segregating=True,
                immigration=immigration,
                throw=False,
            )
            jax.block_until_ready(result)
            hold_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            transition = drifft.Transition(
                fft[destination], frequency_map, source_gene_copies=2 * source
            )
            distribution = transition.advance_segregating(result.distribution)
            distribution = distribution.at[0].add(2 * destination * mutation_rate)
            distribution.block_until_ready()
            boundary_seconds = time.perf_counter() - tick
            if not bool(jnp.all(jnp.isfinite(distribution))):
                raise RuntimeError("nonfinite demographic output")
            intervals.append(
                dict(
                    source_ne=source,
                    destination_ne=destination,
                    generations=duration,
                    hold_seconds=hold_seconds,
                    boundary_seconds=boundary_seconds,
                    converged=bool(result.converged),
                    error_estimate=float(result.error_estimate),
                )
            )
        seconds = time.perf_counter() - start
        report["passes"].append(
            dict(
                label=label,
                seconds=seconds,
                intervals=intervals,
                failed_intervals=sum(not item["converged"] for item in intervals),
                final=distribution.tolist(),
            )
        )
    # macOS reports bytes; Linux reports KiB. CPU-only benchmark on these hosts.
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    return report


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the TSV, launch serial workers, and collect reproducible measurements."""
    if args.demography.resolve() == args.output.resolve():
        raise ValueError("output must differ from the demographic input")
    with args.demography.open() as handle:
        table = csv.DictReader(handle, delimiter="\t")
        if not {"Ne", "T_gen"} <= set(table.fieldnames or ()):
            raise ValueError("demography requires Ne and T_gen columns")
        rows = []
        for line, row in enumerate(table, start=2):
            try:
                rows.append((int(row["Ne"]), int(row["T_gen"])))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"row {line}: Ne and T_gen must both contain integers"
                ) from error
    if any(ne < 1 or generation < 0 for ne, generation in rows):
        raise ValueError("Ne must be positive and T_gen nonnegative")
    if any(a[1] <= b[1] for a, b in zip(rows[:-1], rows[1:], strict=True)):
        raise ValueError("T_gen must be strictly decreasing (oldest first)")
    rows = [row for row in rows if row[1] >= args.cutoff]
    if len(rows) < 2:
        raise ValueError("cutoff must retain at least two demographic boundaries")
    if args.rank is not None and any(args.rank >= 2 * ne for ne, _ in rows):
        raise ValueError("rank must be smaller than every retained gene-copy count")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"demography", "output", "worker", "directory"}
    }
    report: dict[str, Any] = dict(
        schema_version=1,
        history=dict(
            ne=[row[0] for row in rows],
            generations=[a[1] - b[1] for a, b in zip(rows[:-1], rows[1:], strict=True)],
            boundaries=[row[1] for row in rows],
        ),
        settings=settings,
        provenance=dict(
            input_sha256=hashlib.sha256(args.demography.read_bytes()).hexdigest(),
            git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            source_sha256={
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    Path(__file__).resolve(),
                    *sorted((root / "src/drifft").rglob("*.py")),
                ]
            },
            python=sys.version,
            platform=platform.platform(),
            dtype="float64",
            device="cpu",
            compilation_cache=False,
            threads=dict(blas=1, openmp=1, jax="default"),
        ),
        backends={},
    )
    # Initialization is shared, but cannot warm either backend's process-local JIT.
    with tempfile.TemporaryDirectory(prefix="drifft-demography-") as temporary:
        directory = Path(temporary)
        (directory / "config.json").write_text(json.dumps(report))
        environment = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        for mode in ("initialize", "fft", "krylov"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--directory",
                    str(directory),
                ],
                env=environment,
                check=True,
            )
            result = json.loads((directory / f"{mode}.json").read_text())
            if mode == "initialize":
                report["initialization"] = result
            else:
                report["backends"][mode] = result
    reference = report["backends"]["fft"]["passes"][-1]["final"]
    actual = report["backends"]["krylov"]["passes"][-1]["final"]
    difference = sum(abs(a - b) for a, b in zip(actual, reference, strict=True))
    denominator = sum(abs(value) for value in reference)
    report["relative_l1_difference"] = difference / denominator if denominator else None
    report["max_absolute_difference"] = max(
        abs(a - b) for a, b in zip(actual, reference, strict=True)
    )
    report["converged"] = all(
        timed_pass["failed_intervals"] == 0
        for backend in report["backends"].values()
        for timed_pass in backend["passes"]
    )
    return report


def main(argv: list[str] | None = None) -> int:
    r"""Write synchronized demographic benchmark measurements to JSON.

    **Arguments:**

    - `argv`: Command-line arguments, or `None` to read the process arguments.

    **Returns:**

    Exit code 0 on convergence or 1 on runtime or convergence failure.

    **Raises:**

    `SystemExit` with code 2 for invalid input, or code 0 for `--help`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demography", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cutoff",
        type=int,
        default=252,
        help="retain boundaries at least this many generations ago",
    )
    parser.add_argument(
        "--initialization", choices=["equilibrium", "zero"], default="equilibrium"
    )
    parser.add_argument("--mutation-rate", type=float, default=1e-8)
    parser.add_argument("--equilibrium-rtol", type=float, default=1e-10)
    parser.add_argument("--equilibrium-atol", type=float, default=1e-12)
    parser.add_argument("--equilibrium-max-steps", type=int, default=4000)
    parser.add_argument("--krylov-dimension", type=int, default=128)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--worker", choices=["initialize", "fft", "krylov"], help=argparse.SUPPRESS
    )
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.directory is None:
                parser.error("worker requires --directory")
            result = run_worker(
                json.loads((args.directory / "config.json").read_text()),
                args.worker,
                args.directory,
            )
            (args.directory / f"{args.worker}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            return 0
        if args.demography is None or args.output is None:
            parser.error("demography and --output are required")
        if (
            args.cutoff < 0
            or args.equilibrium_max_steps < 1
            or args.krylov_dimension < 2
            or (args.rank is not None and args.rank < 1)
        ):
            parser.error("invalid cutoff, iteration limit, dimension, or rank")
        for name in (
            "mutation_rate",
            "equilibrium_rtol",
            "equilibrium_atol",
            "rtol",
            "atol",
        ):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                parser.error(f"{name} must be finite and nonnegative")
        report = run_comparison(args)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return 0 if report["converged"] else 1
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
