"""Fresh-process and warm CPU propagation timings for drifft and fastDTWF."""

# Imperative Shell: isolated workers, timing, backend adapters, and JSON reports.
# Keep the parent standard-library-only: workers import just their own backend.
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

LOGGER = logging.getLogger(__name__)


def worker(args: argparse.Namespace) -> None:
    """Flush a first-result marker, then measure warm calls in this fresh process."""
    start = time.perf_counter()
    call: Callable[[], Any]
    gene_copies = args.gene_copies
    selection = args.selections[0]
    metadata: dict[str, Any] = {}
    if args.worker == "drifft":
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        import drifft

        imported = time.perf_counter()
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        plan = drifft.build_transition_plan(gene_copies, tail_tol=args.tail_tol)
        jax_mass = jnp.full(gene_copies + 1, 1.0 / (gene_copies + 1), dtype=jnp.float64)
        transition = drifft.Transition(plan, selection=drifft.Underdominance(selection))
        jax.block_until_ready((transition, jax_mass))
        advance = eqx.filter_jit(drifft.Transition.advance)

        def call():
            return advance(transition, jax_mass).block_until_ready()

        packages = (
            "drifft",
            "jax",
            "jaxlib",
            "equinox",
            "lineax",
            "jax-finufft",
            "numpy",
            "scipy",
        )
    else:
        torch = importlib.import_module("torch")
        # Upstream's DoubleTensor annotations predate modern Tensor factories.
        fastdtwf = cast(Any, importlib.import_module("fastDTWF"))
        imported = time.perf_counter()
        torch.set_grad_enabled(False)
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        counts = torch.arange(gene_copies + 1, dtype=torch.float64)
        frequency = counts / gene_copies
        probabilities = (frequency - selection * frequency * (1 - frequency)) / (
            1 - 2 * selection * frequency * (1 - frequency)
        )
        mass = torch.full(
            (gene_copies + 1,), 1.0 / (gene_copies + 1), dtype=torch.float64
        )
        groups = fastdtwf.coarse_grain(
            probabilities, gene_copies, args.tv_sds[0], no_fix=False, sfs=False
        )
        coefficients = (
            torch.lgamma(torch.tensor(gene_copies + 1.0, dtype=torch.float64))
            - torch.lgamma(counts + 1)
            - torch.lgamma(gene_copies - counts + 1)
        )
        call = partial(
            fastdtwf.mat_multiply,
            mass,
            groups,
            probabilities,
            gene_copies,
            args.row_eps,
            no_fix=False,
            sfs=False,
            bcoefs=coefficients,
        )
        packages = ("fastdtwf", "torch", "numba", "numpy", "scipy")
    prepared = time.perf_counter()
    value = call()
    finished = time.perf_counter()
    # Parent stops its launch-to-first-result clock here, before warm repeats,
    # diagnostics, version lookup, and process shutdown.
    print(
        json.dumps(
            {
                "event": "first_result_ready",
                "imports_seconds": imported - start,
                "setup_seconds": prepared - imported,
                "first_call_seconds": finished - prepared,
            }
        ),
        flush=True,
    )
    warm = []
    for _ in range(args.warm_repeats):
        start = time.perf_counter()
        value = call()
        warm.append(time.perf_counter() - start)
    if args.worker == "drifft":
        metadata["plan_bytes"] = sum(leaf.nbytes for leaf in jax.tree.leaves(plan))
        metadata["persistent_compilation_cache"] = False
    else:
        metadata["group_count"] = int(groups.max()) + 1
        metadata["torch_threads"] = torch.get_num_threads()
        metadata["source"] = json.loads(
            importlib.metadata.distribution("fastdtwf").read_text("direct_url.json")
            or "null"
        )
    indices = sorted({0, 1, gene_copies // 2, gene_copies - 1, gene_copies})
    values = value.tolist()
    print(
        json.dumps(
            metadata
            | {
                "warm_seconds": warm,
                "drifft_backend": "chebyshev_nufft"
                if args.worker == "drifft"
                else None,
                "versions": {
                    name: importlib.metadata.version(name) for name in packages
                },
                "loaded_backends": {
                    "jax": "jax" in sys.modules,
                    "torch": "torch" in sys.modules,
                },
                "mass": float(value.sum()),
                "minimum": min(values),
                "sample_indices": indices,
                "sample_values": [values[index] for index in indices],
            }
        ),
        flush=True,
    )


def measure_process(
    args: argparse.Namespace, backend: str, selection: float, tv_sd: float
) -> dict[str, Any]:
    """Time a new interpreter until its first synchronized result is available."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--gene-copies",
        str(args.gene_copies),
        "--selections",
        str(selection),
        "--tv-sds",
        str(tv_sd),
        "--row-eps",
        str(args.row_eps),
        "--tail-tol",
        str(args.tail_tol),
        "--warm-repeats",
        str(args.warm_repeats),
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    started = time.perf_counter()
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Avoid read-ahead: communicate() must receive the second JSON record
        # even when the worker writes both records before our first readline().
        text=False,
        bufsize=0,
        env=environment,
    ) as process:
        assert process.stdout is not None
        marker = process.stdout.readline()
        cold_seconds = time.perf_counter() - started
        output, errors = process.communicate()
        errors = errors.decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"{backend} worker failed: {errors.strip()}")
    first = json.loads(marker.decode("utf-8"))
    if first.pop("event") != "first_result_ready":
        raise RuntimeError("Worker did not emit a first-result marker")
    return (
        first
        | json.loads(output.decode("utf-8"))
        | {
            "process_to_first_result_seconds": cold_seconds,
            "stderr": errors,
        }
    )


def run(args: argparse.Namespace) -> None:
    """Alternate backend order across trials and retain every raw timing sample."""
    runs = []
    for selection in args.selections:
        cases = [("drifft", args.tv_sds[0])] + [("fastdtwf", sd) for sd in args.tv_sds]
        samples: dict[tuple[str, float], list[dict[str, Any]]] = {
            case: [] for case in cases
        }
        for trial in range(args.cold_repeats):
            for backend, tv_sd in cases if trial % 2 == 0 else list(reversed(cases)):
                sample = measure_process(args, backend, selection, tv_sd)
                samples[(backend, tv_sd)].append(sample)
                LOGGER.info(
                    "%s selection=%g tv_sd=%g trial=%d cold=%.3fs warm=%.3fms",
                    backend,
                    selection,
                    tv_sd,
                    trial + 1,
                    sample["process_to_first_result_seconds"],
                    1000 * statistics.median(sample["warm_seconds"]),
                )
        for (backend, tv_sd), measurements in samples.items():
            # Take medians across processes; do not treat correlated warm calls
            # in one process as independent cold-start experiments.
            warm = statistics.median(
                statistics.median(sample["warm_seconds"]) for sample in measurements
            )
            cold = statistics.median(
                sample["process_to_first_result_seconds"] for sample in measurements
            )
            runs.append(
                {
                    "backend": backend,
                    "selection_strength": selection,
                    "tv_sd": tv_sd if backend == "fastdtwf" else None,
                    "cold_median_seconds": cold,
                    "warm_median_seconds": warm,
                    "estimated_total_seconds": {
                        str(count): cold + (count - 1) * warm for count in (1, 10, 100)
                    },
                    "samples": measurements,
                }
            )
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "device": "cpu",
        "input": "uniform",
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the benchmark configuration, run isolated trials, and emit JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-copies", type=int, default=28948)
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 0.0001])
    parser.add_argument("--tv-sds", type=float, nargs="+", default=[0.1, 0.01])
    parser.add_argument("--row-eps", type=float, default=1e-8)
    parser.add_argument("--tail-tol", type=float, default=1e-12)
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument(
        "--threads", type=int, help="PyTorch CPU threads; default is unchanged"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--worker", choices=["drifft", "fastdtwf"], help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.gene_copies < 2 or min(args.cold_repeats, args.warm_repeats) < 1:
        parser.error("gene-copies must be >= 2 and repetition counts must be positive")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in args.selections):
        parser.error("selections must be finite and in [0, 1]")
    if any(not math.isfinite(sd) or sd <= 0 for sd in args.tv_sds):
        parser.error("tv-sds must be finite and positive")
    if len(set(args.tv_sds)) != len(args.tv_sds):
        parser.error("tv-sds must be distinct")
    if not math.isfinite(args.row_eps) or not 0 < args.row_eps < 1:
        parser.error("row-eps must be finite and in (0, 1)")
    if not math.isfinite(args.tail_tol) or not 1e-15 <= args.tail_tol < 0.1:
        parser.error("tail-tol must be finite and in [1e-15, 0.1)")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if args.output and (not args.output.parent.is_dir() or args.output.is_dir()):
        parser.error("output must name a file in an existing directory")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )
    try:
        worker(args) if args.worker else run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error(
            "error: %s (benchmark dependencies: uv sync --group fastdtwf)", error
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
