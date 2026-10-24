"""Fresh-process costs and numerical diagnostics for NUFFT encodings.

The legacy case changes only the degree planner to the previous 4M/d bound.
Each timing worker measures one size, accuracy, encoding and operation; numerical
diagnostics run after timing. Dense power references are limited to M <= 1024.
"""

# pattern: Imperative Shell
import argparse
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
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

LOGGER = logging.getLogger(__name__)


def legacy_degree(size: int, tolerance: float) -> int:
    """Reproduce the previous degree planner for a benchmark-only baseline."""
    lo, hi = 1, size
    while lo < hi:
        mid = (lo + hi) // 2
        if math.log(4 * size / mid) - mid * mid / (2 * size) <= math.log(tolerance):
            hi = mid
        else:
            lo = mid + 1
    return lo


def diagnostics(backend, strength):
    """Evaluate selected full rows, row derivatives and small dense power oracles."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import binom

    import drifft

    size = backend.gene_copies
    low = np.r_[0, 1e-15, 1e-10, 0.1 / size, 1 / size, 2 / size, 0.01, 0.1, 0.5]
    probabilities = np.unique(
        np.r_[low, 1 - low, np.random.default_rng(19).uniform(size=8)]
    )

    @jax.jit
    def row(q):
        return drifft.Transition.from_probabilities(backend, q[None]).advance(
            jnp.ones(1)
        )

    derivative = jax.jit(lambda q: jax.jvp(row, (q,), (jnp.ones_like(q),))[1])
    counts = np.arange(size + 1)
    errors, derivatives, relative_derivatives, masses, negatives = [], [], [], [], []
    for q in probabilities:
        actual = np.asarray(row(jnp.asarray(q)))
        reference = binom.pmf(counts, size, q)
        slope = size * (
            binom.pmf(counts - 1, size - 1, q) - binom.pmf(counts, size - 1, q)
        )
        error = float(np.abs(np.asarray(derivative(jnp.asarray(q))) - slope).sum())
        errors.append(float(np.abs(actual - reference).sum()))
        derivatives.append(error)
        relative_derivatives.append(error / float(np.abs(slope).sum()))
        masses.append(float(abs(actual.sum() - 1)))
        negatives.append(float(-np.minimum(actual, 0).sum()))
    powers = []
    if size <= 1024:
        for selection in sorted({0.0, strength}):
            transition = drifft.Transition(backend, drifft.Underdominance(selection))
            dense = binom.pmf(
                counts, size, np.asarray(transition.probabilities)[:, None]
            )
            for initial in ("singleton", "uniform"):
                mass = (
                    np.eye(1, size + 1, 1).ravel()
                    if initial == "singleton"
                    else np.full(size + 1, 1 / (size + 1))
                )
                actual, reference = jnp.asarray(mass), mass
                for generation in range(1, 101):
                    actual = transition.advance(actual)
                    reference = reference @ dense
                    if generation in (1, 10, 100):
                        output = np.asarray(actual)
                        powers.append(
                            {
                                "selection": selection,
                                "initial": initial,
                                "generations": generation,
                                "l1_error": float(np.abs(output - reference).sum()),
                                "mass_defect": float(abs(output.sum() - 1)),
                                "negative_mass": float(-np.minimum(output, 0).sum()),
                            }
                        )
    return {
        "probabilities": probabilities.tolist(),
        "row_l1": errors,
        "row_l1_max": max(errors),
        "derivative_row_l1": derivatives,
        "derivative_relative_l1_max": max(relative_derivatives),
        "mass_defect_max": max(masses),
        "negative_mass_max": max(negatives),
        "powers": powers,
    }


def run_worker(args):
    """Emit the first-result marker before warm repetitions and diagnostics."""
    started = time.perf_counter()
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import drifft
    import drifft.backends.nufft as nufft_module

    imported = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_compilation_cache", False)
    # Isolated worker only: the baseline retains current execution machinery
    # while restoring the old planner. No third public encoding is introduced.
    size = args.sizes[0]

    def matched_modes(size: int, tolerance: float) -> int:
        """Match Lobatto's output degree, except at exact-fallback boundaries."""
        return min(size, nufft_module._choose_degree(size, tolerance) + 1)

    with ExitStack() as overrides:
        if args.worker == "legacy":
            overrides.enter_context(
                patch.object(nufft_module, "_choose_degree", legacy_degree)
            )
        elif args.worker == "filtered-matched":
            overrides.enter_context(
                patch.object(nufft_module, "_choose_filtered_modes", matched_modes)
            )
        backend = drifft.NUFFT(
            size,
            accuracy=args.accuracies[0],
            encoding="filtered" if args.worker.startswith("filtered") else "lobatto",
            derivative_accuracy=args.derivative_accuracy,
        )
    model = drifft.Transition(backend, drifft.Underdominance(args.selection))
    counts = jnp.arange(size + 1, dtype=jnp.float64)
    vector = (
        jnp.sin(counts * 0.731)
        if args.operation == "expect"
        else jnp.full(size + 1, 1 / (size + 1))
    )
    jax.block_until_ready((model, vector))
    prepared = time.perf_counter()
    arrays, static = eqx.partition((model, vector), eqx.is_array)
    method = getattr(drifft.Transition, args.operation)
    compiled = (
        jax.jit(lambda a: method(*eqx.combine(a, static))).lower(arrays).compile()
    )
    compiled_at = time.perf_counter()
    jax.block_until_ready(compiled(arrays))
    finished = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "first_result_ready",
                "imports_seconds": imported - started,
                "setup_seconds": prepared - imported,
                "compile_seconds": compiled_at - prepared,
                "first_execution_seconds": finished - compiled_at,
            }
        ),
        flush=True,
    )
    warm = []
    for _ in range(args.warm_repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(arrays))
        warm.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    result = {
        "warm_seconds": warm,
        "degree": backend.degree,
        "sample_count": backend.nodes.size,
        "taper_width": backend.taper_width,
        "radius": backend.radius,
        "spectral_bound": backend.spectral_bound,
        "sampling_bound": backend.sampling_bound,
        "plan_bytes": sum(x.nbytes for x in jax.tree.leaves(backend)),
        "xla_temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "persistent_compilation_cache": False,
        "source_path": drifft.__file__,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("jax", "jaxlib", "jax-finufft", "equinox", "numpy", "scipy")
        },
        "diagnostics": diagnostics(backend, args.selection)
        if args.diagnostics
        else None,
    }
    print(json.dumps(result, allow_nan=False), flush=True)


def measure_process(args, variant, size, accuracy, operation, trial):
    """Measure launch-to-first-result with an unbuffered binary protocol."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        variant,
        "--sizes",
        str(size),
        "--accuracies",
        str(accuracy),
        "--operation",
        operation,
        "--selection",
        str(args.selection),
        "--warm-repeats",
        str(args.warm_repeats),
    ]
    if args.diagnostics and operation == "advance" and trial == 0:
        command.append("--diagnostics")
    if args.derivative_accuracy:
        command.append("--derivative-accuracy")
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    started = time.perf_counter()
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=environment,
    ) as process:
        assert process.stdout is not None
        marker = process.stdout.readline()
        cold = time.perf_counter() - started
        output, errors = process.communicate()
        if process.returncode:
            raise RuntimeError(
                f"{variant} worker failed: {errors.decode(errors='replace')}"
            )
    first = json.loads(marker)
    if first.pop("event") != "first_result_ready":
        raise RuntimeError("worker did not emit first-result marker")
    return (
        first
        | json.loads(output)
        | {
            "process_to_first_result_seconds": cold,
            "stderr": errors.decode(errors="replace"),
        }
    )


def run(args):
    """Alternate encoding order and summarize medians across fresh processes."""
    runs = []
    for size in args.sizes:
        for accuracy in args.accuracies:
            for operation in ("expect", "advance"):
                samples = {variant: [] for variant in args.variants}
                for trial in range(args.cold_repeats):
                    order = args.variants if trial % 2 == 0 else args.variants[::-1]
                    for variant in order:
                        sample = measure_process(
                            args, variant, size, accuracy, operation, trial
                        )
                        samples[variant].append(sample)
                        LOGGER.info(
                            "%s M=%d accuracy=%g %s cold=%.3fs warm=%.3fms",
                            variant,
                            size,
                            accuracy,
                            operation,
                            sample["process_to_first_result_seconds"],
                            1000 * statistics.median(sample["warm_seconds"]),
                        )
                for variant, measurements in samples.items():
                    runs.append(
                        {
                            "variant": variant,
                            "M": size,
                            "accuracy": accuracy,
                            "operation": operation,
                            "cold_median_seconds": statistics.median(
                                s["process_to_first_result_seconds"]
                                for s in measurements
                            ),
                            "warm_median_seconds": statistics.median(
                                statistics.median(s["warm_seconds"])
                                for s in measurements
                            ),
                            "samples": measurements,
                        }
                    )
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)


def main(argv=None):
    """Validate benchmark inputs before launching numerical workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--accuracies", type=float, nargs="+", default=[1e-6, 1e-10])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["legacy", "lobatto", "filtered", "filtered-matched"],
        default=["legacy", "lobatto", "filtered"],
    )
    parser.add_argument("--selection", type=float, default=0.1)
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--derivative-accuracy", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker",
        choices=["legacy", "lobatto", "filtered", "filtered-matched"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--operation",
        choices=["expect", "advance"],
        default="advance",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if min(args.sizes) < 2 or min(args.cold_repeats, args.warm_repeats) < 1:
        parser.error("sizes must be >= 2 and repetition counts positive")
    if any(not math.isfinite(a) or not 1e-13 <= a < 0.1 for a in args.accuracies):
        parser.error("accuracies must be finite and in [1e-13, .1)")
    if not math.isfinite(args.selection) or not 0 <= args.selection <= 1:
        parser.error("selection must be finite and in [0, 1]")
    if args.output and (not args.output.parent.is_dir() or args.output.is_dir()):
        parser.error("output must name a file in an existing directory")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run_worker(args) if args.worker else run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
