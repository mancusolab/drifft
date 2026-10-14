"""Synchronized Chebyshev/NUFFT accuracy, compilation and execution benchmark."""

# Imperative Shell: device setup, timing, reference validation and report I/O.
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import resource
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binom

import drifft

LOGGER = logging.getLogger(__name__)


def timed_kernel(fn: Callable, args: tuple, repeats: int):
    """Compile separately, warm up once, then time synchronized execution."""
    start = time.perf_counter()
    dynamic, static = eqx.partition(args, eqx.is_array)
    compiled = (
        jax.jit(lambda arrays: fn(*eqx.combine(arrays, static)))
        .lower(dynamic)
        .compile()
    )
    compilation = time.perf_counter() - start
    value = jax.block_until_ready(compiled(dynamic))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(compiled(dynamic))
        elapsed.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    return value, {
        "compile_seconds": compilation,
        "median_seconds": float(np.median(elapsed)),
        "min_seconds": min(elapsed),
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
    }


def run(args: argparse.Namespace) -> None:
    """Measure source/destination transitions without dense production references."""
    rng = np.random.default_rng(args.seed)
    report = {
        "schema_version": 2,
        "backend": "chebyshev_nufft",
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["jax", "jaxlib", "lineax", "jax-finufft", "numpy", "scipy"]
        },
        "config": vars(args) | {"output": str(args.output)},
        "runs": [],
    }
    for size in args.sizes:
        source = size if args.source_gene_copies is None else args.source_gene_copies
        values = jnp.asarray(rng.normal(size=size + 1))
        mass = jnp.ones(source + 1) / (source + 1)
        start = time.perf_counter()
        plan = drifft.build_transition_plan(
            size,
            accuracy=args.accuracy,
            batch_size=args.batch_size,
            derivative_accuracy=args.derivative_accuracy,
        )
        jax.block_until_ready(plan)
        planning = time.perf_counter() - start
        rows = np.unique(
            np.r_[
                0, 1, source // 2, source - 1, source, rng.integers(0, source + 1, 16)
            ]
        )
        for strength in args.selections:
            transition = drifft.Transition(
                plan, drifft.Underdominance(strength), source_gene_copies=source
            )
            actual, expect_times = timed_kernel(
                drifft.Transition.expect, (transition, values), args.repeats
            )
            propagated, advance_times = timed_kernel(
                drifft.Transition.advance, (transition, mass), args.repeats
            )

            def selection_jvp(s, x, plan=plan, source=source):
                return jax.jvp(
                    lambda t: drifft.Transition(
                        plan, drifft.Underdominance(t), source_gene_copies=source
                    ).expect(x),
                    (s,),
                    (jnp.ones_like(s),),
                )[1]

            tangent, jvp_times = timed_kernel(
                selection_jvp, (jnp.asarray(strength), values), args.repeats
            )
            probabilities = np.asarray(transition.probabilities)[rows]
            # O(sample_rows * M), never a production-size square reference.
            reference = (
                binom.pmf(np.arange(size + 1)[None, :], size, probabilities[:, None])
                @ values
            )
            derivative_q = (
                size
                * binom.pmf(np.arange(size)[None, :], size - 1, probabilities[:, None])
                @ np.diff(values)
            )
            frequencies = rows / source
            dq_ds = (
                frequencies
                * (1 - frequencies)
                * (2 * frequencies - 1)
                / (1 - 2 * strength * frequencies * (1 - frequencies)) ** 2
            )
            measurement = {
                "M": size,
                "source_gene_copies": source,
                "s": strength,
                "plan_seconds": planning,
                "plan_bytes": sum(a.nbytes for a in jax.tree.leaves(plan)),
                "degree": plan.degree,
                "stencil_width": plan.stencil_width,
                "batch_size": plan.batch_size,
                "spectral_bound": plan.spectral_bound,
                "sampling_bound": plan.sampling_bound,
                "nufft_tol": plan.nufft_tol,
                "expect": expect_times,
                "advance": advance_times,
                "selection_jvp": jvp_times,
                "sample_error": float(
                    np.max(np.abs(actual[rows] - reference))
                    / max(1, float(jnp.max(jnp.abs(values))))
                ),
                "sample_jvp_error": float(
                    np.max(np.abs(tangent[rows] - derivative_q * dq_ds))
                ),
                "mass_defect": float(jnp.abs(propagated.sum() - 1)),
                "minimum_mass": float(propagated.min()),
            }
            if size == source and size <= args.solve_max_m:
                result, times = timed_kernel(
                    drifft.Transition.expected_visits, (transition,), args.repeats
                )
                measurement["solve"] = times | {
                    "converged": bool(result.converged),
                    "residual_inf": float(result.residual_norm),
                    "stats": {
                        k: int(v) if v is not None else None
                        for k, v in result.stats.items()
                    },
                }
                if size <= args.dense_max_m:
                    q = np.asarray(transition.probabilities)[1:-1]
                    matrix = binom.pmf(np.arange(1, size)[None, :], size, q[:, None])
                    rhs = np.eye(1, size - 1).ravel()
                    exact = np.linalg.solve(np.eye(size - 1) - matrix.T, rhs)
                    measurement["solve"]["exact_max_error"] = float(
                        np.max(np.abs(result.visits - exact))
                    )
            report["runs"].append(measurement)
            LOGGER.info(
                "M=%d source=%d s=%g expect=%.3fms advance=%.3fms error=%.3g",
                size,
                source,
                strength,
                1e3 * expect_times["median_seconds"],
                1e3 * advance_times["median_seconds"],
                measurement["sample_error"],
            )
        jax.clear_caches()
    report["process_peak_rss_native_units"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    report["rss_units"] = "bytes" if sys.platform == "darwin" else "KiB"
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def main(argv: Sequence[str] | None = None) -> int:
    r"""Run the reproducible benchmark.

    **Arguments:**

    `argv`: CLI arguments, or the process arguments when omitted.

    **Returns:**

    Zero on success, one on runtime/I/O failure. JSON goes to stdout or `--output`;
    progress goes to stderr. Numerical accuracy and convergence are report fields.

    **Raises:**

    `SystemExit(2)` for invalid arguments, `SystemExit(0)` for help.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 10000, 28948, 100000, 1000000]
    )
    parser.add_argument("--source-gene-copies", type=int)
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 1e-4])
    parser.add_argument("--accuracy", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--derivative-accuracy", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense-max-m", type=int, default=512)
    parser.add_argument("--solve-max-m", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or min(args.sizes) < 2 or args.batch_size < 1:
        parser.error("repeats/batch-size must be positive and sizes >= 2")
    if args.source_gene_copies is not None and args.source_gene_copies < 2:
        parser.error("source-gene-copies must be >= 2")
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s in args.selections):
        parser.error("selections must be finite and in [0, 1]")
    if not np.isfinite(args.accuracy) or not 1e-13 <= args.accuracy < 0.1:
        parser.error("accuracy must be finite and in [1e-13, 0.1)")
    if args.seed < 0 or min(args.dense_max_m, args.solve_max_m) < 0:
        parser.error("seed and reference/solve size limits must be nonnegative")
    if args.output and (not args.output.parent.is_dir() or args.output.is_dir()):
        parser.error("output must name a file in an existing directory")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )
    if args.device:
        jax.config.update("jax_platform_name", args.device)
    jax.config.update("jax_enable_x64", True)
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("error: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
