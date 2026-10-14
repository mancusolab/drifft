"""Reproducible CPU/GPU benchmark; emits JSON with synchronized timings."""

# Imperative Shell: timing, device inspection, and report persistence.
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
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binom

import drifft as wf

LOGGER = logging.getLogger(__name__)


class KernelTiming(TypedDict):
    """JSON timing fields; only the backend memory estimate can be unavailable."""

    compile_seconds: float
    median_seconds: float
    min_seconds: float
    temporary_bytes: int | None


def timed_kernel(
    fn: Callable[..., jax.Array], args: tuple[jax.Array, ...], repeats: int
) -> tuple[jax.Array, KernelTiming]:
    """Compile once, warm up, then time synchronized calls and inspect temporaries."""
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compilation = time.perf_counter() - start
    compiled(*args).block_until_ready()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = compiled(*args).block_until_ready()
        elapsed.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    return value, {
        "compile_seconds": compilation,
        "median_seconds": float(np.median(elapsed)),
        "min_seconds": min(elapsed),
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
    }


def run(args: argparse.Namespace) -> None:
    """Run the validated sweep and write one JSON report; log progress to stderr."""
    rng = np.random.default_rng(args.seed)
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["jax", "jaxlib", "lineax", "numpy", "scipy"]
        },
        "config": vars(args) | {"output": str(args.output)},
        "runs": [],
    }
    for M in args.sizes:
        vhost = rng.normal(size=M + 1)
        v = jnp.asarray(vhost)
        rows = np.unique(
            np.r_[0, 1, 2, M // 2, M - 2, M - 1, M, rng.integers(0, M + 1, 64)]
        )
        for order in args.orders:
            for tail_tol in args.tail_tols:
                start = time.perf_counter()
                plan = wf.build_transition_plan(
                    M,
                    interp_order=order,
                    tail_tol=tail_tol,
                    min_mode_count=args.min_mode_count,
                    block_spacing=args.block_spacing,
                )
                jax.block_until_ready(plan)
                planning = time.perf_counter() - start
                plan_bytes = sum(a.nbytes for a in jax.tree.leaves(plan))
                for s in args.selections:
                    scalar = jnp.asarray(s)
                    x = rows / M
                    p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
                    reference = (
                        binom.pmf(np.arange(M + 1)[None, :], M, p[:, None]) @ vhost
                    )
                    fft, fft_times = timed_kernel(
                        lambda s, v, plan=plan: wf.expected_next_values(plan, s, v),
                        (scalar, v),
                        args.repeats,
                    )
                    band, band_times = timed_kernel(
                        lambda s, v, M=M, tail_tol=tail_tol: wf.apply_p_banded(
                            M, s, v, tail_tol=tail_tol
                        ),
                        (scalar, v),
                        args.repeats,
                    )
                    _, transpose_times = timed_kernel(
                        lambda s, v, plan=plan: wf.build_segregating_operator(
                            plan, s
                        ).T.mv(v),
                        (scalar, v[1:-1]),
                        args.repeats,
                    )
                    measurement = {
                        "M": M,
                        "s": s,
                        "order": order,
                        "tail_tol": tail_tol,
                        "plan_seconds": planning,
                        "plan_bytes": plan_bytes,
                        "block_count": len(plan.eta0),
                        "bucket_count": len(plan.buckets),
                        "node_count": plan.total_nodes,
                        "fft": fft_times,
                        "banded": band_times,
                        "transpose": transpose_times,
                        "fft_sample_error": float(
                            np.max(np.abs(np.asarray(fft)[rows] - reference))
                            / max(1, np.max(np.abs(vhost)))
                        ),
                        "banded_sample_error": float(
                            np.max(np.abs(np.asarray(band)[rows] - reference))
                            / max(1, np.max(np.abs(vhost)))
                        ),
                    }
                    if M <= args.dense_max_m:
                        x = np.arange(M + 1) / M
                        p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
                        dense = jnp.asarray(
                            binom.pmf(np.arange(M + 1)[None, :], M, p[:, None])
                        )
                        _, measurement["dense"] = timed_kernel(
                            lambda A, v: A @ v, (dense, v), args.repeats
                        )
                    if M <= args.solve_max_m:
                        measurement["solves"] = {}
                        for solver in ["gmres", "bicgstab"]:
                            start = time.perf_counter()
                            result = wf.expected_visits(
                                plan, scalar, solver=solver, max_steps=args.max_steps
                            )
                            jax.block_until_ready(result)
                            first_seconds = time.perf_counter() - start
                            start = time.perf_counter()
                            result = wf.expected_visits(
                                plan, scalar, solver=solver, max_steps=args.max_steps
                            )
                            jax.block_until_ready(result)
                            elapsed = time.perf_counter() - start
                            solve = {
                                "first_call_seconds": first_seconds,
                                "warm_seconds": elapsed,
                                "converged": bool(result.converged),
                                "status": str(result.status),
                                "residual_inf": float(result.residual_norm),
                                "stats": {
                                    k: int(v)
                                    if np.ndim(v) == 0 and v is not None
                                    else str(v)
                                    for k, v in result.stats.items()
                                },
                                "occupation_sum": float(result.visits.sum()),
                            }
                            if M <= args.dense_max_m:
                                rhs = np.eye(1, M - 1).ravel()
                                Q = np.asarray(dense)[1:-1, 1:-1]
                                exact = np.linalg.solve(np.eye(M - 1) - Q.T, rhs)
                                solve["exact_residual_inf"] = float(
                                    np.max(
                                        np.abs(
                                            np.asarray(result.visits)
                                            - Q.T @ result.visits
                                            - rhs
                                        )
                                    )
                                )
                                solve["occupation_max_error"] = float(
                                    np.max(np.abs(np.asarray(result.visits) - exact))
                                )
                            measurement["solves"][solver] = solve
                    report["runs"].append(measurement)
                    LOGGER.info(
                        "M=%d s=%g order=%d FFT=%.3fms band=%.3fms error=%.3g",
                        M,
                        s,
                        order,
                        1e3 * fft_times["median_seconds"],
                        1e3 * band_times["median_seconds"],
                        measurement["fft_sample_error"],
                    )
        # Keep multi-size sweeps from accumulating compiled executables indefinitely.
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
    r"""Run the benchmark command with reproducible configuration.

    **Arguments:**

    `argv`: Command-line arguments; defaults to the process arguments.

    **Returns:**

    Exit code 0 when the report is written, 1 on runtime/I/O failure. Solver
    nonconvergence remains a diagnostic in the report, not a benchmark failure.
    JSON goes to stdout or `--output`; progress goes to stderr unless `--quiet`.

    **Raises:**

    `SystemExit(2)` for invalid usage; `SystemExit(0)` for `--help`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096, 10000, 65536]
    )
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--orders", type=int, nargs="+", default=[12])
    parser.add_argument("--tail-tols", type=float, nargs="+", default=[1e-12])
    parser.add_argument("--min-mode-count", type=int, default=128)
    parser.add_argument("--block-spacing", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dense-max-m", type=int, default=1024)
    parser.add_argument("--solve-max-m", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "tpu"],
        help="JAX backend (default: JAX automatic selection)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging"
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or min(args.sizes) < 2:
        parser.error("repeats must be positive and sizes must be >= 2")
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s in args.selections):
        parser.error("--selections must be finite and in [0, 1]")
    if any(order < 2 or order > 16 or order % 2 for order in args.orders):
        parser.error("--orders must be even integers from 2 through 16")
    if any(not np.isfinite(tol) or not 1e-15 <= tol < 0.1 for tol in args.tail_tols):
        parser.error("--tail-tols must be finite and in [1e-15, 0.1)")
    if args.min_mode_count < 8:
        parser.error("--min-mode-count must be >= 8")
    if not np.isfinite(args.block_spacing) or not 0 < args.block_spacing <= 4:
        parser.error("--block-spacing must be finite and in (0, 4]")
    if args.max_steps < 1 or args.seed < 0:
        parser.error("--max-steps must be positive and --seed must be nonnegative")
    if args.dense_max_m < 0 or args.solve_max_m < 0:
        parser.error("--dense-max-m and --solve-max-m must be nonnegative")
    if args.output and (not args.output.parent.is_dir() or args.output.is_dir()):
        parser.error("--output must name a file in an existing directory")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )
    # Backend and precision are startup decisions; no library import changes them.
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
