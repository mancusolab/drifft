"""Measure setup and actual successive generations in isolated CPU workers."""

# pattern: Imperative Shell
# Backend imports remain inside workers. NumPy is used only for host-side
# comparison of independent implementations, outside every timing boundary.
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

BACKENDS = ("nufft", "fft", "reference", "fastdtwf", "condensed", "refreshed")


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize one backend, then evolve the same starting mass at each horizon."""
    backend = args.worker
    size = args.gene_copies
    source = args.source_gene_copies or size
    strength = args.selections[0]
    metadata: dict[str, Any] = {}
    initialized = time.perf_counter()
    if backend in {"nufft", "fft", "reference"}:
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        import drifft

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        # Initialize the CPU client outside the epoch setup measurement.
        jnp.zeros(()).block_until_ready()
        initialization_seconds = time.perf_counter() - initialized
        started = time.perf_counter()
        options = {"tail_tol": 1e-12}
        if backend == "reference":
            options = {"accuracy": 1e-13, "tail_tol": 1e-15, "nufft_tol": 1e-15}
        if backend == "reference":
            plan = drifft.NUFFT(size, accuracy=1e-13, tail_tol=1e-15, nufft_tol=1e-15)
        else:
            plan = (drifft.FFT if backend == "fft" else drifft.NUFFT)(
                size, tail_tol=1e-12
            )
        transition = drifft.Transition(
            plan, selection=drifft.Underdominance(strength), source_gene_copies=source
        )
        mass = jnp.full(source + 1, 1 / (source + 1), dtype=jnp.float64)
        if args.initial == "singleton":
            mass = jnp.zeros(source + 1, dtype=jnp.float64).at[1].set(1)
        jax.block_until_ready((transition, mass))
        setup_seconds = time.perf_counter() - started

        dynamic, static = eqx.partition((transition, mass), eqx.is_array)

        def evolve(arrays, count):
            operator, initial = eqx.combine(arrays, static)
            if source != size:
                return operator.advance(initial)
            return jax.lax.fori_loop(
                0, count, lambda _, value: operator.advance(value), initial
            )

        started = time.perf_counter()
        compiled = jax.jit(evolve).lower(dynamic, jnp.asarray(1)).compile()
        compile_seconds = time.perf_counter() - started
        # Exercise the executable once to exclude first-dispatch/native setup
        # from warm propagation, while retaining its cost in the epoch total.
        started = time.perf_counter()
        compiled(dynamic, jnp.asarray(1)).block_until_ready()
        first_dispatch_seconds = time.perf_counter() - started

        def run(count):
            return compiled(dynamic, jnp.asarray(count)).block_until_ready()

        metadata.update(
            source_path=drifft.__file__,
            plan_bytes=sum(x.nbytes for x in jax.tree.leaves(plan)),
            plan_settings=options,
            persistent_compilation_cache=False,
        )
        packages = ("jax", "jaxlib", "equinox", "lineax", "jax-finufft")
    else:
        # PyTorch is installed only with the optional fastdtwf dependency group.
        torch = importlib.import_module("torch")

        fast = cast(Any, importlib.import_module("fastDTWF.fastDTWF"))
        torch.set_grad_enabled(False)
        if args.threads:
            torch.set_num_threads(args.threads)
        selection = torch.tensor(strength, dtype=torch.float64)
        zero = torch.tensor(0.0, dtype=torch.float64)
        condensed = backend != "fastdtwf"
        refresh = 50 if backend == "refreshed" else float("inf")

        def integrate(value, destination, count):
            return fast._integrate_likelihood_constant_size(
                value,
                destination,
                count,
                selection,
                zero,
                zero,
                args.tv_sd,
                args.row_eps,
                False,
                False,
                0.0,
                s_hom=zero,
                use_condensed=condensed,
                refresh_gens=refresh,
            )

        # Warm imports/Numba/native machinery on a tiny unrelated epoch. Actual
        # per-epoch coarse graining and condensed construction remain timed.
        integrate(torch.full((17,), 1 / 17, dtype=torch.float64), 16, 7)
        initialization_seconds = time.perf_counter() - initialized
        started = time.perf_counter()
        mass = torch.full((source + 1,), 1 / (source + 1), dtype=torch.float64)
        if args.initial == "singleton":
            mass.zero_()
            mass[1] = 1
        setup_seconds = time.perf_counter() - started
        compile_seconds = first_dispatch_seconds = 0.0

        def run(count):
            return integrate(mass, size, count)

        metadata.update(
            use_condensed=condensed,
            refresh_generations=50 if backend == "refreshed" else None,
            torch_threads=torch.get_num_threads(),
            setup_included_in_epoch_seconds=True,
            source=json.loads(
                importlib.metadata.distribution("fastdtwf").read_text("direct_url.json")
                or "null"
            ),
        )
        packages = ("fastdtwf", "torch", "numba")
    epochs = []
    for count in args.generations:
        started = time.perf_counter()
        result = run(count)
        elapsed = time.perf_counter() - started
        values = result.tolist()
        epochs.append(
            {
                "generations": count,
                "seconds": elapsed,
                "total_seconds": setup_seconds
                + compile_seconds
                + first_dispatch_seconds
                + elapsed,
                "distribution": values,
                "mass": float(result.sum()),
                "minimum": min(values),
                "lost": values[0],
                "fixed": values[-1],
            }
        )
        print(
            f"{backend} s={strength:g} generations={count}: {elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )
    return metadata | {
        "backend": backend,
        "selection": strength,
        "initialization_seconds": initialization_seconds,
        "setup_seconds": setup_seconds,
        "compile_seconds": compile_seconds,
        "first_dispatch_seconds": first_dispatch_seconds,
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "epochs": epochs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run workers sequentially and compare their full final distributions."""
    import numpy as np

    records = []
    for selection in args.selections:
        selected = []
        for backend in BACKENDS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--gene-copies",
                str(args.gene_copies),
                "--selections",
                str(selection),
                "--generations",
                *map(str, args.generations),
                "--tv-sd",
                str(args.tv_sd),
                "--row-eps",
                str(args.row_eps),
                "--initial",
                args.initial,
            ]
            if args.source_gene_copies:
                command += ["--source-gene-copies", str(args.source_gene_copies)]
            if args.threads:
                command += ["--threads", str(args.threads)]
            process = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
                env=os.environ
                | {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_COMPILATION_CACHE": "false"},
            )
            selected.append(json.loads(process.stdout))
        reference = next(r for r in selected if r["backend"] == "reference")
        for record in selected:
            for row, ref in zip(record["epochs"], reference["epochs"], strict=True):
                observed = np.asarray(row["distribution"])
                expected = np.asarray(ref["distribution"])
                delta = np.abs(observed - expected)
                row["reference_l1"] = float(delta.sum())
                row["reference_max_abs"] = float(delta.max())
                row["reference_segregating_l1"] = float(delta[1:-1].sum())
                row["reference_segregating_relative_l1"] = float(
                    delta[1:-1].sum() / max(float(np.abs(expected[1:-1]).sum()), 1e-300)
                )
                row["mean_frequency"] = float(
                    observed @ np.linspace(0, 1, len(observed))
                )
        # Strip vectors only after every backend has been compared, including
        # the reference itself. The report retains boundary and mass diagnostics.
        for record in selected:
            for row in record["epochs"]:
                del row["distribution"]
        records.extend(selected)
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "timing_repeats": 1,
        "reference": "Chebyshev/NUFFT accuracy=1e-13; cross-check FFT independently",
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "runs": records,
    }


def main() -> None:
    """Validate epoch shapes and numerical settings before importing backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-copies", type=int, default=28948)
    parser.add_argument("--source-gene-copies", type=int)
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 1e-4])
    parser.add_argument(
        "--generations", type=int, nargs="+", default=[100, 200, 500, 1000]
    )
    parser.add_argument(
        "--initial", choices=["uniform", "singleton"], default="uniform"
    )
    parser.add_argument("--tv-sd", type=float, default=0.1)
    parser.add_argument("--row-eps", type=float, default=1e-8)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gene_copies < 2 or (
        args.source_gene_copies is not None and args.source_gene_copies < 2
    ):
        parser.error("population sizes must be >= 2")
    if any(g < 1 for g in args.generations):
        parser.error("generations must be positive")
    if args.source_gene_copies not in (None, args.gene_copies) and args.generations != [
        1
    ]:
        parser.error("a size change must use a single generation (--generations 1)")
    if any(not math.isfinite(s) or not 0 <= s < 1 for s in args.selections):
        parser.error("selection must be finite and in [0, 1)")
    if not math.isfinite(args.tv_sd) or args.tv_sd <= 0:
        parser.error("tv-sd must be finite and positive")
    if not math.isfinite(args.row_eps) or not 0 < args.row_eps < 1:
        parser.error("row-eps must be finite and in (0, 1)")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if args.output and not args.output.parent.is_dir():
        parser.error("output directory must exist")
    try:
        report = worker(args) if args.worker else run(args)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
