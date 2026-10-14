"""Compare FFT and NUFFT in isolated processes."""

# Imperative Shell: imports isolated by backend, synchronized timings and JSON I/O.
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_worker(args):
    """Time the shared Transition API using one concrete backend."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import binom

    import drifft

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    runs = []
    for size in args.sizes:
        start = time.perf_counter()
        backend = (drifft.FFT if args.worker == "fft" else drifft.NUFFT)(size)
        jax.block_until_ready(backend)
        planning = time.perf_counter() - start
        values = jnp.asarray(np.random.default_rng(17).normal(size=size + 1))
        mass = jnp.ones(size + 1) / (size + 1)
        rows = np.unique(
            np.r_[
                0,
                1,
                size // 2,
                size - 1,
                size,
                np.random.default_rng(18).integers(0, size + 1, 16),
            ]
        )
        for strength in args.selections:
            model = drifft.Transition(backend, drifft.Underdominance(strength))
            result = {
                "M": size,
                "s": strength,
                "plan_seconds": planning,
                "plan_bytes": sum(x.nbytes for x in jax.tree.leaves(backend)),
            }
            for name, vector in (("expect", values), ("advance", mass)):
                method = getattr(drifft.Transition, name)
                dynamic, static = eqx.partition((model, vector), eqx.is_array)
                start = time.perf_counter()
                compiled = (
                    jax.jit(
                        lambda arrays, method=method, static=static: method(
                            *eqx.combine(arrays, static)
                        )
                    )
                    .lower(dynamic)
                    .compile()
                )
                compile_seconds = time.perf_counter() - start
                actual = jax.block_until_ready(compiled(dynamic))
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    actual = jax.block_until_ready(compiled(dynamic))
                    times.append(time.perf_counter() - start)
                memory = compiled.memory_analysis()
                result[name] = {
                    "compile_seconds": compile_seconds,
                    "warm_seconds": times,
                    "median_seconds": float(np.median(times)),
                    "xla_temporary_bytes": None
                    if memory is None
                    else memory.temp_size_in_bytes,
                }
                if name == "expect":
                    q = np.asarray(model.probabilities)[rows]
                    reference = (
                        binom.pmf(np.arange(size + 1)[None, :], size, q[:, None])
                        @ values
                    )
                    result["sample_error"] = float(
                        np.max(np.abs(actual[rows] - reference))
                        / np.max(np.abs(values))
                    )
                else:
                    result["mass_defect"] = float(jnp.abs(actual.sum() - 1))
            runs.append(result)
        jax.clear_caches()
    return {
        "source": str(Path(drifft.__file__).resolve()),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["jax", "jaxlib", "equinox", "lineax"]
        },
        "runs": runs,
    }


def main():
    """Run backends sequentially, with one fresh worker per implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 10000, 28948, 100000]
    )
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 1e-4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["fft", "nufft"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.sizes) < 2
        or args.repeats < 1
        or any(not 0 <= s <= 1 for s in args.selections)
    ):
        parser.error("sizes >= 2, repeats >= 1 and selections in [0,1] are required")
    if args.worker:
        print(json.dumps(run_worker(args)))
        return
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "implementations": {},
    }
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for name in ("fft", "nufft"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--sizes",
            *map(str, args.sizes),
            "--selections",
            *map(str, args.selections),
            "--repeats",
            str(args.repeats),
        ]
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True, check=False
        )
        if completed.returncode:
            parser.exit(1, f"{name} worker failed: {completed.stderr.strip()}\n")
        report["implementations"][name] = json.loads(completed.stdout)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
