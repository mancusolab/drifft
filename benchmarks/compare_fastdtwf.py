"""Compare CPU distribution propagation and expected visits with fastDTWF."""

# Imperative Shell: backend adapters, synchronized timing, and JSON reporting.
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

# NumPy/SciPy are confined to independent dense reference calculations and the
# serialized boundary between isolated workers; conversions are not timed.
import numpy as np
from scipy.stats import binom

LOGGER = logging.getLogger(__name__)


def timed_call(
    call: Callable[[], Any], repeats: int, *, synchronize: bool = False
) -> tuple[Any, dict[str, float]]:
    """Report the first call separately from repeated, completed executions."""

    def evaluate():
        value = call()
        if synchronize:
            import jax

            return jax.block_until_ready(value)
        return value

    start = time.perf_counter()
    value = evaluate()
    first = time.perf_counter() - start
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = evaluate()
        durations.append(time.perf_counter() - start)
    return value, {
        "first_call_seconds": first,
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
    }


def loaded_backends() -> dict[str, bool]:
    """Expose actual imports so isolation can be checked in the report."""
    return {
        name: name in sys.modules for name in ("drifft", "jax", "torch", "fastDTWF")
    }


def drifft_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Measure drifft, exporting exact seeded inputs for the fastDTWF worker."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import drifft

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    records = []
    for gene_copies in args.sizes:
        start = time.perf_counter()
        plan = drifft.NUFFT(gene_copies, tail_tol=args.tail_tol)
        jax.block_until_ready(plan)
        planning_seconds = time.perf_counter() - start
        plan_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(plan))
        random_mass = jax.random.uniform(
            jax.random.fold_in(jax.random.key(args.seed), gene_copies),
            (gene_copies + 1,),
            dtype=jnp.float64,
        )
        inputs = {
            "singleton": jnp.zeros(gene_copies + 1, dtype=jnp.float64).at[1].set(1),
            "uniform": jnp.ones(gene_copies + 1, dtype=jnp.float64) / (gene_copies + 1),
            "random": random_mass / random_mass.sum(),
        }
        jax.block_until_ready(inputs)
        advance = eqx.filter_jit(drifft.Transition.advance)
        for selection in args.selections:
            scalar = jnp.asarray(selection, dtype=jnp.float64)
            transition = drifft.Transition(plan, drifft.Underdominance(scalar))
            probabilities = transition.probabilities
            # Copies are outside both timed kernels. fastDTWF reuses these
            # probabilities; drifft recomputes them inside Transition.advance,
            # a conservative timing asymmetry for drifft.
            host_probabilities = np.asarray(probabilities)
            drifft_transitions = {}
            for name, mass in inputs.items():
                value, timing = timed_call(
                    partial(advance, transition, mass),
                    args.repeats,
                    synchronize=True,
                )
                drifft_transitions[name] = (np.asarray(value).tolist(), timing)
            drifft_solve = None
            if gene_copies <= args.solve_max_m:
                result, timing = timed_call(
                    partial(transition.expected_visits, max_steps=args.max_steps),
                    args.repeats,
                    synchronize=True,
                )
                drifft_solve = (
                    np.asarray(result.visits).tolist(),
                    timing
                    | {
                        "converged": bool(result.converged),
                        "status": str(result.status),
                        "operator_residual_inf": float(result.residual_norm),
                    },
                )
            records.append(
                {
                    "gene_copies": gene_copies,
                    "selection": selection,
                    "planning_seconds": planning_seconds,
                    "plan_bytes": plan_bytes,
                    "inputs": {
                        name: np.asarray(mass).tolist() for name, mass in inputs.items()
                    },
                    "probabilities": host_probabilities.tolist(),
                    "transitions": drifft_transitions,
                    "solve": drifft_solve,
                }
            )
        jax.clear_caches()
    return {"records": records, "loaded_backends": loaded_backends()}


def fastdtwf_worker(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Measure identical no-mutation models on CPU, then persist their diagnostics."""
    start = time.perf_counter()
    try:
        torch = importlib.import_module("torch")
        # Upstream annotates dtype aliases (DoubleTensor) as concrete classes;
        # modern torch factories return Tensor. Keep that typing mismatch here.
        fastdtwf = cast(Any, importlib.import_module("fastDTWF"))
    except ImportError as error:
        raise RuntimeError(
            "Install benchmark dependencies with `uv sync --group fastdtwf` "
            "and run with `uv run --group fastdtwf python benchmarks/compare_fastdtwf.py`."
        ) from error
    import_seconds = time.perf_counter() - start
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    report: dict[str, Any] = {
        "schema_version": 1,
        "drifft_backend": "chebyshev_nufft",
        "worker_backends": {
            "drifft": payload["loaded_backends"],
            "fastdtwf": loaded_backends(),
        },
        "device": "cpu",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "drifft",
                "jax",
                "jaxlib",
                "jax-finufft",
                "fastdtwf",
                "torch",
                "numba",
                "numpy",
                "scipy",
            )
        },
        "fastdtwf_source": json.loads(
            importlib.metadata.distribution("fastdtwf").read_text("direct_url.json")
            or "null"
        ),
        "fastdtwf_import_seconds": import_seconds,
        "torch_threads": torch.get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "runs": [],
    }
    for record in payload["records"]:
        gene_copies = record["gene_copies"]
        selection = record["selection"]
        planning_seconds = record["planning_seconds"]
        plan_bytes = record["plan_bytes"]
        inputs = {name: np.asarray(mass) for name, mass in record["inputs"].items()}
        host_probabilities = np.asarray(record["probabilities"])
        torch_probabilities = torch.tensor(host_probabilities, dtype=torch.float64)
        torch_inputs = {
            name: torch.tensor(mass, dtype=torch.float64)
            for name, mass in inputs.items()
        }
        drifft_transitions = {
            name: (np.asarray(value), timing)
            for name, (value, timing) in record["transitions"].items()
        }
        drifft_solve = record["solve"]
        if drifft_solve is not None:
            drifft_solve = (np.asarray(drifft_solve[0]), drifft_solve[1])
        exact = None
        if gene_copies <= args.dense_max_m:
            exact = binom.pmf(
                np.arange(gene_copies + 1)[None, :],
                gene_copies,
                host_probabilities[:, None],
            )
        for tv_sd in args.tv_sds:
            start = time.perf_counter()
            index_sets = fastdtwf.coarse_grain(
                torch_probabilities, gene_copies, tv_sd, no_fix=False, sfs=False
            )
            torch_counts = torch.arange(gene_copies + 1, dtype=torch.float64)
            # The public mat_multiply API accepts precomputed log binomial
            # coefficients. Prepare them outside timed calls, like drifft's plan.
            bcoefs = (
                torch.lgamma(torch.tensor(gene_copies + 1.0, dtype=torch.float64))
                - torch.lgamma(torch_counts + 1)
                - torch.lgamma(gene_copies - torch_counts + 1)
            )
            setup_seconds = time.perf_counter() - start
            measurement: dict[str, Any] = {
                "gene_copies": gene_copies,
                "selection_strength": selection,
                "tv_sd": tv_sd,
                "row_eps": args.row_eps,
                "drifft_plan_seconds": planning_seconds,
                "drifft_plan_bytes": plan_bytes,
                "fastdtwf_setup_seconds": setup_seconds,
                "fastdtwf_group_count": int(index_sets.max()) + 1,
                "transitions": [],
            }
            with torch.no_grad():
                for name, mass in inputs.items():
                    other, timing = timed_call(
                        partial(
                            fastdtwf.mat_multiply,
                            torch_inputs[name],
                            index_sets,
                            torch_probabilities,
                            gene_copies,
                            args.row_eps,
                            no_fix=False,
                            sfs=False,
                            bcoefs=bcoefs,
                        ),
                        args.repeats,
                    )
                    other = other.numpy()
                    ours, our_timing = drifft_transitions[name]
                    comparison: dict[str, Any] = {
                        "input": name,
                        "between_methods_l1": float(np.abs(ours - other).sum()),
                    }
                    for backend, value, metrics in (
                        ("drifft", ours, our_timing),
                        ("fastdtwf", other, timing),
                    ):
                        comparison[backend] = metrics | {
                            "mass": float(value.sum()),
                            "minimum": float(value.min()),
                        }
                        if exact is not None:
                            reference = np.asarray(mass) @ exact
                            comparison[backend]["exact_l1_error"] = float(
                                np.abs(value - reference).sum()
                            )
                    measurement["transitions"].append(comparison)
                if drifft_solve is not None:
                    start = time.perf_counter()
                    sfs_sets = fastdtwf.coarse_grain(
                        torch_probabilities,
                        gene_copies,
                        tv_sd,
                        no_fix=False,
                        sfs=True,
                    )
                    measurement["fastdtwf_sfs_setup_seconds"] = (
                        time.perf_counter() - start
                    )
                    other, timing = timed_call(
                        partial(
                            fastdtwf.equilibrium_solve,
                            sfs_sets,
                            torch_probabilities,
                            gene_copies,
                            args.row_eps,
                            no_fix=False,
                            sfs=True,
                            injection_rate=1.0 / gene_copies,
                        ),
                        args.repeats,
                    )
                    ours, our_timing = drifft_solve
                    other = other.numpy()[1:-1]
                    solves: dict[str, Any] = {
                        "between_methods_max_error": float(np.abs(ours - other).max())
                    }
                    reference_visits = None
                    if exact is not None:
                        system = np.eye(gene_copies - 1) - exact[1:-1, 1:-1].T
                        rhs = np.zeros(gene_copies - 1)
                        rhs[0] = 1
                        reference_visits = np.linalg.solve(system, rhs)
                    for backend, value, metrics in (
                        ("drifft", ours, our_timing),
                        ("fastdtwf", other, timing),
                    ):
                        solves[backend] = metrics | {
                            "visits_sum": float(value.sum()),
                            "finite": bool(np.isfinite(value).all()),
                        }
                        if reference_visits is not None:
                            solves[backend]["exact_max_error"] = float(
                                np.abs(value - reference_visits).max()
                            )
                            solves[backend]["exact_residual_inf"] = float(
                                np.abs(system @ value - rhs).max()
                            )
                    measurement["expected_visits"] = solves
            report["runs"].append(measurement)
            LOGGER.info(
                "gene_copies=%d selection=%g tv_sd=%g completed",
                gene_copies,
                selection,
                tv_sd,
            )
    report["worker_backends"]["fastdtwf"] = loaded_backends()
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def run(args: argparse.Namespace) -> None:
    """Run disjoint numerical backends; serialization is outside kernel timings."""
    config = vars(args) | {"output": None}
    config.pop("worker", None)
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    payload: dict[str, Any] = {"config": config}
    for backend in ("drifft", "fastdtwf"):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                *(["--quiet"] if args.quiet else []),
            ],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if result.returncode:
            raise RuntimeError(
                f"{backend} worker failed ({result.returncode}): {result.stderr.strip()}"
            )
        if result.stderr:
            sys.stderr.write(result.stderr)
        data = json.loads(result.stdout)
        payload["drifft"] = data
    data["config"] = vars(args) | {"output": str(args.output) if args.output else None}
    data["config"].pop("worker", None)
    encoded = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate settings before importing optional dependencies; emit JSON or exit 1/2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--selections", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--tv-sds", type=float, nargs="+", default=[0.1, 0.01])
    parser.add_argument("--row-eps", type=float, default=1e-8)
    parser.add_argument("--tail-tol", type=float, default=1e-12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense-max-m", type=int, default=1024)
    parser.add_argument("--solve-max-m", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--threads", type=int, help="PyTorch CPU threads (default: backend default)"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--worker", choices=["drifft", "fastdtwf"], help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if min(args.sizes) < 2 or args.repeats < 1 or args.max_steps < 1:
        parser.error("sizes must be >= 2; repeats and max-steps must be positive")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in args.selections):
        parser.error("selections must be finite and in [0, 1]")
    if any(not math.isfinite(sd) or sd <= 0 for sd in args.tv_sds):
        parser.error("tv-sds must be finite and positive")
    if not math.isfinite(args.row_eps) or not 0 < args.row_eps < 1:
        parser.error("row-eps must be finite and in (0, 1)")
    if not math.isfinite(args.tail_tol) or not 1e-15 <= args.tail_tol < 0.1:
        parser.error("tail-tol must be finite and in [1e-15, 0.1)")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if min(args.dense_max_m, args.solve_max_m, args.seed) < 0:
        parser.error("dense-max-m, solve-max-m, and seed must be nonnegative")
    if args.output and (not args.output.parent.is_dir() or args.output.is_dir()):
        parser.error("output must name a file in an existing directory")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )
    try:
        if args.worker:
            payload = json.load(sys.stdin)
            worker_args = argparse.Namespace(**payload["config"])
            if args.worker == "drifft":
                sys.stdout.write(json.dumps(drifft_worker(worker_args)) + "\n")
            else:
                fastdtwf_worker(worker_args, payload["drifft"])
        else:
            run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error("error: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
