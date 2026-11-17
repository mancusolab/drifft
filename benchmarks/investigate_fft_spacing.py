"""Isolated FFT spacing experiment; does not change the production backend.

Run with .venv/bin/python benchmarks/investigate_fft_spacing.py --help.
The two patched host builders are scoped to construction in this single-process
research driver. Application, JAX AD, transposition, and error reporting are the
unmodified production implementation. This is not a public backend API.
"""

# Imperative Shell: experiment configuration, scoped prototype injection, timing,
# and JSON output. Pure geometry/reference calculations are local to this study.
import argparse
import importlib.metadata
import json
import math
import platform
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, logit

from drifft import FFT
from drifft._backend import fft as implementation

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def grid(
    M,
    cutoff,
    spacing,
    order,
    sensitivity,
    scale,
    *,
    strides=None,
    atol=1e-10,
    grad_atol=None,
):
    """Reuse the current Sobolev theorem with reciprocal spacing in place of V."""
    centers = []
    if M > 2 * cutoff:
        centers = [cutoff]
        while centers[-1] < M - cutoff:
            j = centers[-1]
            centers.append(
                min(M - cutoff, j + max(1, int(spacing * np.sqrt(j * (1 - j / M)))))
            )
    p = np.asarray(centers, dtype=float) / M
    eta, reciprocal = logit(p), np.asarray(centers, dtype=float) * (1 - p) / scale
    edges = (eta[:-1] + eta[1:]) / 2
    core = np.r_[logit(cutoff / M), edges, logit(1 - cutoff / M)] if centers else []
    intervals, bounds = [], np.zeros(3)
    for i in range(len(centers)):

        def candidate(stride, i=i):
            inverse = reciprocal[i] / stride
            lo = math.floor((core[i] - eta[i]) * inverse) - order // 2
            hi = math.ceil((core[i + 1] - eta[i]) * inverse) + order // 2
            bound = np.asarray(
                implementation._interpolation_bound(
                    M, order, float(eta[i]), float(inverse), lo, hi, bool(sensitivity)
                )
            )
            bound[1:] *= sensitivity
            return lo, hi, bound

        stride = 1
        lo, hi, bound = candidate(stride)
        if strides is not None:
            while stride < 64:
                next_lo, next_hi, next_bound = candidate(2 * stride)
                if next_bound[0] > atol / 4 or (
                    grad_atol is not None and next_bound[1] > grad_atol / 4
                ):
                    break
                stride *= 2
                lo, hi, bound = next_lo, next_hi, next_bound
            strides[centers[i]] = stride
        reciprocal[i] /= stride
        intervals.append((lo, hi))
        bounds = np.maximum(bounds, bound)
    return (centers, eta, reciprocal, edges, intervals), tuple(map(float, bounds))


def centered_cumulant(p, delta):
    """log(1-p+p*exp(delta)) - p*delta without first-order cancellation.

    Reflect to p <= 1/2, then combine expm1(delta)-delta with
    log1p(p*expm1(delta))-p*expm1(delta). Both terms start at order delta**2.
    Taylor tails in these small-argument branches are below binary64 rounding.
    """
    if p > 0.5:
        p, delta = 1 - p, -delta
    t = p * np.expm1(delta)
    small_exp = delta**2 * np.polynomial.polynomial.polyval(
        delta, [1 / math.factorial(k + 2) for k in range(26)]
    )
    exp_remainder = np.where(abs(delta) <= 1, small_exp, np.expm1(delta) - delta)
    small_log = t**2 * np.polynomial.polynomial.polyval(
        t, [(-1.0) ** (k + 1) / (k + 2) for k in range(64)]
    )
    log_remainder = np.where(abs(t) <= 0.5, small_log, np.log1p(t) - t)
    return p * exp_remainder + log_remainder


def block(
    M, j0, lo, hi, rlo, rhi, *, balance, scale, records, stable_factors=False, stride=1
):
    """Same exact DTD factorization and recurrence, with independent spacing."""
    p = np.float64(j0) / M
    eta = np.log(p) - np.log1p(-p)
    h = scale / (M * p * (1 - p))
    rlo, rhi = rlo * stride, rhi * stride
    r = np.arange(rlo, rhi + 1, dtype=float)
    delta = h * r
    loga = delta * j0 - M * np.log1p(p * np.expm1(delta)) + h * r**2 / 2
    left, right = min(lo, j0), max(hi, j0)
    j = np.arange(left, right, dtype=float)
    increments = np.log(M - j) - np.log(j + 1) + eta + h * (j - j0 + 0.5)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    logb = (cumulative - cumulative[j0 - left])[lo - left : hi - left + 1]
    if stable_factors:
        loga = -M * centered_cumulant(p, delta) + h * r**2 / 2
        x = j - j0
        increments = np.log1p(-x / (M - j0)) - np.log1p((x + 1) / j0) + h * (x + 0.5)
        center = j0 - left
        cumulative = np.r_[
            -np.cumsum(increments[:center][::-1])[::-1],
            0.0,
            np.cumsum(increments[center:]),
        ]
        logb = cumulative[lo - left : hi - left + 1]
    logB = (
        -0.5 * np.log(2 * np.pi * M * p * (1 - p))
        + implementation._stirling_correction(M)
        - implementation._stirling_correction(j0)
        - implementation._stirling_correction(M - j0)
    )
    shift = (max(logb) + min(logb)) / 2 if balance else 0.0
    a, b = np.exp(logB + loga + shift), np.exp(logb - shift)
    L, R = hi - lo + 1, rhi - rlo + 1
    nfft = 1 << (R + 2 * L - 3).bit_length()
    d = np.arange(rlo - (hi - j0), rhi - (lo - j0) + 1, dtype=float)
    kernel = np.exp(-h * d * d / 2)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    # Discrete binomial log curvature; include the entire retained interval.
    j = np.arange(max(1, lo), min(M - 1, hi) + 1, dtype=float)
    curvature = float(np.min(np.log1p(1 / j) + np.log1p(1 / (M - j))))
    middle = min(max(M // 2, max(1, lo)), min(M - 1, hi))
    curvature_lower = Fraction(1, middle + 1) + Fraction(1, M - middle + 1)
    amplification = (
        math.sqrt(nfft) * max(a) * np.linalg.norm(b) * np.linalg.norm(kernel)
    )
    qlo, qhi = expit(eta + h * np.array([rlo, rhi]))
    grad_scale = 4 / (stride * h * math.sqrt(min(qlo * (1 - qlo), qhi * (1 - qhi))))
    records.append(
        {
            "j0": j0,
            "h": float(h),
            "rows": R,
            "columns": L,
            "nfft": nfft,
            "curvature_ratio": float(h / curvature),
            "curvature_ratio_upper": implementation._upper(
                Fraction(float(h)) / curvature_lower
            ),
            "amplification": float(amplification),
            "gradient_amplification": float(grad_scale * amplification),
            "stride": stride,
        }
    )
    return implementation._FFTBlock(
        jnp.asarray(b),
        jnp.asarray(a[::stride]),
        jnp.asarray(np.fft.rfft(kernel, nfft)),
        nfft,
    )


def prepare(
    M, scale, cutoff, order, atol, grad_atol, stable_factors=False, separate_grids=False
):
    """Construct an isolated prototype using the production backend container."""
    records = []
    strides = {} if separate_grids else None

    def prepare_grid(M, cutoff, spacing, order, sensitivity=0):
        if strides is not None:
            strides.clear()
        return grid(
            M,
            cutoff,
            spacing,
            order,
            sensitivity,
            scale,
            strides=strides,
            atol=atol,
            grad_atol=grad_atol,
        )

    def prepare_block(*args, **kwargs):
        stride = 1 if strides is None else strides[args[1]]
        return block(
            *args,
            **kwargs,
            scale=scale,
            records=records,
            stable_factors=stable_factors,
            stride=stride,
        )

    start = time.perf_counter()
    with (
        patch.object(implementation, "_fft_grid", prepare_grid),
        patch.object(implementation, "_build_block", prepare_block),
    ):
        backend = FFT(
            M // 2,
            atol=atol,
            grad_atol=grad_atol,
            angular_sensitivity=None if grad_atol is None else 1.0,
            min_mode_count=cutoff,
            interp_order=order,
        )
    if separate_grids:
        updated = []
        for bucket in backend.buckets:
            # Production gathers adjacent convolution outputs. The prototype
            # instead retains every kth output, with the same coarse node API.
            steps = np.array(
                [r["stride"] for r in records if r["nfft"] == bucket.nfft]
            )[:, None]
            indices = np.asarray(bucket.conv_indices)
            offset = indices[:, :1]
            updated.append(
                jnp.asarray(
                    np.where(indices >= offset, offset + steps * (indices - offset), 0)
                )
            )
        backend = eqx.tree_at(
            lambda b: tuple(x.conv_indices for x in b.buckets), backend, tuple(updated)
        )
    jax.block_until_ready(backend)
    return backend, records, time.perf_counter() - start


def reference(M, probabilities):
    """60-digit independent normalized recurrence; exact binary64 input q.

    NumPy longdouble is only binary64 on the test machine. Decimal avoids
    mistaking reference PMF/log-factorial error for an FFT accuracy floor.
    The angular derivative is evaluated in high precision before conversion.
    """
    rows, grads = [], []
    with localcontext() as context:
        context.prec = 60
        for probability in probabilities:
            q = Decimal(float(probability))
            if not 0 < q < 1:
                row = np.zeros(M + 1)
                row[0 if q == 0 else M] = 1
                rows.append(row)
                grads.append(np.zeros(M + 1))
                continue
            odds = q / (1 - q)
            mode = int((M + 1) * q)
            weights = [Decimal(0)] * (M + 1)
            weights[mode] = Decimal(1)
            for j in range(mode, M):
                weights[j + 1] = weights[j] * (M - j) / ((j + 1) / odds)
            for j in range(mode, 0, -1):
                weights[j - 1] = weights[j] * j / ((M - j + 1) * odds)
            total = sum(weights)
            variance_root = (q * (1 - q)).sqrt()
            rows.append([float(w / total) for w in weights])
            grads.append(
                [
                    float(w / total * (j - M * q) / variance_root)
                    for j, w in enumerate(weights)
                ]
            )
    return np.asarray(rows), np.asarray(grads)


def queries(backend, count, *, common=False):
    """Probe boundaries, both sides of block switches, and generic interiors."""
    M = 2 * backend.pop_size
    if common:
        # Identical physical probes for both candidate architectures, including
        # their boundary and representative block interfaces.
        cuts = np.array([c / M for c in (16, 32, 64, 128) if 2 * c < M])
        interfaces = [*cuts, *(1 - cuts)]
        for cutoff in (32, 128):
            if M <= 2 * cutoff:
                continue
            centers = [cutoff]
            while centers[-1] < M - cutoff:
                j = centers[-1]
                centers.append(
                    min(M - cutoff, j + max(1, int(np.sqrt(j * (1 - j / M)))))
                )
            eta = logit(np.asarray(centers) / M)
            edges = expit((eta[:-1] + eta[1:]) / 2)
            interfaces.extend(edges[[0, len(edges) // 2, -1]])
        interfaces = np.array(interfaces)
        interior = expit(np.linspace(logit(1 / M), logit(1 - 1 / M), count // 2))
        return np.unique(
            np.r_[
                0.0,
                0.5,
                1.0,
                np.nextafter(interfaces, 0),
                interfaces,
                np.nextafter(interfaces, 1),
                interior,
            ]
        )
    cutoff = min(backend.min_mode_count / M, 0.5)
    edges = np.r_[cutoff, expit(np.asarray(backend.edges)), 1 - cutoff]
    selected = edges[np.unique(np.linspace(0, len(edges) - 1, count // 4, dtype=int))]
    interior = expit(
        np.linspace(
            logit(max(cutoff / 2, 1 / M)),
            logit(min(1 - cutoff / 2, 1 - 1 / M)),
            count // 2,
        )
    )
    return np.unique(
        np.r_[
            0.0,
            1.0,
            np.nextafter(selected, 0),
            selected,
            np.nextafter(selected, 1),
            interior,
        ]
    )


@eqx.filter_jit
def matrices(backend, q, values):
    """Extract full coefficient and angular-derivative rows by transposition."""

    def apply(values):
        primal, tangent = jax.jvp(
            lambda p: backend.expect(p, values), (q,), (jnp.sqrt(q * (1 - q)),)
        )
        return jnp.stack((primal, tangent))

    return jax.jacrev(apply)(values)


def timed(function, *args, repeats):
    """Synchronize compilation and every repeated application before timing."""
    start = time.perf_counter()
    actual = jax.block_until_ready(function(*args))
    first = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        actual = jax.block_until_ready(function(*args))
        samples.append(time.perf_counter() - start)
    return actual, {
        "first_seconds": first,
        "median_seconds": float(np.median(samples)),
        "samples_seconds": samples,
    }


@eqx.filter_jit
def expect(backend, q, values):
    """Compile the unchanged production expectation with dynamic geometry."""
    return backend.expect(q, values)


@eqx.filter_jit
def gradient(backend, q, values):
    """Return expectations and their angular JVPs."""
    return jax.jvp(lambda p: backend.expect(p, values), (q,), (jnp.sqrt(q * (1 - q)),))


@eqx.filter_jit
def advance(backend, q, values):
    """Compile the production transpose with dynamic geometry."""
    return backend.advance(q, values)


def experiment(M, scale, cutoff, order, args):
    """Measure a plan against independent rows, derivatives, and signed actions."""
    backend, records, planning = prepare(
        M,
        scale,
        cutoff,
        order,
        args.atol,
        args.grad_atol,
        args.stable_factors,
        args.separate_grids,
    )
    result = {
        "M": M,
        "scale": scale,
        "initial_cutoff": cutoff,
        "order": backend.interp_order,
        "cutoff": backend.min_mode_count,
        "direct_width": backend.boundary_width,
        "blocks": len(records),
        "nodes": backend.total_nodes,
        "planning_seconds": planning,
        "plan_bytes": sum(x.nbytes for x in jax.tree.leaves(backend)),
        "fft_work_proxy": sum(
            len(b.a) * b.nfft * math.log2(b.nfft) for b in backend.buckets
        ),
        "max_amplification": max((r["amplification"] for r in records), default=0),
        "max_gradient_amplification": max(
            (r["gradient_amplification"] for r in records), default=0
        ),
        "max_curvature_ratio": max((r["curvature_ratio"] for r in records), default=0),
        "max_curvature_ratio_upper": max(
            (r["curvature_ratio_upper"] for r in records), default=0
        ),
        "interpolation_bound": backend.interpolation_bound,
        "sampling_bound": backend.sampling_bound,
        "interpolation_grad_bound": backend.interpolation_grad_bound,
        "sampling_grad_bound": backend.sampling_grad_bound,
    }
    result["stride_range"] = [
        min((r["stride"] for r in records), default=1),
        max((r["stride"] for r in records), default=1),
    ]
    if not args.plan_only:
        q = queries(backend, args.query_count, common=args.common_queries)
        exact, exact_grad = reference(M, q)
        chunks = []
        for start in range(0, len(q), 16):
            part = q[start : start + 16]
            padded = np.pad(part, (0, 16 - len(part)), mode="edge")
            chunks.append(
                np.asarray(matrices(backend, jnp.asarray(padded), jnp.zeros(M + 1)))[
                    :, : len(part)
                ]
            )
        rows = np.concatenate(chunks, axis=1)
        errors = np.sum(abs(rows[0] - exact), axis=1)
        grad_errors = np.sum(abs(rows[1] - exact_grad), axis=1)
        result.update(
            {
                "query_count": len(q),
                "row_l1_error": float(max(errors)),
                "angular_gradient_row_l1_error": float(max(grad_errors)),
                "worst_value_q": float(q[np.argmax(errors)]),
                "worst_gradient_q": float(q[np.argmax(grad_errors)]),
                "max_mass_error": float(max(abs(rows[0].sum(axis=1) - 1))),
                "most_negative_entry": float(rows[0].min()),
            }
        )
        interior = (q > backend.min_mode_count / M) & (
            q < 1 - backend.min_mode_count / M
        )
        result["interior_row_l1_error"] = float(max(errors[interior], default=0))
        result["interior_angular_gradient_row_l1_error"] = float(
            max(grad_errors[interior], default=0)
        )
        # Accuracy of actual forward action, including worst-case sign vectors.
        rng = np.random.default_rng(20261113)
        signs = np.stack(
            [
                np.sign(rows[0][np.argmax(errors)] - exact[np.argmax(errors)]),
                np.sign(
                    rows[1][np.argmax(grad_errors)] - exact_grad[np.argmax(grad_errors)]
                ),
                rng.choice([-1.0, 1.0], size=M + 1),
            ]
        )
        outputs = np.asarray(
            eqx.filter_jit(jax.vmap(gradient, in_axes=(None, None, 0)))(
                backend, jnp.asarray(q), jnp.asarray(signs)
            )
        )
        result["signed_action_error"] = float(np.max(abs(outputs[0] - signs @ exact.T)))
        result["signed_gradient_action_error"] = float(
            np.max(abs(outputs[1] - signs @ exact_grad.T))
        )
        mass = rng.normal(size=len(q))
        mass /= np.sum(abs(mass))
        advanced = np.asarray(advance(backend, jnp.asarray(q), jnp.asarray(mass)))
        result["signed_advance_l1_error"] = float(
            np.sum(abs(advanced - exact.T @ mass))
        )
        # Uniform full population grid: identical application workload across plans.
        qfull = jnp.linspace(0, 1, M + 1)
        values = jnp.asarray(rng.choice([-1.0, 1.0], size=M + 1))
        for name, function, vector in [
            ("expect", expect, values),
            ("gradient", gradient, values),
            ("advance", advance, jnp.ones(M + 1) / (M + 1)),
        ]:
            _, result[name] = timed(
                function, backend, qfull, vector, repeats=args.repeats
            )
    return result


def main():
    """Run candidates serially and persist completed measurements after each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[128])
    parser.add_argument("--order", type=int)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--grad-atol", type=lambda s: None if s == "none" else float(s), default=1e-8
    )
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--stable-factors", action="store_true")
    parser.add_argument("--separate-grids", action="store_true")
    parser.add_argument("--common-queries", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(M < 2 or M % 2 for M in args.sizes):
        parser.error("sizes must be even gene-copy counts >= 2")
    if any(not math.isfinite(s) or not 0 < s <= 1 for s in args.scales):
        parser.error("refinement scales must be finite and in (0, 1]")
    if min(args.cutoffs) < 8 or args.query_count < 8 or args.repeats < 1:
        parser.error("cutoffs >= 8, query-count >= 8, and repeats >= 1 are required")
    report = {
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["jax", "jaxlib", "equinox", "numpy", "scipy"]
        },
        "config": vars(args) | {"output": str(args.output)},
        "runs": [],
    }
    for M in args.sizes:
        for cutoff in args.cutoffs:
            for scale in args.scales:
                try:
                    run = experiment(M, scale, cutoff, args.order, args)
                except ValueError as error:
                    run = {
                        "M": M,
                        "scale": scale,
                        "initial_cutoff": cutoff,
                        "error": str(error),
                    }
                report["runs"].append(run)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in run.items()
                            if k not in ("expect", "gradient", "advance")
                        }
                    ),
                    flush=True,
                )
                jax.clear_caches()


if __name__ == "__main__":
    main()
