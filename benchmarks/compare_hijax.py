"""Measure HiJAX derivative workloads against an explicitly selected source tree.

Run each revision in a fresh process with the same interpreter. The JSON records
lowering, compilation, synchronized warm execution, and compiler memory estimates;
the companion NPZ retains outputs for cross-revision numerical comparisons.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def main():
    """Benchmark one source checkout without modifying or installing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))

    import equinox
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.special import gammaln

    import drifft
    from drifft._backend._spectral import _transpose_evaluate, evaluate_cosine
    from drifft._backend.fft import _binomial_weights
    from drifft._polynomial import gegenbauer_moments

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    q = jnp.linspace(0.0, 1.0, 2049)
    mass = jnp.cos(jnp.arange(q.size) / 17) / q.size
    coefficients = jnp.sin(jnp.arange(128)) / (1 + jnp.arange(128)) ** 2
    cases = []
    for index, label in ((0, "points"), (1, "values")):

        def moment_jvp(z, v, index=index):
            operands = (z, v)
            return jax.jvp(
                lambda x: gegenbauer_moments(
                    x if index == 0 else z, x if index == 1 else v, rank=128
                ),
                (operands[index],),
                (jnp.ones_like(operands[index]),),
            )

        cases.append((f"moments_jvp_{label}", moment_jvp, (2 * q - 1, mass)))
    for index, label in ((0, "coefficients"), (1, "probabilities")):

        def cosine_jvp(c, q, index=index):
            operands = (c, q)
            return jax.jvp(
                lambda x: evaluate_cosine(
                    x if index == 0 else c, x if index == 1 else q, 1e-12
                ),
                (operands[index],),
                (jnp.ones_like(operands[index]),),
            )

        cases.append((f"cosine_jvp_{label}", cosine_jvp, (coefficients, q)))

    def transpose_jvp(q, m):
        return jax.jvp(lambda x: _transpose_evaluate(q, x, 128, 1e-12), (m,), (m,))

    cases.append(("cosine_transpose_jvp_mass", transpose_jvp, (q, mass)))
    k = jnp.arange(129)
    logchoose = gammaln(129.0) - gammaln(k + 1.0) - gammaln(129.0 - k)

    def binomial_jvp(logchoose, q):
        return jax.jvp(
            lambda logs: _binomial_weights(128, k, logs, q),
            (logchoose,),
            (jnp.ones_like(logchoose),),
        )

    cases.append(("binomial_jvp_logchoose", binomial_jvp, (logchoose, q)))
    for name in ("FFT", "NUFFT", "Jacobi"):
        backend = getattr(drifft, name)(512)
        values = jnp.sin(jnp.arange(1025.0) / 17)

        def expectation_loss(q, v, backend=backend):
            return jnp.mean(backend.expect(q, v))

        cases.append(
            (f"{name}_gradient_values", jax.grad(expectation_loss, 1), (q, values))
        )
        initial = jnp.ones(1025) / 1025

        def epoch_loss(strength, initial, reward, backend=backend):
            transition = drifft.Transition(backend, drifft.Underdominance(strength))
            final = jax.lax.fori_loop(0, 4, lambda _, p: transition.advance(p), initial)
            return final @ reward

        cases.append(
            (
                f"{name}_epoch_selection_grad",
                jax.value_and_grad(epoch_loss),
                (jnp.asarray(0.02), initial, values),
            )
        )

    records, outputs = {}, {}
    for name, fn, operands in cases:
        jax.block_until_ready(operands)
        started = time.perf_counter()
        lowered = jax.jit(fn).lower(*operands)
        lowered_s = time.perf_counter() - started
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter() - started
        for _ in range(5):
            result = jax.block_until_ready(compiled(*operands))
        durations = []
        for _ in range(args.samples):
            started = time.perf_counter()
            result = jax.block_until_ready(compiled(*operands))
            durations.append(time.perf_counter() - started)
        for i, leaf in enumerate(jax.tree.leaves(result)):
            outputs[f"{name}_{i}"] = np.asarray(leaf)
        memory = compiled.memory_analysis()
        records[name] = {
            "lower_s": lowered_s,
            "compile_s": compile_s,
            "warm_median_s": statistics.median(durations),
            "warm_samples_s": durations,
            "memory_bytes": None
            if memory is None
            else {
                field: getattr(memory, field)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        }
        print(
            f"{args.label} {name}: {statistics.median(durations) * 1e3:.3f} ms",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "source": str(args.source.resolve()),
                "imported": drifft.__file__,
                "jax": jax.__version__,
                "equinox": equinox.__version__,
                "platform": platform.platform(),
                "devices": str(jax.devices()),
                "samples": args.samples,
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.output.with_suffix(".npz"), **outputs)


if __name__ == "__main__":
    main()
