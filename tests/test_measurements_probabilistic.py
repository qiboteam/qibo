"""Test circuit measurements when outcome is probabilistic."""

import numpy as np
import pytest

from qibo import Circuit, gates

from .test_measurements import assert_result


@pytest.mark.parametrize("use_samples", [True, False])
def test_probabilistic_measurement(backend, accelerators, use_samples):
    # set single-thread to fix the random values generated from the frequency custom op
    backend.set_threads(1)
    circuit = Circuit(4, accelerators)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=1000)

    backend.set_seed(1234)
    if use_samples:
        # calculates sample tensor directly using `tf.random.categorical`
        # otherwise it uses the frequency-only calculation
        _ = result.samples()

    # update reference values based on backend and device
    decimal_frequencies = (
        {2: 269, 0: 264, 1: 235, 3: 232}
        if backend.platform in ("cupy", "cuquantum")
        else {0: 249, 1: 231, 2: 253, 3: 267}
    )
    assert sum(result.frequencies().values()) == 1000
    assert_result(backend, result, decimal_frequencies=decimal_frequencies)


def test_sample_frequency_agreement(backend):
    # set single-thread to fix the random values generated from the frequency custom op
    backend.set_threads(1)
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=1000)

    backend.set_seed(1234)
    target_frequencies = result.frequencies(binary=False)
    samples = result.samples(binary=False)
    outcomes, counts = np.unique(samples, return_counts=True)
    outcomes = backend.to_numpy(outcomes)
    counts = backend.to_numpy(counts)
    frequencies = dict(zip(outcomes, counts))
    assert frequencies == target_frequencies


@pytest.mark.parametrize("use_samples", [True, False])
def test_unbalanced_probabilistic_measurement(backend, use_samples):
    # set single-thread to fix the random values generated from the frequency custom op
    backend.set_threads(1)
    state = np.array([1, 1, 1, np.sqrt(3)]) / np.sqrt(6)
    circuit = Circuit(2)
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, initial_state=np.copy(state), nshots=1000)

    backend.set_seed(1234)
    if use_samples:
        # calculates sample tensor directly using `tf.random.categorical`
        # otherwise it uses the frequency-only calculation
        _ = result.samples()
    # update reference values based on backend and device
    decimal_frequencies = (
        {3: 509, 0: 170, 2: 167, 1: 154}
        if backend.platform in ("cupy", "cuquantum")
        else {0: 171, 1: 148, 2: 161, 3: 520}
    )
    assert sum(result.frequencies().values()) == 1000
    assert_result(backend, result, decimal_frequencies=decimal_frequencies)


def test_measurements_with_probabilistic_noise(backend):
    """Check measurements when simulating noise with repeated execution."""
    thetas = np.random.random(5)
    circuit = Circuit(5)
    circuit.add((gates.RX(i, t) for i, t in enumerate(thetas)))
    circuit.add(
        gates.PauliNoiseChannel(i, list(zip(["Y", "Z"], [0.2, 0.4]))) for i in range(5)
    )
    circuit.add(gates.M(*range(5)))
    backend.set_seed(123)
    result = backend.execute_circuit(circuit, nshots=20)
    samples = result.samples()

    backend.set_seed(123)
    target_samples = []
    channel_gates = [gates.Y, gates.Z]
    probs = [0.2, 0.4, 0.4]
    for _ in range(20):
        noiseless_circuit = Circuit(5)
        noiseless_circuit.add((gates.RX(i, t) for i, t in enumerate(thetas)))
        for i in range(5):
            index = int(backend.sample_shots(probs, 1)[0])
            if index != len(channel_gates):
                noiseless_circuit.add(channel_gates[index](i))
        noiseless_circuit.add(gates.M(*range(5)))
        result = backend.execute_circuit(noiseless_circuit, nshots=1)
        target_samples.append(backend.to_numpy(result.samples()))
    target_samples = np.concatenate(target_samples, axis=0)
    backend.assert_allclose(samples, target_samples)


@pytest.mark.parametrize(
    "i,probs", [(0, [0.0, 0.0, 0.0]), (1, [0.1, 0.3, 0.2]), (2, [0.5, 0.5, 0.5])]
)
def test_post_measurement_bitflips_on_circuit(backend, accelerators, i, probs):
    """Check bitflip errors on circuit measurements."""
    backend.set_seed(123)
    circuit = Circuit(5, accelerators=accelerators)
    circuit.add([gates.X(0), gates.X(2), gates.X(3)])
    circuit.add(gates.M(0, 1, p0={0: probs[0], 1: probs[1]}))
    circuit.add(gates.M(3, p0=probs[2]))
    result = backend.execute_circuit(circuit, nshots=30)
    freqs = result.frequencies(binary=False)
    targets = (
        [
            {5: 30},
            {5: 17, 4: 5, 7: 4, 1: 2, 6: 2},
            {4: 9, 2: 5, 5: 5, 3: 4, 6: 4, 0: 1, 1: 1, 7: 1},
        ]
        if backend.platform in ("cupy", "cuquantum")
        else [
            {5: 30},
            {5: 18, 4: 5, 7: 4, 1: 2, 6: 1},
            {4: 8, 2: 6, 5: 5, 1: 3, 3: 3, 6: 2, 7: 2, 0: 1},
        ]
    )
    assert freqs == targets[i]


def test_post_measurement_bitflips_on_circuit_result(backend):
    """Check bitflip errors on ``CircuitResult`` objects."""
    thetas = np.random.random(4)
    backend.set_seed(123)
    circuit = Circuit(4)
    circuit.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
    circuit.add(gates.M(0, 1, register_name="a", p0={0: 0.2, 1: 0.4}))
    circuit.add(gates.M(3, register_name="b", p0=0.3))
    result = backend.execute_circuit(circuit, nshots=30)
    samples = result.samples(binary=True)
    register_samples = result.samples(binary=True, registers=True)
    backend.assert_allclose(register_samples["a"], samples[:, :2])
    backend.assert_allclose(register_samples["b"], samples[:, 2:])


@pytest.mark.parametrize(
    "i,p0,p1",
    [
        (0, 0.2, None),
        (1, 0.2, 0.1),
        (2, (0.1, 0.0, 0.2), None),
        (3, {0: 0.2, 1: 0.1, 2: 0.0}, None),
    ],
)
def test_measurementresult_apply_bitflips(backend, i, p0, p1):
    from qibo.result import CircuitResult

    circuit = Circuit(3)
    circuit.add(gates.M(*range(3)))
    state = backend.zero_state(8)
    result = CircuitResult(state, circuit.measurements, backend)
    result._samples = backend.cast(np.zeros((10, 3)), dtype="int32")
    backend.set_seed(123)
    noisy_samples = result.apply_bitflips(p0, p1)
    targets = [
        [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    ]
    noisy_samples = backend.samples_to_decimal(noisy_samples, 3)
    backend.assert_allclose(noisy_samples, targets[i])
