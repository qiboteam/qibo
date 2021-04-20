"""Test :class:`qibo.abstractions.gates.M` when results are probabilistic."""
import sys
import pytest
import numpy as np
import qibo
from qibo import models, gates, K
from qibo.tests_new.test_measurement_gate import assert_result


@pytest.mark.parametrize("use_samples", [True, False])
def test_probabilistic_measurement(backend, accelerators, use_samples):
    original_backend = qibo.get_backend()
    original_threads = qibo.get_threads()
    qibo.set_backend(backend)
    # set single-thread to fix the random values generated from the frequency custom op
    qibo.set_threads(1)
    c = models.Circuit(4, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    K.set_seed(1234)
    if use_samples:
        # calculates sample tensor directly using `tf.random.categorical`
        # otherwise it uses the frequency-only calculation
        _ = result.samples()

    # update reference values based on backend and device
    if "numpy" in K.name:
        decimal_frequencies = {0: 249, 1: 231, 2: 253, 3: 267}
    else:
        if K.gpu_devices: # pragma: no cover
            # CI does not use GPU
            decimal_frequencies = {0: 273, 1: 233, 2: 242, 3: 252}
        else:
            decimal_frequencies = {0: 271, 1: 239, 2: 242, 3: 248}
    assert sum(result.frequencies().values()) == 1000
    assert_result(result, decimal_frequencies=decimal_frequencies)
    qibo.set_backend(original_backend)
    qibo.set_threads(original_threads)


@pytest.mark.parametrize("use_samples", [True, False])
def test_unbalanced_probabilistic_measurement(backend, use_samples):
    original_backend = qibo.get_backend()
    original_threads = qibo.get_threads()
    qibo.set_backend(backend)
    # set single-thread to fix the random values generated from the frequency custom op
    qibo.set_threads(1)
    state = np.array([1, 1, 1, np.sqrt(3)]) / np.sqrt(6)
    c = models.Circuit(2)
    c.add(gates.Flatten(state))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    K.set_seed(1234)
    if use_samples:
        # calculates sample tensor directly using `tf.random.categorical`
        # otherwise it uses the frequency-only calculation
        _ = result.samples()
    # update reference values based on backend and device
    if "numpy" in K.name:
        decimal_frequencies = {0: 171, 1: 148, 2: 161, 3: 520}
    else:
        if K.gpu_devices: # pragma: no cover
            # CI does not use GPU
            decimal_frequencies = {0: 196, 1: 153, 2: 156, 3: 495}
        else:
            decimal_frequencies = {0: 168, 1: 188, 2: 154, 3: 490}
    assert sum(result.frequencies().values()) == 1000
    assert_result(result, decimal_frequencies=decimal_frequencies)
    qibo.set_backend(original_backend)
    qibo.set_threads(original_threads)


def test_measurements_with_probabilistic_noise(backend):
    """Check measurements when simulating noise with repeated execution."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.random.random(5)
    c = models.Circuit(5)
    c.add((gates.RX(i, t) for i, t in enumerate(thetas)))
    c.add((gates.PauliNoiseChannel(i, px=0.0, py=0.2, pz=0.4, seed=123)
           for i in range(5)))
    c.add(gates.M(*range(5)))
    K.set_seed(123)
    samples = c(nshots=20).samples()

    np.random.seed(123)
    K.set_seed(123)
    target_samples = []
    for _ in range(20):
        noiseless_c = models.Circuit(5)
        noiseless_c.add((gates.RX(i, t) for i, t in enumerate(thetas)))
        for i in range(5):
            if np.random.random() < 0.2:
                noiseless_c.add(gates.Y(i))
            if np.random.random() < 0.4:
                noiseless_c.add(gates.Z(i))
        noiseless_c.add(gates.M(*range(5)))
        target_samples.append(noiseless_c(nshots=1).samples())
    target_samples = np.concatenate(target_samples, axis=0)
    np.testing.assert_allclose(samples, target_samples)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("i,probs", [(0, [0.0, 0.0, 0.0]),
                                     (1, [0.1, 0.3, 0.2]),
                                     (2, [0.5, 0.5, 0.5])])
def test_post_measurement_bitflips_on_circuit(backend, accelerators, i, probs):
    """Check bitflip errors on circuit measurements."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    K.set_seed(123)
    c = models.Circuit(5, accelerators=accelerators)
    c.add([gates.X(0), gates.X(2), gates.X(3)])
    c.add(gates.M(0, 1, p0={0: probs[0], 1: probs[1]}))
    c.add(gates.M(3, p0=probs[2]))
    result = c(nshots=30).frequencies(binary=False)
    if "numpy" in K.name:
        targets = [{5: 30}, {5: 18, 4: 5, 7: 4, 1: 2, 6: 1},
                   {4: 8, 2: 6, 5: 5, 1: 3, 3: 3, 6: 2, 7: 2, 0: 1}]
    else:
        targets = [{5: 30}, {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                   {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2}]
    assert result == targets[i]
    qibo.set_backend(original_backend)


def test_post_measurement_bitflips_on_circuit_result(backend):
    """Check bitflip errors on ``CircuitResult`` objects."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.random.random(4)
    c = models.Circuit(4)
    c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
    c.add(gates.M(0, 1, register_name="a"))
    c.add(gates.M(3, register_name="b"))
    result = c(nshots=30)
    samples = result.samples()
    K.set_seed(123)
    noisy_result = result.copy().apply_bitflips({0: 0.2, 1: 0.4, 3: 0.3})
    noisy_samples = noisy_result.samples(binary=True)
    register_samples = noisy_result.samples(binary=True, registers=True)

    K.set_seed(123)
    sprobs = np.array(K.random_uniform(samples.shape))
    flipper = sprobs < np.array([0.2, 0.4, 0.3])
    target_samples = (samples + flipper) % 2
    np.testing.assert_allclose(noisy_samples, target_samples)
    # Check register samples
    np.testing.assert_allclose(register_samples["a"], target_samples[:, :2])
    np.testing.assert_allclose(register_samples["b"], target_samples[:, 2:])
    qibo.set_backend(original_backend)
