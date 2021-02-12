import collections
import numpy as np
import pytest
from qibo import gates, models
from typing import Optional


def test_probabilistic_measurement(accelerators):
    import tensorflow as tf
    tf.random.set_seed(1234)

    c = models.Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    # update reference values based on device
    if tf.config.list_physical_devices("GPU") and not accelerators: # pragma: no cover
        # case not tested in GitHub workflows because it requires GPU
        decimal_freqs = {0: 273, 1: 233, 2: 242, 3: 252}
        binary_freqs = {"00": 273, "01": 233, "10": 242, "11": 252}
    else:
        decimal_freqs = {0: 271, 1: 239, 2: 242, 3: 248}
        binary_freqs = {"00": 271, "01": 239, "10": 242, "11": 248}
    assert sum(binary_freqs.values()) == 1000
    assert_results(result,
                   decimal_frequencies=decimal_freqs,
                   binary_frequencies=binary_freqs)

def test_unbalanced_probabilistic_measurement():
    import tensorflow as tf
    tf.random.set_seed(1234)

    state = np.array([1, 1, 1, np.sqrt(3)]) / 2.0
    c = models.Circuit(2)
    c.add(gates.Flatten(state))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    # update reference values based on device
    if tf.config.list_physical_devices("GPU"): # pragma: no cover
        # case not tested in GitHub workflows because it requires GPU
        decimal_freqs = {0: 196, 1: 153, 2: 156, 3: 495}
        binary_freqs = {"00": 196, "01": 153, "10": 156, "11": 495}
    else:
        decimal_freqs = {0: 168, 1: 188, 2: 154, 3: 490}
        binary_freqs = {"00": 168, "01": 188, "10": 154, "11": 490}
    assert sum(binary_freqs.values()) == 1000
    assert_results(result,
                   decimal_frequencies=decimal_freqs,
                   binary_frequencies=binary_freqs)


def test_measurements_with_probabilistic_noise():
    """Check measurements when simulating noise with repeated execution."""
    import tensorflow as tf
    thetas = np.random.random(5)
    c = models.Circuit(5)
    c.add((gates.RX(i, t) for i, t in enumerate(thetas)))
    c.add((gates.PauliNoiseChannel(i, px=0.0, py=0.2, pz=0.4, seed=123)
           for i in range(5)))
    c.add(gates.M(*range(5)))
    tf.random.set_seed(123)
    result = c(nshots=20)

    np.random.seed(123)
    tf.random.set_seed(123)
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
    target_samples = tf.concat(target_samples, axis=0)
    np.testing.assert_allclose(result.samples(), target_samples)


@pytest.mark.parametrize("probs", [0.2, {0: 0.1, 1: 0.2, 2: 0.8, 3: 0.3}])
def test_post_measurement_bitflips(probs):
    """Check applying bitflips to measurement samples."""
    import tensorflow as tf
    from qibo import K
    from qibo.core import measurements
    qubits = tuple(range(4))
    samples = np.random.randint(0, 2, (20, 4))
    result = measurements.GateResult(qubits, binary_samples=samples)
    tf.random.set_seed(123)
    noisy_result = result.apply_bitflips(probs)

    tf.random.set_seed(123)
    if isinstance(probs, dict):
        probs = np.array([probs[q] for q in qubits])
    sprobs = tf.random.uniform(samples.shape, dtype=K.dtypes('DTYPE')).numpy()
    flipper = sprobs < probs
    target_samples = (samples + flipper) % 2
    np.testing.assert_allclose(noisy_result.samples(), target_samples)


def test_post_measurement_asymmetric_bitflips():
    """Check applying asymmetric bitflips to measurement samples."""
    import tensorflow as tf
    from qibo import K
    from qibo.core import measurements
    qubits = tuple(range(4))
    samples = np.random.randint(0, 2, (20, 4))
    result = measurements.GateResult(qubits, binary_samples=samples)
    p1_map = {0: 0.2, 1: 0.0, 2: 0.0, 3: 0.1}
    tf.random.set_seed(123)
    noisy_result = result.apply_bitflips(p0=0.2, p1=p1_map)

    p0 = 0.2 * np.ones(4)
    p1 = np.array([0.2, 0.0, 0.0, 0.1])
    tf.random.set_seed(123)
    sprobs = tf.random.uniform(samples.shape, dtype=K.dtypes('DTYPE')).numpy()
    target_samples = np.copy(samples).ravel()
    ids = (np.where(target_samples == 0)[0], np.where(target_samples == 1)[0])
    target_samples[ids[0]] = samples.ravel()[ids[0]] + (sprobs < p0).ravel()[ids[0]]
    target_samples[ids[1]] = samples.ravel()[ids[1]] - (sprobs < p1).ravel()[ids[1]]
    target_samples = target_samples.reshape(samples.shape)
    np.testing.assert_allclose(noisy_result.samples(), target_samples)


@pytest.mark.parametrize("probs,target",
                         [([0.0, 0.0, 0.0], {5:30}),
                          ([0.1, 0.3, 0.2], {5:16, 7:10, 6:2, 3: 1, 4: 1}),
                          ([0.5, 0.5, 0.5], {3:6, 5:6, 7:5, 2:4, 4:3, 0:2, 1:2, 6:2})])
def test_post_measurement_bitflips_on_circuit(accelerators, probs, target):
    """Check bitflip errors on circuit measurements."""
    import tensorflow as tf
    tf.random.set_seed(123)
    c = models.Circuit(5, accelerators=accelerators)
    c.add([gates.X(0), gates.X(2), gates.X(3)])
    c.add(gates.M(0, 1, p0={0: probs[0], 1: probs[1]}))
    c.add(gates.M(3, p0=probs[2]))
    result = c(nshots=30).frequencies(binary=False)
    assert result == target


def test_post_measurement_bitflips_on_circuit_result():
    """Check bitflip errors on ``CircuitResult`` objects."""
    import tensorflow as tf
    from qibo import K
    thetas = np.random.random(4)
    c = models.Circuit(4)
    c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
    c.add(gates.M(0, 1, register_name="a"))
    c.add(gates.M(3, register_name="b"))
    result = c(nshots=30)
    tf.random.set_seed(123)
    noisy_result = result.copy().apply_bitflips({0: 0.2, 1: 0.4, 3: 0.3})
    noisy_samples = noisy_result.samples(binary=True)
    register_samples = noisy_result.samples(binary=True, registers=True)

    samples = result.samples().numpy()
    tf.random.set_seed(123)
    sprobs = tf.random.uniform(samples.shape, dtype=K.dtypes('DTYPE'))
    flipper = sprobs.numpy() < np.array([0.2, 0.4, 0.3])
    target_samples = (samples + flipper) % 2
    np.testing.assert_allclose(noisy_samples, target_samples)
    # Check register samples
    np.testing.assert_allclose(register_samples["a"], target_samples[:, :2])
    np.testing.assert_allclose(register_samples["b"], target_samples[:, 2:])


def test_post_measurement_bitflip_errors():
    """Check errors raised by `GateResult.apply_bitflips` and `gates.M`."""
    from qibo.core import measurements
    samples = np.random.randint(0, 2, (20, 3))
    result = measurements.GateResult((0, 1, 3), binary_samples=samples)
    # Passing wrong qubit ids in bitflip error map
    with pytest.raises(KeyError):
        noisy_result = result.apply_bitflips({0: 0.1, 2: 0.2})
    # Passing wrong bitflip error map type
    with pytest.raises(TypeError):
        noisy_result = result.apply_bitflips("test")
    # Passing negative bitflip probability
    with pytest.raises(ValueError):
        noisy_result = result.apply_bitflips(-0.4)
    # Check bitflip error map errors when creating measurement gate
    gate = gates.M(0, 1, p0=2 * [0.1])
    with pytest.raises(ValueError):
        gate = gates.M(0, 1, p0=4 * [0.1])
    with pytest.raises(KeyError):
        gate = gates.M(0, 1, p0={0: 0.1, 2: 0.2})
    with pytest.raises(TypeError):
        gate = gates.M(0, 1, p0="test")
