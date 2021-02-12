import collections
import numpy as np
import pytest
from qibo import gates, models
from typing import Optional


# TODO: Move this to `test_core_measurements.py`
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

# TODO: Move this to `test_core_measurements.py`
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
