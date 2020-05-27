import numpy as np
import pytest
from qibo.models import Circuit
from qibo.tensorflow import gates


def test_circuit_with_noise_gates():
    """Check that ``circuit.with_noise()`` adds the proper noise channels."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    assert noisy_c.depth == 9
    for i in [1, 2, 4, 5, 7, 8]:
        assert isinstance(noisy_c.queue[i], gates.NoiseChannel)


def test_circuit_with_noise_execution():
    """Check ``circuit.with_noise()`` execution."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    target_c = Circuit(2)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.3))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.2, 0.3))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.3))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.2, 0.3))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_with_measurements():
    """Check ``circuit.with_noise() when using measurement noise."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.add(gates.M(0))
    noisy_c = c.with_noise(3 * (0.1,), measurement_noise = (0.3, 0.0, 0.0))

    target_c = Circuit(2)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.1, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.1, 0.1))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.3, 0.0, 0.0))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.1, 0.1))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_noise_map():
    """Check ``circuit.with_noise() when giving noise map."""
    noise_map = {0: (0.1, 0.2, 0.1), 1: (0.2, 0.3, 0.0),
                 2: (0.0, 0.0, 0.0)}

    c = Circuit(3)
    c.add([gates.H(0), gates.H(1), gates.X(2)])
    c.add(gates.M(2))
    noisy_c = c.with_noise(noise_map, measurement_noise = (0.3, 0.0, 0.0))

    target_c = Circuit(3)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.X(2))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.NoiseChannel(2, 0.3, 0.0, 0.0))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_noise_map_exceptions():
    """Check that proper exceptions are raised when noise map is invalid."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3))
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1), 1: (0.3, 0.1)})
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1)})
    with pytest.raises(TypeError):
        noisy_c = c.with_noise({0, 1})
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3, 0.1),
                               measurement_noise=(0.5, 0.0, 0.0))


def test_circuit_with_noise_exception():
    """Check that calling ``with_noise`` in a noisy circuit raises error."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.NoiseChannel(0, px=0.2)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3, 0.0))
