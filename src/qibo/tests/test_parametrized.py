import numpy as np
import pytest
import qibo
from qibo.models import Circuit
from qibo import gates

_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum"]
_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rx_parameter_setter(backend):
    """Check that the parameter setter of RX gate is working properly."""
    def exact_state(theta):
        phase = np.exp(1j * theta / 2.0)
        gate = np.array([[phase.real, -1j * phase.imag],
                        [-1j * phase.imag, phase.real]])
        return gate.dot(np.ones(2)) / np.sqrt(2)

    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RX(0, theta=theta))
    final_state = c().numpy()
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)

    theta = 0.4321
    c.queue[-1].parameter = theta
    final_state = c().numpy()
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_get_parameters():
    c = Circuit(3)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(1, theta=0.456))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0.789, phi=0.987))
    c.add(gates.H(2))
    unitary = np.array([[0.123, 0.123], [0.123, 0.123]])
    c.add(gates.Unitary(unitary, 1))

    params = [0.123, 0.456, (0.789, 0.987), unitary]
    assert params == c.get_parameters()
    params = [0.123, 0.456, (0.789, 0.987), unitary]
    assert params == c.get_parameters()
    params = {c.queue[0]: 0.123, c.queue[1]: 0.456,
              c.queue[3]: (0.789, 0.987), c.queue[5]: unitary}
    assert params == c.get_parameters("dict")
    params = [0.123, 0.456, 0.789, 0.987]
    params.extend(unitary.ravel())
    assert params == c.get_parameters("flatlist")
    with pytest.raises(ValueError):
        c.get_parameters("test")

@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_set_parameters_with_list(backend, accelerators):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(3, accelerators)
    c.add(gates.RX(0, theta=0))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))
    # execute once
    final_state = c()

    params = [0.123, 0.456, (0.789, 0.321)]
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.RY(1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.fSim(0, 2, theta=params[2][0], phi=params[2][1]))
    target_c.add(gates.H(2))

    c.set_parameters(params)
    np.testing.assert_allclose(c(), target_c())

    # Attempt using a flat list / np.ndarray
    new_params = np.random.random(4)
    params = [new_params[0], new_params[1], (new_params[2], new_params[3])]
    target_c.set_parameters(params)
    c.set_parameters(new_params)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_set_parameters_with_unitary(backend, accelerators):
    """Check updating parameters of circuit that contains ``Unitary`` gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(3, accelerators)
    c.add(gates.RX(0, theta=0))
    c.add(gates.Unitary(np.zeros((4, 4)), 1, 2))
    # execute once
    final_state = c()

    params = [0.1234, np.random.random((4, 4))]
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.Unitary(params[1], 1, 2))
    c.set_parameters(params)
    np.testing.assert_allclose(c(), target_c())

    # Attempt using a flat list / np.ndarray
    params = np.random.random(17)
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.Unitary(params[1:].reshape((4, 4)), 1, 2))
    c.set_parameters(params)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_set_parameters_with_dictionary(backend, accelerators):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(3, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(2))
    c.add(gates.ZPow(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.CZPow(0, 2, theta=0))
    c.add(gates.H(2))
    c.add(gates.Unitary(np.eye(2), 1))
    final_state = c()

    params = [0.123, 0.456, 0.789, np.random.random((2, 2))]
    target_c = Circuit(3, accelerators)
    target_c.add(gates.X(0))
    target_c.add(gates.X(2))
    target_c.add(gates.ZPow(0, theta=params[0]))
    target_c.add(gates.RZ(1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.CZPow(0, 2, theta=params[2]))
    target_c.add(gates.H(2))
    target_c.add(gates.Unitary(params[3], 1))

    param_dict = {c.queue[i]: p for i, p in zip([2, 3, 5, 7], params)}
    c.set_parameters(param_dict)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


def test_circuit_set_parameters_errors():
    """Check updating parameters errors."""
    c = Circuit(2)
    c.add(gates.RX(0, theta=0.789))
    c.add(gates.RX(1, theta=0.789))
    c.add(gates.fSim(0, 1, theta=0.123, phi=0.456))

    with pytest.raises(ValueError):
        c.set_parameters({gates.RX(0, theta=1.0): 0.568})
    with pytest.raises(ValueError):
        c.set_parameters([0.12586])
    with pytest.raises(ValueError):
        c.set_parameters(np.random.random(5))
    with pytest.raises(ValueError):
        import tensorflow as tf
        c.set_parameters(tf.random.uniform((6,), dtype=tf.float64))
    with pytest.raises(TypeError):
        c.set_parameters({0.3568})
    fused_c = c.fuse()
    with pytest.raises(TypeError):
        fused_c.set_parameters({gates.RX(0, theta=1.0): 0.568})


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5])
def test_set_parameters_with_variationallayer(backend, accelerators, nqubits):
    """Check updating parameters of variational layer."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    theta = np.random.random(nqubits)
    c = Circuit(nqubits, accelerators)
    pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ, theta))

    target_c = Circuit(nqubits)
    target_c.add((gates.RY(i, theta[i]) for i in range(nqubits)))
    target_c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    np.testing.assert_allclose(c(), target_c())

    # Test setting VariationalLayer using a list
    new_theta = np.random.random(nqubits)
    c.set_parameters([np.copy(new_theta)])
    target_c.set_parameters(np.copy(new_theta))
    np.testing.assert_allclose(c(), target_c())

    # Test setting VariationalLayer using an array
    new_theta = np.random.random(nqubits)
    c.set_parameters(np.copy(new_theta))
    target_c.set_parameters(np.copy(new_theta))
    np.testing.assert_allclose(c(), target_c())

    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5])
def test_set_parameters_with_double_variationallayer(backend, accelerators, nqubits):
    """Check updating parameters of variational layer."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    theta = np.random.random((3, nqubits))
    c = Circuit(nqubits, accelerators)
    pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 theta[0], theta[1]))
    c.add((gates.RX(i, theta[2, i]) for i in range(nqubits)))

    target_c = Circuit(nqubits)
    target_c.add((gates.RY(i, theta[0, i]) for i in range(nqubits)))
    target_c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_c.add((gates.RY(i, theta[1, i]) for i in range(nqubits)))
    target_c.add((gates.RX(i, theta[2, i]) for i in range(nqubits)))
    np.testing.assert_allclose(c(), target_c())

    new_theta = np.random.random(3 * nqubits)
    c.set_parameters(np.copy(new_theta))
    target_c.set_parameters(np.copy(new_theta))
    np.testing.assert_allclose(c(), target_c())

    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_set_parameters_with_gate_fusion(backend, accelerators):
    """Check updating parameters of fused circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    params = np.random.random(9)
    c = Circuit(5, accelerators)
    c.add(gates.RX(0, theta=params[0]))
    c.add(gates.RY(1, theta=params[1]))
    c.add(gates.CZ(0, 1))
    c.add(gates.RX(2, theta=params[2]))
    c.add(gates.RY(3, theta=params[3]))
    c.add(gates.fSim(2, 3, theta=params[4], phi=params[5]))
    c.add(gates.RX(4, theta=params[6]))
    c.add(gates.RZ(0, theta=params[7]))
    c.add(gates.RZ(1, theta=params[8]))

    fused_c = c.fuse()
    np.testing.assert_allclose(c(), fused_c())

    new_params = np.random.random(9)
    new_params_list = list(new_params[:4])
    new_params_list.append((new_params[4], new_params[5]))
    new_params_list.extend(new_params[6:])
    c.set_parameters(new_params_list)
    fused_c.set_parameters(new_params_list)
    np.testing.assert_allclose(c(), fused_c())

    qibo.set_backend(original_backend)
