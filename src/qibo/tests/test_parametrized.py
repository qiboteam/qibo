import numpy as np
import pytest
import qibo
from qibo.models import Circuit
from qibo import gates

_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize("backend", ["custom", "defaulteinsum", "matmuleinsum"])
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
    c.queue[-1].parameters = theta
    final_state = c().numpy()
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_unitary(backend, accelerators, trainable):
    """Check updating parameters of circuit that contains ``Unitary`` gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    params = [0.1234, np.random.random((4, 4))]

    c = Circuit(3, accelerators)
    c.add(gates.RX(0, theta=0))
    if trainable:
        c.add(gates.Unitary(np.zeros((4, 4)), 1, 2, trainable=trainable))
        trainable_params = list(params)
    else:
        c.add(gates.Unitary(params[1], 1, 2, trainable=trainable))
        trainable_params = [params[0]]
    # execute once
    final_state = c()

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.Unitary(params[1], 1, 2))
    c.set_parameters(trainable_params)
    np.testing.assert_allclose(c(), target_c())

    # Attempt using a flat list / np.ndarray
    new_params = np.random.random(17)
    if trainable:
        c.set_parameters(new_params)
    else:
        c.set_parameters(new_params[:1])
        new_params[1:] = params[1].ravel()
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=new_params[0]))
    target_c.add(gates.Unitary(new_params[1:].reshape((4, 4)), 1, 2))
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


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
@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_double_variationallayer(backend, accelerators,
                                                     nqubits, trainable):
    """Check updating parameters of variational layer."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    theta = np.random.random((3, nqubits))
    c = Circuit(nqubits, accelerators)
    pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 theta[0], theta[1],
                                 trainable=trainable))
    c.add((gates.RX(i, theta[2, i]) for i in range(nqubits)))

    target_c = Circuit(nqubits)
    target_c.add((gates.RY(i, theta[0, i]) for i in range(nqubits)))
    target_c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_c.add((gates.RY(i, theta[1, i]) for i in range(nqubits)))
    target_c.add((gates.RX(i, theta[2, i]) for i in range(nqubits)))
    np.testing.assert_allclose(c(), target_c())

    new_theta = np.random.random(3 * nqubits)
    if trainable:
        c.set_parameters(np.copy(new_theta))
    else:
        c.set_parameters(np.copy(new_theta[2 * nqubits:]))
        new_theta[:2 * nqubits] = theta[:2].ravel()
    target_c.set_parameters(np.copy(new_theta))
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_gate_fusion(backend, accelerators, trainable):
    """Check updating parameters of fused circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    params = np.random.random(9)
    c = Circuit(5, accelerators)
    c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=params[1]))
    c.add(gates.CZ(0, 1))
    c.add(gates.RX(2, theta=params[2]))
    c.add(gates.RY(3, theta=params[3], trainable=trainable))
    c.add(gates.fSim(2, 3, theta=params[4], phi=params[5]))
    c.add(gates.RX(4, theta=params[6]))
    c.add(gates.RZ(0, theta=params[7], trainable=trainable))
    c.add(gates.RZ(1, theta=params[8]))

    fused_c = c.fuse()
    np.testing.assert_allclose(c(), fused_c())

    if trainable:
        new_params = np.random.random(9)
        new_params_list = list(new_params[:4])
        new_params_list.append((new_params[4], new_params[5]))
        new_params_list.extend(new_params[6:])
    else:
        new_params = np.random.random(9)
        new_params_list = list(new_params[1:3])
        new_params_list.append((new_params[4], new_params[5]))
        new_params_list.append(new_params[6])
        new_params_list.append(new_params[8])

    c.set_parameters(new_params_list)
    fused_c.set_parameters(new_params_list)
    np.testing.assert_allclose(c(), fused_c())

    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_variable_theta(backend, accelerators):
    """Check that parametrized gates accept `tf.Variable` parameters."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import tensorflow as tf
    from qibo import K
    theta1 = tf.Variable(0.1234, dtype=K.dtypes('DTYPE'))
    theta2 = tf.Variable(0.4321, dtype=K.dtypes('DTYPE'))

    cvar = Circuit(2, accelerators)
    cvar.add(gates.RX(0, theta1))
    cvar.add(gates.RY(1, theta2))
    final_state = cvar().numpy()

    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    target_state = c().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
