"""Test :meth:`qibo.core.circuit.Circuit.get_parameters` and :meth:`qibo.core.circuit.Circuit.set_parameters`."""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit


@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_list(backend, trainable):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    params = [0.123, 0.456, (0.789, 0.321)]
    c = Circuit(3)
    if trainable:
        c.add(gates.RX(0, theta=0, trainable=trainable))
    else:
        c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))
    # execute once
    final_state = c()

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.RY(1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.fSim(0, 2, theta=params[2][0], phi=params[2][1]))
    target_c.add(gates.H(2))

    # Attempt using a flat np.ndarray/list
    for new_params in (np.random.random(4), list(np.random.random(4))):
        if trainable:
            c.set_parameters(new_params)
        else:
            new_params[0] = params[0]
            c.set_parameters(new_params[1:])
    target_params = [new_params[0], new_params[1], (new_params[2], new_params[3])]
    target_c.set_parameters(target_params)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_ungates(backend, trainable, accelerators):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    params = [0.1, 0.2, 0.3, (0.4, 0.5), (0.6, 0.7, 0.8)]
    if trainable:
        trainable_params = list(params)
    else:
        trainable_params = [0.1, 0.3, (0.4, 0.5)]

    c = Circuit(3, accelerators)
    c.add(gates.RX(0, theta=0))
    if trainable:
        c.add(gates.CRY(0, 1, theta=0, trainable=trainable))
    else:
        c.add(gates.CRY(0, 1, theta=params[1], trainable=trainable))
    c.add(gates.CZ(1, 2))
    c.add(gates.U1(2, theta=0))
    c.add(gates.CU2(0, 2, phi=0, lam=0))
    if trainable:
        c.add(gates.U3(1, theta=0, phi=0, lam=0, trainable=trainable))
    else:
        c.add(gates.U3(1, *params[4], trainable=trainable))
    # execute once
    final_state = c()

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.CRY(0, 1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.U1(2, theta=params[2]))
    target_c.add(gates.CU2(0, 2, *params[3]))
    target_c.add(gates.U3(1, *params[4]))
    c.set_parameters(trainable_params)
    np.testing.assert_allclose(c(), target_c())

    # Attempt using a flat list
    npparams = np.random.random(8)
    if trainable:
        trainable_params = np.copy(npparams)
    else:
        npparams[1] = params[1]
        npparams[5:] = params[4]
        trainable_params = np.delete(npparams, [1, 5, 6, 7])
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=npparams[0]))
    target_c.add(gates.CRY(0, 1, theta=npparams[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.U1(2, theta=npparams[2]))
    target_c.add(gates.CU2(0, 2, *npparams[3:5]))
    target_c.add(gates.U3(1, *npparams[5:]))
    c.set_parameters(trainable_params)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_unitary(backend, trainable, accelerators):
    """Check updating parameters of circuit that contains ``Unitary`` gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    params = [0.1234, np.random.random((4, 4))]

    c = Circuit(4, accelerators)
    c.add(gates.RX(0, theta=0))
    if trainable:
        c.add(gates.Unitary(np.zeros((4, 4)), 1, 2, trainable=trainable))
        trainable_params = list(params)
    else:
        c.add(gates.Unitary(params[1], 1, 2, trainable=trainable))
        trainable_params = [params[0]]
    # execute once
    final_state = c()

    target_c = Circuit(4)
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
    target_c = Circuit(4)
    target_c.add(gates.RX(0, theta=new_params[0]))
    target_c.add(gates.Unitary(new_params[1:].reshape((4, 4)), 1, 2))
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_set_parameters_with_variationallayer(backend, nqubits, accelerators):
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


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_double_variationallayer(backend, nqubits, trainable, accelerators):
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


@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_gate_fusion(backend, trainable, accelerators):
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


def test_variable_theta(backend):
    """Check that parametrized gates accept `tf.Variable` parameters."""
    if "numpy" in backend:
        pytest.skip("Numpy backends do not support variable parameters.")

    from qibo import K
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta1 = K.optimization.Variable(0.1234, dtype=K.dtypes('DTYPE'))
    theta2 = K.optimization.Variable(0.4321, dtype=K.dtypes('DTYPE'))

    cvar = Circuit(2)
    cvar.add(gates.RX(0, theta1))
    cvar.add(gates.RY(1, theta2))
    final_state = cvar()

    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    target_state = c()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
