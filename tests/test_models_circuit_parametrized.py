"""Test :meth:`qibo.models.circuit.Circuit.get_parameters` and
:meth:`qibo.models.circuit.Circuit.set_parameters`."""

import sys

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import MissingBackend


def test_rx_parameter_setter(backend):
    """Check the parameter setter of RX gate."""

    def exact_state(theta):
        phase = np.exp(1j * theta / 2.0)
        gate = np.array(
            [[phase.real, -1j * phase.imag], [-1j * phase.imag, phase.real]]
        )
        return gate.dot(np.ones(2)) / np.sqrt(2)

    theta = 0.1234
    gate = gates.RX(0, theta=theta)
    initial_state = backend.cast(np.ones(2) / np.sqrt(2))
    final_state = backend.apply_gate(gate, initial_state, 1)
    target_state = exact_state(theta)
    backend.assert_allclose(final_state, target_state)

    theta = 0.4321
    gate.parameters = theta
    initial_state = backend.cast(np.ones(2) / np.sqrt(2))
    final_state = backend.apply_gate(gate, initial_state, 1)
    target_state = exact_state(theta)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_list(backend, trainable):
    """Check updating parameters of circuit with list."""
    params = [0.123, 0.456, (0.789, 0.321)]
    c = Circuit(3)
    if trainable:
        c.add(gates.RX(0, theta=0.0, trainable=trainable))
    else:
        c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=0.0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0.0, phi=0.0))
    c.add(gates.H(2))
    # execute once
    final_state = backend.execute_circuit(c)

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
    backend.assert_circuitclose(c, target_c)


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_ungates(backend, trainable, accelerators):
    """Check updating parameters of circuit with list."""
    params = [0.1, 0.2, 0.3, (0.4, 0.5), (0.6, 0.7, 0.8)]
    if trainable:
        trainable_params = list(params)
    else:
        trainable_params = [0.1, 0.3, (0.4, 0.5)]

    c = Circuit(3, accelerators)
    c.add(gates.RX(0, theta=0.0))
    if trainable:
        c.add(gates.CRY(0, 1, theta=0.0, trainable=trainable))
    else:
        c.add(gates.CRY(0, 1, theta=params[1], trainable=trainable))
    c.add(gates.CZ(1, 2))
    c.add(gates.U1(2, theta=0.0))
    c.add(gates.CU2(0, 2, phi=0.0, lam=0.0))
    if trainable:
        c.add(gates.U3(1, theta=0.0, phi=0.0, lam=0.0, trainable=trainable))
    else:
        c.add(gates.U3(1, *params[4], trainable=trainable))
    # execute once
    final_state = backend.execute_circuit(c)

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.CRY(0, 1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.U1(2, theta=params[2]))
    target_c.add(gates.CU2(0, 2, *params[3]))
    target_c.add(gates.U3(1, *params[4]))
    c.set_parameters(trainable_params)
    backend.assert_circuitclose(c, target_c)

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
    backend.assert_circuitclose(c, target_c)


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_unitary(backend, trainable, accelerators):
    """Check updating parameters of circuit that contains ``Unitary`` gate."""
    params = [0.1234, np.random.random((4, 4))]
    c = Circuit(4, accelerators)
    c.add(gates.RX(0, theta=0.0))
    if trainable:
        c.add(gates.Unitary(np.zeros((4, 4)), 1, 2, trainable=trainable))
        trainable_params = list(params)
    else:
        c.add(gates.Unitary(params[1], 1, 2, trainable=trainable))
        trainable_params = [params[0]]
    # execute once
    final_state = backend.execute_circuit(c)

    target_c = Circuit(4)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.Unitary(params[1], 1, 2))
    c.set_parameters(trainable_params)
    backend.assert_circuitclose(c, target_c)

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
    backend.assert_circuitclose(c, target_c)


@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_gate_fusion(backend, trainable):
    """Check updating parameters of fused circuit."""
    params = np.random.random(9)
    c = Circuit(5)
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
    backend.assert_circuitclose(fused_c, c)

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
    backend.assert_circuitclose(fused_c, c)


@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_light_cone(backend, trainable):
    """Check updating parameters of light cone circuit."""
    params = np.random.random(4)
    c = Circuit(4)
    c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=params[1]))
    c.add(gates.CZ(0, 1))
    c.add(gates.RX(2, theta=params[2]))
    c.add(gates.RY(3, theta=params[3], trainable=trainable))
    c.add(gates.CZ(2, 3))
    if trainable:
        c.set_parameters(np.random.random(4))
    else:
        c.set_parameters(np.random.random(2))
    target_state = backend.execute_circuit(c).state()
    lc, _ = c.light_cone(1, 2)
    final_state = backend.execute_circuit(lc).state()
    backend.assert_allclose(final_state, target_state)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="no tensorflow-io-0.32.0's wheel available for Windows",
)
def test_variable_theta():
    """Check that parametrized gates accept `tf.Variable` parameters."""
    try:
        from qibo.backends import construct_backend

        backend = construct_backend(backend="qiboml", platform="tensorflow")
    except MissingBackend:  # pragma: no cover
        pytest.skip("Skipping variable test because tensorflow is not available.")

    theta1 = backend.tf.Variable(0.1234, dtype=backend.tf.float64)
    theta2 = backend.tf.Variable(0.4321, dtype=backend.tf.float64)
    cvar = Circuit(2)
    cvar.add(gates.RX(0, theta1))
    cvar.add(gates.RY(1, theta2))
    final_state = backend.execute_circuit(cvar).state()

    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    target_state = backend.execute_circuit(c).state()
    backend.assert_allclose(final_state, target_state)
