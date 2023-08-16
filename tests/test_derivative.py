import math

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
import tensorflow as tf

import qibo
from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend
from qibo.derivative import (
    calculate_circuit_gradients,
    create_hamiltonian,
    parameter_shift,
)
from qibo.models import Circuit
from qibo.symbols import Z

qibo.set_backend("tensorflow")
tf.get_logger().setLevel("ERROR")
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def ansatz_pdf(layers, params, feature):
    for i in range(layers):
        qml.Hadamard(wires=i)

        qml.RZ(params[12 * i + 0] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 1], wires=i)

        qml.RY(params[12 * i + 2] * feature, wires=i)
        qml.RY(params[12 * i + 3], wires=i)

        qml.RZ(params[12 * i + 4] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 5], wires=i)

        qml.RY(params[12 * i + 6] * feature, wires=i)
        qml.RY(params[12 * i + 7], wires=i)

        qml.RZ(params[12 * i + 8] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 9], wires=i)

        qml.RY(params[12 * i + 10] * feature, wires=i)
        qml.RY(params[12 * i + 11], wires=i)

    return qml.expval(qml.PauliZ([1]))


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    for qubit in range(nqubits):
        c.add(qibo.gates.H(q=qubit))

        for _ in range(layers):
            c.add(
                qibo.gates.RZ(
                    q=qubit,
                    theta=Parameter(
                        lambda x, th1: th1 * sp.log(x), [0.1], featurep=[0.1]
                    ),
                )
            )
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(
                qibo.gates.RY(
                    q=qubit,
                    theta=Parameter(lambda x, th1: th1 * x, [0.1], featurep=[0.1]),
                )
            )
            c.add(qibo.gates.RY(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))

        c.add(qibo.gates.M(qubit))

    return c


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.H(q=0))
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


# calculate the exact gradients
def gradient_exact():
    backend = GlobalBackend()

    test_params = tf.Variable(np.linspace(0.1, 1, 3))

    with tf.GradientTape() as tape:
        c = circuit(nqubits=1)
        c.set_parameters(test_params)

        ham = create_hamiltonian(0, 1, GlobalBackend())
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


def test_psr_commuting_gate():
    # hyperparameters
    scale_factor = 0.5
    nshots = None
    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # separating gates
    c = Circuit(1)
    c.add(gates.H(q=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.M(0))

    params = np.array([0.1, 0.2])
    c.set_parameters(params)

    grad_0 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    grad_1 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    # single gate
    c2 = Circuit(1)
    c2.add(gates.H(q=0))
    c2.add(gates.RY(q=0, theta=0))
    c2.add(gates.M(0))

    params = np.array([0.3])
    c2.set_parameters(params)

    grad_2 = parameter_shift(
        circuit=c2,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    assert np.isclose(grad_0, grad_1, atol=1e-5)
    assert np.isclose(grad_1, grad_2, atol=1e-5)


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
def test_psr(backend, nshots, atol):
    grads = gradient_exact()
    scale_factor = 1

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    test_params *= scale_factor
    c.set_parameters(test_params)

    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad_0 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_1 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_2 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)
