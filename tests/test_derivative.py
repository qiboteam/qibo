import numpy as np
import pennylane as qml
import pytest
import tensorflow as tf

import qibo
from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend
from qibo.derivative import (
    Graph,
    Parameter,
    create_hamiltonian,
    generate_fubini,
    parameter_shift,
)
from qibo.models import Circuit
from qibo.symbols import Z

qibo.set_backend("tensorflow")


# defining an observable
def hamiltonian(nqubits):
    return hamiltonians.hamiltonians.SymbolicHamiltonian(
        np.prod([Z(i) for i in range(nqubits)]), backend=GlobalBackend()
    )


dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="autograd")
def ansatz_pdf(params, feature):
    qml.Hadamard(wires=0)

    qml.RZ(params[0] * feature, wires=0)
    qml.RZ(params[1], wires=0)

    qml.RY(params[2] * feature, wires=0)
    qml.RY(params[3], wires=0)

    qml.RZ(params[4] * feature, wires=0)
    qml.RZ(params[5], wires=0)

    qml.RY(params[6] * feature, wires=0)
    qml.RY(params[7], wires=0)

    qml.RZ(params[8] * feature, wires=0)
    qml.RZ(params[9], wires=0)

    qml.RY(params[10] * feature, wires=0)
    qml.RY(params[11], wires=0)

    return qml.expval(qml.PauliZ(0))


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    c.add(qibo.gates.H(q=0))

    for _ in range(layers):
        c.add(qibo.gates.RZ(q=0, theta=0))
        c.add(qibo.gates.RZ(q=0, theta=0))
        c.add(qibo.gates.RY(q=0, theta=0))
        c.add(qibo.gates.RY(q=0, theta=0))

    c.add(qibo.gates.M(0))

    return c


def ansatz_2qubit(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    c.add(qibo.gates.H(q=0))
    c.add(qibo.gates.H(q=1))

    for _ in range(layers):
        c.add(qibo.gates.RY(q=0, theta=0))
        c.add(qibo.gates.RZ(q=1, theta=0))
        c.add(qibo.gates.RY(q=1, theta=0))

    c.add(qibo.gates.CNOT(0, 1))

    c.add(qibo.gates.M(0))

    c.add(qibo.gates.M(1))

    return c


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
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

        ham = hamiltonian(1)
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


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

    test_hamiltonian = hamiltonian(nqubits=1)

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


@pytest.mark.parametrize(
    "layer_num, trainable_qubits_correct, affected_params_correct",
    [(1, [0, 1], [[0, 1], [2, 3]]), (2, [0, 1], [[6, 7], [4, 5]]), (6, [1], [16, 17])],
)
def test_graph(layer_num, trainable_qubits_correct, affected_params_correct):
    circuit = ansatz_2qubit(3, 2)
    print(circuit.draw())
    nqubits = circuit.nqubits
    gates = circuit.queue
    trainable_params = np.linspace(0.1, 1, 18)
    gate_params = [
        trainable_params[i] + trainable_params[i + 1] for i in range(0, 18, 2)
    ]
    trainable_params_index = [[i, i + 1] for i in range(0, 18, 2)]

    graph = Graph(nqubits, gates, trainable_params_index, gate_params)

    graph.build_graph()

    new_circuit, trainable_qubits, affected_params = graph.run_layer(layer_num)

    assert np.allclose(trainable_qubits, trainable_qubits_correct)
    assert np.allclose(affected_params, affected_params_correct)


def loss_func(ypred, ytrue, other_args=None):
    loss = 0
    for i in range(len(ypred)):
        loss += (ytrue[i] - ypred[i]) ** 2

    return loss


def test_natural_gradient():
    params = qml.numpy.asarray([0.1] * 12)
    obs = create_hamiltonian(0, 1, GlobalBackend())

    # create circuit ansatz for two qubits
    circuit = ansatz(3, 1)

    # initialize optimiser with Parameter objects
    initial_parameters = [Parameter(lambda th1: th1, [0.1]) for i in range(12)]
    optimiser = qibo.optimizers.SGD(
        circuit=circuit, parameters=initial_parameters, loss=loss_func
    )

    _ = optimiser.run_circuit(0.1, nshots=1024)

    fubini = generate_fubini(circuit, 1, initial_parameters, 1.0, obs)
    # initialize optimiser with numpy array
    initial_parameters2 = np.full(12, 0.1)
    optimiser2 = qibo.optimizers.SGD(
        circuit=circuit, parameters=initial_parameters, loss=loss_func
    )

    _ = optimiser2.run_circuit(0.1, nshots=1024)

    fubini2 = generate_fubini(circuit, 1, initial_parameters2, 1.0, obs)

    assert np.allclose(optimiser.params, params)

    metric_tensor = qml.metric_tensor(ansatz_pdf, approx="diag")(params, 1.0)

    print(fubini)
    print(metric_tensor)
    print(fubini2)
    assert np.allclose(fubini, metric_tensor)
    assert np.allclose(fubini2, metric_tensor)


if __name__ == "__main__":
    test_graph(6, [1], [16, 17])
