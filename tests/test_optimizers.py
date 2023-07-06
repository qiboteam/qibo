from qibo.optimizers import Graph
import numpy as np
import qibo
import tensorflow as tf
from qibo.optimizers import Parameter
import pennylane as qml
import pytest

def loss(optimizer, params, feature, label):

    parameters = []

    for i in range(0, 12, 2):
        parameters.append(Parameter(params[i] * feature + params[i + 1], [i, i+1]))
    
    results = optimizer.run_circuit(parameters, nshots=1024)

    prediction = 2 * results[0]
    loss = (label - prediction) ** 2

    return loss

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
        c.add(qibo.gates.RY(q=0, theta=0))

    c.add(qibo.gates.M(0))

    return c


@pytest.mark.parametrize(
    "layer_num, trainable_qubits_correct, affected_params_correct",
    [(1, [0,1], [[0,1], [2,3]]), 
     (2, [0,1], [[6,7],[4,5]]),
     (6, [1], [16,17])],
)
def test_graph(layer_num, trainable_qubits_correct, affected_params_correct):

    circuit = ansatz_2qubit(3, 2)

    nqubits = circuit.nqubits
    gates = circuit.queue
    trainable_params = np.linspace(0.1, 1, 18)
    gate_params = [trainable_params[i] + trainable_params[i+1] for i in range(0, 18, 2)]
    trainable_params_index = [[i, i+1] for i in range(0, 18, 2)]

    graph = Graph(nqubits, gates, trainable_params_index, gate_params)

    graph.build_graph()

    new_circuit, trainable_qubits, affected_params = graph.run_layer(layer_num)

    assert np.allclose(trainable_qubits, trainable_qubits_correct)
    assert np.allclose(affected_params, affected_params_correct)


dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev, interface="autograd")
def ansatz_pdf(params, feature):
    qml.Hadamard(wires=0)

    qml.RZ(params[0]*feature, wires=0)
    qml.RZ(params[1], wires=0)

    qml.RY(params[2]*feature, wires=0)
    qml.RY(params[3], wires=0)

    qml.RZ(params[4]*feature, wires=0)
    qml.RZ(params[5], wires=0)

    qml.RY(params[6]*feature, wires=0)
    qml.RY(params[7], wires=0)

    qml.RZ(params[8]*feature, wires=0)
    qml.RZ(params[9], wires=0)

    qml.RY(params[10]*feature, wires=0)
    qml.RY(params[11], wires=0)

    return qml.expval(qml.PauliZ(0))


def test_natural_gradient():

    params = qml.numpy.asarray([0.1]*12)

    # create circuit ansatz for two qubits
    circuit = ansatz(3, 1)

    # initialize optimiser
    initial_parameters = np.full(12, 0.1) 
    optimiser = qibo.optimizers.Optimizer(loss, circuit=circuit, args=(initial_parameters), method="sgd")

    _ = optimiser.one_prediction(1.0)

    fubini = optimiser.generate_fubini(1.0, method="variance")

    assert np.allclose(optimiser.params, params)

    metric_tensor = qml.metric_tensor(ansatz_pdf, approx='diag')(params, 1.0)
    
    assert np.allclose(fubini, metric_tensor)
   