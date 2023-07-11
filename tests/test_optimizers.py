from qibo.derivative import Parameter
import numpy as np
import qibo
import pennylane as qml
import pytest

def loss(optimizer, params, feature, label):

    parameters = []

    for i in range(0, 12, 2):
        parameters.append(Parameter(lambda th1, th2: th1 * feature + th2, [params[i], params[i+1]]))
    
    results = optimizer.run_circuit(parameters, nshots=1024)

    prediction = 2 * results
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
        c.add(qibo.gates.RZ(q=0, theta=0))
        c.add(qibo.gates.RY(q=0, theta=0))
        c.add(qibo.gates.RY(q=0, theta=0))


    c.add(qibo.gates.M(0))

    return c








