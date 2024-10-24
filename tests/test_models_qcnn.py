import math

import numpy as np

from qibo import gates, set_backend
from qibo.models import Circuit
from qibo.models.qcnn import QuantumCNN

num_angles = 21
angles0 = [i * math.pi / num_angles for i in range(num_angles)]


def test_classifier_circuit2():
    """ """
    set_backend("numpy")
    nqubits = 2
    nlayers = int(nqubits / 2)
    init_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)  #

    qcnn = QuantumCNN(nqubits, nlayers, nclasses=2)  # , params=angles0)

    angles = [0] + angles0

    circuit = qcnn.Classifier_circuit(angles)
    qcnn.set_circuit_params(
        angles, has_bias=True
    )  # only to test line 209-210 in qcnn.py

    # circuit = qcnn._circuit
    statevector = circuit(init_state).state()
    real_vector = get_real_vector2()

    # to compare statevector and real_vector
    np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
    np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5)


def get_real_vector2():
    nqubits = 2
    bits = range(nqubits)
    init_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)  #
    angles = angles0

    # convolution
    k = 0
    a = np.dot(
        one_qubit_unitary(nqubits, bits[0], angles[k : k + 3]).unitary(), init_state
    )
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[0], bits[1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[0], bits[1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[0], bits[1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[1], angles[k : k + 3]).unitary(), a)
    k += 3
    # pooling
    ksink = k
    a = np.dot(one_qubit_unitary(nqubits, bits[1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[0], angles[k : k + 3]).unitary(), a)
    a = np.dot(CNOT_unitary(nqubits, bits[0], bits[1]).unitary(), a)
    a = np.dot(
        one_qubit_unitary(nqubits, bits[1], angles[ksink : ksink + 3])
        .invert()
        .unitary(),
        a,
    )

    return a


def test_classifier_circuit4():
    """ """
    set_backend("numpy")
    nqubits = 4
    nlayers = int(nqubits / 2)
    init_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)  #

    qcnn = QuantumCNN(nqubits, nlayers, nclasses=2)
    angles = [0] + angles0 + angles0

    circuit = qcnn.Classifier_circuit(angles)
    statevector = circuit(init_state).state()
    real_vector = get_real_vector4()

    # to compare statevector and real_vector
    np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
    np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5)


def get_real_vector4():
    nqubits = 4
    init_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)  #
    angles = angles0
    bits = range(nqubits)
    # convolution - layer 1
    # to declare matrix array a

    b0 = 0
    b1 = 1
    k = 0
    a = np.dot(
        one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), init_state
    )
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)

    b0 = 2
    b1 = 3
    k = 0
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)

    b0 = 1
    b1 = 2
    k = 0
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)

    b0 = 3
    b1 = 0
    k = 0
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[b0], bits[b1], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[b0], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[b1], angles[k : k + 3]).unitary(), a)

    # pooling - layer 1
    k = 15  # k+=3
    ksink = k
    a = np.dot(one_qubit_unitary(nqubits, bits[2], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[0], angles[k : k + 3]).unitary(), a)
    a = np.dot(CNOT_unitary(nqubits, bits[0], bits[2]).unitary(), a)
    a = np.dot(
        one_qubit_unitary(nqubits, bits[2], angles[ksink : ksink + 3])
        .invert()
        .unitary(),
        a,
    )

    k = 15  # k+=3
    ksink = k
    a = np.dot(one_qubit_unitary(nqubits, bits[3], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[1], angles[k : k + 3]).unitary(), a)
    a = np.dot(CNOT_unitary(nqubits, bits[1], bits[3]).unitary(), a)
    a = np.dot(
        one_qubit_unitary(nqubits, bits[3], angles[ksink : ksink + 3])
        .invert()
        .unitary(),
        a,
    )

    # convolution - layer 2
    k = 0
    a = np.dot(one_qubit_unitary(nqubits, bits[2], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[3], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(RZZ_unitary(nqubits, bits[2], bits[3], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RYY_unitary(nqubits, bits[2], bits[3], angles[k]).unitary(), a)
    k += 1
    a = np.dot(RXX_unitary(nqubits, bits[2], bits[3], angles[k]).unitary(), a)
    k += 1
    a = np.dot(one_qubit_unitary(nqubits, bits[2], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[3], angles[k : k + 3]).unitary(), a)
    k += 3

    # pooling - layer 2
    ksink = k
    a = np.dot(one_qubit_unitary(nqubits, bits[3], angles[k : k + 3]).unitary(), a)
    k += 3
    a = np.dot(one_qubit_unitary(nqubits, bits[2], angles[k : k + 3]).unitary(), a)
    a = np.dot(CNOT_unitary(nqubits, bits[2], bits[3]).unitary(), a)
    a = np.dot(
        one_qubit_unitary(nqubits, bits[3], angles[ksink : ksink + 3])
        .invert()
        .unitary(),
        a,
    )

    return a


def one_qubit_unitary(nqubits, bit, symbols):
    c = Circuit(nqubits)
    c.add(gates.RX(bit, symbols[0]))
    c.add(gates.RY(bit, symbols[1]))
    c.add(gates.RZ(bit, symbols[2]))

    return c


def RXX_unitary(nqubits, bit0, bit1, angle):
    c = Circuit(nqubits)
    c.add(gates.RXX(bit0, bit1, angle))

    return c


def RYY_unitary(nqubits, bit0, bit1, angle):
    c = Circuit(nqubits)
    c.add(gates.RYY(bit0, bit1, angle))

    return c


def RZZ_unitary(nqubits, bit0, bit1, angle):
    c = Circuit(nqubits)
    c.add(gates.RZZ(bit0, bit1, angle))

    return c


def CNOT_unitary(nqubits, bit0, bit1):
    c = Circuit(nqubits)
    c.add(gates.CNOT(bit0, bit1))

    return c


def test_1_qubit_classifier_circuit_error():
    try:
        QuantumCNN(nqubits=1, nlayers=1, nclasses=2)
    except:
        pass


def test_qcnn_training():
    import random

    set_backend("numpy")

    # generate 2 random states and labels for pytest
    data = np.zeros([2, 16])
    for i in range(2):
        data_i = np.random.rand(16)
        data[i] = data_i / np.linalg.norm(data_i)
    labels = [[1], [-1]]

    # test qcnn training
    testbias = np.zeros(1)
    testangles = [random.uniform(0, 2 * np.pi) for i in range(21 * 2)]
    init_theta = np.concatenate((testbias, testangles))
    test_qcnn = QuantumCNN(nqubits=4, nlayers=1, nclasses=2, params=init_theta)
    testcircuit = test_qcnn._circuit
    result = test_qcnn.minimize(
        init_theta, data=data, labels=labels, nshots=10000, method="Powell"
    )

    # test Predictions function
    predictions = []
    for n in range(len(data)):
        predictions.append(test_qcnn.predict(data[n], nshots=10000)[0])

    # test Accuracy function
    predictions.append(1)
    labels = np.array([[1], [-1], [1]])
    test_qcnn.Accuracy(labels, predictions)


def test_two_qubit_ansatz():
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.RX(0, 0))
    c.add(gates.CNOT(1, 0))
    test_qcnn = QuantumCNN(4, 2, 2, twoqubitansatz=c)


def test_two_qubit_ansatz_training():
    # test qibojit case (copy initial state as quick-fix for in-place update)
    set_backend("qibojit")

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.RX(0, 0))
    c.add(gates.CNOT(1, 0))
    test_qcnn = QuantumCNN(4, 2, 2, twoqubitansatz=c)

    data = np.zeros([2, 16])
    for i in range(2):
        data_i = np.random.rand(16)
        data[i] = data_i / np.linalg.norm(data_i)
    labels = [[1], [-1]]

    totalNParams = test_qcnn.nparams_layer * 2
    init_theta = [
        0 for i in range(totalNParams + 1)
    ]  # totalNParams+1 to account for bias parameter.

    result = test_qcnn.minimize(
        init_theta, data=data, labels=labels, nshots=10000, method="Powell"
    )

    # test Predictions function
    predictions = []
    for n in range(len(data)):
        predictions.append(test_qcnn.predict(data[n], nshots=10000)[0])

    # test Accuracy function
    predictions.append(1)
    labels = np.array([[1], [-1], [1]])
    test_qcnn.Accuracy(labels, predictions)
