#!/usr/bin/env python3
from tkinter import N

import numpy as np

import qibo
from qibo import gates, matrices
from qibo.models import Circuit


class QuantumCNN:
    def __init__(
        self, nqubits, nlayers, nclasses=2, RY=True
    ):  ### TODO: modify this to build the QCNN circuit
        """
        Class for a multi-task variational quantum classifier
        Args:
            nclasses: int number of classes to be classified. Default setting of 2 (phases).
            nqubits: int number of qubits employed in the quantum circuit
        """
        self.nclasses = nclasses
        self.nqubits = nqubits
        self.nlayers = nlayers

        # NOTE: magic number: 21=15+6 params per layer of conv and pool
        self.nparams_conv = 15
        self.nparams_pool = 6
        self.nparams_layer = self.nparams_conv + self.nparams_pool
        self.measured_qubits = int(np.ceil(np.log2(self.nclasses)))

        if self.nqubits <= 1:
            raise ValueError("nqubits must be larger than 1")

        if RY:

            def rotations():
                for q in range(self.nqubits):
                    yield gates.RY(q, theta=0)

        else:

            def rotations():
                for q in range(self.nqubits):
                    yield gates.RX(q, theta=0)
                    yield gates.RZ(q, theta=0)
                    yield gates.RX(q, theta=0)

        self._circuit = self.ansatz(nlayers, rotations)

    def quantum_conv_circuit(self, bits, symbols):
        c = Circuit(self.nqubits)
        for first, second in zip(bits[0::2], bits[1::2]):
            c += self.two_qubit_unitary([first, second], symbols)

        # check that there are more than 2 qubits to prevent double conv
        if len(bits) > 2:
            for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
                c += self.two_qubit_unitary([first, second], symbols)
        return c

    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        c = Circuit(self.nqubits)
        for source, sink in zip(source_bits, sink_bits):
            c += self.two_qubit_pool(source, sink, symbols)
        return c

    def ansatz(self, nlayers, rotations, params=None):  ### TODO: QCNN ansatz goes here
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            nlayers: int number of layers of the varitional circuit ansatz
        Returns:
            Circuit implementing the variational ansatz
        """
        """
        Wenjun's version:
            symbols = sympy.symbols('qconv0:63')
            # uses sympy.Symbols to map learnable variables. ? scans incoming circuits and replaces these with ? variables.
            model_circuit.add (quantum_conv_circuit(qubits, symbols[0:15]))
            model_circuit.add (quantum_pool_circuit(qubits[:4], qubits[4:], symbols[15:21]))
            model_circuit.add (quantum_conv_circuit(qubits[4:], symbols[21:36]))
            model_circuit.add (quantum_pool_circuit(qubits[4:6], qubits[6:], symbols[36:42]))
            model_circuit.add (quantum_conv_circuit(qubits[6:], symbols[42:57]))
            model_circuit.add (quantum_pool_circuit([qubits[6]], [qubits[7]],symbols[57:63]))
        """

        nparams_conv = self.nparams_conv
        nparams_layer = self.nparams_layer

        if params is not None:
            symbols = params
        else:
            symbols = [0 for _ in range(nlayers * nparams_layer)]

        nbits = self.nqubits

        qubits = [_ for _ in range(nbits)]
        c = Circuit(self.nqubits)
        for layer in range(nlayers):
            conv_start = int(nbits - nbits / (2**layer))
            pool_start = int(nbits - nbits / (2 ** (layer + 1)))
            param_start = layer * nparams_layer
            c += self.quantum_conv_circuit(
                qubits[conv_start:], symbols[param_start : param_start + nparams_conv]
            )
            c += self.quantum_pool_circuit(
                qubits[conv_start:pool_start],
                qubits[pool_start:],
                symbols[param_start + nparams_conv : param_start + nparams_layer],
            )

        # Measurements
        c.add(gates.M(*[nbits - 1 - i for i in range(self.measured_qubits)]))

        return c

    def one_qubit_unitary(self, bit, symbols):
        """Make a circuit enacting a rotation of the bloch sphere about the X,
        Y and Z axis, that depends on the values in `symbols`.
        Symbols should be a list of length 3.
        """
        c = Circuit(self.nqubits)
        c.add(
            gates.RX(bit, symbols[0])
        )  # question: is  cirq.X(bit)**symbols[0] same as RX(bit,symbols[0])
        c.add(gates.RY(bit, symbols[1]))
        c.add(gates.RZ(bit, symbols[2]))

        return c

    def two_qubit_unitary(self, bits, symbols):
        """Make a circuit that creates an arbitrary two qubit unitary.
        Symbols should be a list of length 15.
        """
        c = Circuit(self.nqubits)
        c += self.one_qubit_unitary(bits[0], symbols[0:3])
        c += self.one_qubit_unitary(bits[1], symbols[3:6])
        # to improve: to define new gates of XX YY and ZZ outside.
        """matrixXX = K.np.kron(matrices.X,matrices.X)
        matrixYY = K.np.kron(matrices.Y,matrices.Y)
        matrixZZ = K.np.kron(matrices.Z,matrices.Z)"""
        """gates.Unitary(matrixXX, 0, 1,name="XX")
        gates.Unitary(matrixYY, 0, 1,name="YY")
        gates.Unitary(matrixZZ, 0, 1,name="ZZ")"""
        """c.add(symbols[6]*gates.Unitary(matrixZZ, 0, 1))
        c.add(symbols[7]*gates.Unitary(matrixYY, 0, 1))
        c.add(symbols[8]*gates.Unitary(matrixXX, 0, 1))"""
        c.add(gates.RZZ(bits[0], bits[1], symbols[6]))
        c.add(gates.RYY(bits[0], bits[1], symbols[7]))
        c.add(gates.RXX(bits[0], bits[1], symbols[8]))

        c += self.one_qubit_unitary(bits[0], symbols[9:12])
        c += self.one_qubit_unitary(bits[1], symbols[12:])

        return c

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        """Make a circuit to do a parameterized 'pooling' operation, which
        attempts to reduce entanglement down from two qubits to just one.
        Symbols should be a list of 6 params.
        """
        pool_circuit = Circuit(self.nqubits)
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit += sink_basis_selector
        pool_circuit += source_basis_selector
        pool_circuit.add(gates.CNOT(source_qubit, sink_qubit))
        pool_circuit += sink_basis_selector.invert()

        return pool_circuit

    def set_circuit_params(self, angles):

        params = list(angles)
        expanded_params = []
        nbits = self.nqubits
        for layer in range(self.nlayers):
            nleft = nbits / (2**layer)
            param_start = layer * self.nparams_layer
            conv_params = params[param_start : param_start + self.nparams_conv]
            pool_params = params[
                param_start + self.nparams_conv : param_start + self.nparams_layer
            ]
            pool_params += pool_params[2::-1]
            expanded_params += conv_params * int(nleft if nleft > 2 else 1)
            expanded_params += pool_params * int(nleft / 2)

        self._circuit.set_parameters(expanded_params)

    def Classifier_circuit(self, theta):
        """
        Args:
            theta: list or numpy.array with the biases and the angles to be used in the circuit
            nlayers: int number of layers of the varitional circuit ansatz
            RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                if False, parameterized Ry gates are used in the circuit (default=False)
        Returns:
            Circuit implementing the variational ansatz for angles "theta"
        """
        bias = np.array(theta[0 : self.measured_qubits])
        angles = theta[self.measured_qubits :]

        self.set_circuit_params(angles)
        return self._circuit

    def Predictions(self, circuit, theta, init_state, nshots=10000):
        """
        Args:
            theta: list or numpy.array with the biases to be used in the circuit
            init_state: numpy.array with the quantum state to be classified
            nshots: int number of runs of the circuit during the sampling process (default=10000)
        Returns:
            numpy.array() with predictions for each qubit, for the initial state
        """
        bias = np.array(theta[0 : self.measured_qubits])
        circuit = circuit(init_state, nshots)
        result = circuit.frequencies(binary=False)
        prediction = np.zeros(self.measured_qubits)
        # print(result)
        for qubit in range(self.measured_qubits):
            for clase in range(self.nclasses):
                binary = bin(clase)[2:].zfill(self.measured_qubits)
                prediction[qubit] += result[clase] * (1 - 2 * int(binary[-qubit - 1]))
        # print(prediction)
        return prediction / nshots + bias

    def square_loss(self, labels, predictions):
        """
        Args:
            labels: list or numpy.array with the qubit labels of the quantum states to be classified
            predictions: list or numpy.array with the qubit predictions for the quantum states to be classified
        Returns:
            numpy.float32 with the value of the square-loss function
        """
        loss = 0
        for l, p in zip(labels, predictions):
            for qubit in range(self.measured_qubits):
                loss += (l[qubit] - p[qubit]) ** 2

        return loss / len(labels)

    def Cost_function(self, theta, data=None, labels=None, nshots=10000):
        """
        Args:
            theta: list or numpy.array with the biases and the angles to be used in the circuit
            nlayers: int number of layers of the varitional circuit ansatz
            data: numpy.array data[page][word]  (this is an array of kets)
            labels: list or numpy.array with the labels of the quantum states to be classified
            nshots: int number of runs of the circuit during the sampling process (default=10000)
        Returns:
            numpy.float32 with the value of the square-loss function
        """
        circ = self.Classifier_circuit(theta)

        Bias = np.array(theta[0 : self.measured_qubits])
        predictions = np.zeros(shape=(len(data), self.measured_qubits))

        for i, text in enumerate(data):
            predictions[i] = self.Predictions(circ, Bias, text, nshots)

        s = self.square_loss(labels, predictions)

        return s

    def minimize(
        self, init_theta, data=None, labels=None, nshots=10000, method="Powell"
    ):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            nlayers: int number of layers of the varitional ansatz
            init_state: numpy.array with the quantum state to be Schmidt-decomposed
            nshots: int number of runs of the circuit during the sampling process (default=10000)
            RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                if False, parameterized Ry gates are used in the circuit (default=True)
            method: str 'classical optimizer for the minimization'. All methods from scipy.optimize.minmize are suported (default='Powell')
        Returns:
            numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles
        """
        from scipy.optimize import minimize

        result = minimize(
            self.Cost_function, init_theta, args=(data, labels, nshots), method=method
        )
        loss = result.fun
        optimal_angles = result.x

        self.set_circuit_params(optimal_angles[self.measured_qubits :])
        return loss, optimal_angles

    def Accuracy(self, labels, predictions, sign=True, tolerance=1e-2):
        """
        Args:
            labels: numpy.array with the labels of the quantum states to be classified
            predictions: numpy.array with the predictions for the quantum states classified
            sign: if True, labels = np.sign(labels) and predictions = np.sign(predictions) (default=True)
            tolerance: float tolerance level to consider a prediction correct (default=1e-2)
        Returns:
            float with the proportion of states classified successfully
        """
        if sign == True:
            labels = [np.sign(label) for label in labels]
            predictions = [np.sign(prediction) for prediction in predictions]

        accur = 0
        for l, p in zip(labels, predictions):
            if np.allclose(l, p, rtol=0.0, atol=tolerance):
                accur += 1

        accur = accur / len(labels)

        return accur
