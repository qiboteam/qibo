#!/usr/bin/env python3
import numpy as np

import qibo
from qibo import Circuit, gates


class QuantumClassifer:
    def __init__(self, nclasses, nqubits, nlayers, RY=True):
        """
        Class for a multi-task variational quantum classifier

        Args:
            nclases: int number of classes to be classified
            nqubits: int number of qubits employed in the quantum circuit
        """
        self.nclasses = nclasses
        self.nqubits = nqubits
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

    def _CZ_gates1(self):
        """Yields CZ gates used in the variational circuit."""
        for q in range(0, self.nqubits - 1, 2):
            yield gates.CZ(q, q + 1)

    def _CZ_gates2(self):
        """Yields CZ gates used in the variational circuit."""
        for q in range(1, self.nqubits - 1, 2):
            yield gates.CZ(q, q + 1)

        yield gates.CZ(0, self.nqubits - 1)

    def ansatz(self, nlayers, rotations):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            nlayers: int number of layers of the varitional circuit ansatz

        Returns:
            Circuit implementing the variational ansatz
        """
        c = Circuit(self.nqubits)
        for _ in range(nlayers):
            c.add(rotations())
            c.add(self._CZ_gates1())
            c.add(rotations())
            c.add(self._CZ_gates2())
        # Final rotations
        c.add(rotations())
        # Measurements
        c.add(gates.M(*range(self.measured_qubits)))

        return c

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

        self._circuit.set_parameters(angles)

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

        for qubit in range(self.measured_qubits):
            for clase in range(self.nclasses):
                binary = bin(clase)[2:].zfill(self.measured_qubits)
                prediction[qubit] += result[clase] * (1 - 2 * int(binary[-qubit - 1]))

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
        if sign:
            labels = [np.sign(label) for label in labels]
            predictions = [np.sign(prediction) for prediction in predictions]

        accur = 0
        for l, p in zip(labels, predictions):
            if np.allclose(l, p, rtol=0.0, atol=tolerance):
                accur += 1

        accur = accur / len(labels)

        return accur
