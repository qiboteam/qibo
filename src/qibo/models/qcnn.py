"""Implement a Quantum Convolutional Neural Network (QCNN) for classification tasks.
The QCNN model was originally proposed in: arXiv:1810.03787 <https://arxiv.org/abs/1810.03787>_
for the identification of quantum phases.

The QuantumCNN class in this module provides methods to construct the QCNN.
"""

import numpy as np

from qibo import gates
from qibo.backends import get_backend
from qibo.models import Circuit


class QuantumCNN:
    """
    Model that implements and trains a variational quantum convolutional network (QCNN) for
    classification tasks.
    The QCNN model was originally proposed in: `arXiv:1810.03787
    <https://arxiv.org/abs/1810.03787>`_ for the identification of quantum phases.

    Args:
        nqubits (int): number of qubits of the input states. Currently supports powers of 2.
        nlayers (int): number of convolutional and pooling layers.
        nclasses (int): number of classes to be classified. Default setting of 2 (phases).
        params: initial list of variational parameters. If not provided, all parameters
            will be initialized to zero.
        twoqubitansatz (:class:`qibo.models.Circuit`): a two qubit ansatz that can be input by
            the user to form the two qubit ansatz used in the convolutional circuit.

    Example:
        .. testcode::

            import math
            import random

            import numpy as np

            from qibo import set_backend
            from qibo.models.qcnn import QuantumCNN

            set_backend("numpy")

            data = np.random.rand(16)
            data = data / np.linalg.norm(data)
            data = [data]
            labels = [[1]]
            testbias = np.zeros(1)
            testangles = [random.uniform(0, 2 * np.pi) for i in range(21 * 2)]
            init_theta = np.concatenate((testbias, testangles))
            test_qcnn = QuantumCNN(nqubits=4, nlayers=1, nclasses=2, params=init_theta)
            testcircuit = test_qcnn._circuit
            result = test_qcnn.minimize(
               init_theta, data=data, labels=labels, nshots=10000, method="Powell"
            )
            predict0 = test_qcnn.predict(data, nshots=10000)

    """

    def __init__(
        self,
        nqubits,
        nlayers,
        nclasses=2,
        params=None,
        twoqubitansatz=None,
        copy_init_state=None,
    ):
        """
        Initializes the QuantumCNN object.

        Args:
            nqubits (int): The number of qubits in the QCNN.
            nlayers (int): The number of layers in the QCNN.
            nclasses (int, optional): The number of classes for the classification task. Defaults to 2.
            params (np.ndarray, optional): The initial parameters for the QCNN. If None, random parameters are generated. Defaults to None.
            twoqubitansatz (qibo.models.circuit.Circuit, optional): A two-qubit ansatz for the convolutional layers. If None, a default ansatz is used. Defaults to None.
            copy_init_state (bool, optional): Whether to copy the initial state for each shot in the simulation. If None, the behavior depends on the backend. Defaults to None.

        Raises:
            ValueError: If nqubits is not larger than 1.
        """

        self.nclasses = nclasses
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.twoqubitansatz = twoqubitansatz

        if copy_init_state is None:
            if "qibojit" in str(get_backend()):
                self.copy_init_state = True
            else:
                self.copy_init_state = False

        if self.twoqubitansatz is None:
            self.nparams_conv = 15
        else:
            self.nparams_conv = len(self.twoqubitansatz.get_parameters())

        self.nparams_pool = 6
        self.nparams_layer = self.nparams_conv + self.nparams_pool
        self.measured_qubits = int(np.ceil(np.log2(self.nclasses)))

        if self.nqubits <= 1:
            raise ValueError("nqubits must be larger than 1")

        self._circuit = self.ansatz(nlayers, params=params)

    def quantum_conv_circuit(self, bits, symbols):
        """
        Internal helper function to construct a single convolutional layer.

        Args:
            bits: list or numpy.array with the qubits that the convolutional layer should apply to.
            symbols: list or numpy.array with the angles to be used in the circuit.
        Returns:
            Circuit for a single convolutional layer
        """
        circuit = Circuit(self.nqubits)
        for first, second in zip(bits[0::2], bits[1::2]):
            circuit += self.two_qubit_unitary([first, second], symbols)

        # check that there are more than 2 qubits to prevent double conv
        if len(bits) > 2:
            for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
                circuit += self.two_qubit_unitary([first, second], symbols)
        return circuit

    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        """
        Internal helper function to construct a single pooling layer.

        Args:
            source_bits: list or numpy.array with the source qubits for the pooling layer.
            sink_bits: list or numpy.array with the sink qubits for the pooling layer.
            symbols: list or numpy.array with the angles to be used in the circuit.
        Returns:
            Circuit for a single pooling layer
        """
        circuit = Circuit(self.nqubits)
        for source, sink in zip(source_bits, sink_bits):
            circuit += self.two_qubit_pool(source, sink, symbols)
        return circuit

    def ansatz(self, nlayers, params=None):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit.
            nlayers: int number of layers of the varitional circuit ansatz.
        Returns:
            Circuit implementing the QCNN variational ansatz.
        """

        nparams_conv = self.nparams_conv
        nparams_layer = self.nparams_layer

        if params is not None:
            symbols = params
        else:
            symbols = [0 for _ in range(nlayers * nparams_layer)]

        nbits = self.nqubits

        qubits = [_ for _ in range(nbits)]
        circuit = Circuit(self.nqubits)
        for layer in range(nlayers):
            conv_start = int(nbits - nbits / (2**layer))
            pool_start = int(nbits - nbits / (2 ** (layer + 1)))
            param_start = layer * nparams_layer
            circuit += self.quantum_conv_circuit(
                qubits[conv_start:], symbols[param_start : param_start + nparams_conv]
            )
            circuit += self.quantum_pool_circuit(
                qubits[conv_start:pool_start],
                qubits[pool_start:],
                symbols[param_start + nparams_conv : param_start + nparams_layer],
            )

        # Measurements
        circuit.add(gates.M(*[nbits - 1 - i for i in range(self.measured_qubits)]))

        return circuit

    def one_qubit_unitary(self, bit, symbols):
        """
        Internal helper function to make a circuit enacting a rotation of the bloch sphere about the X,
        Y and Z axis, that depends on the values in `symbols`.

        Args:
            bit: the qubit to apply the one-qubit unitaries to
            symbols: length 3 array containing the parameters
        Returns:
            Circuit containing the unitaries added to the specified qubit.
        """
        circuit = Circuit(self.nqubits)
        circuit.add(gates.RX(bit, symbols[0]))
        circuit.add(gates.RY(bit, symbols[1]))
        circuit.add(gates.RZ(bit, symbols[2]))

        return circuit

    def two_qubit_unitary(self, bits, symbols):
        """
        Internal helper function to create a circuit consisting of two qubit unitaries.

        Args:
            bits: the two qubits to apply the unitaries to
            symbols: length 15 array containing the parameters
        Returns:
            Circuit containing the unitaries added to the specified qubits.
        """

        circuit = Circuit(self.nqubits)

        if self.twoqubitansatz is None:
            circuit += self.one_qubit_unitary(bits[0], symbols[0:3])
            circuit += self.one_qubit_unitary(bits[1], symbols[3:6])
            circuit.add(gates.RZZ(bits[0], bits[1], symbols[6]))
            circuit.add(gates.RYY(bits[0], bits[1], symbols[7]))
            circuit.add(gates.RXX(bits[0], bits[1], symbols[8]))

            circuit += self.one_qubit_unitary(bits[0], symbols[9:12])
            circuit += self.one_qubit_unitary(bits[1], symbols[12:])

        else:
            circuit.add(self.twoqubitansatz.on_qubits(bits[0], bits[1]))
            circuit.set_parameters(symbols[0 : self.nparams_conv])

        return circuit

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        """
        Internal helper function to create a circuit to do a parameterized 'pooling' operation with controlled unitaries, which
        attempts to reduce entanglement down from two qubits to just one.

        Args:
            source_qubit: the control qubit.
            sink_qubit: the target qubit for the controlled unitaries.
            symbols: array with 6 elements containing the parameters.
        Returns:
            Circuit containing the unitaries added to the specified qubits.
        """
        pool_circuit = Circuit(self.nqubits)
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit += sink_basis_selector
        pool_circuit += source_basis_selector
        pool_circuit.add(gates.CNOT(source_qubit, sink_qubit))
        pool_circuit += sink_basis_selector.invert()

        return pool_circuit

    def set_circuit_params(self, angles, has_bias=False):
        """
        Sets the parameters of the QCNN circuit. Can be used to load previously saved or optimized parameters.

        Args:
            angles: the parameters to be loaded.
            has_bias: specify whether the list of angles contains the bias.

        """
        if not has_bias:
            params = list(angles)
        else:
            self._optimal_angles = angles
            params = list(angles[self.measured_qubits :])

        expanded_params = []
        nbits = self.nqubits
        for layer in range(self.nlayers):
            nleft = nbits / (2**layer)
            param_start = layer * self.nparams_layer
            conv_params = params[param_start : param_start + self.nparams_conv]
            pool_params = params[
                param_start + self.nparams_conv : param_start + self.nparams_layer
            ]
            pool_params += [-pool_params[2], -pool_params[1], -pool_params[0]]
            expanded_params += conv_params * int(nleft if nleft > 2 else 1)
            expanded_params += pool_params * int(nleft / 2)

        self._circuit.set_parameters(expanded_params)

    def Classifier_circuit(self, theta):
        """
        Args:
            theta: list or numpy.array with the biases and the angles to be used in the circuit.
        Returns:
            Circuit implementing the variational ansatz for angles "theta".
        """
        bias = np.array(theta[0 : self.measured_qubits])
        angles = theta[self.measured_qubits :]

        self.set_circuit_params(angles)
        return self._circuit

    def Predictions(self, circuit, theta, init_state, nshots=10000):
        """
        Args:
            theta: list or numpy.array with the biases to be used in the circuit.
            init_state: numpy.array with the quantum state to be classified.
            nshots: int number of runs of the circuit during the sampling process (default=10000).
        Returns:
            numpy.array() with predictions for each qubit, for the initial state.
        """
        bias = np.array(theta[0 : self.measured_qubits])
        if self.copy_init_state:
            init_state_copy = init_state.copy()
        else:
            init_state_copy = init_state
        circuit_exec = circuit(init_state_copy, nshots)
        result = circuit_exec.frequencies(binary=False)
        prediction = np.zeros(self.measured_qubits)

        for qubit in range(self.measured_qubits):
            for clase in range(self.nclasses):
                binary = bin(clase)[2:].zfill(self.measured_qubits)
                prediction[qubit] += result[clase] * (1 - 2 * int(binary[-qubit - 1]))

        return prediction / nshots + bias

    def square_loss(self, labels, predictions):
        """
        Args:
            labels: list or numpy.array with the qubit labels of the quantum states to be classified.
            predictions: list or numpy.array with the qubit predictions for the quantum states to be classified.
        Returns:
            numpy.float32 with the value of the square-loss function.
        """
        loss = 0
        for l, p in zip(labels, predictions):
            for qubit in range(self.measured_qubits):
                loss += (l[qubit] - p[qubit]) ** 2

        return loss / len(labels)

    def Cost_function(self, theta, data=None, labels=None, nshots=10000):
        """
        Args:
            theta: list or numpy.array with the biases and the angles to be used in the circuit.
            data: numpy.array data[page][word]  (this is an array of kets).
            labels: list or numpy.array with the labels of the quantum states to be classified.
            nshots: int number of runs of the circuit during the sampling process (default=10000).
        Returns:
            numpy.float32 with the value of the square-loss function.
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
            init_theta: list or numpy.array with the angles to be used in the circuit.
            data: the training data to be used in the minimization.
            labels: the corresponding ground truth for the training data.
            nshots: int number of runs of the circuit during the sampling process (default=10000).
            method: str 'classical optimizer for the minimization'. All methods from qibo.optimizers.optimize are suported (default='Powell').
        Returns:
            numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles.
        """
        from qibo.optimizers import optimize

        loss, optimal_angles, result = optimize(
            self.Cost_function, init_theta, args=(data, labels, nshots), method=method
        )

        self._optimal_angles = optimal_angles

        self.set_circuit_params(optimal_angles[self.measured_qubits :])
        return loss, optimal_angles

    def Accuracy(self, labels, predictions, sign=True, tolerance=1e-2):
        """
        Args:
            labels: numpy.array with the labels of the quantum states to be classified.
            predictions: numpy.array with the predictions for the quantum states classified.
            sign: if True, labels = np.sign(labels) and predictions = np.sign(predictions) (default=True).
            tolerance: float tolerance level to consider a prediction correct (default=1e-2).
        Returns:
            float with the proportion of states classified successfully.
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

    def predict(self, init_state, nshots=10000):
        """
        This function is used to produce predictions on new input state after the model is trained. Currently it only takes in one input data.

        Args:
            init_state: the input state to be predicted.
            nshots (default=10000): number of shots.
        Returns:
            numpy.array() with predictions for each qubit, for the initial state.
        """
        return self.Predictions(
            self._circuit, self._optimal_angles, init_state, nshots=nshots
        )
