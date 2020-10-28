"""Models for Quantum Circuit PDFs.
    TODO:
        * documentation
        * tests
        * ansatz cleanup
"""
import numpy as np
from qibo import models
from qibo.hamiltonians import Hamiltonian, matrices
from qibo import gates
from qibo.config import raise_error, K, DTYPES


class PDFModel(object):
    """Allocates a PDF object based on a circuit for a specific model ansatz.
    This class can be initialized with single and multiple output.

    Args:
        ansatz (str): the ansatz name.
        layers (int): the number of layers for the ansatz.
        nqubits (int): the circuit size.
        multi_output (boolean): default false, allocates a multi-output model per PDF flavour.
        fuse (boolean): fuse gates.
        meansure_qubits (list): list of qubits for measurement.
    """

    def __init__(self, ansatz, layers, nqubits, multi_output=False, fuse=False, measure_qubits=None):
        if measure_qubits is None:
            self.measure_qubits = nqubits
        else:
            self.measure_qubits = measure_qubits
        try:
            self.circuit, self.rotation, self.nparams = globals(
            )[f"ansatz_{ansatz}"](layers, nqubits)
        except KeyError:
            raise_error(NotImplementedError, "Ansatz not found.")
        if multi_output:
            self.hamiltonian = [qcpdf_hamiltonian(
                nqubits, z_qubit=q) for q in range(self.measure_qubits)]
        else:
            self.hamiltonian = [qcpdf_hamiltonian(nqubits)]
        if fuse:
            self.circuit = self.circuit.fuse()
        self.multi_output = multi_output
        self.layers = layers
        self.nqubits = nqubits

    def _model(self, state, hamiltonian):
        """Internal function for the evaluation of PDFs."""
        z = hamiltonian.expectation(state)
        y = (1 - z) / (1 + z)
        return y

    def predict(self, parameters, x):
        """Predict PDF model from underlying circuit.

        Args:
            parameters: the list of parameters for the gates.
            x (np.array): a numpy array with the points in x to be evaluated.

        Returns:
            A numpy array with the PDF values.
        """
        if len(parameters) != self.nparams:
            raise_error(RuntimeError, 'Mismatch between number of parameters and model size.')
        pdf = np.zeros(shape=(len(x), len(self.hamiltonian)))
        for i, x_value in enumerate(x):
            params = self.rotation(parameters, x_value)
            self.circuit.set_parameters(params)
            state = self.circuit()
            for flavour, flavour_hamiltonian in enumerate(self.hamiltonian):
                pdf[i, flavour] = self._model(state, flavour_hamiltonian)
        return pdf

    def correlation(self, parameters, x):
        """Predict PDF correlations from underlying circuit.
        Args:
            parameters: the list of parameters for the gates.
            x (np.array): a numpy array with the points in x to be evaluated.

        Returns:
            A numpy array with the correlations values.
        """
        if len(parameters) != self.nparams:
            raise_error(RuntimeError, 'Mismatch between number of parameters and model size.')
        correlation = np.zeros(shape=(len(x), len(self.hamiltonian), len(self.hamiltonian)))
        for i, x_value in enumerate(x):
            params = self.rotation(parameters, x_value)
            self.circuit.set_parameters(params)
            state = self.circuit()
            for flavour1 in range(len(self.hamiltonian)):
                for flavour2 in range(flavour1 + 1):
                    h1 = self.hamiltonian[flavour1].matrix
                    h2 = self.hamiltonian[flavour2].matrix
                    H = h1 @ h2
                    hamiltonian = Hamiltonian(self.nqubits, H)

                    correlation[i, flavour1, flavour2] = self._model(state, hamiltonian)

        correlation = 0.5 * (correlation + np.transpose(correlation, axes=[0,2,1]))
        return correlation


def qcpdf_hamiltonian(nqubits, z_qubit=0):
    """Precomputes Hamiltonian.
    Args:
        nqubits (int): number of qubits.
        z_qubit (int): qubit where the Z measurement is applied, must be z_qubit < nqubits
    Returns:
        An Hamiltonian object.
    """
    eye = matrices.I
    if z_qubit == 0:
        h = matrices.Z
        for _ in range(nqubits - 1):
            h = np.kron(eye, h)

    elif z_qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = np.kron(eye, h)
        h = np.kron(matrices.Z, h)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1== z_qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h)


def map_to(x):
    return 2 * np.pi * x


def maplog_to(x):
    return - np.pi * np.log10(x)


def entangler(circuit):
    qubits = circuit.nqubits
    if qubits > 1:
        for q in range(0, qubits, 2):
            circuit.add(gates.CZPow(q, q + 1, theta=0))
    if qubits > 2:
        for q in range(1, qubits + 1, 2):
            circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

def rotation_entangler(qubits, p, theta, i, j):
    if qubits > 1:
        for q in range(0, qubits, 2):
            p[i] = theta[j]
            i += 1
            j += 1
    if qubits > 2:
        for q in range(1, qubits + 1, 2):
            p[i + 1] = theta[j + 1]
            i += 1
            j += 1
    return p, theta, i, j


def ansatz_0(layers, qubits=1, tangling=True):
    """
    2 parameters per layer and qubit: Ry(x + a) Rz(b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if tangling: entangler(circuit)

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i] = theta[i] + map_to(x)
                p[i + 1] = theta[i + 1]
                i += 2

        for _ in range(qubits):
            p[i] = theta[i] + map_to(x)
            p[i + 1] = theta[i + 1]
            i += 2
        return p

    nparams = 2 * (layers) * qubits

    return circuit, rotation, nparams

def ansatz_1(layers, qubits=1):
    """
    3 parameters per layer and qubit: Ry(x)U3(a, b, c)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = map_to(x)
            i += 4
            j += 3
        return p

    nparams = 3 * (layers) * qubits + (layers - 1) * qubits // 2 * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_2(layers, qubits=1):
    """
    3 parameters per layer and qubit:  Ry(log x)U3(a, b, c)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))


        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                i += 4
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = .5 * maplog_to(x)
            i += 4
            j += 3
        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_3(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x + log(x))
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * (.5 * maplog_to(x) + map_to(x))
                i += 4
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = .5 * (.5 * maplog_to(x) + map_to(x))
            i += 4
            j += 3
        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_4(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x) || U3(a, b, c) Ry(log x)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            if l % 2 == 0:
                for q in range(qubits):
                    p[i: i + 3] = theta[j: j + 3]
                    p[i + 3] = map_to(x)
                    i += 4
                    j += 3
            else:
                for q in range(qubits):
                    p[i: i + 3] = theta[j: j + 3]
                    p[i + 3] = .5 * maplog_to(x)
                    i += 4
                    j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
        if layers % 2 == 0:
            for q in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3
        else:
            for q in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                i += 4
                j += 3
        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_5(layers, qubits=1): # Not promising
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x) Rx (log(x))
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RX(q, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))


    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RX(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                p[i + 4] = map_to(x)
                i += 5
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = .5 * maplog_to(x)
            p[i + 4] = map_to(x)
            i += 5
            j += 3
        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_6(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)       CZ
     U3(a, b, c) Ry(log(x))  CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(0, qubits, 2):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                i += 4
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for q in range(0, qubits, 2):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = map_to(x)
            i += 4
            j += 3

            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = .5 * maplog_to(x)
            i += 4
            j += 3

        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))
    return circuit, rotation, nparams


def ansatz_7(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)      CZ
     U3(a, b, c) Rx(log(x)) CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(0, qubits, 2):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RX(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                i += 4
                j += 3
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for q in range(0, qubits, 2):
            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = map_to(x)
            i += 4
            j += 3

            p[i: i + 3] = theta[j: j + 3]
            p[i + 3] = .5 * maplog_to(x)
            i += 4
            j += 3

        return p

    nparams = 3 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_8(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)      CZ
     U3(a, b, c) Ry(log(x)) U3(a, b, c) CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(0, qubits, 2):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
        return p

    nparams = 6 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_9(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)
     U3(a, b, c) Rx(log(x)) U3(a, b, c)
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RX(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(0, qubits, 2):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RX(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
        return p

    nparams = 6 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_10(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x) U3(a, b, c) || U3(a, b, c) Ry(log x) U3(a, b, c)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, q + 1, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))

    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            if l % 2 == 0:
                for q in range(qubits):
                    p[i: i + 3] = theta[j: j + 3]
                    p[i + 3] = map_to(x)
                    p[i + 4: i + 6]= theta[j + 3: j + 5]
                    i += 6
                    j += 5
            else:
                for q in range(qubits):
                    p[i: i + 3] = theta[j: j + 3]
                    p[i + 3] = .5 * maplog_to(x)
                    p[i + 4: i + 6] = theta[j + 3: j + 5]
                    i += 6
                    j += 5
            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
        if layers % 2 == 0:
            for q in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 6] = theta[j + 3: j + 5]
                i += 6
                j += 5
        else:
            for q in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = .5 * maplog_to(x)
                p[i + 4: i + 6] = theta[j + 3: j + 5]
                i += 6
                j += 5
        return p

    nparams = 5 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams

def ansatz_w1(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: Ry(wx + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if tangling: entangler(circuit)
    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(qubits):
                p[i] = theta[j] + theta[j + 1] * map_to(x)
                p[i + 1] = theta[j + 2]
                i += 2
                j += 3


        for q in range(qubits):
            p[i] = theta[j] + theta[j + 1] * map_to(x)
            p[i + 1] = theta[j + 2]
            i += 2
            j += 3
        return p

    nparams = 3 * layers * qubits
    return circuit, rotation, nparams

def ansatz_w2(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: Ry(w log(x) + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if tangling: entangler(circuit)
    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(qubits):
                p[i] = theta[j] + theta[j + 1] * maplog_to(x)
                p[i + 1] = theta[j + 2]
                i += 2
                j += 3
        return p

    nparams = 3 * layers * qubits
    return circuit, rotation, nparams


def ansatz_w3(layers, qubits=1, tangling=True):
    """
    4 parameters per layer and qubit: Ry(wx + a), Rz(vx + b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if tangling: entangler(circuit)
    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(qubits):
                p[i] = theta[j] + theta[j + 1] * map_to(x)
                p[i + 1] = theta[j + 2] + theta[j + 3] * map_to(x)
                i += 2
                j += 4
        return p

    nparams = 4 * layers * qubits
    return circuit, rotation, nparams


def ansatz_w4(layers, qubits=1):
    """
    4 parameters per layer and qubit: Ry(wx + a), Rz(v log(x) + b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if qubits > 1:
            for q in range(0, qubits, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))
        if qubits > 2:
            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZPow(q, (q + 1) % qubits, theta=0))
    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(qubits):
                p[i] = theta[j] + theta[j + 1] * map_to(x)
                p[i + 1] = theta[j + 2] + theta[j + 3] * maplog_to(x)
                i += 2
                j += 4

            if qubits > 1:
                for q in range(0, qubits, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1
            if qubits > 2:
                for q in range(1, qubits + 1, 2):
                    p[i] = theta[j]
                    i += 1
                    j += 1

        for q in range(qubits):
            p[i] = theta[j] + theta[j + 1] * map_to(x)
            p[i + 1] = theta[j + 2] + theta[j + 3] * maplog_to(x)
            i += 2
            j += 4

        return p

    nparams = 4 * layers * qubits + (layers - 1) * int(np.ceil(qubits / 2)) * (int(qubits > 1) + int(qubits > 2))
    return circuit, rotation, nparams


def ansatz_w5(layers, qubits=1, tangling=True):
    """
    4 parameters per layer and qubit: Ry(w log(x) + a), Rz(v log(x) + b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
        if tangling: entangler(circuit)
    for q in range(qubits):
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(qubits):
                p[i] = theta[j] + theta[j + 1] * maplog_to(x)
                p[i + 1] = theta[j + 2] + theta[j + 3] * maplog_to(x)
                i += 2
                j += 4
        return p

    nparams = 4 * layers * qubits
    return circuit, rotation, nparams
