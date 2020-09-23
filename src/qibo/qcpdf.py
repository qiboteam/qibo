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
from qibo.config import raise_error


class PDFModel(object):
    """Allocates a PDF object based on a circuit for a specific model ansatz.
    This class can be initialized with single and multiple output.

    Args:
        ansatz (str): the ansatz name.
        layers (int): the number of layers for the ansatz.
        nqubits (int): the circuit size.
        multi_output (boolean): default false, allocates a multi-output model per PDF flavour.
    """

    def __init__(self, ansatz, layers, nqubits, multi_output=False, tangling=True, measure_qubits=None):
        if measure_qubits is None:
            self.measure_qubits = nqubits
        else:
            self.measure_qubits = measure_qubits
        try:
            self.circuit, self.rotation, self.nparams = globals(
            )[f"ansatz_{ansatz}"](layers, nqubits,tangling=tangling)
        except KeyError:
            raise_error(NotImplementedError, "Ansatz not found.")
        if multi_output:
            self.hamiltonian = [qcpdf_hamiltonian(
                nqubits, z_qubit=q) for q in range(self.measure_qubits)]
        else:
            self.hamiltonian = [qcpdf_hamiltonian(nqubits)]
        self.multi_output = multi_output
        self.layers = layers
        self.nqubits=nqubits

    def _model(self, parameters, x, hamiltonian):
        """Internal function for the evaluation of PDFs."""
        params = self.rotation(parameters, x)
        self.circuit.set_parameters(params)
        c = self.circuit()
        z = hamiltonian.expectation(c)
        y = (1 - z) / (1 + z)
        return y

    def predict(self, parameters, x, force_zero=False):
        """Predict PDF model from underlying circuit.

        Args:
            parameters: the list of parameters for the gates.
            x (np.array): a numpy array with the points in x to be evaluated.
            force_zero (boolean): default false, forces PDF to zero when x=1.

        Returns:
            A numpy array with the PDF values.
        """
        pdf = np.zeros(shape=(len(self.hamiltonian), len(x)))
        for flavour, hamiltonian in enumerate(self.hamiltonian):
            for i in range(len(x)):
                pdf[flavour, i] = self._model(parameters, x[i], hamiltonian)
            if force_zero:
                pdf_at_one = self._model(parameters, 1, hamiltonian)
                pdf[flavour] -= pdf_at_one
        return pdf

    def get_parameters(self):
        """Return circuit parameters."""
        return self.circuit.get_parameters()


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
    return -np.pi * np.log10(x)

def entangler(circuit):
    qubits = circuit.nqubits
    if qubits > 1:
        for q in range(0, qubits, 2):
            circuit.add(gates.CZ(q, q + 1))
    if qubits > 2:
        for q in range(1, qubits + 1, 2):
            circuit.add(gates.CZ(q, (q + 1) % qubits))

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

def ansatz_1(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c)Ry(x)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if tangling: entangler(circuit)


    for q in range(qubits):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))


    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[3 * j: 3 * j + 3]
            p[i + 3] = map_to(x)
            i += 4
            j += 1
        return p

    nparams = 3 * (layers) * qubits

    return circuit, rotation, nparams


def ansatz_2(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(log x)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if tangling: entangler(circuit)

    for q in range(qubits):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers-1):
            for _ in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[3 * j: 3 * j + 3]
            p[i + 3] = -np.pi / 2 * np.log(x)
            i += 4
            j += 1
        return p

    nparams = 3 * layers * qubits

    return circuit, rotation, nparams


def ansatz_3(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x + log(x))
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if tangling: entangler(circuit)

    for q in range(qubits):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for _ in range(layers - 1):
            for _ in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x) + maplog_to(x)
                i += 4
                j += 1

        for _ in range(qubits):
            p[i: i + 3] = theta[3 * j: 3 * j + 3]
            p[i + 3] = map_to(x) + maplog_to(x)
            i += 4
            j += 1
        return p

    nparams = 3 * layers * qubits

    return circuit, rotation, nparams


def ansatz_4(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(pi x)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        if tangling: entangler(circuit)

    for q in range(qubits):
        circuit.add(gates.RZ(q, theta=0))
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
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = map_to(x)
                    i += 4
                    j += 1
            else:
                for q in range(qubits):
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = maplog_to(x)
                    i += 4
                    j += 1

        if layers % 2 == 0:
            for q in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 1
        else:
            for q in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 1
        return p

    nparams = 3 * layers * qubits

    return circuit, rotation, nparams


def ansatz_5(layers, qubits=1, tangling=True):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x) Rx (log(x))
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RX(q, theta=0))

        if tangling: entangler(circuit)


    for q in range(qubits):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RX(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(qubits):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4] = maplog_to(x)
                i += 5
                j += 3

        return p

    nparams = 3 * layers * qubits

    return circuit, rotation, nparams


def ansatz_6(layers, qubits=2, tangling=True):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)       CZ
     U3(a, b, c) Ry(log(x))  CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))

        if tangling: entangler(circuit)

    for q in range(0, qubits, 2):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3
        return p

    nparams = 6 * (layers) * qubits
    return circuit, rotation, nparams


def ansatz_7(layers, qubits=2, tangling=True):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)      CZ
     U3(a, b, c) Rx(log(x)) CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RX(q + 1, theta=0))

        if tangling: entangler(circuit)


    for q in range(0, qubits, 2):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))

        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RX(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3

        return p

    nparams = 6 * layers * qubits

    return circuit, rotation, nparams


def ansatz_8(layers, qubits=2, tangling=True):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)      CZ
     U3(a, b, c) Ry(log(x)) U3(a, b, c) CZ
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))

            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))

        if tangling: entangler(circuit)

    for q in range(0, qubits, 2):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6
        return p

    nparams = 6 * layers * qubits

    return circuit, rotation, nparams


def ansatz_9(layers, qubits=2, tangling=True):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)      
     U3(a, b, c) Rx(log(x)) U3(a, b, c) 
    """
    circuit = models.Circuit(qubits)
    for l in range(layers - 1):
        for q in range(0, qubits, 2):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))

            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RX(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))
            circuit.add(gates.RY(q + 1, theta=0))
            circuit.add(gates.RZ(q + 1, theta=0))

        if tangling: entangler(circuit)

    for q in range(0, qubits, 2):
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RX(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))
        circuit.add(gates.RY(q + 1, theta=0))
        circuit.add(gates.RZ(q + 1, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            for q in range(0, qubits, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4: i + 7] = theta[j + 3: j + 6]
                i += 7
                j += 6
        return p

    nparams = 6 * layers * qubits

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
        for l in range(layers):
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


def ansatz_w4(layers, qubits=1, tangling=True):
    """
    4 parameters per layer and qubit: Ry(wx + a), Rz(v log(x) + b)
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
                p[i + 1] = theta[j + 2] + theta[j + 3] * maplog_to(x)
                i += 2
                j += 4
        return p

    nparams = 4 * layers * qubits
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