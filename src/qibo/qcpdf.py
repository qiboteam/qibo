"""Models for Quantum Circuit PDFs.
    TODO:
        * documentation
        * tests
        * ansatz cleanup
"""
import numpy as np
import qibo.models as models
from qibo.hamiltonians import Hamiltonian, matrices
from qibo import gates
from qibo.config import raise_error


class PDFModel(object):
    """Allocates a PDF object based on a circuit for a specific model ansatz.
    This class can be initialized with single and multiple output.

    Args:
        ansatz (int): the anstaz number.
        layers (int): the number of layers for the ansatz.
        nqubits (int): the circuit size.
        multi_output (boolean): default false, allocates a multi-output model per PDF flavour.
    """

    def __init__(self, ansatz, layers, nqubits, multi_output=False):
        try:
            self.circuit, self.rotation, self.nparams = globals(
            )[f"ansatz_{ansatz}"](layers, nqubits)
        except KeyError:
            raise_error(NotImplementedError, "Ansatz not found.")
        if multi_output:
            self.hamiltonian = [qcpdf_hamiltonian(
                nqubits, z_qubit=q) for q in range(nqubits)]
        else:
            self.hamiltonian = [qcpdf_hamiltonian(nqubits)]
        self.multi_output = multi_output

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
            if _ == z_qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h)


def map_to(x):
    return 2 * np.pi * x


def maplog_to(x):
    return -np.pi * np.log10(x)


def ansatz_1(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c)Ry(x)
    """
    circuit = models.Circuit(qubits)
    if qubits != 1:
        for _ in range(layers):
            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(0, qubits, 2):
                circuit.add(gates.CZ(q, q + 1))

            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
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

            for _ in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 1
            return p

        nparams = 3 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3
            return p

        nparams = 3 * layers
    return circuit, rotation, nparams


def ansatz_2(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(log x)
    """
    circuit = models.Circuit(qubits)
    if qubits != 1:
        for _ in range(layers):
            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(0, qubits, 2):
                circuit.add(gates.CZ(q, q + 1))

            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
                for _ in range(qubits):
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = maplog_to(x)
                    i += 4
                    j += 1

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

        nparams = 3 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3
            return p

        nparams = 3 * layers
    return circuit, rotation, nparams


def ansatz_3(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x + log(x))
    """
    circuit = models.Circuit(qubits)
    if qubits != 1:
        for _ in range(layers):
            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(0, qubits, 2):
                circuit.add(gates.CZ(q, q + 1))

            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
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

            for _ in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x) + maplog_to(x)
                i += 4
                j += 1
            return p

        nparams = 3 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for _ in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x) + maplog_to(x)
                i += 4
                j += 3
            return p

        nparams = 3 * layers
    return circuit, rotation, nparams


def ansatz_4(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(pi x)
    """
    circuit = models.Circuit(qubits)
    if qubits != 1:
        for _ in range(layers):
            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(0, qubits, 2):
                circuit.add(gates.CZ(q, q + 1))

            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

        for q in range(qubits):
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
                for q in range(qubits):
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = map_to(x)
                    i += 4
                    j += 1

                for q in range(qubits):
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = maplog_to(x)
                    i += 4
                    j += 1

            for q in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 1

            for q in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 1
            return p

        nparams = 3 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(0, layers, 2):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3
            return p

        nparams = 3 * layers
    return circuit, rotation, nparams


def ansatz_5(layers, qubits=1):
    """
    3 parameters per layer and qubit: U3(a, b, c) Ry(x) Rx (log(x))
    """
    circuit = models.Circuit(qubits)
    if qubits != 1:
        for l in range(layers):
            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RX(q, theta=0))

            for q in range(0, qubits, 2):
                circuit.add(gates.CZ(q, q + 1))

            for q in range(qubits):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RX(q, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

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
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = map_to(x)
                    p[i + 4] = maplog_to(x)
                    i += 5
                    j += 1

                for q in range(qubits):
                    p[i: i + 3] = theta[3 * j: 3 * j + 3]
                    p[i + 3] = map_to(x)
                    p[i + 4] = maplog_to(x)
                    i += 5
                    j += 1

            for q in range(qubits):
                p[i: i + 3] = theta[3 * j: 3 * j + 3]
                p[i + 3] = map_to(x)
                p[i + 4] = maplog_to(x)
                i += 5
                j += 1
            return p

        nparams = 3 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RX(0, theta=0))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                p[i + 4] = maplog_to(x)
                i += 5
                j += 3
            return p

        nparams = 3 * layers
    return circuit, rotation, nparams


def ansatz_6(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)       CZ
     U3(a, b, c) Ry(log(x))  CZ
    """
    circuit = models.Circuit(qubits)
    if qubits != 2:
        for l in range(layers):
            for q in range(0, qubits, 2):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))
                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))

                circuit.add(gates.CZ(q, q + 1))

                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))
                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

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

        nparams = 6 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))

            circuit.add(gates.CZ(0, 1))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3
            return p

        nparams = 2 * 3 * layers
    return circuit, rotation, nparams


def ansatz_7(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x)      CZ
     U3(a, b, c) Rx(log(x)) CZ
    """
    circuit = models.Circuit(qubits)
    if qubits != 2:
        for l in range(layers):
            for q in range(0, qubits, 2):
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))
                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RX(q + 1, theta=0))

                circuit.add(gates.CZ(q, q + 1))

                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))

                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RY(q + 1, theta=0))
                circuit.add(gates.RZ(q + 1, theta=0))
                circuit.add(gates.RX(q + 1, theta=0))

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

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

        nparams = 6 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))

            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RX(1, theta=0))

            circuit.add(gates.CZ(0, 1))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = map_to(x)
                i += 4
                j += 3

                p[i: i + 3] = theta[j: j + 3]
                p[i + 3] = maplog_to(x)
                i += 4
                j += 3
            return p

        nparams = 2 * 3 * layers
    return circuit, rotation, nparams


def ansatz_8(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)      CZ
     U3(a, b, c) Ry(log(x)) U3(a, b, c) CZ
    """
    circuit = models.Circuit(qubits)
    if qubits != 2:
        for l in range(layers):
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

                circuit.add(gates.CZ(q, q + 1))

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

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

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

        nparams = 6 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))

            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))

            circuit.add(gates.CZ(0, 1))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
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

        nparams = 2 * 6 * layers
    return circuit, rotation, nparams


def ansatz_9(layers, qubits=2):
    assert qubits % 2 == 0
    """
     3 parameters per layer and qubit:
     U3(a, b, c) Ry(x) U3(a, b, c)      CZ
     U3(a, b, c) Rx(log(x)) U3(a, b, c) CZ
    """
    circuit = models.Circuit(qubits)
    if qubits != 2:
        for l in range(layers):
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

                circuit.add(gates.CZ(q, q + 1))

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

            for q in range(1, qubits + 1, 2):
                circuit.add(gates.CZ(q, (q + 1) % qubits))

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

        nparams = 6 * (2 * layers + 1) * qubits
    else:
        for _ in range(layers):
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RZ(0, theta=0))

            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RX(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))
            circuit.add(gates.RY(1, theta=0))
            circuit.add(gates.RZ(1, theta=0))

            circuit.add(gates.CZ(0, 1))

        def rotation(theta, x):
            p = circuit.get_parameters()
            i = 0
            j = 0
            for l in range(layers):
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

        nparams = 2 * 6 * layers
    return circuit, rotation, nparams
