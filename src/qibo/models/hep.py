from qibo import gates, K
from qibo.hamiltonians import Hamiltonian, matrices
from qibo.models.circuit import Circuit
from qibo.config import raise_error


class qPDF:
    """Variational Circuit for Quantum PDFs (qPDF).

    Args:
        ansatz (str): the ansatz name, options are 'Weighted' and 'Fourier'.
        layers (int): the number of layers for the ansatz.
        nqubits (int): the number of qubits for the circuit.
        multi_output (bool): allocates a multi-output model per PDF flavour (default is False).
    """
    def __init__(self, ansatz, layers, nqubits, multi_output=False):
        """Initialize qPDF."""
        if not isinstance(layers, int) or layers < 1: # pragma: no cover
            raise_error(RuntimeError, "Layers must be positive and integer.")
        if not isinstance(nqubits, int) or nqubits < 1: # pragma: no cover
            raise_error(RuntimeError, "Number of qubits must be positive and integer.")
        if not isinstance(multi_output, bool): # pragma: no cover
            raise_error(TypeError, "multi-output must be a boolean.")

        # parse ansatz
        if ansatz == 'Weighted':
            ansatz_function = ansatz_Weighted
        elif ansatz == 'Fourier':
            ansatz_function = ansatz_Fourier
        else: # pragma: no cover
            raise_error(NotImplementedError, f"Ansatz {ansatz} not found.")

        # load ansatz
        self.circuit, self.rotation, self.nparams = ansatz_function(layers, nqubits)

        # load hamiltonian
        if multi_output:
            self.hamiltonian = [qpdf_hamiltonian(
                nqubits, z_qubit=q) for q in range(nqubits)]
        else:
            self.hamiltonian = [qpdf_hamiltonian(nqubits)]

    def _model(self, state, hamiltonian):
        """Internal function for the evaluation of PDFs.

        Args:
            state (numpy.array): state vector.
            hamiltonian (qibo.hamiltonian.Hamiltonian): the Hamiltonian object.

        Returs:
            The qPDF object following the (1-z)/(1+z) structure.
        """
        z = hamiltonian.expectation(state)
        y = (1 - z) / (1 + z)
        return y

    def predict(self, parameters, x):
        """Predict PDF model from underlying circuit.

        Args:
            parameters (numpy.array): list of parameters for the gates.
            x (numpy.array): a numpy array with the points in x to be evaluated.

        Returns:
            A numpy array with the PDF values.
        """
        if len(parameters) != self.nparams: # pragma: no cover
            raise_error(
                RuntimeError, 'Mismatch between number of parameters and model size.')
        pdf = K.qnp.zeros(shape=(len(x), len(self.hamiltonian)), dtype='DTYPE')
        for i, x_value in enumerate(x):
            params = self.rotation(parameters, x_value)
            self.circuit.set_parameters(params)
            state = self.circuit()
            for flavour, flavour_hamiltonian in enumerate(self.hamiltonian):
                pdf[i, flavour] = self._model(state, flavour_hamiltonian)
        return pdf


def qpdf_hamiltonian(nqubits, z_qubit=0):
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
            h = K.np.kron(eye, h)

    elif z_qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = K.np.kron(eye, h)
        h = K.np.kron(matrices.Z, h)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1 == z_qubit:
                h = K.np.kron(matrices.Z, h)
            else:
                h = K.np.kron(eye, h)
    return Hamiltonian(nqubits, h)


def map_to(x):
    """Auxiliary function"""
    return 2 * K.np.pi * x


def maplog_to(x):
    """Auxiliary function"""
    return - K.np.pi * K.np.log10(x)


def ansatz_Fourier(layers, qubits=1):
    """Fourier Ansatz implementation. It is composed by 3 parameters per layer
    and qubit: U3(a, b, c) Ry(x) || U3(a, b, c) Ry(log x).

    Args:
        layers (int): number of layers.
        qubits (int): number of qubits.

    Returns:
        The circuit, the rotation function and the total number of parameters.
    """
    circuit = Circuit(qubits)
    for l in range(layers - 1):
        for q in range(qubits):
            for _ in range(2):
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
        for _ in range(2):
            circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.RZ(q, theta=0))
            circuit.add(gates.RY(q, theta=0))

    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers - 1):
            for q in range(qubits):
                p[i] = map_to(x)
                p[i + 1: i + 3] = theta[j: j + 2]
                i += 3
                j += 2

                p[i] = .5 * maplog_to(x)
                p[i + 1: i + 3] = theta[j: j + 2]
                i += 3
                j += 2
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
            p[i] = .5 * map_to(x)
            p[i + 1: i + 3] = theta[j: j + 2]
            i += 3
            j += 2

            p[i] = .5 * maplog_to(x)
            p[i + 1: i + 3] = theta[j: j + 2]
            i += 3
            j += 2
        return p

    nparams = 4 * layers * qubits + \
        (layers - 1) * int(K.np.ceil(qubits / 2)) * \
        (int(qubits > 1) + int(qubits > 2))

    return circuit, rotation, nparams


def ansatz_Weighted(layers, qubits=1):
    """Fourier Ansatz implementation. 4 parameters per layer and
    qubit: Ry(wx + a), Rz(v log(x) + b)

    Args:
        layers (int): number of layers.
        qubits (int): number of qubits.

    Returns:
        The circuit, the rotation function and the total number of
        parameters.
    """
    circuit = Circuit(qubits)
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

    nparams = 4 * layers * qubits + \
        (layers - 1) * int(K.np.ceil(qubits / 2)) * \
        (int(qubits > 1) + int(qubits > 2))
    return circuit, rotation, nparams
