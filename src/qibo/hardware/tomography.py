import numpy as np
from qibo import matrices
from qibo.config import raise_error
from scipy.linalg import solve, sqrtm
from scipy.optimize import minimize


class Cholesky:
    """Cholesky decomposition of the density matrix.

    Note that this object can be initialized passing either the ``matrix``
    or ``vector`` argument, not both.

    Args:
        matrix (np.ndarray): Cholesky decomposition as a lower triangular
            complex matrix.
        vector (np.ndarray): Cholesky decomposition matrix values re-arranged
            to a real vector.
    """

    def __init__(self, matrix=None, vector=None):
        if matrix is not None and vector is not None:
            raise_error(ValueError, "Cannot initialize Cholesky object using "
                                    "both a matrix and a vector. Please use "
                                    "one of them.")
        if matrix is not None and not isinstance(matrix, np.ndarray):
            raise_error(TypeError, "Matrix must be a ``np.ndarray`` but is {}."
                                   "".format(type(matrix)))
        if vector is not None and not isinstance(vector, np.ndarray):
            raise_error(TypeError, "Vector must be a ``np.ndarray`` but is {}."
                                   "".format(type(matrix)))
        self._matrix = matrix
        self._vector = vector

    @classmethod
    def from_matrix(cls, x):
        """Creates :class:`qibo.numpy.tomography.Cholesky` object from matrix."""
        return cls(matrix=x)

    @classmethod
    def from_vector(cls, x):
        """Creates :class:`qibo.numpy.tomography.Cholesky` object from vector."""
        return cls(vector=x)

    @classmethod
    def decompose(cls, inmatrix):
        """Creates :class:`qibo.numpy.tomography.Cholesky` by decomposing a matrix."""
        D = np.linalg.det(inmatrix)
        M11 = cls.minor(inmatrix, [0, 0], [0, 1])
        M12 = cls.minor(inmatrix, [0, 1], [0, 1])
        M1122 = cls.minor(inmatrix, [[0, 1], [0, 1]], [0, 1])
        M1223 = cls.minor(inmatrix, [[0, 1], [1, 2]], [0, 1])
        M1123 = cls.minor(inmatrix, [[0, 1], [0, 2]], [0, 1])
        tmatrix = np.zeros_like(inmatrix)
        tmatrix[0, 0] = np.sqrt(D / M11)
        tmatrix[1, 0] = M12 / np.sqrt(M11 * M1122)
        tmatrix[1, 1] = np.sqrt(M11 / M1122)
        tmatrix[2, 0] = M1223 / np.sqrt(inmatrix[3,3]) / np.sqrt(M1122)
        tmatrix[2, 1] = M1123 / np.sqrt(inmatrix[3,3]) / np.sqrt(M1122)
        tmatrix[2, 2] = np.sqrt(M1122) / np.sqrt(inmatrix[3,3])
        tmatrix[3, 0] = inmatrix[3,0] / np.sqrt(inmatrix[3,3])
        tmatrix[3, 1] = inmatrix[3,1] / np.sqrt(inmatrix[3,3])
        tmatrix[3, 2] = inmatrix[3,2] / np.sqrt(inmatrix[3,3])
        tmatrix[3, 3] = np.sqrt(inmatrix[3,3])
        return cls(matrix=tmatrix)

    @staticmethod
    def minor(m, a, b):
        """Calculates minor of a matrix.

        Helper method for :class:`qibo.numpy.tomography.Cholesky.decompose`.
        """
        mc = np.copy(m)
        for x, y in zip(a, b):
            mc = np.delete(mc, x, y)
        return np.linalg.det(mc)

    @property
    def matrix(self):
        """Cholesky decomposition represented as a complex lower triangular matrix."""
        if self._matrix is None:
            n = int(np.sqrt(len(self.vector)))
            k = (len(self.vector) - n) // 2
            idx = np.tril_indices(n, k=-1)
            m_re = np.diag(self.vector[:n])
            m_re[idx] = self.vector[n:-k]
            m_im = np.zeros_like(m_re)
            m_im[idx] = self.vector[-k:]
            self._matrix = m_re + 1j * m_im
        return self._matrix

    @property
    def vector(self):
        """Cholesky decomposition represented as a real vector.

        Useful for the MLE minimization in
        :meth:`qibo.numpy.tomography.Tomography.fit`.
        """
        if self._vector is None:
            diag = np.diag(self.matrix).real
            idx = np.tril_indices(len(self.matrix), k=-1)
            tril = self.matrix[idx]
            self._vector = np.concatenate([diag, tril.real, tril.imag])
        return self._vector

    def reconstruct(self):
        """Reconstructs density matrix from its Cholesky decomposition."""
        m = self.matrix.conj().T @ self.matrix
        return np.array(m) / np.trace(m)


class Tomography:
    """Performs density matrix tomography from experimental measurements.

    Follows the tomography approach presented in
    `Phys. Rev. A 81, 062325 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.81.062325>`_.

    Args:
        amplitudes (np.ndarray): Array of shape (16,) with the measured
            amplitudes.
        state (np.ndarray): Optional array of shape (4,) with the state values
            to use in the ``beta`` parameter calculation.
            Relevant only if the default beta calculation is used.
        gates (np.ndarray): Optional array of shape (16, 4, 4) with the gates
            to use for the linear estimation of the density matrix.
            If ``None`` the default gates will be calculated using the given
            ``state``. See
            :meth:`qibo.numpy.tomography.Tomography.default_gates` for more
            details.
        gatesets (list): List of gate indices to use for the linear estimation
            of the density matrix.
            If ``None`` the default configuration will be used. See
            :meth:`qibo.numpy.tomography.Tomography.default_gatesets` for more
            details.
    """

    def __init__(self, amplitudes, state=None, gates=None, gatesets=None):
        self.amplitudes = amplitudes
        self.state = state
        self._gates = gates
        self._gatesets = gatesets

        self._linear = None
        self._fitres = None
        self._fitrho = None

    @property
    def gates(self):
        """Array of gates to use in the linear density matrix estimation."""
        if self._gates is None:
            beta = self.find_beta(self.state)
            self._gates = self.default_gates(beta)
        return self._gates

    @property
    def gatesets(self):
        """List of gate indices to use in the linear density matrix estimation."""
        if self._gatesets is None:
            self._gatesets = self.default_gatesets()
        return self._gatesets

    @staticmethod
    def find_beta(state):
        """Finds beta from state data.

        Beta is then used to define the default gate matrices.
        """
        refer_A = np.array([[1, 1, 1, 1],
                            [1, 1, -1, -1],
                            [1, -1, 1, -1],
                            [1, -1, -1, 1]])
        beta = solve(refer_A, state)
        return np.array(beta).flatten()

    @staticmethod
    def default_gates(beta):
        """Calculates default gate matrices for a given beta."""
        I, X, Y, Z = matrices.I, matrices.X, matrices.Y, matrices.Z
        return np.array([
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Z) + beta[3]*np.kron(Z,Z),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Z) - beta[3]*np.kron(Z,Z),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,Z) - beta[3]*np.kron(Z,Z),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) + beta[2]*np.kron(I,Z) + beta[3]*np.kron(Y,Z),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) + beta[2]*np.kron(I,Y) + beta[3]*np.kron(Y,Y),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) - beta[2]*np.kron(I,X) - beta[3]*np.kron(Y,X),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) - beta[2]*np.kron(I,Z) - beta[3]*np.kron(Y,Z),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) + beta[2]*np.kron(I,Z) - beta[3]*np.kron(X,Z),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) + beta[2]*np.kron(I,Y) - beta[3]*np.kron(X,Y),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) - beta[2]*np.kron(I,X) + beta[3]*np.kron(X,X),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) - beta[2]*np.kron(I,Z) + beta[3]*np.kron(X,Z),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Y) + beta[3]*np.kron(Z,Y),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Y) - beta[3]*np.kron(Z,Y),
            beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,X) - beta[3]*np.kron(Z,X),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,X) + beta[3]*np.kron(Z,X),
            beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,Z) + beta[3]*np.kron(Z,Z)
        ])

    def default_gatesets(self):
        """Calculates the default gate indices."""
        return [([0, 1, 2, 15], (0, 1, 2, 3), (0, 1, 2, 3)),
                ([11, 12, 13, 14], (1, 0, 3, 2), (0, 1, 2, 3)),
                ([3, 6, 7, 10], (2, 3, 0, 1), (0, 1, 2, 3)),
                ([4, 5, 8, 9], (3, 2, 1, 0), (0, 1, 2, 3))]

    @property
    def linear(self):
        """Linear estimation of the density matrix from given measurements."""
        if self._linear is None:
            dtype = self.gates.dtype
            self._linear = np.zeros_like(self.gates[0])
            for i, (seti, seta, setb) in enumerate(self.gatesets):
                setA = self.gates[seti][:, seta, setb]
                minus = sum(np.dot(self.gates[seti][:, sa, sb], self._linear[sb, sa])
                            for _, sa, sb in self.gatesets[:i])
                setB = self.amplitudes[seti].astype(dtype) - minus
                self._linear[setb, seta] = solve(setA, setB)
        return self._linear

    @property
    def fit(self):
        """MLE estimation of the density matrix from given measurements."""
        if self._fitrho is None:
            if self._fitres is None:
                raise_error(ValueError, "Cannot return fitted density matrix "
                                        "before `minimize` is called.")
            self._fitrho = Cholesky.from_vector(self._fitres.x).reconstruct()
        return self._fitrho

    @property
    def success(self):
        """Bool that shows if the MLE minimization was successful."""
        if self._fitres is None:
            raise_error(ValueError, "Cannot return minimization success "
                                    "before `minimize` is called.")
        return self._fitres.success

    def mle(self, x):
        """Calculates the MLE loss.

        Args:
            x (np.ndarray): Cholesky decomposition of the density matrix
                decomposed to a real vector.
        """
        rho = Cholesky.from_vector(x).reconstruct()
        loss = 0
        for gate, amp in zip(self.gates, self.amplitudes):
            tr = np.trace(rho * gate)
            loss += 0.5 * (amp - tr) ** 2 / tr
        return abs(loss)

    def minimize(self, tol=1e-5):
        """Finds density matrix that minimizes MLE."""
        t_guess = Cholesky.decompose(self.linear).vector
        self._fitres = minimize(self.mle, t_guess, tol=tol)
        return self._fitres

    def fidelity(self, theory):
        """Fidelity between the MLE-fitted density matrix and a given one."""
        sqrt_th = sqrtm(theory)
        return abs(np.trace(sqrtm(sqrt_th @ self.fit @ sqrt_th))) * 100

    # Need to do this part dynamically
    @staticmethod
    def gate_sequence(nqubits):
        if nqubits != 2:
            raise_error(NotImplementedError)
        from qibo import gates

        pi = np.pi
        pio2 = pi / 2
        
        return [
            [gates.I(0), gates.I(1)],
            [gates.RX(0, pi), gates.I(1)],
            [gates.I(0), gates.RX(1, pi)],
            [gates.RX(0, pio2), gates.I(1)],
            [gates.RX(0, pio2), gates.RX(1, pio2)],
            [gates.RX(0, pio2), gates.RY(1, pio2)],
            [gates.RX(0, pio2), gates.RX(1, pi)],
            [gates.RY(0, pio2), gates.I(1)],
            [gates.RY(0, pio2), gates.RX(1, pio2)],
            [gates.RY(0, pio2), gates.RY(1, pio2)],
            [gates.RY(0, pio2), gates.RX(1, pi)],
            [gates.I(0), gates.RX(1, pio2)],
            [gates.RX(0, pi), gates.RX(1, pio2)],
            [gates.I(0), gates.RY(1, pio2)],
            [gates.RX(0, pi), gates.RY(1, pio2)],
            [gates.RX(0, pi), gates.RX(1, pi)],
        ]

    @staticmethod
    def basis_states(nqubits):
        if nqubits != 2:
            raise_error(NotImplementedError)

        from qibo import gates
        return [
            [gates.RX(0, np.pi / 2), gates.RX(0, -np.pi / 2)], #|00>
            [gates.RX(0, np.pi)], #|10>
            [gates.RX(1, np.pi)], #|01>
            [gates.RX(1, np.pi), gates.Align(0, 1), gates.RX(0, np.pi)] #|11>
        ]

