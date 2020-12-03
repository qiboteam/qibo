import numpy as np
from qibo import matrices
from scipy.linalg import solve, sqrtm
from scipy.optimize import minimize


class Cholesky:

    def __init__(self, matrix=None, vector=None):
        self._matrix = matrix
        self._vector = vector

    @classmethod
    def from_matrix(cls, x):
        return cls(matrix=x)

    @classmethod
    def from_vector(cls, x):
        return cls(vector=x)

    @classmethod
    def decompose(cls, inmatrix):
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
        mc = np.copy(m)
        for x, y in zip(a, b):
            mc = np.delete(mc, x, y)
        return np.linalg.det(mc)

    @property
    def matrix(self):
        if self._matrix is None:
            n = int(np.sqrt(len(self.vector)))
            k = (len(self.vector) - n) // 2
            idx = np.tril_indices(n, k=-1)
            m_re = np.diag(self.vector[:n])
            m_re[idx] = self.vector[n:-k]
            m_im = np.zeros_like(m_re)
            m_im[idx] = self.vector[-k:]
            return np.matrix(m_re + 1j * m_im)
        return self._matrix

    @property
    def vector(self):
        if self._vector is None:
            diag = np.diag(self.matrix).real
            idx = np.tril_indices(len(self.matrix), k=-1)
            tril = self.matrix[idx]
            self._vector = np.concatenate([diag, tril.real, tril.imag])
        return self._vector

    def reconstruct(self):
        m = self.matrix.H * self.matrix
        return np.array(m) / np.trace(m)


class Tomography:

    def __init__(self, amplitudes, state=None, gates=None, gatesets=None):
        self.amplitudes = amplitudes
        self.state = state
        self._gates = gates
        self._gatesets = gatesets

        self._linear = None
        self._fitres = None

    @property
    def gates(self):
        if self._gates is None:
            beta = self.find_beta(self.state)
            self._gates = self._default_gates(beta)
        return self._gates

    @property
    def gatesets(self):
        if self._gatesets is None:
            self._gatesets = self._default_gatesets()
        return self._gatesets

    @staticmethod
    def find_beta(state):
        """Finds beta from state data."""
        refer_A = np.matrix([[1, 1, 1, 1],
                             [1, 1, -1, -1],
                             [1, -1, 1, -1],
                             [1, -1, -1, 1]])
        beta = solve(refer_A, state)
        return np.array(beta).flatten()

    @staticmethod
    def _default_gates(beta):
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

    def _default_gatesets(self):
        return [([0, 1, 2, 15], (0, 1, 2, 3), (0, 1, 2, 3)),
                ([11, 12, 13, 14], (1, 0, 3, 2), (0, 1, 2, 3)),
                ([3, 6, 7, 10], (2, 3, 0, 1), (0, 1, 2, 3)),
                ([4, 5, 8, 9], (3, 2, 1, 0), (0, 1, 2, 3))]

    @property
    def linear(self):
        """Linear estimation of the density matrix."""
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
        if self._fitres is None:
            raise ValueError
        return Cholesky.from_vector(self._fitres.x).reconstruct()

    @property
    def success(self):
        if self._fitres is None:
            raise ValueError
        return self._fitres.success

    def mle(self, x):
        rho = Cholesky.from_vector(x).reconstruct()
        loss = 0
        for gate, amp in zip(self.gates, self.amplitudes):
            tr = np.trace(rho * gate)
            loss += 0.5 * (amp - tr) ** 2 / tr
        return abs(loss)

    def minimize(self, tol=1e-9):
        """Fits density matrix to minimize MLE."""
        t_guess = Cholesky.decompose(self.linear).vector
        self._fitres = minimize(self.mle, t_guess, tol=tol)
        return self._fitres

    def fidelity(self, theory):
        sqrt_th = sqrtm(theory)
        return abs(np.trace(sqrtm(sqrt_th @ self.fit @ sqrt_th))) * 100
