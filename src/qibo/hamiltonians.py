# -*- coding: utf-8 -*-
from qibo import matrices, K, gates
from qibo.config import raise_error
from qibo.core.hamiltonians import Hamiltonian, SymbolicHamiltonian, TrotterHamiltonian
from qibo.core.terms import HamiltonianTerm
from qibo.models import Circuit, QAOA
import numpy as np


def multikron(matrix_list):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrices (list): List of matrices as ``np.ndarray``s.

    Returns:
        ``np.ndarray`` of the Kronecker product of all ``matrices``.
    """
    h = 1
    for m in matrix_list:
        h = K.np.kron(h, m)
    return h


def _build_spin_model(nqubits, matrix, condition):
    """Helper method for building nearest-neighbor spin model Hamiltonians."""
    h = sum(multikron(
      (matrix if condition(i, j) else matrices.I for j in range(nqubits)))
            for i in range(nqubits))
    return h


def XXZ(nqubits, delta=0.5, dense=True):
    """Heisenberg XXZ model with periodic boundary conditions.

    .. math::
        H = \\sum _{i=0}^N \\left ( X_iX_{i + 1} + Y_iY_{i + 1} + \\delta Z_iZ_{i + 1} \\right ).

    Args:
        nqubits (int): number of quantum bits.
        delta (float): coefficient for the Z component (default 0.5).
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.

    Example:
        .. testcode::

            from qibo.hamiltonians import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits
    """
    if dense:
        condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
        hx = _build_spin_model(nqubits, matrices.X, condition)
        hy = _build_spin_model(nqubits, matrices.Y, condition)
        hz = _build_spin_model(nqubits, matrices.Z, condition)
        matrix = hx + hy + delta * hz
        return Hamiltonian(nqubits, matrix)

    hx = K.np.kron(matrices.X, matrices.X)
    hy = K.np.kron(matrices.Y, matrices.Y)
    hz = K.np.kron(matrices.Z, matrices.Z)
    matrix = hx + hy + delta * hz
    terms = [HamiltonianTerm(matrix, i, i + 1) for i in range(nqubits - 1)]
    terms.append(HamiltonianTerm(matrix, nqubits - 1, 0))
    ham = SymbolicHamiltonian()
    ham.terms = terms
    return ham


def _OneBodyPauli(nqubits, matrix, dense=True, ground_state=None):
    """Helper method for constracting non-interacting X, Y, Z Hamiltonians."""
    if dense:
        condition = lambda i, j: i == j % nqubits
        ham = -_build_spin_model(nqubits, matrix, condition)
        return Hamiltonian(nqubits, ham)

    matrix = - matrix
    terms = [HamiltonianTerm(matrix, i) for i in range(nqubits)]
    ham = SymbolicHamiltonian(ground_state=ground_state)
    ham.terms = terms
    return ham


def X(nqubits, dense=True):
    """Non-interacting Pauli-X Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N X_i.

    Args:
        nqubits (int): number of quantum bits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    from qibo import K
    def ground_state():
        n = K.cast((2 ** nqubits,), dtype='DTYPEINT')
        state = K.ones(n, dtype='DTYPECPX')
        return state / K.sqrt(K.cast(n, dtype=state.dtype))
    return _OneBodyPauli(nqubits, matrices.X, dense, ground_state)


def Y(nqubits, dense=True):
    """Non-interacting Pauli-Y Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Y_i.

    Args:
        nqubits (int): number of quantum bits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    return _OneBodyPauli(nqubits, matrices.Y, dense)


def Z(nqubits, dense=True):
    """Non-interacting Pauli-Z Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Z_i.

    Args:
        nqubits (int): number of quantum bits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    return _OneBodyPauli(nqubits, matrices.Z, dense)


def TFIM(nqubits, h=0.0, dense=True):
    """Transverse field Ising model with periodic boundary conditions.

    .. math::
        H = - \\sum _{i=0}^N \\left ( Z_i Z_{i + 1} + h X_i \\right ).

    Args:
        nqubits (int): number of quantum bits.
        h (float): value of the transverse field.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    if dense:
        condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
        ham = -_build_spin_model(nqubits, matrices.Z, condition)
        if h != 0:
            condition = lambda i, j: i == j % nqubits
            ham -= h * _build_spin_model(nqubits, matrices.X, condition)
        return Hamiltonian(nqubits, ham)

    matrix = -(K.np.kron(matrices.Z, matrices.Z) + h * K.np.kron(matrices.X, matrices.I))
    terms = [HamiltonianTerm(matrix, i, i + 1) for i in range(nqubits - 1)]
    terms.append(HamiltonianTerm(matrix, nqubits - 1, 0))
    ham = SymbolicHamiltonian()
    ham.terms = terms
    return ham


def MaxCut(nqubits, dense=True):
    """Max Cut Hamiltonian.

    .. math::
        H = - \\sum _{i,j=0}^N  \\frac{1 - Z_i Z_j}{2}.

    Args:
        nqubits (int): number of quantum bits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    import sympy as sp

    Z = sp.symbols(f'Z:{nqubits}')
    V = sp.symbols(f'V:{nqubits**2}')
    sham = - sum(V[i * nqubits + j] * (1 - Z[i] * Z[j]) for i in range(nqubits) for j in range(nqubits))
    sham /= 2

    v = K.qnp.ones(nqubits**2, dtype='DTYPEINT')
    smap = {s: (i, matrices.Z) for i, s in enumerate(Z)}
    smap.update({s: (i, v[i]) for i, s in enumerate(V)})

    ham = SymbolicHamiltonian(sham, smap)
    if dense:
        return ham.dense
    return ham


from qibo.symbols import X as tspX
from qibo.symbols import Y as tspY
from qibo.symbols import Z as tspZ
class TSP:
    """
    This is a TSP class that enables us to implement TSP according to
    https://arxiv.org/pdf/1709.03489.pdf by Hadfield (2017).
    Here is an example of how the code can be run.

    num_cities = 3
    distance_matrix = np.random.rand(num_cities, num_cities)
    distance_matrix = distance_matrix.round(1)
    print(distance_matrix)
    small_tsp = TSP(distance_matrix)
    obj_hamil, mixer = small_tsp.TspHamiltonians(dense=False)
    initial_parameters = np.random.uniform(0, 1, 2)
    initial_state = small_tsp.PrepareInitialStateTsp([i for i in range(num_cities)])
    qaoa = QAOA(obj_hamil, mixer=mixer)

    """
    def __init__(self, distance_matrix):
        """
        Args:
            distance_matrix: a numpy matrix encoding the distance matrix.
        """
        self.distance_matrix = distance_matrix
        self.two_to_one = dict()
        self.num_cities = distance_matrix.shape[0]
        counter = 0
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.two_to_one[(i, j)] = counter
                counter += 1


    def PrepareObjTsp(self, dense=True):
        """ This function returns the objective Hamiltonian
        Args:
            dense: Whether the Hamiltonian should be dense.

        Returns: objective Hamiltonian that we wish to minimize the tsp distance.

        """
        form = 0
        for i in range(self.num_cities):
            for u in range(self.num_cities):
                for v in range(self.num_cities):
                    if u != v:
                        form += self.distance_matrix[u, v] * tspZ(self.two_to_one[u, i]) * tspZ(self.two_to_one[v, (i + 1) % self.num_cities])
        ham = SymbolicHamiltonian(form)
        if dense:
            ham = ham.dense
        return ham


    def SPlusTsp(self, u, i):
        """ This is a subroutine required for the mixer performing X+iY

        Args:
            u: this represents the index of the object.
            i: this represents the location of the slot

        Returns: the gate for the subroutine
        """
        return tspX(self.two_to_one[u, i]) + 1j*tspY(self.two_to_one[u,i])


    def SNegTsp(self, u, i):
        """ This is a subroutine required for the mixer performing X-iY

        Args:
            u: this represents the index of the object.
            i: this represents the location of the slot

        Returns: the gate for the subroutine
        """
        return tspX(self.two_to_one[u, i]) - 1j*tspY(self.two_to_one[u,i])


    def TspMixer(self, dense=True):
        """

        Args:
            dense: Indicate whether the Hamiltonian is dense

        Returns: Hamiltonian describing the mixer.

        """
        form = 0
        for i in range(self.num_cities):
            for u in range(self.num_cities):
                for v in range(self.num_cities):
                    if u != v:
                        form += self.SPlusTsp(u, i) * self.SPlusTsp(v, (i+1) % self.num_cities) * self.SNegTsp(u, (i+1)%self.num_cities) * self.SNegTsp(v, i) + self.SNegTsp(u, i) * self.SNegTsp(v, (i+1) % self.num_cities) * self.SPlusTsp(u, (i+1) % self.num_cities) * self.SPlusTsp(v, i)
        ham = SymbolicHamiltonian(form)
        if dense:
            ham = ham.dense
        return ham


    def TspHamiltonians(self, dense=True):
        """

        Args:
            dense: Indicates if the Hamiltonian is dense.

        Returns: Return a pair of Hamiltonian for the objective as well as the mixer.

        """
        return self.PrepareObjTsp(dense), self.TspMixer(dense)


    def PrepareInitialStateTsp(self, ordering):
        """
        To run QAOA by Hadsfield, we need to start from a valid permutation function to ensure feasibility.
        Args:
            ordering is a list which is a permutation from 0 to n-1
        Returns: return an initial state that can be used to start TSP QAOA.
        """
        c = Circuit(len(ordering)**2)
        for i in range(len(ordering)):
            c.add(gates.X(self.two_to_one[ordering[i], i]))
        result = c()
        return result.state(numpy=True)