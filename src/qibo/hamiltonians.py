# -*- coding: utf-8 -*-
from qibo import matrices
from qibo import numpy as qnp
from qibo.config import raise_error
from qibo.core.hamiltonians import Hamiltonian, SymbolicHamiltonian, TrotterHamiltonian


def _build_spin_model(nqubits, matrix, condition):
    """Helper method for building nearest-neighbor spin model Hamiltonians."""
    h = sum(SymbolicHamiltonian.multikron(
      (matrix if condition(i, j) else matrices.I for j in range(nqubits)))
            for i in range(nqubits))
    return h


def XXZ(nqubits, delta=0.5, numpy=False, trotter=False):
    """Heisenberg XXZ model with periodic boundary conditions.

    .. math::
        H = \\sum _{i=0}^N \\left ( X_iX_{i + 1} + Y_iY_{i + 1} + \\delta Z_iZ_{i + 1} \\right ).

    Args:
        nqubits (int): number of quantum bits.
        delta (float): coefficient for the Z component (default 0.5).
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
        trotter (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.base.hamiltonians.TrotterHamiltonian` object, otherwise
            it creates a :class:`qibo.base.hamiltonians.Hamiltonian` object.

    Example:
        ::

            from qibo.hamiltonians import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits
    """
    if trotter:
        hx = qnp.kron(matrices.X, matrices.X)
        hy = qnp.kron(matrices.Y, matrices.Y)
        hz = qnp.kron(matrices.Z, matrices.Z)
        term = Hamiltonian(2, hx + hy + delta * hz, numpy=True)
        terms = {(i, i + 1): term for i in range(nqubits - 1)}
        terms[(nqubits - 1, 0)] = term
        return TrotterHamiltonian.from_dictionary(terms)

    condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
    hx = _build_spin_model(nqubits, matrices.X, condition)
    hy = _build_spin_model(nqubits, matrices.Y, condition)
    hz = _build_spin_model(nqubits, matrices.Z, condition)
    matrix = hx + hy + delta * hz
    return Hamiltonian(nqubits, matrix, numpy=numpy)


def _OneBodyPauli(nqubits, matrix, numpy=False, trotter=False,
                  ground_state=None):
    """Helper method for constracting non-interacting X, Y, Z Hamiltonians."""
    if not trotter:
        condition = lambda i, j: i == j % nqubits
        ham = -_build_spin_model(nqubits, matrix, condition)
        return Hamiltonian(nqubits, ham, numpy=numpy)

    term = Hamiltonian(1, -matrix, numpy=True)
    terms = {(i,): term for i in range(nqubits)}
    return TrotterHamiltonian.from_dictionary(terms, ground_state=ground_state)


def X(nqubits, numpy=False, trotter=False):
    """Non-interacting Pauli-X Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N X_i.

    Args:
        nqubits (int): number of quantum bits.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
        trotter (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.base.hamiltonians.TrotterHamiltonian` object, otherwise
            it creates a :class:`qibo.base.hamiltonians.Hamiltonian` object.
    """
    from qibo import K
    def ground_state():
        n = K.cast(2 ** nqubits, dtype='DTYPEINT')
        state = K.ones(n, dtype='DTYPECPX')
        return state / K.sqrt(K.cast(n, dtype=state.dtype))
    return _OneBodyPauli(nqubits, matrices.X, numpy, trotter, ground_state)


def Y(nqubits, numpy=False, trotter=False):
    """Non-interacting Pauli-Y Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Y_i.

    Args:
        nqubits (int): number of quantum bits.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
        trotter (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.base.hamiltonians.TrotterHamiltonian` object, otherwise
            it creates a :class:`qibo.base.hamiltonians.Hamiltonian` object.
    """
    return _OneBodyPauli(nqubits, matrices.Y, numpy, trotter)


def Z(nqubits, numpy=False, trotter=False):
    """Non-interacting Pauli-Z Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Z_i.

    Args:
        nqubits (int): number of quantum bits.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
        trotter (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.base.hamiltonians.TrotterHamiltonian` object, otherwise
            it creates a :class:`qibo.base.hamiltonians.Hamiltonian` object.
    """
    return _OneBodyPauli(nqubits, matrices.Z, numpy, trotter)


def TFIM(nqubits, h=0.0, numpy=False, trotter=False):
    """Transverse field Ising model with periodic boundary conditions.

    .. math::
        H = - \\sum _{i=0}^N \\left ( Z_i Z_{i + 1} + h X_i \\right ).

    Args:
        nqubits (int): number of quantum bits.
        h (float): value of the transverse field.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
        trotter (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.base.hamiltonians.TrotterHamiltonian` object, otherwise
            it creates a :class:`qibo.base.hamiltonians.Hamiltonian` object.
    """
    if trotter:
        term_matrix = -qnp.kron(matrices.Z, matrices.Z)
        term_matrix -= h * qnp.kron(matrices.X, matrices.I)
        term = Hamiltonian(2, term_matrix, numpy=True)
        terms = {(i, i + 1): term for i in range(nqubits - 1)}
        terms[(nqubits - 1, 0)] = term
        return TrotterHamiltonian.from_dictionary(terms)

    condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
    ham = -_build_spin_model(nqubits, matrices.Z, condition)
    if h != 0:
        condition = lambda i, j: i == j % nqubits
        ham -= h * _build_spin_model(nqubits, matrices.X, condition)
    return Hamiltonian(nqubits, ham, numpy=numpy)
