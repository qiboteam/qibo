from functools import reduce
from typing import Optional, Union

import numpy as np

from qibo import symbols
from qibo.backends import Backend, _check_backend, matrices
from qibo.config import raise_error
from qibo.hamiltonians.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.hamiltonians.terms import HamiltonianTerm


def X(nqubits, dense: bool = True, backend=None):
    """Non-interacting Pauli-:math:`X` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^N \\, X_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.X, dense, backend=backend)


def Y(nqubits, dense: bool = True, backend=None):
    """Non-interacting Pauli-:math:`Y` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^{N} \\, Y_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.Y, dense, backend=backend)


def Z(nqubits, dense: bool = True, backend=None):
    """Non-interacting Pauli-:math:`Z` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^{N} \\, Z_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.Z, dense, backend=backend)


def TFIM(nqubits, h: float = 0.0, dense: bool = True, backend=None):
    """Transverse field Ising model with periodic boundary conditions.

    .. math::
        H = - \\sum _{k=0}^{N} \\, \\left(Z_{k} \\, Z_{k + 1} + h \\, X_{k}\\right) \\, .

    Args:
        nqubits (int): number of qubits.
        h (float, optional): value of the transverse field. Defaults to :math:`0.0`.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")
    backend = _check_backend(backend)
    if dense:
        condition = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        ham = -_build_spin_model(nqubits, backend.matrices.Z, condition, backend)
        if h != 0:
            condition = lambda i, j: i == j % nqubits
            ham -= h * _build_spin_model(
                nqubits, backend.matrices.X, condition, backend
            )
        return Hamiltonian(nqubits, ham, backend=backend)

    term = lambda q1, q2: symbols.Z(q1, backend=backend) * symbols.Z(
        q2, backend=backend
    ) + h * symbols.X(q1, backend=backend)
    form = -1 * sum(term(i, i + 1) for i in range(nqubits - 1)) - term(nqubits - 1, 0)
    ham = SymbolicHamiltonian(form=form, nqubits=nqubits, backend=backend)
    return ham


def MaxCut(
    nqubits,
    dense: bool = True,
    adj_matrix: Optional[Union[list[list[float]], np.ndarray]] = None,
    backend: Optional[Backend] = None,
):
    """Max Cut Hamiltonian.

    .. math::
        H = -\\frac{1}{2} \\, \\sum _{j, k = 0}^{N}  \\, \\left(1 - Z_{j} \\, Z_{k}\\right) \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        adj_matrix (list[list[float]] | np.ndarray): Adjecency matrix of the graph. Defaults to a
            homogeneous fully connected graph with all edges having an equal 1.0 weight.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    if adj_matrix is None:
        adj_matrix = np.ones((nqubits, nqubits))
    elif len(adj_matrix) != nqubits:
        raise_error(
            RuntimeError,
            f"Expected an adjacency matrix of shape ({nqubits},{nqubits}) for a {nqubits}-qubits system.",
        )

    form = -sum(
        adj_matrix[i][j]
        * (1 - symbols.Z(i, backend=backend) * symbols.Z(j, backend=backend))
        for i in range(nqubits)
        for j in range(nqubits)
    )
    form /= 2

    ham = SymbolicHamiltonian(form, nqubits=nqubits, backend=backend)
    if dense:
        return ham.dense
    return ham


def Heisenberg(
    nqubits,
    coupling_constants: Union[float, int, list, tuple],
    external_field_strengths: Union[float, int],
    dense: bool = True,
    backend=None,
):
    """Heisenberg model on a :math:`1`-dimensional periodic lattice.

    The general :math:`n`-qubit Hamiltonian is given by

    .. math::
        \\begin{align}
        H &= -\\sum_{k = 1}^{n} \\, \\left(
            J_{x} \\, X_{k} \\, X_{k + 1}
            + J_{y} \\, Y_{k} \\, Y_{k + 1}
            + J_{z} \\, Z_{k} \\, Z_{k + 1} \\right) \\\\
        &\\quad\\,\\, - \\sum_{k = 1}^{n} \\left(
            h_{x} \\, X_{k} + h_{y} \\, Y_{k} + h_{z} \\, Z_{k}
            \\right) \\, ,
        \\end{align}

    where :math:`\\{J_{x}, \\, J_{y}, \\, J_{z}\\}` are called the ``coupling constants``,
    :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}` are called the ``external field strengths``,
    and :math:`\\{X, \\, Y, \\, Z\\}` are the usual Pauli operators.

    Args:
        nqubits (int): number of qubits.
        coupling_constants (list or tuple or float or int): list or tuple with the
            three coupling constants :math:`\\{J_{x}, J_{y}, J{z}\\}`.
            If ``int`` or ``float``, then :math:`J_{x} = J_{y} = J_{z}`.
        external_field_strength (list or tuple or float or int): list or tuple with the
            external magnetic field strengths :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`.
            If ``int`` or ``float``, then :math:`h_{x} = h_{y} = h_{z}`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg Hamiltonian.
    """
    if isinstance(coupling_constants, (list, tuple)) and len(coupling_constants) != 3:
        raise_error(
            ValueError,
            f"When `coupling_constants` is type `int` or `list`, it must have length == 3.",
        )

    if isinstance(coupling_constants, (float, int)):
        coupling_constants = [coupling_constants] * 3

    if (
        isinstance(external_field_strengths, (list, tuple))
        and len(external_field_strengths) != 3
    ):
        raise_error(
            ValueError,
            f"When `external_field_strengths` is type `int` or `list`, it must have length == 3.",
        )

    if isinstance(external_field_strengths, (float, int)):
        external_field_strengths = [external_field_strengths] * 3

    backend = _check_backend(backend)

    paulis = (symbols.X, symbols.Y, symbols.Z)

    if dense:
        condition = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        matrix = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
        matrix = backend.cast(matrix, dtype=matrix.dtype)
        for ind, pauli in enumerate(paulis):
            double_term = _build_spin_model(
                nqubits, pauli(0, backend=backend).matrix, condition, backend
            )
            double_term = backend.cast(double_term, dtype=double_term.dtype)
            matrix = matrix - coupling_constants[ind] * double_term
            matrix = (
                matrix
                + external_field_strengths[ind]
                * _OneBodyPauli(nqubits, pauli, dense, backend).matrix
            )

        return Hamiltonian(nqubits, matrix, backend=backend)

    def h(symbol):
        return lambda q1, q2: symbol(q1, backend=backend) * symbol(q2, backend=backend)

    def term(q1, q2):
        return sum(
            coeff * h(operator)(q1, q2)
            for coeff, operator in zip(coupling_constants, paulis)
        )

    form = -1 * sum(term(i, i + 1) for i in range(nqubits - 1)) - term(nqubits - 1, 0)
    form -= sum(
        field_strength * pauli(qubit)
        for qubit in range(nqubits)
        for field_strength, pauli in zip(external_field_strengths, paulis)
        if field_strength != 0.0
    )

    ham = SymbolicHamiltonian(form=form, backend=backend)
    return ham


def XXX(
    nqubits,
    coupling_constant: Union[float, int] = 1,
    external_field_strengths: Union[float, int, list, tuple] = [0.5, 0, 0],
    dense: bool = True,
    backend=None,
):
    """Heisenberg :math:`\\mathrm{XXX}` model with periodic boundary conditions.

    The :math:`n`-qubit :math:`\\mathrm{XXX}` Hamiltonian is given by

    .. math::
        H = -J \\, \\sum_{k = 1}^{n} \\, \\left(
            X_{k} \\, X_{k + 1} + Y_{k} \\, Y_{k + 1} + Z_{k} \\, Z_{k + 1}
            \\right)
            - \\sum_{k = 1}^{n} \\left(h_{x} \\, X_{k} + h_{y} \\, Y_{k}
            + h_{z} \\, Z_{k} \\right) \\, ,

    where :math:`J` is the ``coupling_constant``, :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`
    are called the ``external field strengths``, and :math:`\\{X, \\, Y, \\, Z\\}` are the usual
    Pauli operators.

    Args:
        nqubits (int): number of qubits.
        coupling_constant (float or int, optional): coupling constant :math:`J`.
            Defaults to :math:`1`.
        external_field_strengths (float or int or list or tuple, optional): list or tuple with the
            external magnetic field strengths :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`.
            If ``int`` or ``float``, then :math:`h_{x} = h_{y} = h_{z}`.
            Defaults to :math:`[1/2, \\, 0, \\, 0]`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg :math:`\\mathrm{XXX}` Hamiltonian.
    """
    if not isinstance(coupling_constant, (float, int)):
        raise_error(
            TypeError,
            "`coupling_constant` must be type float or int, "
            + f"but it is type {type(coupling_constant)}.",
        )

    return Heisenberg(
        nqubits,
        coupling_constants=coupling_constant,
        external_field_strengths=external_field_strengths,
        dense=dense,
        backend=backend,
    )


def XXZ(nqubits, delta=0.5, dense: bool = True, backend=None):
    """Heisenberg :math:`\\mathrm{XXZ}` model with periodic boundary conditions.

    .. math::
        H = \\sum _{k=0}^N \\, \\left( X_{k} \\, X_{k + 1} + Y_{k} \\, Y_{k + 1}
            + \\delta Z_{k} \\, Z_{k + 1} \\right) \\, .

    Example:
        .. testcode::

            from qibo.hamiltonians import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits

    Args:
        nqubits (int): number of qubits.
        delta (float, optional): coefficient for the :math:`Z` component.
            Defaults to :math:`0.5`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg :math:`\\mathrm{XXZ}` Hamiltonian.
    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")

    return Heisenberg(nqubits, [-1, -1, -delta], 0, dense=dense, backend=backend)


def _multikron(matrix_list, backend):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrix_list (list): List of matrices as ``ndarray``.

    Returns:
        ndarray: Kronecker product of all matrices in ``matrix_list``.
    """
    return reduce(backend.np.kron, matrix_list)


def _build_spin_model(nqubits, matrix, condition, backend):
    """Helper method for building nearest-neighbor spin model Hamiltonians."""
    h = sum(
        reduce(
            backend.np.kron,
            (
                matrix if condition(i, j) else backend.matrices.I()
                for j in range(nqubits)
            ),
        )
        for i in range(nqubits)
    )
    return h


def _OneBodyPauli(nqubits, operator, dense: bool = True, backend=None):
    """Helper method for constructing non-interacting
    :math:`X`, :math:`Y`, and :math:`Z` Hamiltonians."""
    backend = _check_backend(backend)
    if dense:
        condition = lambda i, j: i == j % nqubits
        ham = -_build_spin_model(
            nqubits, operator(0, backend=backend).matrix, condition, backend
        )
        return Hamiltonian(nqubits, ham, backend=backend)

    form = sum([-1 * operator(i, backend=backend) for i in range(nqubits)])
    ham = SymbolicHamiltonian(form=form, backend=backend)
    return ham
