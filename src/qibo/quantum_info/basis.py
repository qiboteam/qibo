from functools import reduce
from itertools import product

import numpy as np

from qibo import gates


def pauli_basis(nqubits: int, normalize: bool = False, backend=None):
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        qubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis ir returned.
            Defaults to False.
        backend (``qibo.backends.abstract.Backend``, optional): Backend for execution.
            If ``None``, defaults to ``GlobalBackend()``.

    Returns:
        list: list with all Pauli matrices forming the basis.
    """

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    I = gates.I(0).asmatrix(backend)
    X = gates.X(0).asmatrix(backend)
    Y = gates.Y(0).asmatrix(backend)
    Z = gates.Z(0).asmatrix(backend)

    basis_single_qubit = [I, X, Y, Z]

    if nqubits == 1:
        basis = basis_single_qubit
    else:
        basis = list(product(basis_single_qubit, repeat=nqubits))
        basis = [reduce(np.kron, matrix) for matrix in basis]

    if normalize:
        basis /= np.sqrt(2**nqubits)

    return basis


def comp_basis_to_pauli(nqubits: int, normalize: bool = False, backend=None):
    """Unitary matrix :math:`U` that converts operators from the Liouville representation
    in the computational basis to the Pauli-Liouville representation.

    The unitary :math:`U` is given by

    ..math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, {|b_{k})}{(P_{k}|} \\,\\, ,

    where :math:`{|A)}` is the column-vectorization of :math:`A`.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
        Pauli basis. Defaults to False.
        backend (``qibo.backends.abstract.Backend``, optional): Backend for execution.
            If ``None``, defaults to ``GlobalBackend()``.

    Returns:
        Unitary matrix :math:`U`.

    """

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    d = 2**nqubits

    paulis = pauli_basis(nqubits)

    if normalize:
        paulis /= np.sqrt(d)

    comp_basis_state = np.zeros((d**2), dtype="complex")
    matrix = np.zeros((d**2, d**2), dtype="complex")
    for index, pauli in enumerate(paulis):
        comp_basis_state[index] = 1.0
        pauli = pauli.flatten("F")
        matrix += np.kron(comp_basis_state.reshape((-1, 1)), np.conj(pauli))
        comp_basis_state[index] = 0.0

    return backend.cast(matrix, dtype=matrix.dtype)


def pauli_to_comp_basis(nqubits: int, normalize: bool = False, backend=None):
    """Unitary matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The unitary :math:`U` is given by

    ..math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, {|P_{k})}{(b_{k}|} \\, .

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
        Pauli basis. Defaults to False.
        backend (``qibo.backends.abstract.Backend``, optional): Backend for execution.
            If ``None``, defaults to ``GlobalBackend()``.

    Returns:
        Unitary matrix :math:`U`.
    """

    return np.transpose(np.conj(comp_basis_to_pauli(nqubits, normalize=normalize)))
