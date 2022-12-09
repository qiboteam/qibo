from functools import reduce
from itertools import product

import numpy as np

from qibo.config import raise_error
from qibo.gates.gates import I, X, Y, Z


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

    if nqubits <= 0:
        raise_error(ValueError, "nqubits must be a positive int.")

    if not isinstance(normalize, bool):
        raise_error(
            TypeError,
            f"normalize must be type bool, but it is type {type(normalize)} instead.",
        )

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    basis_single_qubit = [
        I(0).asmatrix(backend),
        X(0).asmatrix(backend),
        Y(0).asmatrix(backend),
        Z(0).asmatrix(backend),
    ]

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

    paulis = pauli_basis(nqubits, normalize, backend)

    comp_basis_state = np.zeros((d**2, 1), dtype="complex")
    matrix = np.zeros((d**2, d**2), dtype="complex")
    for index, pauli in enumerate(paulis):
        comp_basis_state[index, 0] = 1.0
        pauli = np.reshape(pauli, (1, -1))
        matrix += np.kron(comp_basis_state, np.conj(pauli))
        comp_basis_state[index, 0] = 0.0

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

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    matrix = np.transpose(np.conj(comp_basis_to_pauli(nqubits, normalize, backend)))

    return backend.cast(matrix, dtype=matrix.dtype)
