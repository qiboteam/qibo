import math
from itertools import combinations, product

import numpy as np

from qibo import symbols
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian


def commutator(a, b):
    """Compute commutator between two arrays."""
    return a @ b - b @ a


def variance(a, state):
    """Calculates the variance of a matrix A with respect to a state:
    Var($A$) = $\\langle\\mu|A^2|\\mu\rangle-\\langle\\mu|A|\\mu\rangle^2$"""
    b = a @ a
    return state.conj().T @ b @ state - (state.conj().T @ a @ state) ** 2


def covariance(a, b, state):
    """This is a generalization of the notion of covariance, needed for the polynomial expansion of the energy fluctuation,
    applied to two operators A and B with respect to a state:
    Cov($A,B$) = $\\langle\\mu|AB|\\mu\rangle-\\langle\\mu|A|\\mu\rangle\\langle\\mu|B|\\mu\rangle$
    """
        
    c = a @ b + b @ a
    return (
        state.conj().T @ c @ state
        - 2 * state.conj().T @ a @ state * state.conj().T @ b @ state
    )


def generate_Z_operators(nqubits: int, backend=None):
    """Generate a dictionary containing 1) all possible products of Pauli Z operators for L = n_qubits and 2) their respective names.
    Return: Dictionary with operator names (str) as keys and operators (np.array) as values

     Example:
        .. testcode::

            from qibo.models.dbi.utils import generate_Z_operators
            from qibo.models.dbi.double_bracket import DoubleBracketIteration
            from qibo.quantum_info import random_hermitian
            from qibo.hamiltonians import Hamiltonian
            import numpy as np

            nqubits = 4
            h0 = random_hermitian(2**nqubits)
            dbi = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))
            generate_Z = generate_Z_operators(nqubits)
            Z_ops = list(generate_Z.values())

            delta_h0 = dbi.diagonal_h_matrix
            dephasing_channel = (sum([Z_op @ h0 @ Z_op for Z_op in Z_ops])+h0)/2**nqubits
            norm_diff = np.linalg.norm(delta_h0 - dephasing_channel)
    """

    backend = _check_backend(backend)
    # list of tuples, e.g. ('Z','I','Z')
    combination_strings = product("ZI", repeat=nqubits)
    output_dict = {}

    for zi_string_combination in combination_strings:
        # except for the identity
        if "Z" in zi_string_combination:
            op_name = "".join(zi_string_combination)
            tensor_op = str_to_symbolic(op_name)
            # append in output_dict
            output_dict[op_name] = SymbolicHamiltonian(
                tensor_op, backend=backend
            ).dense.matrix
    return output_dict


def str_to_symbolic(name: str):
    """Convert string into symbolic hamiltonian.
    Example:
        .. testcode::

            from qibo.models.dbi.utils import str_to_symbolic
            op_name = "ZYXZI"
            # returns 5-qubit symbolic hamiltonian
            ZIXZI_op = str_to_symbolic(op_name)
    """
    tensor_op = 1
    for qubit, char in enumerate(name):
        tensor_op *= getattr(symbols, char)(qubit)
    return tensor_op


def cs_angle_sgn(dbi_object, d):
    """Calculates the sign of Cauchy-Schwarz Angle :math:`\\langle W(Z), W({\\rm canonical}) \\rangle_{\\rm HS}`."""
    norm = np.trace(
        np.dot(
            np.conjugate(
                dbi_object.commutator(dbi_object.diagonal_h_matrix, dbi_object.h.matrix)
            ).T,
            dbi_object.commutator(d, dbi_object.h.matrix),
        )
    )
    return np.sign(norm)


def decompose_into_Pauli_basis(h_matrix: np.array, pauli_operators: list):
    """finds the decomposition of hamiltonian `h_matrix` into Pauli-Z operators"""
    nqubits = int(np.log2(h_matrix.shape[0]))

    decomposition = []
    for Z_i in pauli_operators:
        expect = np.trace(h_matrix @ Z_i) / 2**nqubits
        decomposition.append(expect)
    return decomposition


def generate_pauli_index(nqubits, order):
    if order == 1:
        return list(range(nqubits))
    elif order > 1:
        indices = list(range(nqubits))
        return indices + [
            comb for i in range(2, order + 1) for comb in combinations(indices, i)
        ]
    else:
        raise ValueError("Order must be a positive integer")


def generate_pauli_operator_dict(
    nqubits: int, parameterization_order: int = 1, symbols_pauli=symbols.Z
):
    pauli_index = generate_pauli_index(nqubits, order=parameterization_order)
    pauli_operators = [
        generate_Pauli_operators(nqubits, symbols_pauli, index) for index in pauli_index
    ]
    return {index: operator for index, operator in zip(pauli_index, pauli_operators)}


def diagonal_min_max(matrix: np.array):
    L = int(np.log2(matrix.shape[0]))
    D = np.linspace(np.min(np.diag(matrix)), np.max(np.diag(matrix)), 2**L)
    D = np.diag(D)
    return D


def generate_Pauli_operators(nqubits, symbols_pauli, positions):
    # generate matrix of an nqubit-pauli operator with `symbols_pauli` at `positions`
    if isinstance(positions, int):
        return SymbolicHamiltonian(
            symbols_pauli(positions), nqubits=nqubits
        ).dense.matrix
    else:
        terms = [symbols_pauli(pos) for pos in positions]
        return SymbolicHamiltonian(math.prod(terms), nqubits=nqubits).dense.matrix
