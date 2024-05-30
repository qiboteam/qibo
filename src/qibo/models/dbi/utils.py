import math
from enum import Enum, auto
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
    .. math::
        Var(A) = \\langle\\mu|A^2|\\mu\\rangle-\\langle\\mu|A|\\mu\\rangle^2"""
    b = a @ a
    return state.conj().T @ b @ state - (state.conj().T @ a @ state) ** 2


def covariance(a, b, state):
    """This is a generalization of the notion of covariance, needed for the polynomial expansion of the energy fluctuation,
    applied to two operators A and B with respect to a state:
    .. math::
        Cov(A,B) = \\langle\\mu|AB|\\mu\\rangle-\\langle\\mu|A|\\mu\\rangle\\langle\\mu|B|\\mu\\rangle
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
    """Finds the decomposition of hamiltonian `h_matrix` into Pauli-Z operators"""
    nqubits = int(np.log2(h_matrix.shape[0]))

    decomposition = []
    for Z_i in pauli_operators:
        expect = np.trace(h_matrix @ Z_i) / 2**nqubits
        decomposition.append(expect)
    return decomposition


def generate_pauli_index(nqubits, order):
    """
    Generate all possible combinations of qubits for a given order of Pauli operators.
    """
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
    nqubits: int, parameterization_order: int = 1, symbols_pauli: symbols = symbols.Z
):
    """Generates a dictionary containing Pauli `symbols_pauli` operators of locality `parameterization_order` for `nqubits` qubits.

    Args:
        nqubits (int): number of qubits in the system.
        parameterization_order (int, optional): the locality of the operators generated. Defaults to 1.
        symbols_pauli (qibo.symbols, optional): the symbol of the intended Pauli operator. Defaults to symbols.Z.

    Returns:
        pauli_operator_dict (dictionary): dictionary with structure {"operator_name": operator}

    Example:
        pauli_operator_dict = generate_pauli_operator_dict)
    """
    pauli_index = generate_pauli_index(nqubits, order=parameterization_order)
    pauli_operators = [
        generate_Pauli_operators(nqubits, symbols_pauli, index) for index in pauli_index
    ]
    return {index: operator for index, operator in zip(pauli_index, pauli_operators)}


def diagonal_min_max(matrix: np.array):
    """
    Generate a diagonal matrix D with the same diagonal elements as `matrix` but with minimum and maximum values.
    (may be deprecated as a useful ansatz for D)
    """
    L = int(np.log2(matrix.shape[0]))
    D = np.linspace(np.min(np.diag(matrix)), np.max(np.diag(matrix)), 2**L)
    D = np.diag(D)
    return D


def generate_Pauli_operators(nqubits, symbols_pauli, positions):
    """Generate matrix of an nqubit-pauli operator with `symbols_pauli` at `positions`"""
    if isinstance(positions, int):
        return SymbolicHamiltonian(
            symbols_pauli(positions), nqubits=nqubits
        ).dense.matrix
    else:
        terms = [symbols_pauli(pos) for pos in positions]
        return SymbolicHamiltonian(math.prod(terms), nqubits=nqubits).dense.matrix


class ParameterizationTypes(Enum):
    """Define types of parameterization for diagonal operator."""

    pauli = auto()
    """Uses Pauli-Z operators (magnetic field)."""
    computational = auto()
    """Uses computational basis."""


def params_to_diagonal_operator(
    params: np.array,
    nqubits: int,
    parameterization: ParameterizationTypes = ParameterizationTypes.pauli,
    pauli_parameterization_order: int = 1,
    normalize: bool = False,
    pauli_operator_dict: dict = None,
):
    """Creates the $D$ operator for the double-bracket iteration ansatz depending on the parameterization type."""
    if parameterization is ParameterizationTypes.pauli:
        # raise error if dimension mismatch
        if len(params) != len(pauli_operator_dict):
            raise ValueError(
                f"Dimension of params ({len(params)}) mismatches the given parameterization order ({pauli_parameterization_order})"
            )
        d = sum(
            [params[i] * list(pauli_operator_dict.values())[i] for i in range(nqubits)]
        )
    elif parameterization is ParameterizationTypes.computational:
        d = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            d[i, i] = params[i]
    else:
        raise ValueError(f"Parameterization type not recognized.")
    if normalize:
        d = d / np.linalg.norm(d)
    return d


def off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n):
    """Return the Taylor expansion coefficients of off-diagonal norm of `dbi_object.h` with respect to double bracket rotation duration `s`."""
    Gamma_list = dbi_object.generate_Gamma_list(n + 2, d)
    sigma_Gamma_list = list(map(dbi_object.sigma, Gamma_list))
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients for rotation with [W,H] and H
    c1 = exp_list.reshape((-1, 1, 1)) * sigma_Gamma_list[1:]
    c2 = exp_list.reshape((-1, 1, 1)) * sigma_Gamma_list[:-1]
    # product coefficient
    trace_coefficients = [0] * (2 * n + 1)
    for k in range(n + 1):
        for j in range(n + 1):
            power = k + j
            product_matrix = c1[k] @ c2[j]
            trace_coefficients[power] += 2 * np.trace(product_matrix)
    # coefficients from high to low (n:0)
    coef = list(reversed(trace_coefficients[: n + 1]))
    return coef


def least_squares_polynomial_expansion_coef(dbi_object, d, n: int = 3):
    """Return the Taylor expansion coefficients of least square cost of `dbi_object.h` and diagonal operator `d` with respect to double bracket rotation duration `s`."""
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n + 1, d)
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients
    coef = np.empty(n)
    for i in range(n):
        coef[i] = np.real(
            exp_list[i] * np.trace(dbi_object.backend.cast(d) @ Gamma_list[i + 1])
        )
    coef = list(reversed(coef))
    return coef


def energy_fluctuation_polynomial_expansion_coef(
    dbi_object, d: np.array = None, n: int = 3, state=0
):
    """Return the Taylor expansion coefficients of energy fluctuation of `dbi_object` with respect to double bracket rotation duration `s`."""
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n + 1, d)
    # coefficients
    coef = np.empty(3)
    state_cast = dbi_object.backend.cast(state)
    state_dag = dbi_object.backend.cast(state.conj().T)

    def variance(a):
        """Calculates the variance of a matrix A with respect to a state:
        Var($A$) = $\\langle\\mu|A^2|\\mu\rangle-\\langle\\mu|A|\\mu\rangle^2$"""
        b = a @ a
        return state_dag @ b @ state_cast - (state_dag @ a @ state_cast) ** 2

    def covariance(a, b):
        """This is a generalization of the notion of covariance, needed for the polynomial expansion of the energy fluctuation,
        applied to two operators A and B with respect to a state:
        Cov($A,B$) = $\\langle\\mu|AB|\\mu\rangle-\\langle\\mu|A|\\mu\rangle\\langle\\mu|B|\\mu\rangle$
        """

        c = a @ b + b @ a
        return (
            state_dag @ c @ state_cast
            - 2 * state_dag @ a @ state_cast * state_dag @ b @ state_cast
        )

    coef[0] = np.real(2 * covariance(Gamma_list[0], Gamma_list[1]))
    coef[1] = np.real(2 * variance(Gamma_list[1]))
    coef[2] = np.real(
        covariance(Gamma_list[0], Gamma_list[3])
        + 3 * covariance(Gamma_list[1], Gamma_list[2])
    )
    coef = list(reversed(coef))
    return coef
