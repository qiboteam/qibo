from copy import deepcopy
from itertools import product
from typing import Optional

from hyperopt import hp, tpe

from qibo import symbols
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)


def generate_Z_operators(nqubits: int):
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
    # list of tupples, e.g. ('Z','I','Z')
    combination_strings = product("ZI", repeat=nqubits)
    output_dict = {}

    for op_string_tup in combination_strings:
        # except for the identity
        if "Z" in op_string_tup:
            op_name = "".join(op_string_tup)
            tensor_op = str_to_symbolic(op_name)
            # append in output_dict
            output_dict[op_name] = SymbolicHamiltonian(tensor_op).dense.matrix
    return output_dict


def str_to_symbolic(name: str):
    """Converts string into symbolic hamiltonian
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


def select_best_dbr_generator(
    dbi_object: DoubleBracketIteration,
    d_list: list,
    step: Optional[float] = None,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 200,
    compare_canonical: bool = True,
):
    """Selects the best double bracket rotation generator from a list.

    Args:
        dbi_object (_DoubleBracketIteration): The object intended for double bracket iteration.
        d_list (list): List of diagonal operators (np.array) to run from.
        step (float): Fixed iteration duration.
            Defaults to ``None``, uses hyperopt.
        step_min (float): Minimally allowed iteration duration.
        step_max (float): Maximally allowed iteration duration.
        max_evals (int): Maximally allowed number of evaluation in hyperopt.
        compare_canonical (bool): If `True`, the optimal diagonal operator chosen from "d_list" is compared with the canonical bracket.

    Returns:
        The index of the optimal diagonal operator and respective step duration.
    """
    norms_off_diagonal_restriction = []
    optimal_steps = []
    for d in d_list:
        # prescribed step durations
        h_before = deepcopy(dbi_object.h)
        if step is None:
            step = dbi_object.hyperopt_step(
                step_min=step_min,
                step_max=step_max,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=max_evals,
                d=d,
            )
        dbi_object(step=step, d=d)
        optimal_steps.append(step)
        norms_off_diagonal_restriction.append(dbi_object.off_diagonal_norm)
        dbi_object.h = h_before
    # canonical
    if compare_canonical is True:
        generator_type = dbi_object.mode
        dbi_object.mode = DoubleBracketGeneratorType.canonical
        if step is None:
            step = dbi_object.hyperopt_step(
                step_min=step_min,
                step_max=step_max,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=max_evals,
            )
        dbi_object(step=step)
        optimal_steps.append(step)
        norms_off_diagonal_restriction.append(dbi_object.off_diagonal_norm)
        dbi_object.h = deepcopy(h_before)
        dbi_object.mode = generator_type
    # find best d
    idx_max_loss = norms_off_diagonal_restriction.index(
        min(norms_off_diagonal_restriction)
    )
    step_optimal = optimal_steps[idx_max_loss]
    return idx_max_loss, step_optimal


def select_best_dbr_generator_and_run(
    dbi_object: DoubleBracketIteration,
    d_list: list,
    step: Optional[float] = None,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 200,
    compare_canonical: bool = True,
):
    """Run double bracket iteration with generator chosen from a list."""
    idx_max_loss, step_optimal = select_best_dbr_generator(
        dbi_object,
        d_list,
        step=step,
        step_min=step_min,
        step_max=step_max,
        max_evals=max_evals,
        compare_canonical=compare_canonical,
    )
    # run with optimal d
    if idx_max_loss == len(d_list) and compare_canonical is True:
        # canonical
        generator_type = dbi_object.mode
        dbi_object.mode = DoubleBracketGeneratorType.canonical
        dbi_object(step=step_optimal)
        dbi_object.mode = generator_type
    else:
        d_optimal = d_list[idx_max_loss]
        dbi_object(step=step_optimal, d=d_optimal)
    return idx_max_loss, step_optimal
