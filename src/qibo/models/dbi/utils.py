from copy import deepcopy
from itertools import product
from typing import Optional

import numpy as np
from hyperopt import hp, tpe

from qibo import symbols
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
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
    mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.single_commutator,
):
    """Selects the best double bracket rotation generator from a list and runs the

    Args:
        dbi_object (_DoubleBracketIteration): The target DoubleBracketIteration object.
        d_list (list): List of diagonal operators (np.array) to run from.
        step (float): Fixed iteration duration.
            Defaults to ``None``, uses hyperopt.
        step_min (float): Minimally allowed iteration duration.
        step_max (float): Maximally allowed iteration duration.
        max_evals (int): Maximally allowed number of evaluation in hyperopt.
        compare_canonical (bool): If `True`, the optimal diagonal operator chosen from "d_list" is compared with the canonical bracket.
        mode (_DoubleBracketGeneratorType): DBI generator type used for the selection.

    Returns:
        The updated dbi_object, index of the optimal diagonal operator, respective step duration, and evolution direction.
    """
    norms_off_diagonal_restriction = [
        dbi_object.off_diagonal_norm for _ in range(len(d_list))
    ]
    optimal_steps = [0 for _ in range(len(d_list))]
    flip_list = [1 for _ in range(len(d_list))]
    for i, d in enumerate(d_list):
        # prescribed step durations
        dbi_eval = deepcopy(dbi_object)
        flip_list[i] = CS_angle_sgn(dbi_eval, d)
        if flip_list[i] != 0:
            if step is None:
                step_best = dbi_eval.hyperopt_step(
                    d=flip_list[i] * d,
                    step_min=step_min,
                    step_max=step_max,
                    space=hp.uniform,
                    optimizer=tpe,
                    max_evals=max_evals,
                )
            else:
                step_best = step
            dbi_eval(step=step_best, d=flip_list[i] * d)
            optimal_steps[i] = step_best
            norms_off_diagonal_restriction[i] = dbi_eval.off_diagonal_norm
    # canonical
    if compare_canonical is True:
        flip_list.append(1)
        dbi_eval = deepcopy(dbi_object)
        dbi_eval.mode = DoubleBracketGeneratorType.canonical
        if step is None:
            step_best = dbi_eval.hyperopt_step(
                step_min=step_min,
                step_max=step_max,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=max_evals,
            )
        else:
            step_best = step
        dbi_eval(step=step_best)
        optimal_steps.append(step_best)
        norms_off_diagonal_restriction.append(dbi_eval.off_diagonal_norm)
    # find best d
    idx_max_loss = norms_off_diagonal_restriction.index(
        min(norms_off_diagonal_restriction)
    )
    flip = flip_list[idx_max_loss]
    step_optimal = optimal_steps[idx_max_loss]
    dbi_eval = deepcopy(dbi_object)
    if idx_max_loss == len(d_list) and compare_canonical is True:
        # canonical
        dbi_eval(step=step_optimal, mode=DoubleBracketGeneratorType.canonical)

    else:
        d_optimal = flip * d_list[idx_max_loss]
        dbi_eval(step=step_optimal, d=d_optimal)
    return dbi_eval, idx_max_loss, step_optimal, flip


def cs_angle_sgn(dbi_object, d):
    """Calculates the sign of Cauchy-Schwarz Angle $$<W(Z), W(canonical)>_{HS}$$"""
    norm = np.trace(
        np.dot(
            np.conjugate(
                dbi_object.commutator(dbi_object.diagonal_h_matrix, dbi_object.h.matrix)
            ).T,
            dbi_object.commutator(d, dbi_object.h.matrix),
        )
    )
    return np.sign(norm)
