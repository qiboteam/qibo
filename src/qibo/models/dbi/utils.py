from copy import deepcopy
from itertools import product
from typing import Optional

import numpy as np
from hyperopt import hp, tpe

from qibo import symbols
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
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


def select_best_dbr_generator(
    dbi_object: DoubleBracketIteration,
    d_list: list,
    step: Optional[float] = None,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 200,
    compare_canonical: bool = True,
):
    """Selects the best double bracket rotation generator from a list and execute the rotation.

    Args:
        dbi_object (`DoubleBracketIteration`): the target DoubleBracketIteration object.
        d_list (list): list of diagonal operators (np.array) to run from.
        step (float): fixed iteration duration.
            Defaults to ``None``, uses hyperopt.
        step_min (float): minimally allowed iteration duration.
        step_max (float): maximally allowed iteration duration.
        max_evals (int): maximally allowed number of evaluation in hyperopt.
        compare_canonical (bool): if `True`, the optimal diagonal operator chosen from "d_list" is compared with the canonical bracket.

    Returns:
        The updated dbi_object, index of the optimal diagonal operator, respective step duration, and evolution direction.
    """
    norms_off_diagonal_restriction = [
        dbi_object.off_diagonal_norm for _ in range(len(d_list))
    ]
    optimal_steps, flip_list = [], []
    for i, d in enumerate(d_list):
        # prescribed step durations
        dbi_eval = deepcopy(dbi_object)
        flip_list.append(cs_angle_sgn(dbi_eval, d))
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
            optimal_steps.append(step_best)
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
    idx_max_loss = np.argmin(norms_off_diagonal_restriction)
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
