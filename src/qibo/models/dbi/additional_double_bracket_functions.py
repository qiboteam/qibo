from copy import copy, deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt import hp, tpe

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.symbols import I, X, Z


def visualize_matrix(matrix, title=""):
    """Visualize absolute values of a matrix in a heatmap form."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    try:
        im = ax.imshow(np.absolute(matrix), cmap="inferno")
    except TypeError:
        im = ax.imshow(np.absolute(matrix.get()), cmap="inferno")
    fig.colorbar(im, ax=ax)


def visualize_drift(h0, h):
    """Visualize drift of the evolved hamiltonian w.r.t. h0."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(r"Drift: $|\hat{H}_0 - \hat{H}_{1}|$")
    try:
        im = ax.imshow(np.absolute(h0 - h), cmap="inferno")
    except TypeError:
        im = ax.imshow(np.absolute((h0 - h).get()), cmap="inferno")

    fig.colorbar(im, ax=ax)


def plot_histories(loss_histories: list, steps: list, labels: list = None):
    """Plot off-diagonal norm histories over a sequential evolution."""
    plt.figure(figsize=(5, 5 * 6 / 8))
    if len(steps) == 1:
        # fixed_step
        x_axis = [i * steps[0] for i in range(len(loss_histories))]
    else:
        x_axis = [sum(steps[:k]) for k in range(1, len(steps) + 1)]
    plt.plot(x_axis, loss_histories, "-o")

    x_labels_rounded = [round(x, 2) for x in x_axis]
    x_labels_rounded = [0] + x_labels_rounded[0:5] + [max(x_labels_rounded)]
    x_labels_rounded.pop(3)
    plt.xticks(x_labels_rounded)

    y_labels_rounded = [round(y, 1) for y in loss_histories]
    y_labels_rounded = y_labels_rounded[0:5] + [min(y_labels_rounded)]
    plt.yticks(y_labels_rounded)

    if labels is not None:
        labels_copy = copy(labels)
        labels_copy.insert(0, "Initial")
        for i, label in enumerate(labels_copy):
            plt.text(x_axis[i], loss_histories[i], label)

    plt.grid()
    plt.xlabel(r"Flow duration $s$")
    plt.title("Loss function histories")


def generate_Z_operators(nqubits: int):
    """Generate a dictionary containing 1) all possible products of Pauli Z operators for L = n_qubits and 2) their respective names.
    Return: Dictionary with the following keys

        - *"Z_operators"*
        - *"Z_words"*

     Example:
        .. testcode::

            from qibo.models.dbi.additional_double_bracket_functions import generate_Z_operators
            from qibo.models.dbi.double_bracket import DoubleBracketIteration
            from qibo.quantum_info import random_hermitian
            from qibo.hamiltonians import Hamiltonian
            import numpy as np

            nqubits = 4
            h0 = random_hermitian(2**nqubits)
            dbi = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))
            generate_Z = generate_Z_operators(4)
            Z_ops = generate_Z["Z_operators"]
            Z_words = generate_Z["Z_operators"]

            delta_h0 = dbi.diagonal_h_matrix
            dephasing_channel = (sum([Z_op @ h0 @ Z_op for Z_op in Z_ops])+h0)/2**nqubits
            norm_diff = np.linalg.norm(delta_h0 - dephasing_channel)
            print(norm_diff)
    """
    combination_strings = product("ZI", repeat=nqubits)
    operator_map = {"Z": Z, "I": I}
    operators = []
    operators_words = []

    for op_string in combination_strings:
        tensor_op = 1
        # except for the identity
        if "Z" in op_string:
            for qubit, char in enumerate(op_string):
                if char in operator_map:
                    tensor_op *= operator_map[char](qubit)
            op_string_cat = "".join(op_string)
            operators_words.append(op_string_cat)
            # append np.array operators
            operators.append(SymbolicHamiltonian(tensor_op).dense.matrix)
    return {"Z_operators": operators, "Z_words": operators_words}


def select_best_dbr_generator(
    dbi_object: DoubleBracketIteration,
    d_list: list,
    step: float = None,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 100,
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
    h_before = deepcopy(dbi_object.h)
    norms_off_diagonal_restriction = []
    optimal_steps = []
    for d in d_list:
        # prescribed step durations
        if step is not None:
            dbi_object(step=step, d=d)
        # compute step durations using hyperopt
        else:
            step = dbi_object.hyperopt_step(
                step_min=step_min,
                step_max=step_max,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=max_evals,
                d=d,
            )
        optimal_steps.append(step)
        norms_off_diagonal_restriction.append(dbi_object.off_diagonal_norm)
        dbi_object.h = deepcopy(h_before)
    # canonical
    if compare_canonical is True:
        generator_type = dbi_object.mode
        dbi_object.mode = DoubleBracketGeneratorType.canonical
        if step is not None:
            dbi_object(step=step)
        else:
            step = dbi_object.hyperopt_step(
                step_min=step_min,
                step_max=step_max,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=max_evals,
            )
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
    step: float = None,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 100,
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
    if idx_max_loss == len(d_list):
        # canonical
        generator_type = dbi_object.mode
        dbi_object.mode = DoubleBracketGeneratorType.canonical
        dbi_object(step=step_optimal)
        dbi_object.mode = generator_type
    else:
        d_optimal = d_list[idx_max_loss]
        dbi_object(step=step_optimal, d=d_optimal)
    return idx_max_loss, step_optimal
