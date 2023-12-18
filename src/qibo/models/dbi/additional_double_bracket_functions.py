from copy import deepcopy
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
    """Visualize hamiltonian in a heatmap form."""
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
    ax.set_title(r"Drift: $|\hat{H}_0 - \hat{H}_{\ell}|$")
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
        labels_copy = labels
        labels_copy.insert(0, "Initial")
        for i, label in enumerate(labels_copy):
            plt.text(x_axis[i], loss_histories[i], label)

    plt.grid()
    plt.xlabel(r"Flow duration $s$")
    plt.title("Loss function histories")


def generate_Z_operators(n_qubits: int):
    """Generate a list of local_Z operators with n_qubits and their respective names."""
    combination_strings = product("ZI", repeat=n_qubits)
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


def iteration_from_list(
    class_dbi: DoubleBracketIteration,
    d_list: list,
    step: float = None,
    compare_canonical: bool = True,
):
    """Performs 1 double-bracket iteration using the optimal generator from operator_list.
    Returns the index of the optimal operator
    """
    h_before = deepcopy(class_dbi.h)
    off_diagonal_norms = []
    for d in d_list:
        # fixed step
        if step is not None:
            class_dbi(step=step, d=d)
        # hyperopt
        else:
            step = class_dbi.hyperopt_step(
                step_min=1e-5,
                step_max=1,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=100,
                d=d,
            )
        off_diagonal_norms.append(class_dbi.off_diagonal_norm)
        class_dbi.h = deepcopy(h_before)
    # canonical
    if compare_canonical is True:
        generator_type = class_dbi.mode
        class_dbi.mode = DoubleBracketGeneratorType.canonical
        if step is not None:
            class_dbi(step=step)
        else:
            step = class_dbi.hyperopt_step(
                step_min=1e-5,
                step_max=1,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=100,
            )
        off_diagonal_norms.append(class_dbi.off_diagonal_norm)
        class_dbi.h = deepcopy(h_before)
        class_dbi.mode = generator_type
    # find best d
    idx_max_loss = off_diagonal_norms.index(min(off_diagonal_norms))
    # run with optimal d
    if idx_max_loss == len(d_list):
        # canonical
        generator_type = class_dbi.mode
        class_dbi.mode = DoubleBracketGeneratorType.canonical
        if step is not None:
            class_dbi(step=step)
        else:
            step = class_dbi.hyperopt_step(
                step_min=1e-5,
                step_max=1,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=100,
            )
        class_dbi.mode = generator_type
    else:
        d_optimal = d_list[idx_max_loss]
        # fixed step
        if step is not None:
            class_dbi(step=step, d=d_optimal)
        # hyperopt
        else:
            step = class_dbi.hyperopt_step(
                step_min=1e-5,
                step_max=1,
                space=hp.uniform,
                optimizer=tpe,
                max_evals=100,
                d=d_optimal,
            )

    return idx_max_loss, step
