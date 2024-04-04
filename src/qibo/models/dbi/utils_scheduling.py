import math
from copy import deepcopy
from functools import partial
from typing import Optional
from enum import Enum, auto
from qibo.models.dbi.double_bracket import DoubleBracketCost
from qibo.models.dbi.utils import off_diagonal_norm_polynomial_expansion_coef, least_squares_polynomial_expansion_coef, energy_fluctuation_polynomial_expansion_coef

import hyperopt
import numpy as np

from qibo.models.dbi.double_bracket import DoubleBracketCost

error = 1e-3


def grid_search_step(
    dbi_object,
    step_min: float = 1e-5,
    step_max: float = 1,
    num_evals: int = 100,
    space: Optional[np.array] = None,
    d: Optional[np.array] = None,
):
    """
    Greedy optimization of the iteration step.

    Args:
        step_min: lower bound of the search grid;
        step_max: upper bound of the search grid;
        mnum_evals: number of iterations between step_min and step_max;
        d: diagonal operator for generating double-bracket iterations.

    Returns:
        (float): optimized best iteration step (minimizing off-diagonal norm).
    """
    if space is None:
        space = np.linspace(step_min, step_max, num_evals)

    if d is None:
        d = dbi_object.diagonal_h_matrix

    loss_list = [dbi_object.loss(step, d=d) for step in space]

    idx_max_loss = np.argmin(loss_list)
    return space[idx_max_loss]


def hyperopt_step(
    dbi_object,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 500,
    space: callable = None,
    optimizer: callable = None,
    look_ahead: int = 1,
    d: Optional[np.array] = None,
):
    """
    Optimize iteration step using hyperopt.

    Args:
        step_min: lower bound of the search grid;
        step_max: upper bound of the search grid;
        max_evals: maximum number of iterations done by the hyperoptimizer;
        space: see hyperopt.hp possibilities;
        optimizer: see hyperopt algorithms;
        look_ahead: number of iteration steps to compute the loss function;
        d: diagonal operator for generating double-bracket iterations.

    Returns:
        (float): optimized best iteration step (minimizing loss function).
    """
    if space is None:
        space = hyperopt.hp.uniform
    if optimizer is None:
        optimizer = hyperopt.tpe
    if d is None:
        d = dbi_object.diagonal_h_matrix

    space = space("step", step_min, step_max)

    best = hyperopt.fmin(
        fn=partial(dbi_object.loss, d=d, look_ahead=look_ahead),
        space=space,
        algo=optimizer.suggest,
        max_evals=max_evals,
    )
    return best["step"]


def polynomial_step(
    dbi_object,
    n: int = 2,
    n_max: int = 5,
    d: np.array = None,
    coef: Optional[list] = None,
    cost: DoubleBracketCost = None,
):
    r"""
    Optimizes iteration step by solving the n_th order polynomial expansion of the loss function.
    e.g. $n=2$: $2\Trace(\sigma(\Gamma_1 + s\Gamma_2 + s^2/2\Gamma_3)\sigma(\Gamma_0 + s\Gamma_1 + s^2/2\Gamma_2))
    Args:
        n (int, optional): the order to which the loss function is expanded. Defaults to 4.
        n_max (int, optional): maximum order allowed for recurring calls of `polynomial_step`. Defaults to 5.
        d (np.array, optional): diagonal operator, default as $\delta(H)$.
        backup_scheduling (`DoubleBracketScheduling`): the scheduling method to use in case no real positive roots are found.
    """
    if cost is None:
        cost = dbi_object.cost

    if d is None:
        d = dbi_object.diagonal_h_matrix

    if n > n_max:
        raise ValueError(
            "No solution can be found with polynomial approximation. Increase `n_max` or use other scheduling methods."
        )
    if coef is None:
        if cost is DoubleBracketCost.off_diagonal_norm:
            coef = off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n)
        elif cost is DoubleBracketCost.least_squares:
            coef = least_squares_polynomial_expansion_coef(dbi_object, d, n)
        elif cost is DoubleBracketCost.energy_fluctuation:
            coef = energy_fluctuation_polynomial_expansion_coef(
                dbi_object, d, n, dbi_object.state
            )
        else:
            raise ValueError(f"Cost function {cost} not recognized.")

    roots = np.roots(coef)
    real_positive_roots = [
        np.real(root) for root in roots if np.imag(root) < error and np.real(root) > 0
    ]
    # solution exists, return minimum s
    if len(real_positive_roots) > 0:
        sol = min(real_positive_roots)
        for s in real_positive_roots:
            if dbi_object.loss(s, d) < dbi_object.loss(sol, d):
                sol = s
        return sol
    # solution does not exist, return None
    else:
        return None

