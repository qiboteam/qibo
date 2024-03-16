import math
from functools import partial
from typing import Optional

import hyperopt
import numpy as np

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
    verbose: bool = False,
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
        verbose: level of verbosity;
        d: diagonal operator for generating double-bracket iterations.

    Returns:
        (float): optimized best iteration step (minimizing off-diagonal norm).
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
        verbose=verbose,
    )
    return best["step"]


def polynomial_step(
    dbi_object,
    n: int = 2,
    n_max: int = 5,
    d: np.array = None,
    coef: Optional[list] = None,
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

    if d is None:
        d = dbi_object.diagonal_h_matrix

    if n > n_max:
        raise ValueError(
            "No solution can be found with polynomial approximation. Increase `n_max` or use other scheduling methods."
        )
    if coef is None:
        coef = off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n)
    roots = np.roots(coef)
    real_positive_roots = [
        np.real(root) for root in roots if np.imag(root) < error and np.real(root) > 0
    ]
    # solution exists, return minimum s
    if len(real_positive_roots) > 0:
        return min(real_positive_roots)
    # solution does not exist, return None
    else:
        return None


def off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    W = dbi_object.commutator(d, dbi_object.sigma(dbi_object.h.matrix))
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
