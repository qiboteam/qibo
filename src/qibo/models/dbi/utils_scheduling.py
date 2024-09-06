import math
from typing import Optional

import numpy as np
import optuna

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
    self,
    step_min: float = 1e-5,
    step_max: float = 1,
    max_evals: int = 1000,
    look_ahead: int = 1,
    verbose: bool = False,
    d: np.array = None,
    optimizer: optuna.samplers.BaseSampler = None,
):
    """
    Optimize iteration step using Optuna.

    Args:
        step_min: lower bound of the search grid;
        step_max: upper bound of the search grid;
        max_evals: maximum number of trials done by the optimizer;
        look_ahead: number of iteration steps to compute the loss function;
        verbose: level of verbosity;
        d: diagonal operator for generating double-bracket iterations;
        optimizer: Optuna sampler for the search algorithm (e.g.,
            optuna.samplers.TPESampler()).
            See: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html

    Returns:
        (float): optimized best iteration step.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        step = trial.suggest_float("step", step_min, step_max)
        return self.loss(step, d=d, look_ahead=look_ahead)

    if optimizer is None:
        optimizer = optuna.samplers.TPESampler()

    study = optuna.create_study(direction="minimize", sampler=optimizer)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=verbose)

    return study.best_params["step"]


def polynomial_step(
    dbi_object,
    n: int = 2,
    n_max: int = 5,
    d: np.array = None,
    coef: Optional[list] = None,
    cost: Optional[str] = None,
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
        coef = dbi_object.cost_expansion(d=d, n=n)
    roots = np.roots(coef)
    real_positive_roots = [
        np.real(root) for root in roots if np.imag(root) < 1e-3 and np.real(root) > 0
    ]
    # solution exists, return minimum s
    if len(real_positive_roots) > 0:
        losses = [dbi_object.loss(step=root, d=d) for root in real_positive_roots]
        return real_positive_roots[losses.index(min(losses))]
    # solution does not exist, return None
    else:
        return None


def simulated_annealing_step(
    dbi_object,
    d: Optional[np.array] = None,
    initial_s=None,
    step_min=1e-5,
    step_max=1,
    s_jump_range=None,
    s_jump_range_divident=5,
    initial_temp=1,
    cooling_rate=0.85,
    min_temp=1e-5,
    max_iter=200,
):
    """
    Perform a single step of simulated annealing optimization.

    Parameters:
        dbi_object: DBI object
            The object representing the problem to be optimized.
        d: Optional[np.array], optional
            The diagonal matrix 'd' used in optimization. If None, it uses the diagonal
            matrix 'diagonal_h_matrix' from dbi_object.
        initial_s: float or None, optional
            Initial value for 's', the step size. If None, it is initialized using
            polynomial_step function with 'n=4'. If 'polynomial_step' returns None,
            'initial_s' is set to 'step_min'.
        step_min: float, optional
            Minimum value for the step size 's'.
        step_max: float, optional
            Maximum value for the step size 's'.
        s_jump_range: float or None, optional
            Range for the random jump in step size. If None, it's calculated based on
            'step_min', 'step_max', and 's_jump_range_divident'.
        s_jump_range_divident: int, optional
            Dividend to determine the range for random jump in step size.
        initial_temp: float, optional
            Initial temperature for simulated annealing.
        cooling_rate: float, optional
            Rate at which temperature decreases in simulated annealing.
        min_temp: float, optional
            Minimum temperature threshold for termination of simulated annealing.
        max_iter: int, optional
            Maximum number of iterations for simulated annealing.

    Returns:
        float:
            The optimized step size 's'.
    """

    if d is None:
        d = dbi_object.diagonal_h_matrix
    if initial_s is None:
        initial_s = polynomial_step(dbi_object=dbi_object, d=d, n=4)
        # TODO: implement test to catch this if statement
        if initial_s is None:  # pragma: no cover
            initial_s = step_min
    if s_jump_range is None:
        s_jump_range = (step_max - step_min) / s_jump_range_divident
    current_s = initial_s
    current_loss = dbi_object.loss(d=d, step=current_s)
    temp = initial_temp

    for _ in range(max_iter):
        candidate_s = max(
            step_min,
            min(
                current_s + np.random.uniform(-1 * s_jump_range, s_jump_range), step_max
            ),
        )
        candidate_loss = dbi_object.loss(d=d, step=candidate_s)

        # Calculate change in loss
        delta_loss = candidate_loss - current_loss

        # Determine if the candidate solution is an improvement
        if delta_loss < 0 or np.random.rand() < math.exp(-delta_loss / temp):
            current_s = candidate_s
            current_loss = candidate_loss
        # Cool down
        temp *= cooling_rate
        if temp < min_temp or current_s > step_max or current_s < step_min:
            break

    return current_s
