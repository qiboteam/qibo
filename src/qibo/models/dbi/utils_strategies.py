import hyperopt

from qibo.models.dbi.double_bracket import *
from qibo.models.dbi.utils import cs_angle_sgn
from qibo.models.dbi.utils_analytical import *
from qibo.models.dbi.utils_scheduling import polynomial_step


def select_best_dbr_generator(
    dbi_object: DoubleBracketIteration,
    d_list: list,
    step: Optional[float] = None,
    compare_canonical: bool = True,
    scheduling: DoubleBracketScheduling = None,
    **kwargs,
):
    """Selects the best double bracket rotation generator from a list and execute the rotation.

    Args:
        dbi_object (`DoubleBracketIteration`): the target DoubleBracketIteration object.
        d_list (list): list of diagonal operators (np.array) to run from.
        step (float): fixed iteration duration.
            Defaults to ``None``, optimize with `scheduling` method and `choose_step` function.
        compare_canonical (boolean): if `True`, the diagonalization effect with operators from `d_list` is compared with the canonical bracket.
        scheduling (`DoubleBracketScheduling`): scheduling method for finding the optimal step.

    Returns:
        The updated dbi_object, index of the optimal diagonal operator, respective step duration, and evolution direction.
    """
    if scheduling is None:
        scheduling = dbi_object.scheduling
    norms_off_diagonal_restriction = [dbi_object.off_diagonal_norm] * (len(d_list) + 1)
    optimal_steps = np.zeros(len(d_list) + 1)
    flip_list = np.ones(len(d_list) + 1)
    for i, d in enumerate(d_list):
        # prescribed step durations
        dbi_eval = deepcopy(dbi_object)
        flip_list[i] = cs_angle_sgn(dbi_eval, d)
        if flip_list[i] != 0:
            if step is None:
                step_best = dbi_eval.choose_step(
                    d=flip_list[i] * d, scheduling=scheduling, **kwargs
                )
            else:
                step_best = step
            dbi_eval(step=step_best, d=flip_list[i] * d)
            optimal_steps[i] = step_best
            norms_off_diagonal_restriction[i] = dbi_eval.off_diagonal_norm
    # canonical
    if compare_canonical is True:
        dbi_eval = deepcopy(dbi_object)
        dbi_eval.mode = DoubleBracketGeneratorType.canonical
        if step is None:
            step_best = dbi_eval.choose_step(scheduling=scheduling, **kwargs)
        else:
            step_best = step
        dbi_eval(step=step_best)
        optimal_steps[-1] = step_best
        norms_off_diagonal_restriction[-1] = dbi_eval.off_diagonal_norm
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


def gradient_descent_pauli(
    dbi_object: DoubleBracketIteration,
    d_coef: list,
    d: Optional[np.array] = None,
    pauli_operator_dict: dict = None,
    parameterization_order: int = 1,
    n: int = 3,
    onsite_Z_ops: Optional[list] = None,
    lr_min: float = 1e-5,
    lr_max: float = 1,
    max_evals: int = 100,
    space: callable = None,
    optimizer: callable = None,
    verbose: bool = False,
    use_ds: bool = True,
):
    """calculate the elements of one gradient descent step on `dbi_object`.

    Args:
        dbi_object (DoubleBracketIteration): the target dbi object
        d_coef (list): the initial decomposition of `d` into Pauli-Z operators
        d (np.array, optional): the initial diagonal operator. Defaults to None.
        n_taylor (int, optional): the highest order to expand the loss function derivative. Defaults to 2.
        onsite_Z_ops (list, optional): list of onsite-Z operators. Defaults to None.
        lr_min (float, optional): the minimal gradient step. Defaults to 1e-5.
        lr_max (float, optional): the maximal gradient step. Defaults to 1.
        max_evals (int, optional): the max number of evaluations for `hyperopt` to find the optimal gradient step `lr`. Defaults to 100.
        space (callable, optional): the search space for `hyperopt`. Defaults to None.
        optimizer (callable, optional): optimizer for `hyperopt`. Defaults to None.
        verbose (bool, optional): option to print out the 'hyperopt' progress. Defaults to False.
        use_ds (bool, optional): if False, ds is set to 0. Defaults to True.

    Returns:
        the optimal step found, coeffcients of `d` in Pauli-Z basis, matrix of `d`

    """
    nqubits = int(np.log2(dbi_object.h.matrix.shape[0]))
    if pauli_operator_dict is None:
        pauli_operator_dict = generate_pauli_operator_dict(
            nqubits, parameterization_order
        )

    grad, s = gradient_Pauli(
        dbi_object, d, n=n, pauli_operator_dict=pauli_operator_dict, use_ds=use_ds
    )
    # optimize gradient descent step with hyperopt
    if space is None:
        space = hyperopt.hp.loguniform("lr", np.log(lr_min), np.log(lr_max))
    if optimizer is None:
        optimizer = hyperopt.tpe

    def func_loss_to_lr(lr):
        d_coef_eval = [d_coef[j] - grad[j] * lr for j in range(nqubits)]
        d_eval = sum(
            [
                d_coef_eval[i] * list(pauli_operator_dict.values())[i]
                for i in range(nqubits)
            ]
        )
        return dbi_object.loss(step=s, d=d_eval)

    best = hyperopt.fmin(
        fn=func_loss_to_lr,
        space=space,
        algo=optimizer.suggest,
        max_evals=max_evals,
        verbose=verbose,
    )
    lr = best["lr"]

    d_coef = [d_coef[j] - grad[j] * lr for j in range(nqubits)]
    d = sum([d_coef[i] * list(pauli_operator_dict.values())[i] for i in range(nqubits)])
    return s, d_coef, d


def gradientDiagonal(dbi_object, d, H, n=3):
    # Gradient of potential function with respect to diagonal elements of D (full-diagonal ansatz)
    grad = np.zeros(len(d))
    for i in range(len(d)):
        s = polynomial_step(dbi_object, n=3, d=d)
        derivative = dpolynomial_diDiagonal(dbi_object, s, d, H, i)
        grad[i] = d[i, i] - derivative
    return grad


def gradient_ascent(dbi_object, d, step, iterations):
    H = dbi_object.h.matrix
    loss = np.zeros(iterations + 1)
    grad = np.zeros((iterations, len(d)))
    dbi_eval = deepcopy(dbi_object)
    s = polynomial_step(dbi_object, n=3, d=d)
    dbi_eval(s, d=d)
    loss[0] = dbi_eval(d)
    diagonals = np.empty((len(d), iterations + 1))
    diagonals[:, 0] = np.diag(d)

    for i in range(iterations):
        dbi_eval = deepcopy(dbi_object)
        grad[i, :] = gradientDiagonal(dbi_object, d, H)
        for j in range(len(d)):
            d[j, j] = d[j, j] - step * grad[i, j]
        s = polynomial_step(dbi_object, n=3, d=d)
        dbi_eval(s, d=d)
        loss[i + 1] = dbi_eval.least_squares(d)
        diagonals[:, i + 1] = np.diag(d)

    return d, loss, grad, diagonals
