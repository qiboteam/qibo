import hyperopt

from qibo.backends import _check_backend
from qibo.models.dbi.double_bracket import *
from qibo.models.dbi.utils import *
from qibo.models.dbi.utils_gradients import *
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
        d = dbi_eval.backend.cast(d)
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


def gradient_pauli(
    dbi_object,
    d: np.array,
    pauli_operator_dict: dict,
    use_ds=False,
    n=3,
    backend=None,
    **kwargs,
):
    r"""Calculate the gradient of loss function with respect to onsite Pauli-Z coefficients
    Args:
        dbi_object (DoubleBracketIteration): the target dbi object
        d (np.array): the diagonal operator
        n_taylor (int): the highest order of the taylore expansion of  w.r.t `s`
        onsite_Z_ops (list): list of Pauli-Z operators
        taylor_coef (list): coefficients of `s` in the taylor expansion of math:`\\frac{\\partial ||\sigma(e^{sW}He^{-sW})||^2}{\\partial s}`
        use_ds (boolean): if False, ds is set to 0
    """
    backend = _check_backend(backend)
    # n is the highest order for calculating s

    # pauli_index is the list of positions \mu
    pauli_operators = list(pauli_operator_dict.values())
    num_paul = len(pauli_operators)
    grad = []
    coef = off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n=n)
    s = polynomial_step(dbi_object, n=5, d=d)

    a, b, c = coef[len(coef) - 3 :]
    for i, operator in enumerate(pauli_operators):
        da, db, dc, ds = ds_di_pauli(
            dbi_object, d=d, Z_i=backend.cast(operator), taylor_coef=[a, b, c]
        )
        if use_ds is True:
            ds = 0
        grad.append(
            s**3 / 3 * da
            + s**2 / 2 * db
            + 2 * s * dc
            + s**2 * ds * a
            + s * ds * b
            + 2 * ds * c
        )
    grad = backend.to_numpy(grad)
    grad = grad / np.linalg.norm(grad)
    return grad, s


def gradient_descent_pauli(
    dbi_object: DoubleBracketIteration,
    d_coef: list,
    d: Optional[np.array] = None,
    pauli_operator_dict: dict = None,
    parameterization_order: int = 1,
    n: int = 3,
    lr_min: float = 1e-5,
    lr_max: float = 1,
    max_evals: int = 100,
    space: callable = None,
    optimizer: callable = None,
    verbose: bool = False,
    use_ds: bool = True,
    backend=None,
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
    backend = _check_backend(backend)
    nqubits = int(np.log2(dbi_object.h.matrix.shape[0]))
    if pauli_operator_dict is None:
        pauli_operator_dict = generate_pauli_operator_dict(
            nqubits,
            parameterization_order,
            backend=backend,
        )

    grad, s = gradient_pauli(
        dbi_object,
        d,
        n=n,
        pauli_operator_dict=pauli_operator_dict,
        use_ds=use_ds,
        backend=backend,
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