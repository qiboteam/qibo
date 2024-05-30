import hyperopt

from qibo.backends import _check_backend
from qibo.models.dbi.double_bracket import *
from qibo.models.dbi.utils import *


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


def gradient_numerical(
    dbi_object: DoubleBracketIteration,
    d_params: list,
    parameterization: ParameterizationTypes,
    s: float = 1e-2,
    delta: float = 1e-3,
    backend=None,
    **kwargs,
):
    r"""
    Gradient of the DBI with respect to the parametrization of D. A simple finite difference is used to calculate the gradient.

    Args:
        dbi_object(DoubleBracketIteration): DoubleBracketIteration object.
        d_params(np.array): Parameters for the ansatz (note that the dimension must be 2**nqubits for full ansazt and nqubits for Pauli ansatz).
        delta(float): Step size for numerical gradient.
    Returns:
        grad(np.array): Gradient of the D operator.
    """
    backend = _check_backend(backend)
    nqubits = dbi_object.nqubits
    grad = np.zeros(len(d_params))
    d = params_to_diagonal_operator(
        d_params, nqubits, parameterization=parameterization, **kwargs
    )
    for i in range(len(d_params)):
        params_new = backend.cast(d_params, copy=True)
        params_new[i] += delta
        d_new = params_to_diagonal_operator(
            params_new, nqubits, parameterization=parameterization, **kwargs
        )
        # find the increment of a very small step
        grad[i] = (dbi_object.loss(s, d_new) - dbi_object.loss(s, d)) / delta
    return grad


def gradient_descent(
    dbi_object: DoubleBracketIteration,
    iterations: int,
    d_params: list,
    parameterization: ParameterizationTypes,
    pauli_operator_dict: list = None,
    pauli_parameterization_order: int = 1,
    normalize: bool = False,
    lr_min: float = 1e-5,
    lr_max: float = 1,
    max_evals: int = 100,
    space: callable = None,
    optimizer: callable = None,
    verbose: bool = False,
    backend=None,
):
    backend = _check_backend(backend)
    nqubits = dbi_object.nqubits
    # use polynomial scheduling for analytical solutions
    d = params_to_diagonal_operator(
        d_params,
        nqubits,
        parameterization=parameterization,
        pauli_operator_dict=pauli_operator_dict,
        normalize=normalize,
    )
    loss_hist = [dbi_object.loss(0.0, d=d)]
    d_params_hist = [d_params]
    s_hist = [0]
    if parameterization is ParameterizationTypes.pauli and pauli_operator_dict is None:
        pauli_operator_dict = generate_pauli_operator_dict(
            nqubits=nqubits,
            parameterization_order=pauli_parameterization_order,
            backend=backend,
        )
    # first step
    s = dbi_object.choose_step(d=d)
    dbi_object(step=s, d=d)
    for _ in range(iterations):
        grad = gradient_numerical(
            dbi_object,
            d_params,
            parameterization,
            pauli_operator_dict=pauli_operator_dict,
            pauli_parameterization_order=pauli_parameterization_order,
            normalize=normalize,
            backend=backend,
        )

        # set up hyperopt to find optimal lr
        def func_loss_to_lr(lr):
            d_params_eval = [d_params[j] - grad[j] * lr for j in range(len(grad))]
            d_eval = params_to_diagonal_operator(
                d_params_eval,
                nqubits,
                parameterization=parameterization,
                pauli_operator_dict=pauli_operator_dict,
                normalize=normalize,
            )
            return dbi_object.loss(step=s, d=d_eval)

        if space is None:
            space = hyperopt.hp.loguniform("lr", np.log(lr_min), np.log(lr_max))
        if optimizer is None:
            optimizer = hyperopt.tpe
        best = hyperopt.fmin(
            fn=func_loss_to_lr,
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )
        lr = best["lr"]

        d_params = [d_params[j] - grad[j] * lr for j in range(len(grad))]
        d = params_to_diagonal_operator(
            d_params,
            nqubits,
            parameterization=parameterization,
            pauli_operator_dict=pauli_operator_dict,
            normalize=normalize,
        )
        s = dbi_object.choose_step(d=d)
        dbi_object(step=s, d=d)

        # record history
        loss_hist.append(dbi_object.loss(0.0, d=d))
        d_params_hist.append(d_params)
        s_hist.append(s)
    return loss_hist, d_params_hist, s_hist
