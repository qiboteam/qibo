from copy import deepcopy

from qibo.models.dbi.utils_analytical import *
from qibo.models.dbi.utils_scheduling import polynomial_step


def gradientDiagonalEntries(
    dbi_object, params, d_type=d_ansatz_type.element_wise, delta=1e-4
):
    r"""
    Gradient of the DBI with respect to the parametrization of D. A simple finite difference is used to calculate the gradient.

    Args:
        dbi_object(DoubleBracketIteration): DoubleBracketIteration object.
        params(np.array): Parameters for the ansatz (note that the dimension must be 2**nqubits for full ansazt and nqubits for Pauli ansatz).
        h(np.array): Hamiltonian.
        d_type(d_ansatz_type): Ansatz used for the D operator. Options are 'Full' and '1-local'.
        delta(float): Step size for numerical gradient.
    Returns:
        grad(np.array): Gradient of the D operator.
    """

    grad = np.zeros(len(params))
    d = d_ansatz(params, d_type)
    for i in range(len(params)):
        params_new = deepcopy(params)
        params_new[i] += delta
        d_new = d_ansatz(params_new, d_type)
        grad[i] = (dbi_object.loss(0.0, d_new) - dbi_object.loss(0.0, d)) / delta
    return grad


def gradient_descent_dbr_d_ansatz(
    dbi_object,
    params,
    nmb_iterations,
    lr=1e-2,
    d_type=d_ansatz_type.element_wise,
    normalize=True,
):
    r"""
    Optimizes the D operator using gradient descent evaluated at the at the rotaion angle found using the polynomial expansion.
    - Declare variables
    - Calculate initial loss
    - Iterate, learning at each the optimal D and measure loss
    - Return values
    Args:
        dbi_object(DoubleBracketIteration): DoubleBracketIteration object.
        params(np.array): Initial parameters for the ansatz (note that the dimension must be 2**nqubits for full ansazt and nqubits for Pauli ansatz).
        nmb_iterations(int): Number of gradient descent iterations.
        lr(float): Learning rate.
        d_type(d_ansatz_type): Ansatz used for the D operator.
        normalize(bool): If True, the D operator is normalized at each iteration.
    Returns:
        d(np.array): Optimized D operator.
        loss(np.array): Loss function evaluated at each iteration.
        grad(np.array): Gradient evaluated at each iteration.
        params_hist(np.array): Parameters evaluated at each iteration.
    """

    d = d_ansatz(params, d_type, normalization=normalize)
    loss = np.zeros(nmb_iterations + 1)
    grad = np.zeros((nmb_iterations, len(params)))
    dbi_new = deepcopy(dbi_object)
    s = polynomial_step(dbi_object, n=3, d=d)
    dbi_new(s, d=d)
    loss[0] = dbi_new.loss(0.0, d)
    params_hist = np.empty((len(params), nmb_iterations + 1))
    params_hist[:, 0] = params

    for i in range(nmb_iterations):
        dbi_new = deepcopy(dbi_object)
        grad[i, :] = gradientDiagonalEntries(dbi_object, params, d_type=d_type)
        for j in range(len(params)):
            params[j] = params[j] - lr * grad[i, j]
        d = d_ansatz(params, d_type, normalization=normalize)
        s = polynomial_step(dbi_new, n=3, d=d)
        dbi_new(s, d=d)
        loss[i + 1] = dbi_new.loss(0.0, d=d)
        params_hist[:, i + 1] = params

    return d, loss, grad, params_hist
