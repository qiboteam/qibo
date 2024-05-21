from qibo.models.dbi.utils_analytical import *
from qibo.models.dbi.utils_scheduling import polynomial_step


def gradient_Pauli(
    dbi_object,
    d: np.array,
    pauli_operator_dict: dict,
    use_ds=False,
    n=3,
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
    # n is the highest order for calculating s

    # pauli_index is the list of positions \mu
    pauli_operators = list(pauli_operator_dict.values())
    num_paul = len(pauli_operators)
    grad = np.zeros(num_paul)
    coef = off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n=n)
    s = polynomial_step(dbi_object, n=3, d=d)

    a, b, c = coef[len(coef) - 3 :]

    for i, operator in enumerate(pauli_operators):
        da, db, dc, ds = ds_di_pauli(
            dbi_object, d=d, Z_i=operator, taylor_coef=[a, b, c]
        )
        if use_ds is True:
            ds = 0
        grad[i] = (
            s**3 / 3 * da
            + s**2 / 2 * db
            + 2 * s * dc
            + s**2 * ds * a
            + s * ds * b
            + 2 * ds * c
        )
    grad = np.array(grad)
    grad = grad / np.linalg.norm(grad)
    return grad, s


def derivative_scalar_product_dbr_approx_element_wise_ansatz(dbi_object, d, h, i):
    r"""
    TODO: add formula and explain terms
    Gradient wrt the ith diagonal elements of D.
    We make Double_bracket rotation with duration given by the minimzer of the ´polynomial_step´ function.
    Gradient of the Taylor expansion of the least squares loss function as a function of $s$ the duration of Double-Bracket rotation element-wise ansatz:
    $\partial_{D_{ii}} \text{Tr}(H_k@D) \approx \sum_{k=0}^{n} \frac{1}{k!!} \partial_{D_ii}\text{Tr}(\Gamma_{k}D)$.
    Args:
        dbi_object(DoubleBracketIteration): DoubleBracketIteration object.
        d(np.array): D operator.
        h(np.array): Hamiltonian.
        i(int): Index of the diagonal element of D.
    Returns:
        derivative(float): Derivative of the polynomial expansion with respect to the ith diagonal elements of D.
    """
    derivative = 0
    s = polynomial_step(dbi_object, n=3, d=d)
    dD_di = np.zeros(d.shape)
    gamma_list = dbi_object.generate_Gamma_list(4, d)
    dD_di[i, i] = 1
    dGamma = [commutator(dD_di, h)]
    derivative += np.real(
        np.trace(gamma_list[0] @ dD_di)
        + np.trace(dGamma[0] @ d + gamma_list[1] @ dD_di) * s
    )
    for n in range(2, 4):
        dGamma.append(dGamma_diDiagonal(d, h, n, i, dGamma, gamma_list))
        derivative += np.real(
            np.trace(dGamma[-1] @ d + gamma_list[n] @ dD_di) * s**n / math.factorial(n)
        )

    return derivative


def gradientDiagonalEntries(
    dbi_object, params, h, analytic=True, d_type=d_ansatz_type.element_wise, delta=1e-4
):
    r"""
    Gradient of the DBI with respect to the parametrization of D. If analytic is True, the analytical gradient of the polynomial expansion of the DBI is used.
    As the analytical gradient is applied on the polynomial expansion of the cost function, the numerical gradients may be more accurate.

    Args:
        dbi_object(DoubleBracketIteration): DoubleBracketIteration object.
        params(np.array): Parameters for the ansatz (note that the dimension must be 2**nqubits for full ansazt and nqubits for Pauli ansatz).
        h(np.array): Hamiltonian.
        analytic(bool): If True, the gradient is calculated analytically, otherwise numerically.
        d_type(d_ansatz_type): Ansatz used for the D operator. Options are 'Full' and '1-local'.
        delta(float): Step size for numerical gradient.
    Returns:
        grad(np.array): Gradient of the D operator.
    """

    grad = np.zeros(len(params))
    d = d_ansatz(params, d_type)
    if analytic == True:
        for i in range(len(params)):
            derivative = derivative_scalar_product_dbr_approx_element_wise_ansatz(
                dbi_object, d, h, i
            )
            grad[i] = d[i, i] - derivative
    else:
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
    analytic=True,
    d_type=d_ansatz_type.element_wise,
    normalize=False,
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
        analytic(bool): If True, the gradient is calculated analytically, otherwise numerically.
        d_type(d_ansatz_type): Ansatz used for the D operator.
        normalize(bool): If True, the D operator is normalized at each iteration.
    Returns:
        d(np.array): Optimized D operator.
        loss(np.array): Loss function evaluated at each iteration.
        grad(np.array): Gradient evaluated at each iteration.
        params_hist(np.array): Parameters evaluated at each iteration.
    """

    h = dbi_object.h.matrix
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
        grad[i, :] = gradientDiagonalEntries(
            dbi_object, params, h, analytic=analytic, d_type=d_type
        )
        for j in range(len(params)):
            params[j] = params[j] - lr * grad[i, j]
        d = d_ansatz(params, d_type, normalization=normalize)
        s = polynomial_step(dbi_new, n=3, d=d)
        dbi_new(s, d=d)
        loss[i + 1] = dbi_new.loss(0.0, d=d)
        params_hist[:, i + 1] = params

    return d, loss, grad, params_hist
