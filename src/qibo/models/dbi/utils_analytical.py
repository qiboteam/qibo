import math
from copy import deepcopy
from enum import Enum, auto
from typing import Optional

import numpy as np

from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.dbi.utils import commutator, covariance, variance
from qibo.models.dbi.utils_scheduling import polynomial_step


def dGamma_di_pauli(dbi_object, n: int, Z_i: np.array, d: np.array):
    """Computes the derivatives $\frac{\\partial \\Gamma_n}{\\partial \alpha_i}$ where the diagonal operator $D=\\sum \alpha_i Z_i$.

    Args:
        dbi_object (DoubleBracketIteration): the target dbi object
        n (int): the number of nested commutators in `Gamma`
        i (int/tupple): the index of onsite-Z coefficient
        d (np.array): the diagonal operator

    Returns:
        (list): [dGamma_0_di, dGamma_1_di, ..., dGamma_n_di]
    """
    nqubits = int(np.log2(dbi_object.h.matrix.shape[0]))
    dGamma_di = [np.zeros((2**nqubits, 2**nqubits))] * (n + 1)
    Gamma_list = dbi_object.generate_Gamma_list(n=n + 2, d=d)
    W = dbi_object.commutator(dbi_object.backend.cast(d), dbi_object.h.matrix)
    dW_di = dbi_object.commutator(dbi_object.backend.cast(Z_i), dbi_object.h.matrix)
    for k in range(n + 1):
        if k == 0:
            continue
        elif k == 1:
            dGamma_di[k] = dW_di
        else:
            dGamma_di[k] = dbi_object.commutator(
                dW_di, Gamma_list[k - 1]
            ) + dbi_object.commutator(W, dGamma_di[k - 1])
    return dGamma_di


def ds_di_pauli(
    dbi_object,
    d: np.array,
    Z_i: np.array,
    taylor_coef: Optional[list] = None,
):
    r"""Return the derivatives of the first 3 polynomial coefficients with respect to onsite Pauli-Z coefficients\
        Args:
            dbi_object (DoubleBracketIteration): the target dbi object
            d (np.array): the diagonal operator
            i (int): the index of onsite-Z coefficient
            taylor_coef (list): coefficients of `s` in the taylor expansion of math:`\\frac{\\partial ||\sigma(e^{sW}He^{-sW})||^2}{\\partial s}`, from the highest order to the lowest.
            onsite_Z_ops (list): onsite Z operators of `dbi_object.h`
        Returns:
            floats da, db, dc, ds
    """
    # generate the list of derivatives w.r.t ith Z operator coefficient
    dGamma_di = dGamma_di_pauli(dbi_object, n=4, Z_i=Z_i, d=d)
    Gamma_list = dbi_object.generate_Gamma_list(n=4, d=d)

    def derivative_product(k1, k2):
        r"""Calculate the derivative of a product $\sigma(\Gamma(n1,i))@\sigma(\Gamma(n2,i))"""
        return dbi_object.sigma(dGamma_di[k1]) @ dbi_object.sigma(
            Gamma_list[k2]
        ) + dbi_object.sigma(dbi_object.sigma(Gamma_list[k1])) @ dbi_object.sigma(
            dGamma_di[k2]
        )

    # calculate the derivatives of s polynomial coefficients
    da = np.trace(3 * derivative_product(1, 2) + 3 * derivative_product(3, 0))
    db = np.trace(2 * derivative_product(1, 1) + 2 * derivative_product(0, 2))
    dc = np.trace(2 * derivative_product(1, 0))

    ds = 0
    if taylor_coef != None:
        a, b, c = taylor_coef[len(taylor_coef) - 3 :]
        delta = b**2 - 4 * a * c
        ddelta = 2 * (b * db - 2 * (a * dc + da * c))

        ds = (-db + 0.5 * ddelta / np.sqrt(delta)) * a - (-b + np.sqrt(delta)) * da
        ds /= 2 * a**2

    return da, db, dc, ds


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
    s = dbi_object.choose_step(
        d=d,
        **kwargs,
    )

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


def dGamma_diDiagonal(d, h, n, i, dGamma, gamma_list):
    r"""
    Gradient of the nth gamma operator with respect to the ith diagonal elements of D.
    $Gamma_{n} = [W,[W,...,[W,H]]...]]$,
    $\frac{\partial Gamma_{n}}{\partial D_{ii}} = \partial_{D_{ii}} W\Gamma_{n-1}-\partial_{D_{ii}}\Gamma_{n-1} W$.
    and thus is can be computed recursively.
    Args:
        d(np.array): D operator.
        h(np.array): Hamiltonian.
        n(int): nth Gamma operator.
        i(int): Index of the diagonal element of D.
        dGamma(list): List of the n-1 derivatives of the gamma operators (better to keep them in memory than to calculate at each iteration).
        gamma_list(list): List of the n gamma operators.
    Returns:
        (float): Derivative of the nth gamma operator with respect to the ith diagonal elements of D.
    """
    dD_di = np.zeros(d.shape)
    dD_di[i, i] = 1
    dW_di = commutator(commutator(dD_di, h), gamma_list[n - 1])
    w = commutator(d, h)
    return dW_di + commutator(w, dGamma[-1])


def dpolynomial_diDiagonal(dbi_object, s, d, H, i):
    # Derivative of polynomial approximation of potential function with respect to diagonal elements of d (full-diagonal ansatz)
    # Formula can be expanded easily to any order, with n=3 corresponding to cubic approximation
    derivative = 0
    A = np.zeros(d.shape)
    Gamma_list = dbi_object.generate_Gamma_list(4, d)
    A[i, i] = 1
    dGamma = [commutator(A, H)]
    derivative += np.real(
        np.trace(Gamma_list[0] @ A) + np.trace(dGamma[0] @ d + Gamma_list[1] @ A) * s
    )
    for n in range(2, 4):
        dGamma.append(dGamma_diDiagonal(d, H, n, i, dGamma, Gamma_list))
        derivative += np.real(
            np.trace(dGamma[-1] @ d + Gamma_list[n] @ A) * s**n / math.factorial(n)
        )

    return derivative


def off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    W = dbi_object.commutator(
        dbi_object.backend.cast(d), dbi_object.sigma(dbi_object.h.matrix)
    )
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


def least_squares_polynomial_expansion_coef(dbi_object, d: np.array = None, n: int = 3):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n + 1, d)
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients
    coef = np.empty(n)
    for i in range(n):
        coef[i] = np.real(exp_list[i] * np.trace(d @ Gamma_list[i + 1]))
    coef = list(reversed(coef))
    return coef


def energy_fluctuation_polynomial_expansion_coef(
    dbi_object, d: np.array = None, n: int = 3, state=0
):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n + 1, d)
    # coefficients
    coef = np.empty(3)
    coef[0] = np.real(2 * covariance(Gamma_list[0], Gamma_list[1], state))
    coef[1] = np.real(2 * variance(Gamma_list[1], state))
    coef[2] = np.real(
        covariance(Gamma_list[0], Gamma_list[3], state)
        + 3 * covariance(Gamma_list[1], Gamma_list[2], state)
    )
    coef = list(reversed(coef))
    return coef


class d_ansatz_type(Enum):

    element_wise = auto()
    local_1 = auto()


def d_ansatz(params: np.array, d_type: d_ansatz_type, normalization: bool = False):
    r"""
    Creates the $D$ operator for the double-bracket iteration ansatz depending on the type of parameterization.
    If $\alpha_i$ are our parameters and d the number of qubits then:

    element_wise: $D = \sum_{i=0}^{2^d} \alpha_i |i\rangle \langle i|$
    local_1: $D = \sum_{i=1}^{d} \alpha_i Z_i$
    Args:
        params(np.array): parameters for the ansatz.
        d_type(d_ansatz type): type of parameterization for the ansatz.
        normalization(bool): If True, the diagonal is normalized to 1.
    """

    if d_type is d_ansatz_type.element_wise:
        d = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            d[i, i] = params[i]

    elif d_type is d_ansatz_type.local_1:

        op_list = [params[i] * symbols.Z(i) for i in range(len(params))]
        symbolHam = op_list[0]
        for i in range(len(params) - 1):
            symbolHam += op_list[i + 1]

        d = SymbolicHamiltonian(symbolHam, nqubits=len(params))
        d = d.dense.matrix
    else:
        raise ValueError(f"Parameterization type {type} not recognized.")
    if normalization:
        d = d / np.linalg.norm(d)
    return d


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
