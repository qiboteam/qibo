import math
from copy import deepcopy
from typing import Optional

import numpy as np

from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.dbi.utils import *


def d_gamma_di_pauli(dbi_object, n: int, Z_i: np.array, d: np.array):
    """Computes the derivatives $\frac{\\partial \\Gamma_n}{\\partial \alpha_i}$ where the diagonal operator $D=\\sum \alpha_i Z_i$.

    Args:
        dbi_object (DoubleBracketIteration): the target dbi object
        n (int): the number of nested commutators in `Gamma`
        i (int/tupple): the index of onsite-Z coefficient
        d (np.array): the diagonal operator

    Returns:
        (list): [d_gamma_0_di, d_gamma_1_di, ..., d_gamma_n_di]
    """
    nqubits = int(np.log2(dbi_object.h.matrix.shape[0]))
    d_gamma_di = [np.zeros((2**nqubits, 2**nqubits))] * (n + 1)
    gamma_list = dbi_object.generate_gamma_list(n=n + 2, d=d)
    W = dbi_object.commutator(dbi_object.backend.cast(d), dbi_object.h.matrix)

    dW_di = dbi_object.commutator(dbi_object.backend.cast(Z_i), dbi_object.h.matrix)
    for k in range(n + 1):
        if k == 0:
            continue
        elif k == 1:
            d_gamma_di[k] = dW_di
        else:
            d_gamma_di[k] = dbi_object.commutator(
                dW_di, gamma_list[k - 1]
            ) + dbi_object.commutator(W, d_gamma_di[k - 1])
    return d_gamma_di


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
    d_gamma_di = d_gamma_di_pauli(dbi_object, n=4, Z_i=Z_i, d=d)
    gamma_list = dbi_object.generate_gamma_list(n=4, d=d)

    def derivative_product(k1, k2):
        r"""Calculate the derivative of a product $\sigma(\Gamma(n1,i))@\sigma(\Gamma(n2,i))"""
        return dbi_object.sigma(d_gamma_di[k1]) @ dbi_object.sigma(
            gamma_list[k2]
        ) + dbi_object.sigma(dbi_object.sigma(gamma_list[k1])) @ dbi_object.sigma(
            d_gamma_di[k2]
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


def d_gamma_d_diagonal(d, h, n, i, d_gamma, gamma_list):
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
        d_gamma(list): List of the n-1 derivatives of the gamma operators (better to keep them in memory than to calculate at each iteration).
        gamma_list(list): List of the n gamma operators.
    Returns:
        (float): Derivative of the nth gamma operator with respect to the ith diagonal elements of D.
    """
    dD_di = np.zeros(d.shape)
    dD_di[i, i] = 1
    dW_di = commutator(commutator(dD_di, h), gamma_list[n - 1])
    w = commutator(d, h)
    return dW_di + commutator(w, d_gamma[-1])


def d_polynomial_d_diagonal(dbi_object, s, d, H, i):
    # Derivative of polynomial approximation of potential function with respect to diagonal elements of d (full-diagonal ansatz)
    # Formula can be expanded easily to any order, with n=3 corresponding to cubic approximation
    derivative = 0
    A = np.zeros(d.shape)
    gamma_list = dbi_object.generate_gamma_list(4, d)
    A[i, i] = 1
    d_gamma = [commutator(A, H)]
    derivative += np.real(
        np.trace(gamma_list[0] @ A) + np.trace(d_gamma[0] @ d + gamma_list[1] @ A) * s
    )
    for n in range(2, 4):
        d_gamma.append(d_gamma_d_diagonal(d, H, n, i, d_gamma, gamma_list))
        derivative += np.real(
            np.trace(d_gamma[-1] @ d + gamma_list[n] @ A) * s**n / math.factorial(n)
        )

    return derivative


def off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n):
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    W = dbi_object.commutator(
        dbi_object.backend.cast(d), dbi_object.sigma(dbi_object.h.matrix)
    )
    gamma_list = dbi_object.generate_gamma_list(n + 2, d)
    sigma_gamma_list = list(map(dbi_object.sigma, gamma_list))
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients for rotation with [W,H] and H
    c1 = exp_list.reshape((-1, 1, 1)) * sigma_gamma_list[1:]
    c2 = exp_list.reshape((-1, 1, 1)) * sigma_gamma_list[:-1]
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


def least_squares_polynomial_expansion_coef(dbi_object, d, n: int = 3):
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    gamma_list = dbi_object.generate_gamma_list(n + 1, d)
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients
    coef = np.empty(n)
    for i in range(n):
        coef[i] = np.real(
            exp_list[i] * np.trace(dbi_object.backend.cast(d) @ gamma_list[i + 1])
        )
    coef = list(reversed(coef))
    return coef


def energy_fluctuation_polynomial_expansion_coef(
    dbi_object, d: np.array = None, n: int = 3, state=0
):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    gamma_list = dbi_object.generate_gamma_list(n + 1, d)
    # coefficients
    coef = np.empty(3)
    state_cast = dbi_object.backend.cast(state)
    state_dag = dbi_object.backend.cast(state.conj().T)

    def variance(a):
        """Calculates the variance of a matrix A with respect to a state:
        Var($A$) = $\\langle\\mu|A^2|\\mu\rangle-\\langle\\mu|A|\\mu\rangle^2$"""
        b = a @ a
        return state_dag @ b @ state_cast - (state_dag @ a @ state_cast) ** 2

    def covariance(a, b):
        """This is a generalization of the notion of covariance, needed for the polynomial expansion of the energy fluctuation,
        applied to two operators A and B with respect to a state:
        Cov($A,B$) = $\\langle\\mu|AB|\\mu\rangle-\\langle\\mu|A|\\mu\rangle\\langle\\mu|B|\\mu\rangle$
        """

        c = a @ b + b @ a
        return (
            state_dag @ c @ state_cast
            - 2 * state_dag @ a @ state_cast * state_dag @ b @ state_cast
        )

    coef[0] = np.real(2 * covariance(gamma_list[0], gamma_list[1]))
    coef[1] = np.real(2 * variance(gamma_list[1]))
    coef[2] = np.real(
        covariance(gamma_list[0], gamma_list[3])
        + 3 * covariance(gamma_list[1], gamma_list[2])
    )
    coef = list(reversed(coef))
    return coef


def gradient_diagonal_entries(dbi_object, params, delta=1e-4):
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
    d = element_wise_d(params)
    for i in range(len(params)):
        params_new = deepcopy(params)
        params_new[i] += delta
        d_new = element_wise_d(params_new)
        grad[i] = (dbi_object.loss(0.0, d_new) - dbi_object.loss(0.0, d)) / delta
    return grad


def gradient_descent_dbr_d_ansatz(
    dbi_object,
    params,
    nmb_iterations,
    lr=1e-2,
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

    d = element_wise_d(params, normalization=normalize)
    loss = np.zeros(nmb_iterations + 1)
    grad = np.zeros((nmb_iterations, len(params)))
    dbi_eval = deepcopy(dbi_object)
    s = dbi_eval.choose_step(d=d)
    dbi_eval(s, d=d)
    loss[0] = dbi_eval.loss(0.0, d)
    params_hist = np.empty((len(params), nmb_iterations + 1))
    params_hist[:, 0] = params

    for i in range(nmb_iterations):
        dbi_eval = deepcopy(dbi_object)
        grad[i, :] = gradient_diagonal_entries(dbi_eval, params)
        for j in range(len(params)):
            params[j] = params[j] - lr * grad[i, j]
        d = element_wise_d(params, normalization=normalize)
        s = dbi_eval.choose_step(d=d)
        loss[i + 1] = dbi_eval.loss(s, d=d)
        params_hist[:, i + 1] = params

    return d, loss, grad, params_hist
