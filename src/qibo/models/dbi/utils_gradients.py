import math
from copy import deepcopy
from enum import Enum, auto

import numpy as np

from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.dbi.utils import commutator
from qibo.models.dbi.utils_scheduling import polynomial_step


class d_ansatz_type(Enum):

    element_wise = auto()
    local_1 = auto()
    # local_2 = auto() # for future implementation
    # ising = auto() # for future implementation


def d_ansatz(params: np.array, d_type: d_ansatz_type == d_ansatz_type.element_wise):
    r"""
    Creates the $D$ operator for the double-bracket iteration ansatz depending on the type of parameterization.
    If $\alpha_i$ are our parameters and d the number of qubits then:

    element_wise: $D = \sum_{i=0}^{2^d} \alpha_i |i\rangle \langle i|$
    local_1: $D = \sum_{i=1}^{d} \alpha_i Z_i$
    Args:
        params(np.array): parameters for the ansatz.
        d_type(d_ansatz type): type of parameterization for the ansatz.
    """

    if d_type is d_ansatz_type.element_wise:
        d = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            d[i, i] = params[i]

    elif d_type is d_ansatz_type.local_1:

        op_list = [params[i] * symbols.Z(i) for i in len(params)]
        from functools import reduce

        d = SymbolicHamiltonian(reduce(symbols.Z.add, op_list), nqubits=len(params))
        d.dense.matrix
    else:
        raise ValueError(f"Parameterization type {type} not recognized.")

    return d


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


# def dpolynomial_diDiagonal(dbi_object, d, h, i): #element_wise_ansatz
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
            grad[i] = (dbi_object.cost(d_new) - dbi_object.cost(d)) / delta
    return grad


def gradient_descent_dbr_d_ansatz(
    dbi_object,
    params,
    nmb_iterations,
    lr=1e-2,
    analytic=True,
    d_type=d_ansatz_type.element_wise,
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
    Returns:
        d(np.array): Optimized D operator.
        loss(np.array): Loss function evaluated at each iteration.
        grad(np.array): Gradient evaluated at each iteration.
        params_hist(np.array): Parameters evaluated at each iteration.
    """

    h = dbi_object.h.matrix
    d = d_ansatz(params, d_type)
    loss = np.zeros(nmb_iterations + 1)
    grad = np.zeros((nmb_iterations, len(params)))
    dbi_new = deepcopy(dbi_object)
    s = polynomial_step(dbi_object, n=3, d=d)
    dbi_new(s, d=d)
    loss[0] = dbi_new.least_squares(d)
    params_hist = np.empty((len(params), nmb_iterations + 1))
    params_hist[:, 0] = params

    for i in range(nmb_iterations):
        dbi_new = deepcopy(dbi_object)
        grad[i, :] = gradientDiagonalEntries(
            dbi_object, params, h, analytic=analytic, ansatz=d_type
        )
        for j in range(len(params)):
            params[j] = params[j] - lr * grad[i, j]
        d = d_ansatz(params, d_type)
        s = polynomial_step(dbi_new, n=5, d=d)
        dbi_new(s, d=d)
        loss[i + 1] = dbi_new.cost(d)
        params_hist[:, i + 1] = params

    return d, loss, grad, params_hist
