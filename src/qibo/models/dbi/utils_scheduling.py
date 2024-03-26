import math
from functools import partial
from typing import Optional
from copy import deepcopy
import hyperopt
import numpy as np


error = 1e-3

def commutator(A, B):
    """Compute commutator between two arrays."""
    return A@B-B@A

def variance(A, state):
    """Calculates the variance of a matrix A with respect to a state: Var($A$) = $\langle\mu|A^2|\mu\rangle-\langle\mu|A|\mu\rangle^2$"""
    B = A@A
    return B[state,state]-A[state,state]**2

def covariance(A, B, state):
    """Calculates the covariance of two matrices A and B with respect to a state: Cov($A,B$) = $\langle\mu|AB|\mu\rangle-\langle\mu|A|\mu\rangle\langle\mu|B|\mu\rangle$"""
    C = A@B+B@A
    return C[state,state]-2*A[state,state]*B[state,state]

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
        (float): optimized best iteration step (minimizing loss function).
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
    cost: str = None,
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
        cost = dbi_object.cost.name
        
    if d is None:
        d = dbi_object.diagonal_h_matrix

    if n > n_max:
        raise ValueError(
            "No solution can be found with polynomial approximation. Increase `n_max` or use other scheduling methods."
        )
    if coef is None:
        if cost == "off_diagonal_norm":
            coef = off_diagonal_norm_polynomial_expansion_coef(dbi_object, d, n)
        elif cost == "least_squares":
            coef = least_squares_polynomial_expansion_coef(dbi_object, d, n)
        elif cost == "energy_fluctuation":
            coef = energy_fluctuation_polynomial_expansion_coef(dbi_object, d, n, dbi_object.state)
        else:
            raise ValueError(f"Cost function {cost} not recognized.")
    
    roots = np.roots(coef)
    real_positive_roots = [
        np.real(root) for root in roots if np.imag(root) < error and np.real(root) > 0
    ]
    # solution exists, return minimum s
    if len(real_positive_roots) > 0:
        sol = min(real_positive_roots)
        for s in real_positive_roots:
            if dbi_object.loss(s, d) < dbi_object.loss(sol, d):
                sol = s
        return sol
    # solution does not exist, return None
    else:
        return None

def d_ansatz(params, type = 'Full'):
    r"""
    Creates the $D$ operator for the double-bracket iteration ansatz depending on the type of parameterization.
    Args:
        params(np.array): parameters for the ansatz.
        type(str): type of parameterization, 'Full' or 'Pauli'
        (Full being each entry parametrized and Pauli being a linear combination of Z_i matrix).
    """

    if type == 'Full':
        d = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            d[i, i] = params[i]

    if type == 'Pauli':
        d = np.zeros((2**len(params), 2**len(params)))
        Z = np.array([[1, 0], [0, -1]])
        for i in range(len(params)):
            I1 = np.eye(2**i)
            I2 = np.eye(2**(len(params)-i-1))
            d += params[i]*np.kron(I1,np.kron(Z,I2))
    
    return d


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

def least_squares_polynomial_expansion_coef(dbi_object, d, n):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n+1, d)
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    # coefficients
    coef = np.empty(n)
    for i in range(n):
        coef[i] = np.real(exp_list[i]*np.trace(d@Gamma_list[i+1]))
    coef = list(reversed(coef))
    return coef

#TODO: add a general expansion formula not stopping at 3rd order
def energy_fluctuation_polynomial_expansion_coef(dbi_object, d, n, state):
    if d is None:
        d = dbi_object.diagonal_h_matrix
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_Gamma_list(n+1, d)
    # coefficients
    coef = np.empty(3)
    coef[0] = np.real(2*covariance(Gamma_list[0], Gamma_list[1],state))
    coef[1] = np.real(2*variance(Gamma_list[1],state)+2*covariance(Gamma_list[0],Gamma_list[2],state))
    coef[2] = np.real(covariance(Gamma_list[0], Gamma_list[3],state)+3*covariance(Gamma_list[1], Gamma_list[2],state))
    coef = list(reversed(coef))
    return coef


# D GRADIENTS
def dGamma_diDiagonal(dbi_object, d, H, n, i,dGamma, Gamma_list):
    # Derivative of gamma with respect to diagonal elements of D (full-diagonal ansatz)
    A = np.zeros(d.shape)
    A[i,i] = 1
    B = commutator(commutator(A,H),Gamma_list[n-1])
    W = commutator(d,H)
    return B + commutator(W,dGamma[-1])

def dpolynomial_diDiagonal(dbi_object, d,H,i):
    # Derivative of polynomial approximation of potential function with respect to diagonal elements of d (full-diagonal ansatz)
    # Formula can be expanded easily to any order, with n=3 corresponding to cubic approximation
    derivative = 0
    s = polynomial_step(dbi_object, n=3, d=d)
    A = np.zeros(d.shape)
    Gamma_list = dbi_object.generate_Gamma_list(4, d)
    A[i,i] = 1
    dGamma = [commutator(A,H)]
    derivative += np.real(np.trace(Gamma_list[0]@A)+np.trace(dGamma[0]@d+Gamma_list[1]@A)*s)
    for n in range(2,4):
        dGamma.append(dGamma_diDiagonal(dbi_object,d,H,n,i,dGamma,Gamma_list))
        derivative += np.real(np.trace(dGamma[-1]@d + Gamma_list[n]@A)*s**n/math.factorial(n))

    return derivative

def gradientDiagonal(dbi_object, params, H, analytic = True, ansatz = 'Full', h = 1e-4):
    # Gradient of potential function with respect to diagonal elements of D (full-diagonal ansatz)
    grad = np.zeros(len(params))
    d = d_ansatz(params, ansatz)
    if analytic == True:
        for i in range(len(params)):
            derivative = dpolynomial_diDiagonal(dbi_object,d,H,i)
            grad[i] = d[i,i]-derivative
    else:
        for i in range(len(params)):
            params_new = deepcopy(params)
            params_new[i] += h
            d_new = d_ansatz(params_new, ansatz)
            grad[i] = (dbi_object.least_squares(d_new)-dbi_object.least_squares(d))/h
    return grad

def gradient_ascent(dbi_object, params, step, iterations, analytic = True, ansatz = 'Full'):
    H = dbi_object.h.matrix
    d = d_ansatz(params,ansatz)
    loss = np.zeros(iterations+1)
    grad = np.zeros((iterations,len(params)))
    dbi_new = deepcopy(dbi_object)
    s = polynomial_step(dbi_object, n = 3, d=d)
    dbi_new(s,d=d)
    loss[0] = dbi_new.least_squares(d)
    params_hist = np.empty((len(params),iterations+1))
    params_hist[:,0] = params

    for i in range(iterations):
        dbi_new = deepcopy(dbi_object)
        grad[i,:] = gradientDiagonal(dbi_object, params, H, analytic=analytic, ansatz=ansatz)
        for j in range(len(params)):
            params[j] = params[j] - step*grad[i,j]
        d = d_ansatz(params,ansatz)     
        s = polynomial_step(dbi_object, n = 3, d=d)
        dbi_new(s,d=d)
        loss[i+1] = dbi_new.least_squares(d)
        params_hist[:,i+1] = params
        
    return d,loss,grad,params_hist