from qibo.models.dbi.utils import *


def dGamma_di_Pauli(dbi_object, n: int, Z_i: np.array, d: np.array):
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
    W = dbi_object.commutator(d, dbi_object.h.matrix)
    dW_di = dbi_object.commutator(Z_i, dbi_object.h.matrix)
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


def ds_di_Pauli(
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
    dGamma_di = dGamma_di_Pauli(dbi_object, n=4, Z_i=Z_i, d=d)
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
        da, db, dc, ds = ds_di_Pauli(
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


def dGamma_diDiagonal(d, H, n, i, dGamma, Gamma_list):
    # Derivative of gamma with respect to diagonal elements of D (full-diagonal ansatz)
    A = np.zeros(d.shape)
    A[i, i] = 1
    B = commutator(commutator(A, H), Gamma_list[n - 1])
    W = commutator(d, H)
    return B + commutator(W, dGamma[-1])


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
