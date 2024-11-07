"""Submodule with entropy measures."""

from typing import List, Tuple, Union

import numpy as np

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.linalg_operations import matrix_power, partial_trace
from qibo.quantum_info.metrics import _check_hermitian, purity


def shannon_entropy(prob_dist, base: float = 2, backend=None):
    """Calculate the Shannon entropy of a discrete random variable.


    For a discrete random variable :math:`\\chi` that has values :math:`x` in the set
    :math:`\\mathcal{X}` with probability distribution :math:`\\operatorname{p}(x)`,
    the base-:math:`b` Shannon entropy is defined as

    .. math::
        \\operatorname{H}_{b}(\\chi) = - \\sum_{x \\in \\mathcal{X}}
            \\, \\operatorname{p}(x) \\, \\log_{b}(\\operatorname{p}(x)) \\, ,

    where :math:`d = \\text{dim}(\\mathcal{H})` is the dimension of the
    Hilbert space :math:`\\mathcal{H}`, :math:`b` is the log base,
    and :math:`0 \\log_{b}(0) \\equiv 0, \\,\\, \\forall \\, b`.

    Args:
        prob_dist (ndarray or list): probability array
            :math:`\\{\\operatorname{p(x)}\\}_{x \\in \\mathcal{X}}`.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Shannon entropy :math:`\\operatorname{H}_{b}`.
    """
    backend = _check_backend(backend)

    if isinstance(prob_dist, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist = backend.cast(prob_dist, dtype=np.float64)

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if len(prob_dist.shape) != 1:
        raise_error(
            TypeError,
            f"Probability array must have dims (k,) but it has {prob_dist.shape}.",
        )

    if len(prob_dist) == 0:
        raise_error(TypeError, "Empty array.")

    if any(prob_dist < 0) or any(prob_dist > 1.0):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    total_sum = backend.np.sum(prob_dist)

    if np.abs(float(total_sum) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Probability array must sum to 1.")

    log_prob = backend.np.where(
        prob_dist != 0, backend.np.log2(prob_dist) / np.log2(base), 0.0
    )

    shan_entropy = -backend.np.sum(prob_dist * log_prob)

    # absolute value if entropy == 0.0 to avoid returning -0.0
    shan_entropy = backend.np.abs(shan_entropy) if shan_entropy == 0.0 else shan_entropy

    return np.real(float(shan_entropy))


def classical_relative_entropy(prob_dist_p, prob_dist_q, base: float = 2, backend=None):
    """Calculate the (classical) relative entropy between two discrete random variables.

    Given two random variables, :math:`\\chi` and :math:`\\upsilon`,
    that admit values :math:`x` in the set :math:`\\mathcal{X}` with respective probabilities
    :math:`\\operatorname{p}(x)` and :math:`\\operatorname{q}(x)`, then their base-:math:`b`
    relative entropy is given by

    .. math::
        \\operatorname{D}_{b}(\\chi \\, \\| \\, \\upsilon) =
            \\sum_{x \\in \\mathcal{X}} \\, \\operatorname{p}(x) \\,
            \\log_{b}\\left( \\frac{\\operatorname{p}(x)}{\\operatorname{q}(x)} \\right) \\, .

    The classical relative entropy is also known as the
    `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.

    Args:
        prob_dist_p (ndarray or list): discrete probability
            :math:`\\{\\operatorname{p}(x)\\}_{x\\in\\mathcal{X}}`.
        prob_dist_q (ndarray or list): discrete probability
            :math:`\\{\\operatorname{q}(x)\\}_{x\\in\\mathcal{X}}`.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical relative entropy :math:`\\operatorname{D}_{b}`.
    """
    backend = _check_backend(backend)
    prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)
    prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    if (len(prob_dist_p.shape) != 1) or (len(prob_dist_q.shape) != 1):
        raise_error(
            TypeError,
            "Probability arrays must have dims (k,) but have "
            + f"dims {prob_dist_p.shape} and {prob_dist_q.shape}.",
        )

    if (len(prob_dist_p) == 0) or (len(prob_dist_q) == 0):
        raise_error(TypeError, "At least one of the arrays is empty.")

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if (any(prob_dist_p < 0) or any(prob_dist_p > 1.0)) or (
        any(prob_dist_q < 0) or any(prob_dist_q > 1.0)
    ):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )
    total_sum_p = backend.np.sum(prob_dist_p)

    total_sum_q = backend.np.sum(prob_dist_q)

    if np.abs(float(total_sum_p) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "First probability array must sum to 1.")

    if np.abs(float(total_sum_q) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Second probability array must sum to 1.")

    entropy_p = -1 * shannon_entropy(prob_dist_p, base=base, backend=backend)

    log_prob_q = backend.np.where(
        prob_dist_q != 0.0, backend.np.log2(prob_dist_q) / np.log2(base), -np.inf
    )

    log_prob = backend.np.where(prob_dist_p != 0.0, log_prob_q, 0.0)

    relative = backend.np.sum(prob_dist_p * log_prob)

    return entropy_p - relative


def classical_mutual_information(
    prob_dist_joint, prob_dist_p, prob_dist_q, base: float = 2, backend=None
):
    """Calculate the (classical) mutual information between two random variables.

    Let :math:`\\chi` and :math:`\\upsilon` be two discrete random variables that
    have values :math:`x \\in \\mathcal{X}` and :math:`y \\in \\mathcal{Y}`, respectively.
    Then, their base-:math:`b` mutual information is given by

    .. math::
        \\operatorname{I}_{b}(\\chi, \\, \\upsilon) = \\operatorname{H}_{b}(\\chi)
            + \\operatorname{H}_{b}(\\upsilon)
            - \\operatorname{H}_{b}(\\chi, \\, \\upsilon) \\, ,

    where :math:`\\operatorname{H}_{b}(\\cdot)` is the :func:`qibo.quantum_info.shannon_entropy`,
    and :math:`\\operatorname{H}_{b}(\\chi, \\, \\upsilon)` represents the joint Shannon entropy
    of the two random variables.

    Args:
        prob_dist_joint (ndarray): joint discrete probability
            :math:`\\{\\operatorname{p}(x, \\, y)\\}_{x\\in\\mathcal{X},y\\in\\mathcal{Y}}`.
        prob_dist_p (ndarray): marginal discrete probability
            :math:`\\{\\operatorname{p}(x)\\}_{x\\in\\mathcal{X}}`.
        prob_dist_q (ndarray): marginal discrete probability
            :math:`\\{\\operatorname{q}(y)\\}_{y\\in\\mathcal{Y}}`.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Mutual information :math:`\\operatorname{I}_{b}`.
    """
    return (
        shannon_entropy(prob_dist_p, base, backend)
        + shannon_entropy(prob_dist_q, base, backend)
        - shannon_entropy(prob_dist_joint, base, backend)
    )


def classical_renyi_entropy(
    prob_dist, alpha: Union[float, int], base: float = 2, backend=None
):
    """Calculate the (classical) Rényi entropy of a discrete random variable.

    Let :math:`\\chi` be a discrete random variable that has values :math:`x`
    in the set :math:`\\mathcal{X}` with probability :math:`\\operatorname{p}(x)`.
    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)`,
    the (classical) base-:math:`b` Rényi entropy of :math:`\\chi` is defined as

    .. math::
        \\operatorname{H}_{\\alpha}^{\\text{re}}(\\chi) = \\frac{1}{1 - \\alpha} \\,
            \\log_{b}\\left( \\sum_{x} \\, \\operatorname{p}^{\\alpha}(x) \\right) \\, ,

    where :math:`\\|\\cdot\\|_{\\alpha}` is the vector :math:`\\alpha`-norm.

    A special case is the limit :math:`\\alpha \\to 1`, in which the classical Rényi entropy
    coincides with the :func:`qibo.quantum_info.shannon_entropy`.

    Another special case is the limit :math:`\\alpha \\to 0`, where the function is
    reduced to :math:`\\log_{b}\\left(|\\operatorname{p}|\\right)`, with :math:`|\\operatorname{p}|`
    being the support of :math:`\\operatorname{p}`.
    This is known as the `Hartley entropy <https://en.wikipedia.org/wiki/Hartley_function>`_
    (also known as *Hartley function* or *max-entropy*).

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-\\log_{b}(\\max_{x}(\\operatorname{p}(x)))`, which is called the
    `min-entropy <https://en.wikipedia.org/wiki/Min-entropy>`_.

    Args:
        prob_dist (ndarray): discrete probability
            :math:`\\{\\operatorname{p}(x)\\}_{x\\in\\mathcal{X}}`.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical Rényi entropy :math:`\\operatorname{H}_{\\alpha}^{\\text{re}}`.
    """
    backend = _check_backend(backend)
    prob_dist = backend.cast(prob_dist, dtype=np.float64)

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if len(prob_dist.shape) != 1:
        raise_error(
            TypeError,
            f"Probability array must have dims (k,) but it has {prob_dist.shape}.",
        )

    if len(prob_dist) == 0:
        raise_error(TypeError, "Empty array.")

    if any(prob_dist < 0) or any(prob_dist > 1.0):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    total_sum = backend.np.sum(prob_dist)

    if np.abs(float(total_sum) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Probability array must sum to 1.")

    if alpha == 0.0:
        return np.log2(len(prob_dist)) / np.log2(base)

    if alpha == 1.0:
        return shannon_entropy(prob_dist, base=base, backend=backend)

    if alpha == np.inf:
        return -1 * backend.np.log2(max(prob_dist)) / np.log2(base)

    total_sum = backend.np.sum(prob_dist**alpha)

    renyi_ent = (1 / (1 - alpha)) * backend.np.log2(total_sum) / np.log2(base)

    return renyi_ent


def classical_relative_renyi_entropy(
    prob_dist_p, prob_dist_q, alpha: Union[float, int], base: float = 2, backend=None
):
    """Calculate the (classical) relative Rényi entropy between two discrete random variables.

    This function is also known as
    `Rényi divergence <https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R%C3%A9nyi_divergence>`_.

    Let :math:`\\chi` and :math:`\\upsilon` be two discrete random variables
    that admit values :math:`x` in the set :math:`\\mathcal{X}` with respective probabilities
    :math:`\\operatorname{p}(x)` and :math:`\\operatorname{q}(x)`.
    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)`, the (classical) relative
    Rényi entropy is defined as

    .. math::
        \\operatorname{D}_{\\alpha,b}^{\\text{re}}(\\chi \\, \\| \\, \\upsilon) =
            \\frac{1}{\\alpha - 1} \\, \\log_{b}\\left( \\sum_{x} \\,
            \\frac{\\operatorname{p}^{\\alpha}(x)}{\\operatorname{q}^{\\alpha - 1}(x)} \\right)
            \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the classical Rényi divergence
    coincides with the :func:`qibo.quantum_info.classical_relative_entropy`.

    Another special case is the limit :math:`\\alpha \\to 1/2`, where the function is
    reduced to :math:`-2 \\log_{b}\\left(\\sum_{x\\in\\mathcal{X}} \\,
    \\sqrt{\\operatorname{p}(x) \\, \\operatorname{q}(x)} \\right)`.
    The sum inside the :math:`\\log_{b}` is known as the
    `Bhattacharyya coefficient <https://en.wikipedia.org/wiki/Bhattacharyya_distance>`_.

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`\\log_{b}\\,\\max_{x\\in\\mathcal{X}}\\,(\\operatorname{p}(x) \\, \\operatorname{q}(x))`.

    Args:
        prob_dist_p (ndarray or list): discrete probability
            :math:`\\{\\operatorname{p}(x)\\}_{x\\in\\mathcal{X}}`.
        prob_dist_q (ndarray or list): discrete probability
            :math:`\\{\\operatorname{q}(x)\\}_{x\\in\\mathcal{X}}`.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical relative Rényi entropy :math:`D_{\\alpha,b}^{\\text{re}}`.
    """
    backend = _check_backend(backend)
    prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)
    prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    if (len(prob_dist_p.shape) != 1) or (len(prob_dist_q.shape) != 1):
        raise_error(
            TypeError,
            "Probability arrays must have dims (k,) but have "
            + f"dims {prob_dist_p.shape} and {prob_dist_q.shape}.",
        )

    if (len(prob_dist_p) == 0) or (len(prob_dist_q) == 0):
        raise_error(TypeError, "At least one of the arrays is empty.")

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if (any(prob_dist_p < 0) or any(prob_dist_p > 1.0)) or (
        any(prob_dist_q < 0) or any(prob_dist_q > 1.0)
    ):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    total_sum_p = backend.np.sum(prob_dist_p)
    total_sum_q = backend.np.sum(prob_dist_q)

    if np.abs(float(total_sum_p) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "First probability array must sum to 1.")

    if np.abs(float(total_sum_q) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Second probability array must sum to 1.")

    if alpha == 0.5:
        total_sum = backend.np.sqrt(prob_dist_p * prob_dist_q)
        total_sum = backend.np.sum(total_sum)

        return -2 * backend.np.log2(total_sum) / np.log2(base)

    if alpha == 1.0:
        return classical_relative_entropy(
            prob_dist_p, prob_dist_q, base=base, backend=backend
        )

    if alpha == np.inf:
        return backend.np.log2(max(prob_dist_p / prob_dist_q)) / np.log2(base)

    prob_p = prob_dist_p**alpha
    prob_q = prob_dist_q ** (1 - alpha)

    total_sum = backend.np.sum(prob_p * prob_q)

    return (1 / (alpha - 1)) * backend.np.log2(total_sum) / np.log2(base)


def classical_tsallis_entropy(prob_dist, alpha: float, base: float = 2, backend=None):
    """Calculate the (classical) Tsallis entropy for a discrete random variable.

    This is defined as

    .. math::
        \\begin{align}
        \\operatorname{H}_{\\alpha}^{\\text{ts}}(\\chi) &= -\\sum_{x\\in\\mathcal{X}} \\,
            \\operatorname{p}^{\\alpha}(x) \\, \\ln_{\\alpha}\\left(\\operatorname{p}(x)\\right)
            \\\\
        &= \\frac{1}{\\alpha - 1} \\,\\left(1 - \\sum_{x} \\, \\operatorname{p}^{\\alpha}(x)
            \\right) \\, ,
        \\end{align}

    where :math:`\\ln_{\\alpha}(x) \\equiv (x^{1-\\alpha} - 1) / (1 - \\alpha)`
    is the so-called :math:`\\alpha`-logarithm.

    Args:
        prob_dist (ndarray): discrete probability
            :math:`\\{\\operatorname{p}(x)\\}_{x\\in\\mathcal{X}}`.
        alpha (float or int): entropic index.
        base (float): the base of the :math:`\\log`. Used when :math:`\\alpha = 1`.
            Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical Tsallis entropy :math:`\\operatorname{H}_{\\alpha}^{\\text{ts}}`.
    """
    backend = _check_backend(backend)

    if isinstance(prob_dist, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist = backend.cast(prob_dist, dtype=np.float64)

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if len(prob_dist.shape) != 1:
        raise_error(
            TypeError,
            f"Probability array must have dims (k,) but it has {prob_dist.shape}.",
        )

    if len(prob_dist) == 0:
        raise_error(TypeError, "Empty array.")

    if any(prob_dist < 0) or any(prob_dist > 1.0):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    total_sum = backend.np.sum(prob_dist)

    if np.abs(float(total_sum) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Probability array must sum to 1.")

    if alpha == 1.0:
        return shannon_entropy(prob_dist, base=base, backend=backend)

    total_sum = prob_dist**alpha
    total_sum = backend.np.sum(total_sum)

    return (1 / (alpha - 1)) * (1 - total_sum)


def classical_relative_tsallis_entropy(
    prob_dist_p, prob_dist_q, alpha: float, base: float = 2, backend=None
):
    """Calculate the (classical) relative Tsallis entropy between two discrete random variables.

    Given a discrete random variable :math:`\\chi` that has values :math:`x` in the set
    :math:`\\mathcal{X}` with probability :math:`\\mathrm{p}(x)` and a discrete random variable
    :math:`\\upsilon` that has the values :math:`x` in the same set :math:`\\mathcal{X}` with
    probability :math:`\\mathrm{q}(x)`, their relative Tsallis entropy is given by

    .. math::
        D_{\\alpha}^{\\text{ts}}(\\chi \\, \\| \\, \\upsilon) = \\sum_{x \\in \\mathcal{X}} \\,
            \\mathrm{p}^{\\alpha}(x) \\, \\ln_{\\alpha}
            \\left( \\frac{\\mathrm{p}(x)}{\\mathrm{q}(x)} \\right) \\, ,

    where :math:`\\ln_{\\alpha}(x) \\equiv (x^{1 - \\alpha} - 1) / (1 - \\alpha)`
    is the so-called :math:`\\alpha`-logarithm.

    When :math:`\\alpha = 1`, this funciton reduces to reduces to
    :class:`qibo.quantum_info.classical_relative_entropy`.

    Args:
        prob_dist_p (ndarray or list): discrete probability
            :math:`\\{\\operatorname{p}(x)\\}`.
        prob_dist_q (ndarray or list): discrete probability
            :math:`\\{\\operatorname{q}(x)\\}`.
        alpha (float): entropic index.
        base (float): the base of the :math:`\\log`. Used when :math:`\\alpha = 1`.
            Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical Tsallis relative entropy :math:`D_{\\alpha}^{\\text{ts}}`.
    """
    if alpha == 1.0:
        return classical_relative_entropy(prob_dist_p, prob_dist_q, base, backend)

    backend = _check_backend(backend)

    if isinstance(prob_dist_p, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)

    if isinstance(prob_dist_q, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    element_wise = prob_dist_p**alpha
    element_wise = element_wise * _q_logarithm(prob_dist_p / prob_dist_q, alpha)

    return backend.np.sum(element_wise)


def von_neumann_entropy(
    state,
    base: float = 2,
    check_hermitian: bool = False,
    return_spectrum: bool = False,
    backend=None,
):
    """Calculate the von Neumann entropy of a quantum state.

    Given a quantum ``state`` :math:`\\rho`, the base-:math:`b` von Neumann entropy is

    .. math::
        S_{b}(\\rho) = -\\text{Tr}\\left(\\rho \\, \\log_{b}(\\rho)\\right) \\, .

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        base (float, optional): the base of the :math:`\\log`. Defaults to :math:`2`.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian .
            Defaults to ``False``.
        return_spectrum: if ``True``, returns :math:`S_{b}(\\rho)` and
            :math:`-\\log_{b}(\\text{eigenvalues}(\\rho))`.
            If ``False``, returns only :math:`S_{b}(\\rho)`.
            Default is ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: von Neumann entropy :math:`\\operatorname{S}_{b}`.
    """
    backend = _check_backend(backend)

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if not isinstance(check_hermitian, bool):
        raise_error(
            TypeError,
            f"check_hermitian must be type bool, but it is type {type(check_hermitian)}.",
        )

    if purity(state, backend=backend) == 1.0:
        if return_spectrum:
            return 0.0, backend.cast([0.0], dtype=float)

        return 0.0

    eigenvalues = backend.calculate_eigenvalues(
        state,
        hermitian=(not check_hermitian or _check_hermitian(state, backend=backend)),
    )

    log_prob = backend.np.where(
        backend.np.real(eigenvalues) > 0.0,
        backend.np.log2(eigenvalues) / np.log2(base),
        0.0,
    )

    ent = -backend.np.sum(eigenvalues * log_prob)
    # absolute value if entropy == 0.0 to avoid returning -0.0
    ent = backend.np.abs(ent) if ent == 0.0 else backend.np.real(ent)

    if return_spectrum:
        log_prob = backend.cast(log_prob, dtype=log_prob.dtype)
        return ent, -log_prob

    return ent


def relative_von_neumann_entropy(
    state,
    target,
    base: float = 2,
    check_hermitian: bool = False,
    precision_tol: float = 1e-14,
    backend=None,
):
    """Calculate the relative von Neumann entropy  between two quantum states.

    Given two quantum states :math:`\\rho` and :math:`\\sigma`, it is defined as

    .. math::
        \\Delta_{b}(\\rho \\, \\| \\, \\sigma) = \\text{Tr}\\left(\\rho \\,
            \\log_{b}(\\rho)\\right) - \\text{Tr}\\left(\\rho \\,
            \\log_{b}(\\sigma)\\right)

    It is also known as the *quantum relative entropy*.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        base (float, optional): the base of the :math:`\\log`. Defaults to :math:`2`.
        check_hermitian (bool, optional): If ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian .
            Defaults to ``False``.
        precision_tol (float, optional): Used when entropy is calculated via engenvalue
            decomposition. Eigenvalues that are smaller than ``precision_tol`` in absolute value
            are set to :math:`0`. Defaults to :math:`10^{-14}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Relative von Neumann entropy :math:`\\Delta_{b}`.
    """
    backend = _check_backend(backend)
    state = backend.cast(state)
    target = backend.cast(target)

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if (
        (len(target.shape) >= 3)
        or (len(target) == 0)
        or (len(target.shape) == 2 and target.shape[0] != target.shape[1])
    ):
        raise_error(
            TypeError,
            f"target must have dims either (k,) or (k,k), but have dims {target.shape}.",
        )

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if not isinstance(check_hermitian, bool):
        raise_error(
            TypeError,
            f"check_hermitian must be type bool, but it is type {type(check_hermitian)}.",
        )

    if purity(state, backend=backend) == 1.0 and purity(target, backend=backend) == 1.0:
        return 0.0

    if len(state.shape) == 1:
        state = backend.np.outer(state, backend.np.conj(state))

    if len(target.shape) == 1:
        target = backend.np.outer(target, backend.np.conj(target))

    eigenvalues_state, eigenvectors_state = backend.calculate_eigenvectors(
        state,
        hermitian=(not check_hermitian or _check_hermitian(state, backend=backend)),
    )
    eigenvalues_target, eigenvectors_target = backend.calculate_eigenvectors(
        target,
        hermitian=(not check_hermitian or _check_hermitian(target, backend=backend)),
    )

    overlaps = backend.np.conj(eigenvectors_state.T) @ eigenvectors_target
    overlaps = backend.np.abs(overlaps) ** 2

    log_state = backend.np.where(
        backend.np.real(eigenvalues_state) > precision_tol,
        backend.np.log2(eigenvalues_state) / np.log2(base),
        0.0,
    )
    log_target = backend.np.where(
        backend.np.real(eigenvalues_target) > precision_tol,
        backend.np.log2(eigenvalues_target) / np.log2(base),
        0.0,
    )

    log_target = overlaps @ log_target

    log_target = backend.np.where(eigenvalues_state != 0.0, log_target, 0.0)

    entropy_state = backend.np.sum(eigenvalues_state * log_state)

    relative = backend.np.sum(eigenvalues_state * log_target)

    return float(backend.np.real(entropy_state - relative))


def mutual_information(
    state,
    partition: Union[List[int], Tuple[int, ...]],
    base: float = 2,
    check_hermitian: bool = False,
    backend=None,
):
    """Calculate the mutual information over two partitions of a quantum state.

    Given a bipartite quantum state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    its base-:math:`b` mutual information over those two partitions is given by

    .. math::
        \\mathcal{I}_{b}(\\rho) = \\operatorname{S}_{b}(\\rho_{A})
            + \\operatorname{S}_{b}(\\rho_{B}) - \\operatorname{S}_{b}(\\rho) \\, ,

    where :math:`\\rho_{A} = \\text{Tr}_{B}(\\rho)` is the reduced density matrix of qubits
    in partition :math:`A`, :math:`\\rho_{B} = \\text{Tr}_{A}(\\rho)` is the reduced density
    matrix of qubits in partition :math:`B`, and :math:`\\operatorname{S}_{b}(\\cdot)`
    is the base-:math:`b` :func:`qibo.quantum_info.von_neumann_entropy`.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        partition (list or tuple): indices of qubits in partition :math:`A`.
            Partition :math:`B` is assumed to contain the remaining qubits.
        base (float, optional): the base of the :math:`\\log`. Defaults to :math:`2`.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian . Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Mutual information :math:`\\mathcal{I}_{b}`.
    """
    nqubits = np.log2(len(state))

    if not nqubits.is_integer():
        raise_error(ValueError, f"dimensions of ``state`` must be a power of 2.")

    partition_b = set(list(range(int(nqubits)))) ^ set(list(partition))

    state_a = partial_trace(state, partition_b, backend)
    state_b = partial_trace(state, partition, backend)

    return (
        von_neumann_entropy(state_a, base, check_hermitian, False, backend)
        + von_neumann_entropy(state_b, base, check_hermitian, False, backend)
        - von_neumann_entropy(state, base, check_hermitian, False, backend)
    )


def renyi_entropy(state, alpha: Union[float, int], base: float = 2, backend=None):
    """Calculate the Rényi entropy of a quantum state.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)`, the Rényi entropy
    :math:`\\operatorname{H}_{\\alpha,b}`
    is defined as

    .. math::
        \\operatorname{S}_{\\alpha,b}^{\\text{re}}(\\rho) = \\frac{1}{1 - \\alpha} \\,
            \\log_{b}\\left(\\rho^{\\alpha} \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Rényi entropy
    coincides with the :func:`qibo.quantum_info.von_neumann_entropy`.

    Another special case is the limit :math:`\\alpha \\to 0`, where the function is
    reduced to :math:`\\log_{b}\\left(d\\right)`, with :math:`d = 2^{n}`
    being the dimension of the Hilbert space in which ``state`` :math:`\\rho` lives in.
    This is known as the `Hartley entropy <https://en.wikipedia.org/wiki/Hartley_function>`_
    (also known as *Hartley function* or *max-entropy*).

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-\\log_{b}(\\|\\rho\\|_{\\infty})`, with :math:`\\|\\cdot\\|_{\\infty}`
    being the `spectral norm
    <https://en.wikipedia.org/wiki/Matrix_norm#Matrix_norms_induced_by_vector_p-norms>`_.
    This is known as the `min-entropy <https://en.wikipedia.org/wiki/Min-entropy>`_.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Rényi entropy :math:`\\operatorname{S}_{\\alpha,b}^{\\text{re}}`.
    """
    backend = _check_backend(backend)

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if abs(purity(state, backend=backend) - 1.0) < PRECISION_TOL:
        return 0.0

    if alpha == 0.0:
        return np.log2(len(state)) / np.log2(base)

    if alpha == 1.0:
        return von_neumann_entropy(state, base=base, backend=backend)

    if alpha == np.inf:
        return (
            -1
            * backend.np.log2(backend.calculate_norm_density_matrix(state, order=2))
            / np.log2(base)
        )

    log = backend.np.log2(backend.np.trace(matrix_power(state, alpha, backend=backend)))

    return (1 / (1 - alpha)) * log / np.log2(base)


def relative_renyi_entropy(
    state, target, alpha: Union[float, int], base: float = 2, backend=None
):
    """Calculate the relative Rényi entropy between two quantum states.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)` and two quantum states
    :math:`\\rho` and :math:`\\sigma`, the base-:math:`b` relative Rényi entropy
    is defined as

    .. math::
        \\Delta_{\\alpha,b}^{\\text{re}}(\\rho \\, \\| \\, \\sigma) = \\frac{1}{\\alpha - 1}
            \\, \\log_{b}\\left( \\text{Tr}\\left( \\rho^{\\alpha} \\,
            \\sigma^{1 - \\alpha} \\right) \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Rényi entropy
    coincides with the :func:`qibo.quantum_info.relative_von_neumann_entropy`.

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-2 \\, \\log_{b}(\\|\\sqrt{\\rho} \\, \\sqrt{\\sigma}\\|_{1})`,
    with :math:`\\|\\cdot\\|_{1}` being the
    `Schatten 1-norm <https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms>`_.
    This is known as the `min-relative entropy <https://arxiv.org/abs/1310.7178>`_.

    .. note::
        Function raises ``NotImplementedError`` when ``target`` :math:`\\sigma`
        is a pure state and :math:`\\alpha > 1`. This is due to the fact that
        it is not possible to calculate :math:`\\sigma^{1 - \\alpha}` when
        :math:`\\alpha > 1` and :math:`\\sigma` is a projector, i.e. a singular matrix.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the :math:`\\log`. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Relative Rényi entropy :math:`\\Delta_{\\alpha,,b}^{\\text{re}}`.
    """
    backend = _check_backend(backend)
    state = backend.cast(state)
    target = backend.cast(target)
    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if (
        (len(target.shape) >= 3)
        or (len(target) == 0)
        or (len(target.shape) == 2 and target.shape[0] != target.shape[1])
    ):
        raise_error(
            TypeError,
            f"target must have dims either (k,) or (k,k), but have dims {target.shape}.",
        )

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    purity_target = purity(target, backend=backend)
    if (
        abs(purity(state, backend=backend) - 1.0) < PRECISION_TOL
        and abs(purity_target - 1) < PRECISION_TOL
    ):
        return 0.0

    if alpha > 1.0 and abs(purity_target - 1) < PRECISION_TOL:
        raise_error(
            NotImplementedError,
            "It is not possible to invert a singular matrix. ``target`` is a pure state and alpha > 1.",
        )

    if len(state.shape) == 1:
        state = backend.np.outer(state, backend.np.conj(state))

    if alpha == 1.0:
        return relative_von_neumann_entropy(state, target, base, backend=backend)

    if alpha == np.inf:
        new_state = matrix_power(state, 0.5, backend=backend)
        new_target = matrix_power(target, 0.5, backend=backend)

        log = backend.np.log2(
            backend.calculate_norm_density_matrix(new_state @ new_target, order=1)
        )

        return -2 * log / np.log2(base)

    log = matrix_power(state, alpha, backend=backend)
    log = log @ matrix_power(target, 1 - alpha, backend=backend)
    log = backend.np.log2(backend.np.trace(log))

    return (1 / (alpha - 1)) * log / np.log2(base)


def tsallis_entropy(state, alpha: float, base: float = 2, backend=None):
    """Calculate the Tsallis entropy of a quantum state.

    .. math::
        \\operatorname{S}_{\\alpha}^{\\text{ts}}(\\rho) =
            \\frac{\\text{Tr}(\\rho^{\\alpha}) - 1}{1 - \\alpha}

    When :math:`\\alpha = 1`, the functions defaults to
    :func:`qibo.quantum_info.von_neumann_entropy`.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        alpha (float or int): entropic index.
        base (float, optional): the base of the :math:`\\log`. Used when :math:`\\alpha = 1`.
            Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Tsallis entropy :math:`\\operatorname{S}_{\\alpha}^{\\text{ts}}`.
    """
    backend = _check_backend(backend)

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError, f"alpha must be type float, but it is type {type(alpha)}."
        )

    if alpha < 0.0:
        raise_error(ValueError, "alpha must a non-negative float.")

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if abs(purity(state, backend=backend) - 1.0) < PRECISION_TOL:
        return 0.0

    if alpha == 1.0:
        return von_neumann_entropy(state, base=base, backend=backend)

    return (1 / (1 - alpha)) * (
        backend.np.trace(matrix_power(state, alpha, backend=backend)) - 1
    )


def relative_tsallis_entropy(
    state,
    target,
    alpha: Union[float, int],
    base: float = 2,
    check_hermitian: bool = False,
    backend=None,
):
    """Calculate the relative Tsallis entropy between two quantum states.

    For :math:`\\alpha \\in [0, \\, 2]` and quantum states :math:`\\rho` and
    :math:`\\sigma`, the relative Tsallis entropy is defined as

    .. math::
        \\Delta_{\\alpha}^{\\text{ts}}(\\rho, \\, \\sigma) = \\frac{1 -
            \\text{Tr}\\left(\\rho^{\\alpha} \\, \\sigma^{1 - \\alpha}\\right)}{1 - \\alpha}
            \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Tsallis entropy
    coincides with the :func:`qibo.quantum_info.relative_von_neumann_entropy`.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        alpha (float or int): entropic index :math:`\\alpha \\in [0, \\, 2]`.
        base (float, optional): the base of the :math:`\\log`. Used when :math:`\\alpha = 1`.
            Defaults to :math:`2`.
        check_hermitian (bool, optional): Used when :math:`\\alpha = 1`.
            If ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian .
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: Relative Tsallis entropy :math:`\\Delta_{\\alpha}^{\\text{ts}}`.

    References:
        1. S. Abe, *Nonadditive generalization of the quantum Kullback-Leibler
        divergence for measuring the degree of purification*,
        `Phys. Rev. A 68, 032302 <https://doi.org/10.1103/PhysRevA.68.032302>`_.

        2. S. Furuichi, K. Yanagi, and K. Kuriyama,
        *Fundamental properties of Tsallis relative entropy*,
        `J. Math. Phys., Vol. 45, Issue 12, pp. 4868-4877 (2004)
        <https://doi.org/10.1063/1.1805729>`_ .
    """
    if alpha == 1.0:
        return relative_von_neumann_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )

    if not isinstance(alpha, (float, int)):
        raise_error(
            TypeError,
            f"``alpha`` must be type float or int, but it is type {type(alpha)}.",
        )

    if alpha < 0.0 or alpha > 2.0:
        raise_error(
            ValueError, f"``alpha`` must be in the interval [0, 2], but it is {alpha}."
        )

    if alpha < 1.0:
        alpha = 2 - alpha

    factor = 1 - alpha

    if len(state.shape) == 1:
        state = backend.np.outer(state, backend.np.conj(state.T))

    if len(target.shape) == 1:
        target = backend.np.outer(target, backend.np.conj(target.T))

    trace = matrix_power(state, alpha, backend=backend)
    trace = trace @ matrix_power(target, factor, backend=backend)
    trace = backend.np.trace(trace)

    return (1 - trace) / factor


def entanglement_entropy(
    state,
    partition: Union[List[int], Tuple[int, ...]],
    base: float = 2,
    check_hermitian: bool = False,
    return_spectrum: bool = False,
    backend=None,
):
    """Calculate the entanglement entropy of a bipartite quantum state.


    Given a bipartite quantum state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    its base-:math:`b` entanglement entropy is given by

    .. math::
        \\operatorname{S}_{b}^{\\text{ent}}(\\rho) \\equiv \\operatorname{S}_{b}(\\rho_{A}) =
            -\\text{Tr}\\left(\\rho_{A} \\, \\log_{b}(\\rho_{A})\\right) \\, ,

    where :math:`\\rho_{A} = \\text{Tr}_{B}(\\rho)` is the reduced density matrix calculated
    by tracing out the ``partition`` :math:`B`.

    Args:
        state (ndarray): statevector or density matrix.
        partition (list or tuple): qubits in the partition :math:`B` to be traced out.
        base (float, optional): the base of the :math:`\\log`. Defaults to :math:`2`.
        check_hermitian (bool, optional): if ``True``, checks if :math:`\\rho_{A}` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian . Default: ``False``.
        return_spectrum: if ``True``, returns ``entropy`` and eigenvalues of ``state``.
            If ``False``, returns only ``entropy``. Default is ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Entanglement entropy :math:`\\operatorname{S}_{b}^{\\text{ent}}`.
    """
    backend = _check_backend(backend)

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if (
        (len(state.shape) not in [1, 2])
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    reduced_density_matrix = partial_trace(state, partition, backend=backend)

    entropy_entanglement = von_neumann_entropy(
        reduced_density_matrix,
        base=base,
        check_hermitian=check_hermitian,
        return_spectrum=return_spectrum,
        backend=backend,
    )

    return entropy_entanglement


def _q_logarithm(x, q: float):
    """Generalization of logarithm function necessary for classical (relative) Tsallis entropy."""
    factor = 1 - q
    return (x**factor - 1) / factor
