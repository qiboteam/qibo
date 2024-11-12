"""Submodule with entropy measures."""

from typing import Union

import numpy as np

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.linalg_operations import matrix_power, partial_trace
from qibo.quantum_info.metrics import _check_hermitian, purity


def shannon_entropy(prob_dist, base: float = 2, backend=None):
    """Calculate the Shannon entropy of a probability array :math:`\\mathbf{p}`, which is given by

    .. math::
        H(\\mathbf{p}) = - \\sum_{k = 0}^{d^{2} - 1} \\, p_{k} \\, \\log_{b}(p_{k}) \\, ,

    where :math:`d = \\text{dim}(\\mathcal{H})` is the dimension of the
    Hilbert space :math:`\\mathcal{H}`, :math:`b` is the log base (default 2),
    and :math:`0 \\log_{b}(0) \\equiv 0`.

    Args:
        prob_dist (ndarray or list): a probability array :math:`\\mathbf{p}`.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Shannon entropy :math:`H(\\mathcal{p})`.
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
    """Calculates the relative entropy between two discrete probability distributions.

    For probabilities :math:`\\mathbf{p}` and :math:`\\mathbf{q}`, it is defined as

    .. math::
        D(\\mathbf{p} \\, \\| \\, \\mathbf{q}) = \\sum_{x} \\, \\mathbf{p}(x) \\,
            \\log\\left( \\frac{\\mathbf{p}(x)}{\\mathbf{q}(x)} \\right) \\, .

    The classical relative entropy is also known as the
    `Kullback-Leibler (KL) divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical relative entropy between :math:`\\mathbf{p}` and :math:`\\mathbf{q}`.
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
    """Calculates the classical mutual information of two random variables.

    Given two random variables :math:`(X, \\, Y)`, their mutual information is given by

    .. math::
        I(X, \\, Y) \\equiv H(p(x)) + H(q(y)) - H(p(x, \\, y)) \\, ,

    where :math:`p(x, \\, y)` is the joint probability distribution of :math:`(X, Y)`,
    :math:`p(x)` is the marginal probability distribution of :math:`X`,
    :math:`q(y)` is the marginal probability distribution of :math:`Y`,
    and :math:`H(\\cdot)` is the :func:`qibo.quantum_info.entropies.shannon_entropy`.

    Args:
        prob_dist_joint (ndarray): joint probability distribution :math:`p(x, \\, y)`.
        prob_dist_p (ndarray): marginal probability distribution :math:`p(x)`.
        prob_dist_q (ndarray): marginal probability distribution :math:`q(y)`.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Mutual information :math:`I(X, \\, Y)`.
    """
    return (
        shannon_entropy(prob_dist_p, base, backend)
        + shannon_entropy(prob_dist_q, base, backend)
        - shannon_entropy(prob_dist_joint, base, backend)
    )


def classical_renyi_entropy(
    prob_dist, alpha: Union[float, int], base: float = 2, backend=None
):
    """Calculates the classical Rényi entropy :math:`H_{\\alpha}` of a discrete probability distribution.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)` and probability distribution
    :math:`\\mathbf{p}`, the classical Rényi entropy is defined as

    .. math::
        H_{\\alpha}(\\mathbf{p}) = \\frac{1}{1 - \\alpha} \\, \\log\\left( \\sum_{x}
            \\, \\mathbf{p}^{\\alpha}(x) \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the classical Rényi entropy
    coincides with the :func:`qibo.quantum_info.entropies.shannon_entropy`.

    Another special case is the limit :math:`\\alpha \\to 0`, where the function is
    reduced to :math:`\\log\\left(|\\mathbf{p}|\\right)`, with :math:`|\\mathbf{p}|`
    being the support of :math:`\\mathbf{p}`.
    This is known as the `Hartley entropy <https://en.wikipedia.org/wiki/Hartley_function>`_
    (also known as *Hartley function* or *max-entropy*).

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-\\log(\\max_{x}(\\mathbf{p}(x)))`, which is called the
    `min-entropy <https://en.wikipedia.org/wiki/Min-entropy>`_.

    Args:
        prob_dist (ndarray): discrete probability distribution.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical Rényi entropy :math:`H_{\\alpha}`.
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
    """Calculates the classical relative Rényi entropy between two discrete probability distributions.

    This function is also known as
    `Rényi divergence <https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R%C3%A9nyi_divergence>`_.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)` and probability distributions
    :math:`\\mathbf{p}` and :math:`\\mathbf{q}`, the classical relative Rényi entropy is defined as

    .. math::
        H_{\\alpha}(\\mathbf{p} \\, \\| \\, \\mathbf{q}) = \\frac{1}{\\alpha - 1} \\,
            \\log\\left( \\sum_{x} \\, \\frac{\\mathbf{p}^{\\alpha}(x)}
            {\\mathbf{q}^{\\alpha - 1}(x)} \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the classical Rényi divergence
    coincides with the :func:`qibo.quantum_info.entropies.classical_relative_entropy`.

    Another special case is the limit :math:`\\alpha \\to 1/2`, where the function is
    reduced to :math:`-2 \\log\\left(\\sum_{x} \\, \\sqrt{\\mathbf{p}(x) \\, \\mathbf{q}(x)} \\right)`.
    The sum inside the :math:`\\log` is known as the
    `Bhattacharyya coefficient <https://en.wikipedia.org/wiki/Bhattacharyya_distance>`_.

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`\\log(\\max_{x}(\\mathbf{p}(x) \\, \\mathbf{q}(x))`.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical relative Rényi entropy :math:`H_{\\alpha}(\\mathbf{p} \\, \\| \\, \\mathbf{q})`.
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
    """Calculates the classical Tsallis entropy for a discrete probability distribution.

    This is defined as

    .. math::
        S_{\\alpha}(\\mathbf{p}) = \\frac{1}{\\alpha - 1} \\,
            \\left(1 - \\sum_{x} \\, \\mathbf{p}^{\\alpha}(x) \\right)

    Args:
        prob_dist (ndarray): discrete probability distribution.
        alpha (float or int): entropic index.
        base (float): the base of the log. Used when ``alpha=1.0``.
            Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Classical Tsallis entropy :math:`S_{\\alpha}(\\mathbf{p})`.
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
    """Calculate the classical relative Tsallis entropy between two discrete probability distributions.

    Given a discrete random variable :math:`\\chi` that has values :math:`x` in the set
    :math:`\\mathcal{X}` with probability :math:`\\mathrm{p}(x)` and a discrete random variable
    :math:`\\upsilon` that has the values :math:`x` in the same set :math:`\\mathcal{X}` with
    probability :math:`\\mathrm{q}(x)`, their relative Tsallis entropy is given by

    .. math::
        D_{\\alpha}^{\\text{ts}}(\\chi \\, \\| \\, \\upsilon) = \\sum_{x \\in \\mathcal{X}} \\,
            \\mathrm{p}^{\\alpha}(x) \\, \\ln_{\\alpha}
            \\left( \\frac{\\mathrm{p}(x)}{\\mathrm{q}(x)} \\right) \\, ,

    where :math:`\\ln_{\\alpha}(x) \\equiv \\frac{x^{1 - \\alpha} - 1}{1 - \\alpha}`
    is the so-called :math:`\\alpha`-logarithm. When :math:`\\alpha = 1`, it reduces to
    :class:`qibo.quantum_info.entropies.classical_relative_entropy`.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        alpha (float): entropic index.
        base (float): the base of the log used when :math:`\\alpha = 1`. Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Tsallis relative entropy :math:`D_{\\alpha}^{\\text{ts}}`.
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
    """Calculates the von-Neumann entropy :math:`S(\\rho)` of a quantum ``state`` :math:`\\rho`.

    It is given by

    .. math::
        S(\\rho) = - \\text{tr}\\left[\\rho \\, \\log(\\rho)\\right]

    Args:
        state (ndarray): statevector or density matrix.
        base (float, optional): the base of the log. Defaults to :math:`2`.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian .
            Defaults to ``False``.
        return_spectrum: if ``True``, returns ``entropy`` and
            :math:`-\\log_{\\textup{b}}(\\textup{eigenvalues})`, where :math:`b` is ``base``.
            If ``False``, returns only ``entropy``. Default is ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: The von-Neumann entropy :math:`S` of ``state`` :math:`\\rho`.
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
    """Calculates the relative von Neumann entropy  between two quantum states.

    Also known as *quantum relative entropy*, :math:`S(\\rho \\, \\| \\, \\sigma)` is given by

    .. math::
        S(\\rho \\, \\| \\, \\sigma) = \\text{tr}\\left[\\rho \\, \\log(\\rho)\\right]
            - \\text{tr}\\left[\\rho \\, \\log(\\sigma)\\right]

    where ``state`` :math:`\\rho` and ``target`` :math:`\\sigma` are two quantum states.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        base (float, optional): the base of the log. Defaults to :math:`2`.
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
        float: Relative (von-Neumann) entropy :math:`S(\\rho \\, \\| \\, \\sigma)`.
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
    state, partition, base: float = 2, check_hermitian: bool = False, backend=None
):
    """Calculates the mutual information of a bipartite state.

    Given a qubit ``partition`` :math:`A`, the mutual information
    of state :math:`\\rho` is given by

    .. math::
        I(\\rho) \\equiv S(\\rho_{A}) + S(\\rho_{B}) - S(\\rho) \\, ,

    where :math:`B` is the remaining qubits that are not in partition :math:`A`,
    and :math:`S(\\cdot)` is the :func:`qibo.quantum_info.von_neumann_entropy`.

    Args:
        state (ndarray): statevector or density matrix.
        partition (Union[List[int], Tuple[int]]): indices of qubits in partition :math:`A`.
        base (float, optional): the base of the log. Defaults to :math:`2`.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian . Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Mutual information :math:`I(\\rho)` of ``state`` :math:`\\rho`.
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
    """Calculates the Rényi entropy :math:`H_{\\alpha}` of a quantum state :math:`\\rho`.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)`, the Rényi entropy is defined as

    .. math::
        H_{\\alpha}(\\rho) = \\frac{1}{1 - \\alpha} \\, \\log\\left( \\rho^{\\alpha} \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Rényi entropy
    coincides with the :func:`qibo.quantum_info.entropies.entropy`.

    Another special case is the limit :math:`\\alpha \\to 0`, where the function is
    reduced to :math:`\\log\\left(d\\right)`, with :math:`d = 2^{n}`
    being the dimension of the Hilbert space in which ``state`` :math:`\\rho` lives in.
    This is known as the `Hartley entropy <https://en.wikipedia.org/wiki/Hartley_function>`_
    (also known as *Hartley function* or *max-entropy*).

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-\\log(\\|\\rho\\|_{\\infty})`, with :math:`\\|\\cdot\\|_{\\infty}`
    being the `spectral norm <https://en.wikipedia.org/wiki/Matrix_norm#Matrix_norms_induced_by_vector_p-norms>`_.
    This is known as the `min-entropy <https://en.wikipedia.org/wiki/Min-entropy>`_.

    Args:
        state (ndarray): statevector or density matrix.
        alpha (float or int): order of the Rényi entropy.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Rényi entropy :math:`H_{\\alpha}`.
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
            * backend.np.log2(backend.calculate_matrix_norm(state, order=2))
            / np.log2(base)
        )

    log = backend.np.log2(backend.np.trace(matrix_power(state, alpha, backend=backend)))

    return (1 / (1 - alpha)) * log / np.log2(base)


def relative_renyi_entropy(
    state, target, alpha: Union[float, int], base: float = 2, backend=None
):
    """Calculates the relative Rényi entropy between two quantum states.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)` and quantum states
    :math:`\\rho` and :math:`\\sigma`, the relative Rényi entropy is defined as

    .. math::
        H_{\\alpha}(\\rho \\, \\| \\, \\sigma) = \\frac{1}{\\alpha - 1} \\,
            \\log\\left( \\textup{tr}\\left( \\rho^{\\alpha} \\,
            \\sigma^{1 - \\alpha} \\right) \\right) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Rényi entropy
    coincides with the :func:`qibo.quantum_info.entropies.relative_von_neumann_entropy`.

    In the limit :math:`\\alpha \\to \\infty`, the function reduces to
    :math:`-2 \\, \\log(\\|\\sqrt{\\rho} \\, \\sqrt{\\sigma}\\|_{1})`,
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
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Relative Rényi entropy :math:`H_{\\alpha}(\\rho \\, \\| \\, \\sigma)`.
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
            backend.calculate_matrix_norm(new_state @ new_target, order=1)
        )

        return -2 * log / np.log2(base)

    log = matrix_power(state, alpha, backend=backend)
    log = log @ matrix_power(target, 1 - alpha, backend=backend)
    log = backend.np.log2(backend.np.trace(log))

    return (1 / (alpha - 1)) * log / np.log2(base)


def tsallis_entropy(state, alpha: float, base: float = 2, backend=None):
    """Calculates the Tsallis entropy of a quantum state.

    .. math::
        S_{\\alpha}(\\rho) = \\frac{1}{1 - \\alpha} \\,
            \\left( \\text{tr}(\\rho^{\\alpha}) - 1 \\right)

    When :math:`\\alpha = 1`, the functions defaults to
    :func:`qibo.quantum_info.entropies.entropy`.

    Args:
        state (ndarray): statevector or density matrix.
        alpha (float or int): entropic index.
        base (float, optional): the base of the log. Used when ``alpha=1.0``.
            Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Tsallis entropy :math:`S_{\\alpha}(\\rho)`.
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
            \\text{tr}\\left(\\rho^{\\alpha} \\, \\sigma^{1 - \\alpha}\\right)}{1 - \\alpha} \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the Tsallis entropy
    coincides with the :func:`qibo.quantum_info.entropies.relative_von_neumann_entropy`.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        alpha (float or int): entropic index :math:`\\alpha \\in [0, \\, 2]`.
        base (float, optional): the base of the log used when :math:`\\alpha = 1`.
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
    bipartition,
    base: float = 2,
    check_hermitian: bool = False,
    return_spectrum: bool = False,
    backend=None,
):
    """Calculates the entanglement entropy :math:`S` of bipartition :math:`A`
    of ``state`` :math:`\\rho`. This is given by

    .. math::
        S(\\rho_{A}) = -\\text{tr}(\\rho_{A} \\, \\log(\\rho_{A})) \\, ,

    where :math:`\\rho_{A} = \\text{tr}_{B}(\\rho)` is the reduced density matrix calculated
    by tracing out the ``bipartition`` :math:`B`.

    Args:
        state (ndarray): statevector or density matrix.
        bipartition (list or tuple or ndarray): qubits in the subsystem to be traced out.
        base (float, optional): the base of the log. Defaults to :math: `2`.
        check_hermitian (bool, optional): if ``True``, checks if :math:`\\rho_{A}` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian . Default: ``False``.
        return_spectrum: if ``True``, returns ``entropy`` and eigenvalues of ``state``.
            If ``False``, returns only ``entropy``. Default is ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Entanglement entropy :math:`S` of ``state`` :math:`\\rho`.
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

    reduced_density_matrix = partial_trace(state, bipartition, backend=backend)

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
