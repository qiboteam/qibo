"""Submodule with entropy measures."""

import numpy as np

from qibo.backends import GlobalBackend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.metrics import _check_hermitian_or_not_gpu, purity


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
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        (float): Shannon entropy :math:`H(\\mathcal{p})`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    if np.abs(np.sum(prob_dist) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Probability array must sum to 1.")

    if base == 2:
        log_prob = np.where(prob_dist != 0.0, np.log2(prob_dist), 0.0)
    elif base == 10:
        log_prob = np.where(prob_dist != 0, np.log10(prob_dist), 0.0)
    elif base == np.e:
        log_prob = np.where(prob_dist != 0, np.log(prob_dist), 0.0)
    else:
        log_prob = np.where(prob_dist != 0, np.log(prob_dist) / np.log(base), 0.0)

    shan_entropy = -np.sum(prob_dist * log_prob)

    # absolute value if entropy == 0.0 to avoid returning -0.0
    shan_entropy = np.abs(shan_entropy) if shan_entropy == 0.0 else shan_entropy

    return complex(shan_entropy).real


def classical_relative_entropy(prob_dist_p, prob_dist_q, base: float = 2, backend=None):
    """Calculates the relative entropy between two discrete probability distributions.

    For probabilities :math:`\\mathbf{p}` and :math:`\\mathbf{q}`, it is defined as

    ..math::
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: Classical relative entropy between :math:`\\mathbf{p}` and :math:`\\mathbf{q}`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if isinstance(prob_dist_p, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)
    if isinstance(prob_dist_q, list):
        # np.float64 is necessary instead of native float because of tensorflow
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
    if np.abs(np.sum(prob_dist_p) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "First probability array must sum to 1.")

    if np.abs(np.sum(prob_dist_q) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Second probability array must sum to 1.")

    entropy_p = -1 * shannon_entropy(prob_dist_p, base=base, backend=backend)

    if base == 2:
        log_prob_q = np.where(prob_dist_q != 0.0, np.log2(prob_dist_q), -np.inf)
    elif base == 10:
        log_prob_q = np.where(prob_dist_q != 0.0, np.log10(prob_dist_q), -np.inf)
    elif base == np.e:
        log_prob_q = np.where(prob_dist_q != 0.0, np.log(prob_dist_q), -np.inf)
    else:
        log_prob_q = np.where(
            prob_dist_q != 0.0, np.log(prob_dist_q) / np.log(base), -np.inf
        )

    log_prob = np.where(prob_dist_p != 0.0, log_prob_q, 0.0)

    relative = np.sum(prob_dist_p * log_prob)

    return entropy_p - relative


def classical_renyi_entropy(prob_dist, alpha: float, base: float = 2, backend=None):
    """Calculates the classical Rényi entropy :math:`H_{\\alpha}` of a discrete probability distribution.

    For :math:`\\alpha \\in (0, \\, 1) \\cup (1, \\, \\infty)` and probability distribution
    :math:`\\mathbf{p}`, the Rényi entropy is defined as

    .. math::
        H_{\\alpha}(\\mathbf{p}) = \\frac{1}{1 - \\alpha} \\, \\log\\left( \\sum_{x}
            \\, \\mathbf{p}^{\\alpha}(x) \\right) \\, .

    For :math:`\\alpha \\in \\{0, 1, \\infty \\}`, it is further defined that

    .. math::
        H_{\\alpha}(\\mathbf{p}) = \\lim_{\\beta \\to \\alpha} \\, H_{\\beta}(\\mathbf{p}) \\, .

    A special case is the limit :math:`\\alpha \\to 1`, in which the classical Rényi entropy
    coincides with the :func:`qibo.quantum_info.entropies.shannon_entropy`.

    Args:
        prob_dist (ndarray): discrete probability distribution.
        alpha (float): order of the Rényi entropy.
            If :math:`\\alpha = 1`, defaults to :func:`qibo.quantum_info.entropies.shannon_entropy`.
            If :math:`\\alpha = \\infty`, defaults to the
            `min-entropy <https://en.wikipedia.org/wiki/Min-entropy>`_.
        base (float): the base of the log. Defaults to  :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: Rényi entropy :math:`H_{\\alpha}`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if isinstance(prob_dist, list):
        # np.float64 is necessary instead of native float because of tensorflow
        prob_dist = backend.cast(prob_dist, dtype=np.float64)

    if not isinstance(alpha, float):
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

    if np.abs(np.sum(prob_dist) - 1.0) > PRECISION_TOL:
        raise_error(ValueError, "Probability array must sum to 1.")

    if alpha == 1.0:
        return shannon_entropy(prob_dist, base=base, backend=backend)

    if alpha == np.inf:
        return -1 * np.log2(max(prob_dist)) / np.log2(base)

    renyi_ent = (1 / (1 - alpha)) * np.log2(np.sum(prob_dist**alpha)) / np.log2(base)

    return renyi_ent


def entropy(
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: The von-Neumann entropy :math:`S` of ``state`` :math:`\\rho`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    if purity(state) == 1.0:
        if return_spectrum:
            return 0.0, backend.cast([1.0], dtype=float)

        return 0.0

    if not check_hermitian or _check_hermitian_or_not_gpu(state, backend=backend):
        eigenvalues = np.linalg.eigvalsh(state)
    else:
        eigenvalues = np.linalg.eigvals(state)

    if base == 2:
        log_prob = np.where(eigenvalues > 0, np.log2(eigenvalues), 0.0)
    elif base == 10:
        log_prob = np.where(eigenvalues > 0, np.log10(eigenvalues), 0.0)
    elif base == np.e:
        log_prob = np.where(eigenvalues > 0, np.log(eigenvalues), 0.0)
    else:
        log_prob = np.where(eigenvalues > 0, np.log(eigenvalues) / np.log(base), 0.0)

    ent = -np.sum(eigenvalues * log_prob)
    # absolute value if entropy == 0.0 to avoid returning -0.0
    ent = np.abs(ent) if ent == 0.0 else ent

    ent = float(ent)

    if return_spectrum:
        log_prob = backend.cast(log_prob, dtype=log_prob.dtype)
        return ent, -log_prob

    return ent


def relative_entropy(
    state, target, base: float = 2, check_hermitian: bool = False, backend=None
):
    """Calculates the relative entropy :math:`S(\\rho \\, \\| \\, \\sigma)` between ``state`` :math:`\\rho` and ``target`` :math:`\\sigma`.

    It is given by

    .. math::
        S(\\rho \\, \\| \\, \\sigma) = \\text{tr}\\left[\\rho \\, \\log(\\rho)\\right]
            - \\text{tr}\\left[\\rho \\, \\log(\\sigma)\\right]

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        target (ndarray): statevector or density matrix :math:`\\sigma`.
        base (float, optional): the base of the log. Defaults to :math:`2`.
        check_hermitian (bool, optional): If ``True``, checks if ``state`` is Hermitian.
            If ``False``, it assumes ``state`` is Hermitian .
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: Relative (von-Neumann) entropy :math:`S(\\rho \\, \\| \\, \\sigma)`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    if purity(state) == 1.0 and purity(target) == 1.0:
        return 0.0

    if not check_hermitian or _check_hermitian_or_not_gpu(state, backend=backend):
        eigenvalues_state = np.linalg.eigvalsh(state)
    else:
        eigenvalues_state = np.linalg.eigvals(state)

    if not check_hermitian or _check_hermitian_or_not_gpu(target, backend=backend):
        eigenvalues_target = np.linalg.eigvalsh(target)
    else:
        eigenvalues_target = np.linalg.eigvals(target)

    if base == 2:
        log_state = np.where(eigenvalues_state > 0, np.log2(eigenvalues_state), 0.0)
        log_target = np.where(
            eigenvalues_target > 0, np.log2(eigenvalues_target), -np.inf
        )
    elif base == 10:
        log_state = np.where(eigenvalues_state > 0, np.log10(eigenvalues_state), 0.0)
        log_target = np.where(
            eigenvalues_target > 0, np.log10(eigenvalues_target), -np.inf
        )
    elif base == np.e:
        log_state = np.where(eigenvalues_state > 0, np.log(eigenvalues_state), 0.0)
        log_target = np.where(
            eigenvalues_target > 0, np.log(eigenvalues_target), -np.inf
        )
    else:
        log_state = np.where(
            eigenvalues_state > 0, np.log(eigenvalues_state) / np.log(base), 0.0
        )
        log_target = np.where(
            eigenvalues_target > 0, np.log(eigenvalues_target) / np.log(base), -np.inf
        )

    log_target = np.where(eigenvalues_state != 0.0, log_target, 0.0)

    entropy_state = np.sum(eigenvalues_state * log_state)

    relative = np.sum(eigenvalues_state * log_target)

    return float(entropy_state - relative)


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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        float: Entanglement entropy :math:`S` of ``state`` :math:`\\rho`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    nqubits = int(np.log2(state.shape[0]))

    reduced_density_matrix = (
        backend.partial_trace(state, bipartition, nqubits)
        if len(state.shape) == 1
        else backend.partial_trace_density_matrix(state, bipartition, nqubits)
    )

    entropy_entanglement = entropy(
        reduced_density_matrix,
        base=base,
        check_hermitian=check_hermitian,
        return_spectrum=return_spectrum,
        backend=backend,
    )

    return entropy_entanglement
