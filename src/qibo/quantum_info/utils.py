"""Utility functions for the Quantum Information module."""

from functools import reduce
from re import finditer

import numpy as np

from qibo.backends import GlobalBackend
from qibo.config import PRECISION_TOL, raise_error

# Phases corresponding to all 24 single-qubit Clifford gates
ONEQUBIT_CLIFFORD_PARAMS = [
    (0, 0, 0, 0),
    (np.pi, 1, 0, 0),
    (np.pi, 0, 1, 0),
    (np.pi, 0, 0, 1),
    (np.pi / 2, 1, 0, 0),
    (-np.pi / 2, 1, 0, 0),
    (np.pi / 2, 0, 1, 0),
    (-np.pi / 2, 0, 1, 0),
    (np.pi / 2, 0, 0, 1),
    (-np.pi / 2, 0, 0, 1),
    (np.pi, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
    (np.pi, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
    (np.pi, -1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)),
    (np.pi, 0, -1 / np.sqrt(2), 1 / np.sqrt(2)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
]


def hamming_weight(bitstring, return_indexes: bool = False):
    """Calculates the Hamming weight of a bitstring.

    Args:
        bitstring (int or str or tuple or list or ndarray): bitstring to calculate the
            weight, either in binary or integer representation.
        return_indexes (bool, optional): If ``True``, returns the indexes of the
            non-zero elements. Defaults to ``False``.

    Returns:
        (int or list): Hamming weight of bitstring or list of indexes of non-zero elements.
    """
    if not isinstance(return_indexes, bool):
        raise_error(
            TypeError,
            f"return_indexes must be type bool, but it is type {type(return_indexes)}",
        )

    if isinstance(bitstring, int):
        bitstring = f"{bitstring:b}"
    elif isinstance(bitstring, (list, tuple, np.ndarray)):
        bitstring = "".join([str(bit) for bit in bitstring])

    indexes = [item.start() for item in finditer("1", bitstring)]

    if return_indexes:
        return indexes

    weight = len(indexes)

    return weight


def shannon_entropy(probability_array, base: float = 2, backend=None):
    """Calculate the Shannon entropy of a probability array :math:`\\mathbf{p}`, which is given by

    .. math::
        H(\\mathbf{p}) = - \\sum_{k = 0}^{d^{2} - 1} \\, p_{k} \\, \\log_{b}(p_{k}) \\, ,

    where :math:`d = \\text{dim}(\\mathcal{H})` is the dimension of the
    Hilbert space :math:`\\mathcal{H}`, :math:`b` is the log base (default 2),
    and :math:`0 \\log_{b}(0) \\equiv 0`.

    Args:
        probability_array (ndarray or list): a probability array :math:`\\mathbf{p}`.
        base (float): the base of the log. Default: 2.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        (float): The Shannon entropy :math:`H(\\mathcal{p})`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if isinstance(probability_array, list):
        probability_array = backend.cast(probability_array, dtype=float)

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if len(probability_array.shape) != 1:
        raise_error(
            TypeError,
            f"Probability array must have dims (k,) but it has {probability_array.shape}.",
        )

    if len(probability_array) == 0:
        raise_error(TypeError, "Empty array.")

    if any(probability_array < 0) or any(probability_array > 1.0):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    if (np.sum(probability_array) > 1.0 + PRECISION_TOL) or (
        np.sum(probability_array) < 1.0 - PRECISION_TOL
    ):
        raise_error(ValueError, "Probability array must sum to 1.")

    if base == 2:
        log_prob = np.where(probability_array != 0.0, np.log2(probability_array), 0.0)
    elif base == 10:
        log_prob = np.where(probability_array != 0, np.log10(probability_array), 0.0)
    elif base == np.e:
        log_prob = np.where(probability_array != 0, np.log(probability_array), 0.0)
    else:
        log_prob = np.where(
            probability_array != 0, np.log(probability_array) / np.log(base), 0.0
        )

    entropy = -np.sum(probability_array * log_prob)

    # absolute value if entropy == 0.0 to avoid returning -0.0
    entropy = np.abs(entropy) if entropy == 0.0 else entropy

    return complex(entropy).real


def hellinger_distance(prob_dist_p, prob_dist_q, validate: bool = False, backend=None):
    """Calculate the Hellinger distance :math:`H(p, q)` between
    two discrete probability distributions, :math:`\\mathbf{p}` and :math:`\\mathbf{q}`.
    It is defined as

    .. math::
        H(\\mathbf{p} \\, , \\, \\mathbf{q}) = \\frac{1}{\\sqrt{2}} \\, \\|
            \\sqrt{\\mathbf{p}} - \\sqrt{\\mathbf{q}} \\|_{2}

    where :math:`\\|\\cdot\\|_{2}` is the Euclidean norm.

    Args:
        prob_dist_p: (discrete) probability distribution :math:`p`.
        prob_dist_q: (discrete) probability distribution :math:`q`.
        validate (bool): if True, checks if :math:`p` and :math:`q` are proper
            probability distributions. Default: False.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        (float): Hellinger distance :math:`H(p, q)`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if isinstance(prob_dist_p, list):
        prob_dist_p = backend.cast(prob_dist_p, dtype=float)
    if isinstance(prob_dist_q, list):
        prob_dist_q = backend.cast(prob_dist_q, dtype=float)

    if (len(prob_dist_p.shape) != 1) or (len(prob_dist_q.shape) != 1):
        raise_error(
            TypeError,
            "Probability arrays must have dims (k,) but have "
            + f"dims {prob_dist_p.shape} and {prob_dist_q.shape}.",
        )

    if (len(prob_dist_p) == 0) or (len(prob_dist_q) == 0):
        raise_error(TypeError, "At least one of the arrays is empty.")

    if validate:
        if (any(prob_dist_p < 0) or any(prob_dist_p > 1.0)) or (
            any(prob_dist_q < 0) or any(prob_dist_q > 1.0)
        ):
            raise_error(
                ValueError,
                "All elements of the probability array must be between 0. and 1..",
            )
        if (np.sum(prob_dist_p) > 1.0 + PRECISION_TOL) or (
            np.sum(prob_dist_p) < 1.0 - PRECISION_TOL
        ):
            raise_error(ValueError, "First probability array must sum to 1.")

        if (np.sum(prob_dist_q) > 1.0 + PRECISION_TOL) or (
            np.sum(prob_dist_q) < 1.0 - PRECISION_TOL
        ):
            raise_error(ValueError, "Second probability array must sum to 1.")

    distance = backend.calculate_norm(
        np.sqrt(prob_dist_p) - np.sqrt(prob_dist_q)
    ) / np.sqrt(2)

    distance = float(distance)

    return distance


def hellinger_fidelity(prob_dist_p, prob_dist_q, validate: bool = False, backend=None):
    """Calculate the Hellinger fidelity between two discrete
    probability distributions, :math:`p` and :math:`q`. The fidelity is
    defined as :math:`(1 - H^{2}(p, q))^{2}`, where :math:`H(p, q)`
    is the Hellinger distance.

    Args:
        prob_dist_p: (discrete) probability distribution :math:`p`.
        prob_dist_q: (discrete) probability distribution :math:`q`.
        validate (bool): if True, checks if :math:`p` and :math:`q` are proper
            probability distributions. Default: False.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        (float): Hellinger fidelity.

    """
    distance = hellinger_distance(prob_dist_p, prob_dist_q, validate, backend=backend)

    return (1 - distance**2) ** 2


def haar_integral(nqubits: int, t: int, samples: int, backend=None):
    """Returns the integral over pure states over the Haar measure.

    .. math::
        \\int_{\\text{Haar}} d\\psi \\, \\left(|\\psi\\rangle\\right.\\left.
            \\langle\\psi|\\right)^{\\otimes t}

    Args:
        nqubits (int): Number of qubits.
        t (int): power that defines the :math:`t`-design.
        samples (int): number of samples to estimate the integral.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        array: Estimation of the Haar integral.
    """

    if isinstance(nqubits, int) is False:
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if isinstance(t, int) is False:
        raise_error(TypeError, f"t must be type int, but it is type {type(t)}.")

    if isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    from qibo.quantum_info.random_ensembles import random_statevector

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = 2**nqubits

    rand_unit_density = np.zeros((dim**t, dim**t), dtype=complex)
    rand_unit_density = backend.cast(rand_unit_density, dtype=rand_unit_density.dtype)
    for _ in range(samples):
        haar_state = np.reshape(
            random_statevector(dim, haar=True, backend=backend), (-1, 1)
        )

        rho = haar_state @ np.conj(np.transpose(haar_state))

        rand_unit_density += reduce(np.kron, [rho] * t)

    integral = rand_unit_density / samples

    return integral


def pqc_integral(circuit, t: int, samples: int, backend=None):
    """Returns the integral over pure states generated by uniformly sampling
    in the parameter space described by a parameterized circuit.

    .. math::
        \\int_{\\Theta} d\\psi \\, \\left(|\\psi_{\\theta}\\rangle\\right.\\left.
            \\langle\\psi_{\\theta}|\\right)^{\\otimes t}

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        t (int): power that defines the :math:`t`-design.
        samples (int): number of samples to estimate the integral.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Estimation of the integral.
    """

    if isinstance(t, int) is False:
        raise_error(TypeError, f"t must be type int, but it is type {type(t)}.")

    if isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    from qibo.gates import I

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    circuit.density_matrix = True
    dim = 2**circuit.nqubits

    rand_unit_density = np.zeros((dim**t, dim**t), dtype=complex)
    rand_unit_density = backend.cast(rand_unit_density, dtype=rand_unit_density.dtype)
    for _ in range(samples):
        params = np.random.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        circuit.set_parameters(params)

        rho = backend.execute_circuit(circuit).state()

        rand_unit_density += reduce(np.kron, [rho] * t)

    integral = rand_unit_density / samples

    return integral
