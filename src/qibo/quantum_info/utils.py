"""Utility functions for the Quantum Information module."""

from functools import reduce
from re import finditer

import numpy as np

from qibo import matrices
from qibo.backends import GlobalBackend
from qibo.config import PRECISION_TOL, raise_error


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

    return len(indexes)


def hadamard_transform(array, implementation: str = "fast", backend=None):
    """Calculates the (fast) Hadamard Transform :math:`\\text{HT}` of a
    :math:`2^{n}`-dimensional vector or :math:`2^{n} \times 2^{n}` matrix :math:`A`,
    where :math:`n` is the number of qubits in the system. If :math:`A` is a vector, then

    .. math::
        \\text{HT}(A) = \\frac{1}{2**{n / 2}} \\, H^{\\otimes n} \\, A \\,

    where :math:`H` is the :class:`qibo.gates.H` gate. If :math:`A` is a matrix, then

    .. math::
        \\text{HT}(A) = \\frac{1}{2**{n}} \\, H^{\\otimes n} \\, A \\, H^{\\otimes n} \\, .

    Args:
        array (ndarray): array or matrix.
        implementation (str, optional): if ``"regular"``, it uses the straightforward
            implementation of the algorithm with computational complexity of
            :math:`\\mathcal{O}(2^{2n})` for vectors and :math:`\\mathcal{O}(2^{3n})`
            for matrices. If ``"fast"``, computational complexity is
            :math:`\\mathcal{O}(n \\, 2^{n})` in both cases.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        ndarray: (Fast) Hadamard Transform of ``array``.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if (
        len(array.shape) not in [1, 2]
        or (len(array.shape) == 1 and np.log2(array.shape[0]).is_integer() is False)
        or (
            len(array.shape) == 2
            and (
                np.log2(array.shape[0]).is_integer() is False
                or np.log2(array.shape[1]).is_integer() is False
            )
        )
    ):
        raise_error(
            TypeError,
            f"array must have shape (2**n,) or (2**n, 2**n), but it has shape {array.shape}.",
        )

    if isinstance(implementation, str) is False:
        raise_error(
            TypeError,
            f"implementation must be type str, but it is type {type(implementation)}.",
        )

    if implementation not in ["fast", "regular"]:
        raise_error(
            ValueError,
            f"implementation must be either `regular` or `fast`, but it is {implementation}.",
        )

    if implementation == "regular":
        nqubits = int(np.log2(array.shape[0]))
        hadamards = np.real(reduce(np.kron, [matrices.H] * nqubits))
        hadamards /= 2 ** (nqubits / 2)
        hadamards = backend.cast(hadamards, dtype=hadamards.dtype)

        array = hadamards @ array

        if len(array.shape) == 2:
            array = array @ hadamards

        return array

    array = _hadamard_transform_1d(array)

    if len(array.shape) == 2:
        array = _hadamard_transform_1d(np.transpose(array))
        array = np.transpose(array)

    # needed for the tensorflow backend
    array = backend.cast(array, dtype=array.dtype)

    return array


def shannon_entropy(probability_array, base: float = 2, backend=None):
    """Calculate the Shannon entropy of a probability array :math:`\\mathbf{p}`, which is given by

    .. math::
        H(\\mathbf{p}) = - \\sum_{k = 0}^{d^{2} - 1} \\, p_{k} \\, \\log_{b}(p_{k}) \\, ,

    where :math:`d = \\text{dim}(\\mathcal{H})` is the dimension of the
    Hilbert space :math:`\\mathcal{H}`, :math:`b` is the log base (default 2),
    and :math:`0 \\log_{b}(0) \\equiv 0`.

    Args:
        probability_array (ndarray or list): a probability array :math:`\\mathbf{p}`.
        base (float): the base of the log. Defaults to  :math:`2`.
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

    if np.abs(np.sum(probability_array) - 1.0) > PRECISION_TOL:
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
        if np.abs(np.sum(prob_dist_p) - 1.0) > PRECISION_TOL:
            raise_error(ValueError, "First probability array must sum to 1.")

        if np.abs(np.sum(prob_dist_q) - 1.0) > PRECISION_TOL:
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


def haar_integral(nqubits: int, power_t: int, samples: int, backend=None):
    """Returns the integral over pure states over the Haar measure.

    .. math::
        \\int_{\\text{Haar}} d\\psi \\, \\left(|\\psi\\rangle\\right.\\left.
            \\langle\\psi|\\right)^{\\otimes t}

    Args:
        nqubits (int): Number of qubits.
        power_t (int): power that defines the :math:`t`-design.
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

    if isinstance(power_t, int) is False:
        raise_error(
            TypeError, f"power_t must be type int, but it is type {type(power_t)}."
        )

    if isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    from qibo.quantum_info.random_ensembles import (  # pylint: disable=C0415
        random_statevector,
    )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = 2**nqubits

    rand_unit_density = np.zeros((dim**power_t, dim**power_t), dtype=complex)
    rand_unit_density = backend.cast(rand_unit_density, dtype=rand_unit_density.dtype)
    for _ in range(samples):
        haar_state = np.reshape(
            random_statevector(dim, haar=True, backend=backend), (-1, 1)
        )

        rho = haar_state @ np.conj(np.transpose(haar_state))

        rand_unit_density += reduce(np.kron, [rho] * power_t)

    integral = rand_unit_density / samples

    return integral


def pqc_integral(circuit, power_t: int, samples: int, backend=None):
    """Returns the integral over pure states generated by uniformly sampling
    in the parameter space described by a parameterized circuit.

    .. math::
        \\int_{\\Theta} d\\psi \\, \\left(|\\psi_{\\theta}\\rangle\\right.\\left.
            \\langle\\psi_{\\theta}|\\right)^{\\otimes t}

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        power_t (int): power that defines the :math:`t`-design.
        samples (int): number of samples to estimate the integral.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Estimation of the integral.
    """

    if isinstance(power_t, int) is False:
        raise_error(
            TypeError, f"power_t must be type int, but it is type {type(power_t)}."
        )

    if isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    circuit.density_matrix = True
    dim = 2**circuit.nqubits

    rand_unit_density = np.zeros((dim**power_t, dim**power_t), dtype=complex)
    rand_unit_density = backend.cast(rand_unit_density, dtype=rand_unit_density.dtype)
    for _ in range(samples):
        params = np.random.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        circuit.set_parameters(params)

        rho = backend.execute_circuit(circuit).state()

        rand_unit_density += reduce(np.kron, [rho] * power_t)

    integral = rand_unit_density / samples

    return integral


def _hadamard_transform_1d(array):
    # necessary because of tf.EagerTensor
    # does not accept item assignment
    array_copied = np.copy(array)

    indexes = [2**k for k in range(int(np.log2(len(array_copied))))]
    for index in indexes:
        for k in range(0, len(array_copied), 2 * index):
            for j in range(k, k + index):
                # copy necessary because of cupy backend
                elem_1 = np.copy(array_copied[j])
                elem_2 = np.copy(array_copied[j + index])
                array_copied[j] = elem_1 + elem_2
                array_copied[j + index] = elem_1 - elem_2
        array_copied /= 2.0

    return array_copied
