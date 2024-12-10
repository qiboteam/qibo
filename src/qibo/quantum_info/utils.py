"""Utility functions for the Quantum Information module."""

from functools import reduce
from itertools import permutations
from math import factorial
from re import finditer
from typing import Optional, Union

import numpy as np

from qibo import matrices
from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error


def hamming_weight(
    bitstring: Union[int, str, list, tuple], return_indexes: bool = False
):
    """Calculates the Hamming weight of a bitstring.

    The Hamming weight of a bistring is the number of :math:'1's that the bistring contains.

    Args:
        bitstring (int or str or tuple or list): bitstring to calculate the
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

    if not isinstance(bitstring, (int, str, list, tuple, np.ndarray)):
        raise_error(
            TypeError,
            "bitstring must be either type int, list, tuple, or numpy.ndarray. "
            f"However, it is type {type(bitstring)}.",
        )

    if isinstance(bitstring, int):
        bitstring = f"{bitstring:b}"
    elif isinstance(bitstring, (list, tuple, np.ndarray)):
        bitstring = "".join([str(bit) for bit in bitstring])

    indexes = [item.start() for item in finditer("1", bitstring)]

    if return_indexes:
        return indexes

    return len(indexes)


def hamming_distance(
    bitstring_1: Union[int, str, list, tuple],
    bitstring_2: Union[int, str, list, tuple],
    return_indexes: bool = False,
):
    """Calculates the Hamming distance between two bistrings.

    This is done by calculating the Hamming weight
    (:func:`qibo.quantum_info.utils.hamming_weight`) of ``| bitstring_1 - bitstring_2 |``.

    Args:
        bitstring_1 (int or str or list or tuple): fisrt bistring.
        bitstring_2 (int or str or list or tuple): second bitstring.
        return_indexes (bool, optional): If ``True``, returns the indexes of the
            non-zero elements. Defaults to ``False``.

    Returns:
        int or list: Hamming distance or list of indexes of non-zero elements.
    """
    if not isinstance(return_indexes, bool):
        raise_error(
            TypeError,
            f"return_indexes must be type bool, but it is type {type(return_indexes)}",
        )

    if not isinstance(bitstring_1, (int, str, list, tuple)):
        raise_error(
            TypeError,
            "bitstring_1 must be either type int, list, tuple, or numpy.ndarray. "
            f"However, it is type {type(bitstring_1)}.",
        )

    if not isinstance(bitstring_2, (int, str, list, tuple)):
        raise_error(
            TypeError,
            "bitstring_2 must be either type int, list, tuple, or numpy.ndarray. "
            f"However, it is type {type(bitstring_2)}.",
        )

    if isinstance(bitstring_1, (list, tuple)):
        bitstring_1 = "".join(bitstring_1)

    if isinstance(bitstring_2, (list, tuple)):
        bitstring_2 = "".join(bitstring_2)

    nbits = max(len(bitstring_1), len(bitstring_2))

    bitstring_1 = "0" * (nbits - len(bitstring_1)) + bitstring_1
    bitstring_2 = "0" * (nbits - len(bitstring_2)) + bitstring_2

    difference = np.array(list(bitstring_1), dtype=int) - np.array(
        list(bitstring_2), dtype=int
    )
    difference = np.abs(difference)
    difference = difference.astype(str)
    difference = "".join(difference)

    return hamming_weight(difference, return_indexes=return_indexes)


def hadamard_transform(array, implementation: str = "fast", backend=None):
    """Calculates the (fast) Hadamard Transform :math:`\\text{HT}` of a
    :math:`2^{n}`-dimensional vector or :math:`2^{n} \\times 2^{n}` matrix :math:`A`,
    where :math:`n` is the number of qubits in the system. If :math:`A` is a vector, then

    .. math::
        \\text{HT}(A) = \\frac{1}{2^{n / 2}} \\, H^{\\otimes n} \\, A \\,

    where :math:`H` is the :class:`qibo.gates.H` gate. If :math:`A` is a matrix, then

    .. math::
        \\text{HT}(A) = \\frac{1}{2^{n}} \\, H^{\\otimes n} \\, A \\, H^{\\otimes n} \\, .

    Args:
        array (ndarray): array or matrix.
        implementation (str, optional): if ``"regular"``, it uses the straightforward
            implementation of the algorithm with computational complexity of
            :math:`\\mathcal{O}(2^{2n})` for vectors and :math:`\\mathcal{O}(2^{3n})`
            for matrices. If ``"fast"``, computational complexity is
            :math:`\\mathcal{O}(n \\, 2^{n})` in both cases.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: (Fast) Hadamard Transform of ``array``.
    """
    backend = _check_backend(backend)

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

    array = _hadamard_transform_1d(array, backend=backend)

    if len(array.shape) == 2:
        array = _hadamard_transform_1d(array.T, backend=backend).T

    # needed for the tensorflow backend
    array = backend.cast(array, dtype=array.dtype)

    return array


def hellinger_distance(prob_dist_p, prob_dist_q, validate: bool = False, backend=None):
    """Calculates the Hellinger distance :math:`H` between two discrete probability distributions.

    For probabilities :math:`\\mathbf{p}` and :math:`\\mathbf{q}`, it is defined as

    .. math::
        H(\\mathbf{p} \\, , \\, \\mathbf{q}) = \\frac{1}{\\sqrt{2}} \\, \\|
            \\sqrt{\\mathbf{p}} - \\sqrt{\\mathbf{q}} \\|_{2}

    where :math:`\\|\\cdot\\|_{2}` is the Euclidean norm.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        validate (bool, optional): If ``True``, checks if :math:`p` and :math:`q` are proper
            probability distributions. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (float): Hellinger distance :math:`H(p, q)`.
    """
    backend = _check_backend(backend)

    if isinstance(prob_dist_p, list):
        prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)
    if isinstance(prob_dist_q, list):
        prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

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
        if backend.np.abs(backend.np.sum(prob_dist_p) - 1.0) > PRECISION_TOL:
            raise_error(ValueError, "First probability array must sum to 1.")

        if backend.np.abs(backend.np.sum(prob_dist_q) - 1.0) > PRECISION_TOL:
            raise_error(ValueError, "Second probability array must sum to 1.")

    distance = float(
        backend.calculate_vector_norm(
            backend.np.sqrt(prob_dist_p) - backend.np.sqrt(prob_dist_q)
        )
        / np.sqrt(2)
    )

    return distance


def hellinger_fidelity(prob_dist_p, prob_dist_q, validate: bool = False, backend=None):
    """Calculates the Hellinger fidelity between two discrete probability distributions.

    For probabilities :math:`p` and :math:`q`, the fidelity is defined as

    .. math::
        (1 - H^{2}(p, q))^{2} \\, ,

    where :math:`H(p, q)` is the :func:`qibo.quantum_info.utils.hellinger_distance`.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        validate (bool, optional): if ``True``, checks if :math:`p` and :math:`q` are proper
            probability distributions. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Hellinger fidelity.

    """
    backend = _check_backend(backend)

    distance = hellinger_distance(prob_dist_p, prob_dist_q, validate, backend=backend)

    return (1 - distance**2) ** 2


def hellinger_shot_error(
    prob_dist_p, prob_dist_q, nshots: int, validate: bool = False, backend=None
):
    """Calculates the Hellinger fidelity error between two discrete probability distributions estimated from finite statistics.

    It is calculated propagating the probability error of each state of the system.
    The complete formula is:

    .. math::
        \\frac{1 - H^{2}(p, q)}{\\sqrt{nshots}} \\, \\sum_{k} \\,
            \\left(\\sqrt{p_{k} \\, (1 - q_{k})} + \\sqrt{q_{k} \\, (1 - p_{k})}\\right)

    where :math:`H(p, q)` is the :func:`qibo.quantum_info.utils.hellinger_distance`,
    and :math:`1 - H^{2}(p, q)` is the square root of the
    :func:`qibo.quantum_info.utils.hellinger_fidelity`.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        nshots (int): number of shots we used to run the circuit to obtain :math:`p` and :math:`q`.
        validate (bool, optional): if ``True``, checks if :math:`p` and :math:`q` are proper
            probability distributions. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Hellinger fidelity error.

    """
    backend = _check_backend(backend)

    if isinstance(prob_dist_p, list):
        prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)

    if isinstance(prob_dist_q, list):
        prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    hellinger_error = hellinger_fidelity(
        prob_dist_p, prob_dist_q, validate=validate, backend=backend
    )
    hellinger_error = np.sqrt(hellinger_error / nshots) * backend.np.sum(
        np.sqrt(prob_dist_q * (1 - prob_dist_p))
        + np.sqrt(prob_dist_p * (1 - prob_dist_q))
    )

    return hellinger_error


def total_variation_distance(
    prob_dist_p, prob_dist_q, validate: bool = False, backend=None
):
    """Calculate the total variation distance between two discrete probability distributions.

    For probabilities :math:`p` and :math:`q`, the total variation distance is defined as

    .. math::
        \\operatorname{TVD}(p, \\, q) = \\frac{1}{2} \\, \\|p - q\\|_{1}
            = \\frac{1}{2} \\, \\sum_{x} \\, \\left|p(x) - q(x)\\right| \\, ,

    where :math:`\\|\\cdot\\|_{1}` detones the :math:`\\ell_{1}`-norm.

    Args:
        prob_dist_p (ndarray or list): discrete probability distribution :math:`p`.
        prob_dist_q (ndarray or list): discrete probability distribution :math:`q`.
        validate (bool, optional): if ``True``, checks if :math:`p` and :math:`q` are proper
            probability distributions. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        float: Total variation distance measure.
    """
    backend = _check_backend(backend)

    if isinstance(prob_dist_p, list):
        prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)

    if isinstance(prob_dist_q, list):
        prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    if validate:
        if (any(prob_dist_p < 0) or any(prob_dist_p > 1.0)) or (
            any(prob_dist_q < 0) or any(prob_dist_q > 1.0)
        ):
            raise_error(
                ValueError,
                "All elements of the probability array must be between 0. and 1..",
            )
        if backend.np.abs(backend.np.sum(prob_dist_p) - 1.0) > PRECISION_TOL:
            raise_error(ValueError, "First probability array must sum to 1.")

        if backend.np.abs(backend.np.sum(prob_dist_q) - 1.0) > PRECISION_TOL:
            raise_error(ValueError, "Second probability array must sum to 1.")

    tvd = backend.calculate_vector_norm(prob_dist_p - prob_dist_q, order=1)

    return tvd / 2


def haar_integral(
    nqubits: int,
    power_t: int,
    samples: Optional[int] = None,
    backend=None,
):
    """Returns the integral over pure states over the Haar measure.

    .. math::
        \\int_{\\text{Haar}} d\\psi \\, \\left(|\\psi\\rangle\\right.\\left.
            \\langle\\psi|\\right)^{\\otimes t}

    Args:
        nqubits (int): Number of qubits.
        power_t (int): power that defines the :math:`t`-design.
        samples (int, optional): If ``None``, estimated the integral exactly.
            Otherwise, number of samples to estimate the integral via sampling.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        array: Estimation of the Haar integral.

    .. note::
        The ``exact=True`` method is implemented using Lemma 34 of
        `Kliesch and Roth (2020) <https://arxiv.org/abs/2010.05925>`_.
    """

    if isinstance(nqubits, int) is False:
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if isinstance(power_t, int) is False:
        raise_error(
            TypeError, f"power_t must be type int, but it is type {type(power_t)}."
        )

    if samples is not None and isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    backend = _check_backend(backend)

    dim = 2**nqubits

    if samples is not None:
        from qibo.quantum_info.random_ensembles import (  # pylint: disable=C0415
            random_statevector,
        )

        rand_unit_density = np.zeros((dim**power_t, dim**power_t), dtype=complex)
        rand_unit_density = backend.cast(
            rand_unit_density, dtype=rand_unit_density.dtype
        )
        for _ in range(samples):
            haar_state = backend.np.reshape(
                random_statevector(dim, backend=backend), (-1, 1)
            )

            rho = haar_state @ backend.np.conj(haar_state).T

            rand_unit_density = rand_unit_density + reduce(
                backend.np.kron, [rho] * power_t
            )

        integral = rand_unit_density / samples

        return integral

    normalization = factorial(dim - 1) / factorial(dim - 1 + power_t)

    permutations_list = list(permutations(np.arange(power_t) + power_t))
    permutations_list = [
        tuple(np.arange(power_t)) + indices for indices in permutations_list
    ]

    identity = np.eye(dim**power_t, dtype=float)
    identity = np.reshape(identity, (dim,) * (2 * power_t))
    identity = backend.cast(identity, dtype=identity.dtype)

    integral = np.zeros((dim**power_t, dim**power_t), dtype=float)
    integral = backend.cast(integral, dtype=integral.dtype)
    for indices in permutations_list:
        integral = integral + backend.np.reshape(
            backend.np.transpose(identity, indices), (-1, dim**power_t)
        )
    integral = integral * normalization

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
            the current backend. Defaults to ``None``.

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

    backend = _check_backend(backend)

    circuit.density_matrix = True
    dim = 2**circuit.nqubits

    rand_unit_density = np.zeros((dim**power_t, dim**power_t), dtype=complex)
    rand_unit_density = backend.cast(rand_unit_density, dtype=rand_unit_density.dtype)
    for _ in range(samples):
        params = np.random.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        circuit.set_parameters(params)

        rho = backend.execute_circuit(circuit).state()

        rand_unit_density = rand_unit_density + reduce(np.kron, [rho] * power_t)

    integral = rand_unit_density / samples

    return integral


def _hadamard_transform_1d(array, backend=None):
    # necessary because of tf.EagerTensor
    # does not accept item assignment
    backend = _check_backend(backend)
    array_copied = backend.np.copy(array)

    indexes = [2**k for k in range(int(np.log2(len(array_copied))))]
    for index in indexes:
        for k in range(0, len(array_copied), 2 * index):
            for j in range(k, k + index):
                # copy necessary because of cupy backend
                elem_1 = backend.np.copy(array_copied[j])
                elem_2 = backend.np.copy(array_copied[j + index])
                array_copied[j] = elem_1 + elem_2
                array_copied[j + index] = elem_1 - elem_2
        array_copied /= 2.0

    return array_copied
