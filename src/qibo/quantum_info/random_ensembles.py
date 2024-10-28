"""Module with functions that create random quantum and classical objects."""

import math
import warnings
from functools import cache
from typing import Optional, Union

import numpy as np
from scipy.stats import rv_continuous

from qibo import Circuit, gates, matrices
from qibo.backends import NumpyBackend, _check_backend_and_local_state
from qibo.config import MAX_ITERATIONS, PRECISION_TOL, raise_error
from qibo.quantum_info.basis import comp_basis_to_pauli
from qibo.quantum_info.superoperator_transformations import (
    choi_to_chi,
    choi_to_kraus,
    choi_to_liouville,
    choi_to_pauli,
    choi_to_stinespring,
    vectorization,
)


class _ProbabilityDistributionGaussianLoader(rv_continuous):
    """Probability density function for sampling phases of
    the RBS gates as a function of circuit depth."""

    def _pdf(self, theta: float, depth: int):
        amplitude = 2 * math.gamma(2 ** (depth - 1)) / math.gamma(2 ** (depth - 2)) ** 2

        probability = abs(math.sin(theta) * math.cos(theta)) ** (2 ** (depth - 1) - 1)

        return amplitude * probability / 4


class _probability_distribution_sin(rv_continuous):  # pragma: no cover
    def _pdf(self, theta: float):
        return 0.5 * np.sin(theta)

    def _cdf(self, theta: float):
        return np.sin(theta / 2) ** 2

    def _ppf(self, theta: float):
        return 2 * np.arcsin(np.sqrt(theta))


def uniform_sampling_U3(ngates: int, seed=None, backend=None):
    """Samples parameters for Haar-random :class:`qibo.gates.U3`.

    Args:
        ngates (int): Total number of :math:`U_{3}` gates to be sampled.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: array of shape (``ngates``, :math:`3`).
    """
    if not isinstance(ngates, int):
        raise_error(
            TypeError, f"ngates must be type int, but it is type {type(ngates)}."
        )
    elif ngates <= 0:
        raise_error(ValueError, f"ngates must be non-negative, but it is {ngates}.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    sampler = _probability_distribution_sin(a=0, b=np.pi, seed=local_state)
    phases = local_state.random((ngates, 3))
    phases[:, 0] = sampler.rvs(size=len(phases[:, 0]))
    phases[:, 1] = phases[:, 1] * 2 * np.pi
    phases[:, 2] = phases[:, 2] * 2 * np.pi

    phases = backend.cast(phases, dtype=phases.dtype)

    return phases


def random_gaussian_matrix(
    dims: int,
    rank: Optional[int] = None,
    mean: float = 0,
    stddev: float = 1,
    seed=None,
    backend=None,
):
    """Generates a random Gaussian Matrix.

    Gaussian matrices are matrices where each entry is
    sampled from a Gaussian probability distribution

    .. math::"haar",
        p(x) = \\frac{1}{\\sqrt{2 \\, \\pi} \\, \\sigma} \\,
            \\exp{\\left(-\\frac{(x - \\mu)^{2}}{2\\,\\sigma^{2}}\\right)}

    with mean :math:`\\mu` and standard deviation :math:`\\sigma`.

    Args:
        dims (int): dimension of the matrix.
        rank (int, optional): rank of the matrix. If ``None``, then
            ``rank == dims``. Default: ``None``.
        mean (float, optional): mean of the Gaussian distribution. Defaults to 0.
        stddev (float, optional): standard deviation of the Gaussian distribution.
            Defaults to ``1``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Random Gaussian matrix with dimensions ``(dims, rank)``.
    """

    if dims <= 0:
        raise_error(ValueError, "dims must be type int and positive.")

    if rank is None:
        rank = dims
    else:
        if rank > dims:
            raise_error(
                ValueError, f"rank ({rank}) cannot be greater than dims ({dims})."
            )
        elif rank <= 0:
            raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")

    if stddev is not None and stddev <= 0.0:
        raise_error(ValueError, "stddev must be a positive float.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    dims = (dims, rank)

    matrix = 1.0j * local_state.normal(loc=mean, scale=stddev, size=dims)
    matrix += local_state.normal(loc=mean, scale=stddev, size=dims)
    matrix = backend.cast(matrix, dtype=matrix.dtype)

    return matrix


def random_hermitian(
    dims: int,
    semidefinite: bool = False,
    normalize: bool = False,
    seed=None,
    backend=None,
):
    """Generates a random Hermitian matrix :math:`H`, i.e.
    a random matrix such that :math:`H = H^{\\dagger}.`

    Args:
        dims (int): dimension of the matrix.
        semidefinite (bool, optional): if ``True``, returns a Hermitian matrix that
            is also positive semidefinite. Defaults to ``False``.
        normalize (bool, optional): if ``True`` and ``semidefinite=False``, returns
            a Hermitian matrix with eigenvalues in the interval
            :math:`[-1, \\, 1]`. If ``True`` and ``semidefinite=True``,
            interval is :math:`[0, \\, 1]`. Defaults to ``False``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Hermitian matrix :math:`H` with dimensions ``(dims, dims)``.
    """

    if dims <= 0:
        raise_error(ValueError, f"dims ({dims}) must be type int and positive.")

    if not isinstance(semidefinite, bool) or not isinstance(normalize, bool):
        raise_error(TypeError, "semidefinite and normalize must be type bool.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    matrix = random_gaussian_matrix(dims, dims, seed=local_state, backend=backend)

    if semidefinite:
        matrix = backend.np.matmul(backend.np.conj(matrix).T, matrix)
    else:
        matrix = (matrix + backend.np.conj(matrix).T) / 2

    if normalize:
        matrix = matrix / np.linalg.norm(backend.to_numpy(matrix))

    return matrix


def random_unitary(dims: int, measure: Optional[str] = None, seed=None, backend=None):
    """Returns a random Unitary operator :math:`U`, i.e.
    a random operator such that :math:`U^{-1} = U^{\\dagger}`.

    Args:
        dims (int): dimension of the matrix.
        measure (str, optional): probability measure in which to sample the unitary
            from. If ``None``, functions returns :math:`\\exp{(-i \\, H)}`, where
            :math:`H` is a Hermitian operator. If ``"haar"``, returns an Unitary
            matrix sampled from the Haar measure. Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Unitary matrix :math:`U` with dimensions ``(dims, dims)``.
    """

    if dims <= 0:
        raise_error(ValueError, "dims must be type int and positive.")

    if measure is not None:
        if not isinstance(measure, str):
            raise_error(
                TypeError, f"measure must be type str but it is type {type(measure)}."
            )
        if measure != "haar":
            raise_error(ValueError, f"measure {measure} not implemented.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    if measure == "haar":
        unitary = random_gaussian_matrix(dims, dims, seed=local_state, backend=backend)
        # Tensorflow experi
        Q, R = backend.np.linalg.qr(unitary)
        D = backend.np.diag(R)
        D = D / backend.np.abs(D)
        R = backend.np.diag(D)
        unitary = backend.np.matmul(Q, R)
    elif measure is None:
        from scipy.linalg import expm

        H = random_hermitian(dims, seed=seed, backend=NumpyBackend())
        unitary = expm(-1.0j * H / 2)
        unitary = backend.cast(unitary, dtype=unitary.dtype)

    return unitary


def random_quantum_channel(
    dims: int,
    representation: str = "liouville",
    measure: Optional[str] = None,
    rank: Optional[int] = None,
    order: str = "row",
    normalize: bool = False,
    precision_tol: Optional[float] = None,
    validate_cp: bool = True,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    seed=None,
    backend=None,
):
    """Creates a random superoperator from an unitary operator in one of the
    supported superoperator representations.

    Args:
        dims (int): dimension of the :math:`n`-qubit operator, i.e. :math:`\\text{dims}=2^{n}`.
        representation (str, optional): If ``"chi"``, returns a random channel in the
            Chi representation. If ``"choi"``, returns channel in Choi representation.
            If ``"kraus"``, returns Kraus representation of channel. If ``"liouville"``,
            returns Liouville representation. If ``"pauli"``, returns Pauli-Liouville
            representation. If "pauli-<pauli_order>" or "chi-<pauli_order>", (e.g. "pauli-IZXY"),
            returns it in the Pauli basis with the corresponding order of single-qubit Pauli elements
            (see :func:`qibo.quantum_info.pauli_basis`). If ``"stinespring"``,
            returns random channel in the Stinespring representation. Defaults to ``"liouville"``.
        measure (str, optional): probability measure in which to sample the unitary
            from. If ``None``, functions returns :math:`\\exp{(-i \\, H)}`, where
            :math:`H` is a Hermitian operator. If ``"haar"``, returns an Unitary
            matrix sampled from the Haar measure. If ``"bcsz"``, it samples an unitary
            from the BCSZ distribution with Kraus ``rank``. Defaults to ``None``.
        rank (int, optional): used when ``measure=="bcsz"``. Rank of the matrix.
            If ``None``, then ``rank==dims``. Defaults to ``None``.
        order (str, optional): If ``"row"``, vectorization is performed row-wise.
            If ``"column"``, vectorization is performed column-wise. If ``"system"``,
            a block-vectorization is performed. Defaults to ``"row"``.
        normalize (bool, optional): used when ``representation="chi"`` or
            ``representation="pauli"``. If ``True`` assumes the normalized Pauli basis.
            If ``False``, it assumes unnormalized Pauli basis. Defaults to ``False``.
        precision_tol (float, optional): if ``representation="kraus"``, it is the
            precision tolerance for eigenvalues found in the spectral decomposition
            problem. Any eigenvalue :math:`\\lambda <` ``precision_tol`` is set
            to 0 (zero). If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        validate_cp (bool, optional): used when ``representation="stinespring"``.
            If ``True``, checks if the Choi representation of superoperator
            used as intermediate step is a completely positive map.
            If ``False``, it assumes that it is completely positive (and Hermitian).
            Defaults to ``True``.
        nqubits (int, optional): used when ``representation="stinespring"``.
            Total number of qubits in the system that is interacting with
            the environment. Must be equal or greater than the number of
            qubits that Kraus representation of the system superoperator acts on.
            If ``None``, defaults to the number of qubits in the Kraus operators.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): used when ``representation="stinespring"``.
            Statevector representing the initial state of the enviroment.
            If ``None``, it assumes the environment in its ground state.
            Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Superoperator representation of a random unitary gate.
    """
    if not isinstance(representation, str):
        raise_error(
            TypeError,
            f"representation must be type str, but it is type {type(representation)}",
        )

    if representation not in [
        "chi",
        "choi",
        "kraus",
        "liouville",
        "pauli",
        "stinespring",
    ]:
        if (
            ("chi-" not in representation and "pauli-" not in representation)
            or len(representation.split("-")) != 2
            or set(representation.split("-")[1]) != {"I", "X", "Y", "Z"}
        ):
            raise_error(ValueError, f"representation {representation} not implemented.")

    if measure == "bcsz" and order not in ["row", "column"]:
        raise_error(
            NotImplementedError, f"order {order} not implemented for measure {measure}."
        )

    backend, local_state = _check_backend_and_local_state(seed, backend)

    if measure == "bcsz":
        super_op = _super_op_from_bcsz_measure(
            dims=dims, rank=rank, order=order, seed=local_state, backend=backend
        )
    else:
        super_op = random_unitary(dims, measure, local_state, backend)
        super_op = vectorization(super_op, order=order, backend=backend)
        super_op = backend.np.outer(super_op, backend.np.conj(super_op))

    if "chi" in representation:
        pauli_order = "IXYZ"
        if "-" in representation:
            pauli_order = representation.split("-")[1]
        super_op = choi_to_chi(
            super_op,
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )
    elif representation == "kraus":
        super_op = choi_to_kraus(
            super_op,
            precision_tol=precision_tol,
            order=order,
            validate_cp=False,
            backend=backend,
        )
    elif representation == "liouville":
        super_op = choi_to_liouville(super_op, order=order, backend=backend)
    elif "pauli" in representation:
        pauli_order = "IXYZ"
        if "-" in representation:
            pauli_order = representation.split("-")[1]
        super_op = choi_to_pauli(
            super_op,
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )
    elif representation == "stinespring":
        super_op = choi_to_stinespring(
            super_op,
            precision_tol=precision_tol,
            order=order,
            validate_cp=validate_cp,
            nqubits=nqubits,
            initial_state_env=initial_state_env,
            backend=backend,
        )

    return super_op


def random_statevector(dims: int, seed=None, backend=None):
    """Creates a random statevector :math:`\\ket{\\psi}`.

    .. math::
        \\ket{\\psi} = \\sum_{k = 0}^{d - 1} \\, \\sqrt{p_{k}} \\,
            e^{i \\phi_{k}} \\, \\ket{k} \\, ,

    where :math:`d` is ``dims``, and :math:`p_{k}` and :math:`\\phi_{k}` are, respectively,
    the probability and phase corresponding to the computational basis state :math:`\\ket{k}`.

    Args:
        dims (int): dimension of the matrix.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Random statevector :math:`\\ket{\\psi}`.
    """

    if dims <= 0:
        raise_error(ValueError, "dim must be of type int and >= 1")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    backend, local_state = _check_backend_and_local_state(seed, backend)

    state = backend.cast(local_state.standard_normal(dims).astype(complex))
    state = state + 1.0j * backend.cast(local_state.standard_normal(dims))
    state = state / backend.np.linalg.norm(state)

    return state


def random_density_matrix(
    dims: int,
    rank: Optional[int] = None,
    pure: bool = False,
    metric: str = "hilbert-schmidt",
    basis: Optional[str] = None,
    normalize: bool = False,
    order: str = "row",
    seed=None,
    backend=None,
):
    """Creates a random density matrix :math:`\\rho`. If ``pure=True``,

    .. math::
        \\rho = \\ketbra{\\psi}{\\psi} \\, ,

    where :math:`\\ket{\\psi}` is a :func:`qibo.quantum_info.random_statevector`.
    If ``pure=False``, then

    .. math::
        \\rho = \\sum_{k} \\, p_{k} \\, \\ketbra{\\psi_{k}}{\\psi_{k}} \\, .

    is a mixed state.

    Args:
        dims (int): dimension of the matrix.
        rank (int, optional): rank of the matrix. If ``None``, then ``rank == dims``.
            Defaults to ``None``.
        pure (bool, optional): if ``True``, returns a pure state. Defaults to ``False``.
        metric (str, optional): metric to sample the density matrix from. Options:
            ``"hilbert-schmidt"``, ``"ginibre"``, and ``"bures"``.
            Note that, by definition, ``rank`` defaults to ``None``
            when ``metric=="hilbert-schmidt"``. Defaults to ``"hilbert-schmidt"``.
        basis (str, optional): if ``None``, returns random density matrix in the
            computational basis. If ``"pauli-<pauli_order>"``, (e.g. ``"pauli-IZXY"``),
            returns it in the Pauli basis with the corresponding order of single-qubit
            Pauli elements (see :func:`qibo.quantum_info.pauli_basis`).
            Defaults to ``None``.
        normalize(bool, optional): used when ``basis="pauli-<pauli-order>"``. If ``True``
            returns random density matrix in the normalized Pauli basis. If ``False``,
            returns state in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): used when ``basis="pauli-<pauli-order>"``. If ``"row"``,
            vectorization of Pauli basis is performed row-wise. If ``"column"``,
            vectorization is performed column-wise. If ``"system"``, system-wise
            vectorization is performed. Default is ``"row"``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: Random density matrix :math:`\\rho`.
    """

    if dims <= 0:
        raise_error(ValueError, "dims must be type int and positive.")

    if rank is not None and rank > dims:
        raise_error(ValueError, f"rank ({rank}) cannot be greater than dims ({dims}).")

    if rank is not None and rank <= 0:
        raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")

    if rank is not None and not isinstance(rank, int):
        raise_error(TypeError, f"rank must be type int, but it is type {type(rank)}.")

    if not isinstance(pure, bool):
        raise_error(TypeError, f"pure must be type bool, but it is type {type(pure)}.")

    if not isinstance(metric, str):
        raise_error(
            TypeError, f"metric must be type str, but it is type {type(metric)}."
        )
    if metric not in ["hilbert-schmidt", "ginibre", "bures"]:
        raise_error(ValueError, f"metric {metric} not implemented.")

    if basis is not None and not isinstance(basis, str):
        raise_error(TypeError, f"basis must be type str, but it is type {type(basis)}.")
    elif basis is not None and basis not in ["pauli"]:
        if (
            "pauli-" not in basis
            or len(basis.split("-")) != 2
            or set(basis.split("-")[1]) != {"I", "X", "Y", "Z"}
        ):
            raise_error(ValueError, f"basis {basis} nor recognized.")

    if not isinstance(normalize, bool):
        raise_error(
            TypeError, f"normalize must be type bool, but it is type {type(normalize)}."
        )
    elif normalize is True and basis is None:
        raise_error(ValueError, "normalize cannot be True when basis=None.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    if metric == "hilbert-schmidt":
        rank = None

    if pure:
        state = random_statevector(dims, seed=local_state, backend=backend)
        state = backend.np.outer(state, backend.np.conj(state).T)
    else:
        if metric in ["hilbert-schmidt", "ginibre"]:
            state = random_gaussian_matrix(
                dims, rank, mean=0, stddev=1, seed=local_state, backend=backend
            )
            state = backend.np.matmul(
                state, backend.np.transpose(backend.np.conj(state), (1, 0))
            )
            state = state / backend.np.trace(state)
        else:
            nqubits = int(np.log2(dims))
            state = backend.identity_density_matrix(nqubits, normalize=False)
            state += random_unitary(dims, seed=local_state, backend=backend)
            state = backend.np.matmul(
                state,
                random_gaussian_matrix(dims, rank, seed=local_state, backend=backend),
            )
            state = backend.np.matmul(
                state, backend.np.transpose(backend.np.conj(state), (1, 0))
            )
            state /= backend.np.trace(state)

    state = backend.cast(state, dtype=state.dtype)

    if basis is not None:
        pauli_order = basis.split("-")[1]
        unitary = comp_basis_to_pauli(
            int(np.log2(dims)),
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )
        state = unitary @ vectorization(state, order=order, backend=backend)

    return state


def random_clifford(
    nqubits: int,
    return_circuit: bool = True,
    density_matrix: bool = False,
    seed=None,
    backend=None,
):
    """Generates a random :math:`n`-qubit Clifford operator, where :math:`n` is ``nqubits``.
    For the mathematical details, see Reference [1].

    Args:
        nqubits (int): number of qubits.
        return_circuit (bool, optional): if ``True``, returns a :class:`qibo.models.Circuit`
            object. If ``False``, returns an ``ndarray`` object. Defaults to ``True``.
        density_matrix (bool, optional): used when ``return_circuit=True``. If `True`,
            the circuit would evolve density matrices. Defaults to ``False``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        (ndarray or :class:`qibo.models.Circuit`): Random Clifford operator.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.
    """

    if isinstance(nqubits, int) is False:
        raise_error(
            TypeError,
            f"nqubits must be type int, but it is type {type(nqubits)}.",
        )

    if nqubits <= 0:
        raise_error(ValueError, "nqubits must be a positive integer.")

    if not isinstance(return_circuit, bool):
        raise_error(
            TypeError,
            f"return_circuit must be type bool, but it is type {type(return_circuit)}.",
        )

    backend, local_state = _check_backend_and_local_state(seed, backend)

    hadamards, permutations = _sample_from_quantum_mallows_distribution(
        nqubits, local_state=local_state
    )

    delta_matrix = np.eye(nqubits, dtype=int)
    delta_matrix_prime = np.copy(delta_matrix)

    gamma_matrix_prime = local_state.integers(0, 2, size=nqubits)
    gamma_matrix_prime = np.diag(gamma_matrix_prime)

    gamma_matrix = local_state.integers(0, 2, size=nqubits)
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = np.diag(gamma_matrix)

    # filling off-diagonal elements of gammas and deltas matrices
    for j in range(nqubits):
        for k in range(j + 1, nqubits):
            b = local_state.integers(0, 2)
            gamma_matrix_prime[k, j] = b
            gamma_matrix_prime[j, k] = b

            b = local_state.integers(0, 2)
            delta_matrix_prime[k, j] = b

            if hadamards[k] == 1 and hadamards[j] == 1:  # pragma: no cover
                b = local_state.integers(0, 2)
                gamma_matrix[k, j] = b
                gamma_matrix[j, k] = b
                if permutations[k] > permutations[j]:
                    b = local_state.integers(0, 2)
                    delta_matrix[k, j] = b

            if hadamards[k] == 0 and hadamards[j] == 1:
                b = local_state.integers(0, 2)
                delta_matrix[k, j] = b
                if permutations[k] > permutations[j]:
                    b = local_state.integers(0, 2)
                    gamma_matrix[k, j] = b
                    gamma_matrix[j, k] = b

            if (
                hadamards[k] == 1
                and hadamards[j] == 0
                and permutations[k] < permutations[j]
            ):  # pragma: no cover
                b = local_state.integers(0, 2)
                gamma_matrix[k, j] = b
                gamma_matrix[j, k] = b

            if (
                hadamards[k] == 0
                and hadamards[j] == 0
                and permutations[k] < permutations[j]
            ):  # pragma: no cover
                b = local_state.integers(0, 2)
                delta_matrix[k, j] = b

    # get first element of the Borel group
    clifford_circuit = _operator_from_hadamard_free_group(
        gamma_matrix, delta_matrix, density_matrix
    )

    # Apply permutated Hadamard layer
    for qubit, had in enumerate(hadamards):
        if had == 1:
            clifford_circuit.add(gates.H(int(permutations[qubit])))

    # get second element of the Borel group
    clifford_circuit += _operator_from_hadamard_free_group(
        gamma_matrix_prime,
        delta_matrix_prime,
        density_matrix,
        random_pauli(
            nqubits,
            depth=1,
            return_circuit=True,
            density_matrix=density_matrix,
            seed=local_state,
            backend=backend,
        ),
    )

    if return_circuit is False:
        clifford_circuit = clifford_circuit.unitary(backend=backend)

    return clifford_circuit


def random_pauli(
    qubits,
    depth: int,
    max_qubits: Optional[int] = None,
    subset: Optional[list] = None,
    return_circuit: bool = True,
    density_matrix: bool = False,
    seed=None,
    backend=None,
):
    """Creates random Pauli operator(s).

    Pauli operators are sampled from the single-qubit Pauli set
    :math:`\\{I, \\, X, \\, Y, \\, Z\\}`.

    Args:
        qubits (int or list or ndarray): if ``int`` and ``max_qubits=None``, the
            number of qubits. If ``int`` and ``max_qubits != None``, qubit index
            in which the Pauli sequence will act. If ``list`` or ``ndarray``,
            indexes of the qubits for the Pauli sequence to act.
        depth (int): length of the sequence of Pauli gates.
        max_qubits (int, optional): total number of qubits in the circuit.
            If ``None``, ``max_qubits = max(qubits)``. Defaults to ``None``.
        subset (list, optional): list containing a subset of the 4 single-qubit
            Pauli operators. If ``None``, defaults to the complete set.
            Defaults to ``None``.
        return_circuit (bool, optional): if ``True``, returns a :class:`qibo.models.Circuit`
            object. If ``False``, returns an ``ndarray`` with shape (qubits, depth, 2, 2)
            that contains all Pauli matrices that were sampled. Defaults to ``True``.
        density_matrix (bool, optional): used when ``return_circuit=True``. If `True`,
            the circuit would evolve density matrices. Defaults to ``False``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        (ndarray or :class:`qibo.models.Circuit`): all sampled Pauli operators.

    """

    if (
        not isinstance(qubits, int)
        and not isinstance(qubits, list)
        and not isinstance(qubits, np.ndarray)
    ):
        raise_error(
            TypeError,
            f"qubits must be either type int, list or ndarray, but it is type {type(qubits)}.",
        )

    if isinstance(qubits, int) and qubits < 0:
        raise_error(ValueError, "qubits must be a non-negative integer.")

    if isinstance(qubits, int) is False and any(q < 0 for q in qubits):
        raise_error(ValueError, "qubit indexes must be non-negative integers.")

    if isinstance(depth, int) and depth <= 0:
        raise_error(ValueError, "depth must be a positive integer.")

    if isinstance(max_qubits, int) and max_qubits <= 0:
        raise_error(ValueError, "max_qubits must be a positive integer.")

    if max_qubits is not None:
        if isinstance(qubits, int) and qubits >= max_qubits:
            raise_error(
                ValueError,
                f"qubit index ({qubits}) must be < max_qubits ({max_qubits}).",
            )
        elif not isinstance(qubits, int) and any(q >= max_qubits for q in qubits):
            raise_error(ValueError, "all qubit indexes must be < max_qubits.")

    if not isinstance(return_circuit, bool):
        raise_error(
            TypeError,
            f"return_circuit must be type bool, but it is type {type(return_circuit)}.",
        )

    if subset is not None and not isinstance(subset, list):
        raise_error(
            TypeError, f"subset must be type list, but it is type {type(subset)}."
        )

    if subset is not None and any(isinstance(item, str) is False for item in subset):
        raise_error(
            TypeError,
            "subset argument must be a subset of strings in the set ['I', 'X', 'Y', 'Z'].",
        )

    backend, local_state = _check_backend_and_local_state(seed, backend)

    complete_set = (
        {"I": gates.I, "X": gates.X, "Y": gates.Y, "Z": gates.Z}
        if return_circuit
        else {"I": matrices.I, "X": matrices.X, "Y": matrices.Y, "Z": matrices.Z}
    )

    if subset is None:
        subset = complete_set
    else:
        subset = {key: complete_set[key] for key in subset}

    keys = list(subset.keys())

    if max_qubits is None:
        if isinstance(qubits, int):
            max_qubits = qubits
            qubits = range(qubits)
        else:
            max_qubits = int(max(qubits)) + 1
    else:
        if isinstance(qubits, int):
            qubits = [qubits]

    indexes = local_state.integers(0, len(subset), size=(len(qubits), depth))
    indexes = [[keys[item] for item in row] for row in indexes]

    if return_circuit:
        gate_grid = Circuit(max_qubits, density_matrix=density_matrix)
        for qubit, row in zip(qubits, indexes):
            for column_item in row:
                if subset[column_item] != gates.I:
                    gate_grid.add(subset[column_item](qubit))
    else:
        gate_grid = backend.cast(
            [[subset[column_item] for column_item in row] for row in indexes]
        )

    return gate_grid


def random_pauli_hamiltonian(
    nqubits: int,
    max_eigenvalue: Optional[Union[int, float]] = None,
    normalize: bool = False,
    pauli_order: str = "IXYZ",
    seed=None,
    backend=None,
):
    """Generates a random Hamiltonian in the Pauli basis.

    Args:
        nqubits (int): number of qubits.
        max_eigenvalue (int or float, optional): fixes the value of the
            largest eigenvalue. Defaults to ``None``.
        normalize (bool, optional): If ``True``, fixes the gap of the
            Hamiltonian as ``1.0``. Moreover, if ``True``, then ``max_eigenvalue``
            must be ``> 1.0``. Defaults to ``False``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements in the basis. Defaults to "IXYZ".
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple(ndarray, ndarray): Hamiltonian in the Pauli basis and its corresponding eigenvalues.
    """
    if isinstance(nqubits, int) is False:
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )
    elif nqubits <= 0:
        raise_error(ValueError, "nqubits must be a positive int.")

    if isinstance(max_eigenvalue, (int, float)) is False and normalize is True:
        raise_error(
            TypeError,
            f"when normalize=True, max_eigenvalue must be type float, "
            + f"but it is {type(max_eigenvalue)}.",
        )
    elif (
        isinstance(max_eigenvalue, (int, float)) is False and max_eigenvalue is not None
    ):
        raise_error(
            TypeError,
            f"max_eigenvalue must be type float, but it is {type(max_eigenvalue)}.",
        )

    if isinstance(normalize, bool) is False:
        raise_error(
            TypeError,
            f"normalize must be type bool, but it is type {type(normalize)}.",
        )
    elif normalize is True and max_eigenvalue <= 1.0:
        raise_error(
            ValueError,
            "when normalize=True, gap is = 1, thus max_eigenvalue must be > 1.",
        )

    backend, local_state = _check_backend_and_local_state(seed, backend)

    d = 2**nqubits

    hamiltonian = random_hermitian(d, normalize=True, seed=local_state, backend=backend)

    eigenvalues, eigenvectors = backend.calculate_eigenvectors(hamiltonian)
    if backend.platform == "tensorflow":
        eigenvalues = backend.to_numpy(eigenvalues)
        eigenvectors = backend.to_numpy(eigenvectors)

    if normalize is True:
        eigenvalues = eigenvalues - eigenvalues[0]

        eigenvalues = eigenvalues / eigenvalues[1]

        shift = 2
        eigenvectors[:, shift:] = (
            eigenvectors[:, shift:] * max_eigenvalue / eigenvalues[-1]
        )
        eigenvalues[shift:] = eigenvalues[shift:] * max_eigenvalue / eigenvalues[-1]

        hamiltonian = np.zeros((d, d), dtype=complex)
        hamiltonian = backend.cast(hamiltonian, dtype=hamiltonian.dtype)
        # excluding the first eigenvector because first eigenvalue is zero
        for eigenvalue, eigenvector in zip(
            eigenvalues[1:], backend.np.transpose(eigenvectors, (1, 0))[1:]
        ):
            hamiltonian = hamiltonian + eigenvalue * backend.np.outer(
                eigenvector, backend.np.conj(eigenvector)
            )

    U = comp_basis_to_pauli(
        nqubits, normalize=True, pauli_order=pauli_order, backend=backend
    )

    hamiltonian = backend.np.real(U @ vectorization(hamiltonian, backend=backend))

    return hamiltonian, eigenvalues


def random_stochastic_matrix(
    dims: int,
    bistochastic: bool = False,
    diagonally_dominant: bool = False,
    precision_tol: Optional[float] = None,
    max_iterations: Optional[int] = None,
    seed=None,
    backend=None,
):
    """Creates a random stochastic matrix.

    Args:
        dims (int): dimension of the matrix.
        bistochastic (bool, optional): if ``True``, matrix is row- and column-stochastic.
            If ``False``, matrix is row-stochastic. Defaults to ``False``.
        diagonally_dominant (bool, optional): if ``True``, matrix is strictly diagonally
            dominant. Defaults to ``False``.
        precision_tol (float, optional): tolerance level for how much each probability
            distribution can deviate from summing up to ``1.0``. If ``None``,
            it defaults to ``qibo.config.PRECISION_TOL``. Defaults to ``None``.
        max_iterations (int, optional): when ``bistochastic=True``, maximum number
            of iterations used to normalize all rows and columns simultaneously.
            If ``None``, defaults to ``qibo.config.MAX_ITERATIONS``.
            Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a statevectorgenerator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ndarray: a random stochastic matrix.

    """
    if dims <= 0:
        raise_error(ValueError, "dims must be type int and positive.")

    if not isinstance(bistochastic, bool):
        raise_error(
            TypeError,
            f"bistochastic must be type bool, but it is type {type(bistochastic)}.",
        )

    if not isinstance(diagonally_dominant, bool):
        raise_error(
            TypeError,
            f"diagonally_dominant must be type bool, but it is type {type(diagonally_dominant)}.",
        )

    if precision_tol is not None:
        if not isinstance(precision_tol, float):
            raise_error(
                TypeError,
                f"precision_tol must be type float, but it is type {type(precision_tol)}.",
            )
        if precision_tol < 0.0:
            raise_error(ValueError, "precision_tol must be non-negative.")

    if max_iterations is not None:
        if not isinstance(max_iterations, int):
            raise_error(
                TypeError,
                f"max_iterations must be type int, but it is type {type(precision_tol)}.",
            )
        if max_iterations <= 0.0:
            raise_error(ValueError, "max_iterations must be a positive int.")

    backend, local_state = _check_backend_and_local_state(seed, backend)

    if precision_tol is None:
        precision_tol = PRECISION_TOL
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    matrix = local_state.random(size=(dims, dims))
    if diagonally_dominant:
        matrix /= dims**2
        for k, row in enumerate(matrix):
            row = np.delete(row, obj=k)
            matrix[k, k] = 1 - np.sum(row)
    row_sum = np.sum(matrix, axis=1)

    row_sum = matrix.sum(axis=1)

    if bistochastic:
        column_sum = matrix.sum(axis=0)
        count = 0
        while count <= max_iterations - 1 and (
            (
                np.any(row_sum >= 1 + precision_tol)
                or np.any(row_sum <= 1 - precision_tol)
            )
            or (
                np.any(column_sum >= 1 + precision_tol)
                or np.any(column_sum <= 1 - precision_tol)
            )
        ):
            matrix = matrix / matrix.sum(axis=0)
            matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
            row_sum = matrix.sum(axis=1)
            column_sum = matrix.sum(axis=0)
            count += 1
        if count == max_iterations:
            warnings.warn("Reached max iterations.", RuntimeWarning)
    else:
        matrix = matrix / np.outer(row_sum, [1] * dims)

    matrix = backend.cast(matrix, dtype=matrix.dtype)

    return matrix


def _sample_from_quantum_mallows_distribution(nqubits: int, local_state):
    """Using the quantum Mallows distribution, samples a binary array
    representing a layer of Hadamard gates as well as an array with permutated
    qubit indexes. For more details, see Reference [1].

    Args:
        nqubits (int): number of qubits.
        local_state (:class:`numpy.random.Generator`): a generator of
            random numbers

    Returns:
        (``ndarray``, ``ndarray`): tuple of binary ``ndarray`` and ``ndarray`` of indexes.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
            structure of the Clifford group*.
            `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.

    """
    mute_index = list(range(nqubits))

    exponents = np.arange(nqubits, 0, -1, dtype=np.int64)
    powers = 4**exponents
    powers[powers == 0] = np.iinfo(np.int64).max

    r = local_state.uniform(0, 1, size=nqubits)

    indexes = -1 * (np.ceil(np.log2(r + (1 - r) / powers)))

    hadamards = 1 * (indexes < exponents)

    permutations = np.zeros(nqubits, dtype=int)
    for l, (index, m) in enumerate(zip(indexes, exponents)):
        k = index if index < m else 2 * m - index - 1
        k = int(k)
        permutations[l] = mute_index[k]
        del mute_index[k]

    return hadamards, permutations


@cache
def _create_S(q):
    return gates.S(int(q))


@cache
def _create_CZ(cq, tq):
    return gates.CZ(int(cq), int(tq))


@cache
def _create_CNOT(cq, tq):
    return gates.CNOT(int(cq), int(tq))


def _operator_from_hadamard_free_group(
    gamma_matrix, delta_matrix, density_matrix: bool = False, pauli_operator=None
):
    """Calculates an element :math:`F` of the Hadamard-free group :math:`\\mathcal{F}_{n}`,
    where :math:`n` is the number of qubits ``nqubits``. For more details,
    see Reference [1].

    Args:
        gamma_matrix (ndarray): :math:`\\, n \\times n \\,` binary matrix.
        delta_matrix (ndarray): :math:`\\, n \\times n \\,` binary matrix.
        density_matrix (bool, optional): used when ``return_circuit=True``. If `True`,
            the circuit would evolve density matrices. Defaults to ``False``.
        pauli_operator (:class:`qibo.models.Circuit`, optional): a :math:`n`-qubit
            Pauli operator. If ``None``, it is assumed to be the Identity.
            Defaults to ``None``.

    Returns:
        :class:`qibo.models.Circuit`: element of the Hadamard-free group.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.
    """
    if gamma_matrix.shape != delta_matrix.shape:  # pragma: no cover
        raise_error(
            ValueError,
            "gamma_matrix and delta_matrix must have shape (nqubits, nqubits), "
            + f"but {gamma_matrix.shape} != {delta_matrix.shape}",
        )

    nqubits = len(gamma_matrix)
    circuit = Circuit(nqubits, density_matrix=density_matrix)

    if pauli_operator is not None:
        circuit += pauli_operator

    idx = np.tril_indices(nqubits, k=-1)
    gamma_ones = gamma_matrix[idx].nonzero()[0]
    delta_ones = delta_matrix[idx].nonzero()[0]

    S_gates = [_create_S(q) for q in np.diag(gamma_matrix).nonzero()[0]]
    CZ_gates = [
        _create_CZ(cq, tq) for cq, tq in zip(idx[1][gamma_ones], idx[0][gamma_ones])
    ]
    CNOT_gates = [
        _create_CNOT(cq, tq) for cq, tq in zip(idx[1][delta_ones], idx[0][delta_ones])
    ]

    circuit.add(S_gates + CZ_gates + CNOT_gates)

    return circuit


def _super_op_from_bcsz_measure(dims: int, rank: int, order: str, seed, backend):
    """Helper function for :func:qibo.quantum_info.random_ensembles.random_quantum_channel.
    Generates a channel from the BCSZ measure.

    Args:
        dims (int): dimension of the :math:`n`-qubit operator, i.e. :math:`\\text{dims}=2^{n}`.
        rank (int, optional): used when ``measure=="bcsz"``. Rank of the matrix.
            If ``None``, then ``rank==dims``. Defaults to ``None``.
        order (str, optional): If ``"row"``, vectorization is performed row-wise.
            If ``"column"``, vectorization is performed column-wise. Defaults to ``"row"``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    nqubits = int(np.log2(dims))

    super_op = random_gaussian_matrix(
        dims**2, rank=rank, mean=0, stddev=1, seed=seed, backend=backend
    )
    super_op = super_op @ backend.np.conj(super_op).T

    # partial trace implemented with einsum
    super_op_reduced = np.einsum(
        "ijik->jk", np.reshape(backend.to_numpy(super_op), (dims,) * 4)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(super_op_reduced)

    eigenvalues = np.sqrt(1.0 / eigenvalues)

    operator = np.zeros((dims, dims), dtype=complex)
    operator = backend.cast(operator, dtype=operator.dtype)
    for eigenvalue, eigenvector in zip(
        backend.cast(eigenvalues), backend.cast(eigenvectors).T
    ):
        operator = operator + eigenvalue * backend.np.outer(
            eigenvector, backend.np.conj(eigenvector)
        )

    if order == "row":
        operator = backend.np.kron(
            backend.identity_density_matrix(nqubits, normalize=False), operator
        )
    if order == "column":
        operator = backend.np.kron(
            operator, backend.identity_density_matrix(nqubits, normalize=False)
        )

    super_op = operator @ super_op @ operator

    return super_op
