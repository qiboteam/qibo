from functools import reduce

import numpy as np

from qibo import gates
from qibo.config import MAX_ITERATIONS, PRECISION_TOL, raise_error
from qibo.models import Circuit
from qibo.quantum_info.utils import ONEQUBIT_CLIFFORD_PARAMS


def random_gaussian_matrix(
    dims: int,
    rank: int = None,
    mean: float = 0,
    stddev: float = 1,
    seed=None,
):
    """Generates a random Gaussian Matrix.

    Gaussian matrices are matrices where each entry is
    sampled from a Gaussian probability distribution

    .. math::
        p(x) = \\frac{1}{\\sqrt{2 \\, \\pi} \\, \\sigma} \\, \\exp{\\left(-\\frac{(x - \\mu)^{2}}{2\\,\\sigma^{2}}\\right)}

    with mean :math:`\\mu` and standard deviation :math:`\\sigma`.

    Args:
        dims (int): dimension of the matrix.
        rank (int, optional): rank of the matrix. If ``None``, then ``rank == dims``. Default: ``None``.
        mean (float, optional): mean of the Gaussian distribution. Default is 0.
        stddev (float, optional): standard deviation of the Gaussian distribution. Default is 1.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): Random Gaussian matrix with dimensions ``(dims, rank)``.

    """

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

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
        raise_error(ValueError, f"stddev must be a positive float.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, f"seed must be either type int or numpy.random.Generator."
        )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    dims = (dims, rank)

    matrix = local_state.normal(
        loc=mean, scale=stddev, size=dims
    ) + 1.0j * local_state.normal(loc=mean, scale=stddev, size=dims)

    return matrix


def random_hermitian(
    dims: int, semidefinite: bool = False, normalize: bool = False, seed=None
):
    """Generates a random Hermitian matrix :math:`H`, i.e.
    a random matrix such that :math:`H = H^{\\dagger}.`

    Args:
        dims (int): dimension of the matrix.
        semidefinite (bool, optional): if ``True``, returns a Hermitian matrix that
            is also positive semidefinite. Default: ``False``.
        normalize (bool, optional): if ``True`` and ``semidefinite=False``, returns
            a Hermitian matrix with eigenvalues in the interval
            :math:`[-1, \\,1]`. If ``True`` and ``semidefinite=True``,
            interval is :math:`[0,\\,1]`. Default: ``False``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): Hermitian matrix :math:`H` with dimensions ``(dims, dims)``.

    """

    if dims <= 0:
        raise_error(ValueError, f"dims ({dims}) must be type int and positive.")

    if not isinstance(semidefinite, bool) or not isinstance(normalize, bool):
        raise_error(TypeError, f"semidefinite and normalize must be type bool.")

    matrix = random_gaussian_matrix(dims, dims, seed=seed)

    if semidefinite:
        matrix = np.dot(np.transpose(np.conj(matrix)), matrix)
    else:
        matrix = (matrix + np.transpose(np.conj(matrix))) / 2

    if normalize:
        matrix = matrix / np.linalg.norm(matrix)

    return matrix


def random_unitary(dims: int, measure: str = None, seed=None):
    """Returns a random Unitary operator :math:`U`,, i.e.
    a random operator such that :math:`U^{-1} = U^{\\dagger}`.

    Args:
        dims (int): dimension of the matrix.
        measure (str, optional): probability measure in which to sample the unitary from.
            If ``None``, functions returns :math:`\\exp{(-i \\, H)}`, where :math:`H`
            is a Hermitian operator. If ``"haar"``, returns an Unitary matrix
            sampled from the Haar measure. Default: ``None``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): Unitary matrix :math:`U` with dimensions ``(dims, dims)``.

    """

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if measure is not None:
        if not isinstance(measure, str):
            raise_error(
                TypeError, f"measure must be type str but it is type {type(measure)}."
            )
        if measure != "haar":
            raise_error(ValueError, f"measure {measure} not implemented.")

    if measure == "haar":
        gaussian_matrix = random_gaussian_matrix(dims, dims, seed=seed)

        Q, R = np.linalg.qr(gaussian_matrix)
        D = np.diag(R)
        D = D / np.abs(D)
        R = np.diag(D)
        unitary = np.dot(Q, R)
    elif measure is None:
        from scipy.linalg import expm

        H = random_hermitian(dims, seed=seed)
        unitary = expm(-1.0j * H / 2)

    return unitary


def random_statevector(dims: int, haar: bool = False, seed=None):
    """Creates a random statevector :math:`\\ket{\\psi}`.

    .. math::
        \\ket{\\psi} = \\sum_{k = 0}^{d - 1} \\, \\sqrt{p_{k}} \\, e^{i \\phi_{k}} \\, \\ket{k} \\, ,

    where :math:`d` is ``dims``, and :math:`p_{k}` and :math:`\\phi_{k}` are, respectively,
    the probability and phase corresponding to the computational basis state :math:`\\ket{k}`.

    Args:
        dims (int): dimension of the matrix.
        haar (bool, optional): if ``True``, statevector is created by sampling a Haar random unitary
            :math:`U_{\\text{haar}}` and acting with it on a random computational basis state
            :math:`\\ket{k}`, i.e. :math:`\\ket{\\psi} = U_{\\text{haar}} \\ket{k}`. Default: ``False``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): Random statevector :math:`\\ket{\\psi}`.

    """

    if dims <= 0:
        raise_error(ValueError, "dim must be of type int and >= 1")

    if not isinstance(haar, bool):
        raise_error(TypeError, f"haar must be type bool, but it is type {type(haar)}.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, f"seed must be either type int or numpy.random.Generator."
        )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    if not haar:
        probabilities = local_state.random(dims)
        probabilities = probabilities / np.sum(probabilities)
        phases = 2 * np.pi * local_state.random(dims)
        state = np.sqrt(probabilities) * np.exp(1.0j * phases)
    else:
        # select a random column of a haar random unitary
        k = local_state.integers(low=0, high=dims)
        state = random_unitary(dims, measure="haar", seed=seed)[:, k]

    return state


def random_density_matrix(
    dims,
    rank: int = None,
    pure: bool = False,
    metric: str = "Hilbert-Schmidt",
    seed=None,
):
    """Creates a random density matrix :math:`\\rho`.

    Args:
        dims (int): dimension of the matrix.
        rank (int, optional): rank of the matrix. If ``None``, then ``rank == dims``. Default: ``None``.
        pure (bool, optional): if ``True``, returns a pure state. Default: ``False``.
        metric (str, optional): metric to sample the density matrix from. Options:
            ``"Hilbert-Schmidt"`` and ``"Bures"``. Default: ``"Hilbert-Schmidt"``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): Random density matrix :math:`\\rho`.

    """

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if rank is not None and rank > dims:
        raise_error(ValueError, f"rank ({rank}) cannot be greater than dims ({dims}).")

    if rank is not None and rank <= 0:
        raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")

    if not isinstance(pure, bool):
        raise_error(TypeError, f"pure must be type bool, but it is type {type(pure)}.")

    if not isinstance(metric, str):
        raise_error(
            TypeError, f"metric must be type str, but it is type {type(metric)}."
        )

    if pure:
        state = random_statevector(dims, seed=seed)
        state = np.outer(state, np.transpose(np.conj(state)))
    else:
        if metric == "Hilbert-Schmidt":
            state = random_gaussian_matrix(dims, rank, seed=seed)
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        elif metric == "Bures":
            state = np.eye(dims) + random_unitary(dims, seed=seed)
            state = np.dot(state, random_gaussian_matrix(dims, rank, seed=seed))
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        else:
            raise_error(ValueError, f"metric {metric} not found.")

    return state


def _clifford_unitary(phase, x, y, z):
    """Returns a parametrized single-qubit Clifford gate,
    where possible parameters are defined in
    ``qibo.quantum_info.utils.ONEQUBIT_CLIFFORD_PARAMS``.

    Args:
        phase (float) : An angle.
        x (float) : prefactor.
        y (float) : prefactor.
        z (float) : prefactor.

    Returns:
        (ndarray): Clifford unitary with dimensions (2, 2).

    """

    return np.array(
        [
            [
                np.cos(phase / 2) - 1.0j * z * np.sin(phase / 2),
                -y * np.sin(phase / 2) - 1.0j * x * np.sin(phase / 2),
            ],
            [
                y * np.sin(phase / 2) - 1.0j * x * np.sin(phase / 2),
                np.cos(phase / 2) + 1.0j * z * np.sin(phase / 2),
            ],
        ]
    )


def random_clifford(
    qubits, return_circuit: bool = False, fuse: bool = False, seed=None
):
    """Generates random Clifford operator(s).

    Args:
        qubits (int or list or ndarray): if ``int``, the number of qubits for the Clifford.
            If ``list`` or ``ndarray``, indexes of the qubits for the Clifford to act on.
        return_circuit (bool, optional): if ``True``, returns a ``qibo.gates.Unitary`` object.
            If ``False``, returns an ``ndarray`` object. Default: ``False``.
        fuse (bool, optional): if ``False``, returns an ``ndarray`` with one Clifford gate per qubit.
            If ``True``, returns the tensor product of the Clifford gates that were
            sampled. Default: ``False``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray or ``qibo.gates.Unitary``): Random Clifford operator(s).

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

    if isinstance(qubits, int) and qubits <= 0:
        raise_error(ValueError, f"qubits must be a positive integer.")

    if (isinstance(qubits, list) or isinstance(qubits, np.ndarray)) and any(
        q < 0 for q in qubits
    ):
        raise_error(ValueError, f"qubit indexes must be non-negative integers.")

    if not isinstance(return_circuit, bool):
        raise_error(
            TypeError,
            f"return_circuit must be type bool, but it is type {type(return_circuit)}.",
        )

    if not isinstance(fuse, bool):
        raise_error(TypeError, f"fuse must be type bool, but it is type {type(fuse)}.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, f"seed must be either type int or numpy.random.Generator."
        )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    if isinstance(qubits, int):
        qubits = range(qubits)

    parameters = local_state.integers(0, len(ONEQUBIT_CLIFFORD_PARAMS), len(qubits))

    unitaries = [_clifford_unitary(*ONEQUBIT_CLIFFORD_PARAMS[p]) for p in parameters]

    if return_circuit == True:
        # tensor product of all gates generated
        unitaries = reduce(np.kron, unitaries)

        unitaries = gates.Unitary(unitaries, *qubits)
    else:
        if len(unitaries) == 1:
            unitaries = unitaries[0]
        elif fuse:
            unitaries = reduce(np.kron, unitaries)
        elif not fuse:
            unitaries = np.array(unitaries)

    return unitaries


def random_pauli(
    qubits,
    depth: int,
    max_qubits: int = None,
    subset: list = None,
    return_circuit: bool = True,
    seed=None,
):
    """Creates random Pauli operators.

    Pauli operators are sampled from the single-qubit Pauli set :math:`\\{I, X, Y, Z\\}`.

    Args:
        qubits (int or list or ndarray): if ``int`` and ``max_qubits=None``, the number of qubits.
            If ``int`` and ``max_qubits != None``, qubit index in which the Pauli sequence will act.
            If ``list`` or ``ndarray``, indexes of the qubits for the Pauli sequence to act.
        depth (int): length of the sequence of Pauli gates.
        max_qubits (int, optional): total number of qubits in the circuit. If ``None``,
            ``max_qubits = max(qubits)``. Default: ``None``.
        subset (list, optional): list containing a subset of the 4 single-qubit Pauli
            operators. If ``None``, defaults to the complete set. Default: ``None``.
        return_circuit (bool, optional): if ``True``, returns a ``qibo.models.Circuit`` object.
            If ``False``, returns an ``ndarray`` with shape (qubits, depth, 2, 2) that contains
            all Pauli matrices that were sampled. Default: ``True``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray or ``qibo.models.Circuit``): all sampled Pauli operators.

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
        raise_error(ValueError, f"qubits must be a non-negative integer.")

    if not isinstance(qubits, int) and any(q < 0 for q in qubits):
        raise_error(ValueError, f"qubit indexes must be non-negative integers.")

    if isinstance(depth, int) and depth <= 0:
        raise_error(ValueError, f"depth must be a positive integer.")

    if isinstance(max_qubits, int) and max_qubits <= 0:
        raise_error(ValueError, f"max_qubits must be a positive integer.")

    if max_qubits is not None:
        if isinstance(qubits, int) and qubits >= max_qubits:
            raise_error(
                ValueError,
                f"qubit index ({qubits}) must be < max_qubits ({max_qubits}).",
            )
        elif not isinstance(qubits, int) and any(q >= max_qubits for q in qubits):
            raise_error(ValueError, f"all qubit indexes must be < max_qubits.")

    if not isinstance(return_circuit, bool):
        raise_error(
            TypeError,
            f"return_circuit must be type bool, but it is type {type(return_circuit)}.",
        )

    if subset is not None and not isinstance(subset, list):
        raise_error(
            TypeError, f"subset must be type list, but it is type {type(subset)}."
        )

    if subset is not None and any(isinstance(item, str) == False for item in subset):
        raise_error(
            TypeError,
            f"subset argument must be a subset of strings in the set ['I', 'X', 'Y', 'Z'].",
        )

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, f"seed must be either type int or numpy.random.Generator."
        )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    complete_set = {"I": gates.I, "X": gates.X, "Y": gates.Y, "Z": gates.Z}

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
        gate_grid = Circuit(max_qubits)
        for qubit, row in zip(qubits, indexes):
            for column_item in row:
                gate_grid.add(subset[column_item](qubit))
    else:
        gate_grid = np.array(
            [
                [subset[column_item](qubit).matrix for column_item in row]
                for qubit, row in zip(qubits, indexes)
            ]
        )

    return gate_grid


def random_stochastic_matrix(
    dims: int,
    bistochastic: bool = False,
    precision_tol: float = None,
    max_iterations: int = None,
    seed=None,
):
    """Creates a random stochastic matrix.

    Args:
        dims (int): dimension of the matrix.
        bistochastic (bool, optional): if ``True``, matrix is row- and column-stochastic.
            If ``False``, matrix is row-stochastic. Default: ``False``.
        precision_tol (float, optional): tolerance level for how much each probability
            distribution can deviate from summing up to ``1.0``. If ``None``,
            it defaults to ``qibo.config.PRECISION_TOL``. Default: ``None``.
        max_iterations (int, optional): when ``bistochastic=True``, maximum number of iterations
            used to normalize all rows and columns simultaneously. If ``None``,
            defaults to ``qibo.config.MAX_ITERATIONS``. Default: ``None``.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of random numbers
            or a fixed seed to initialize a generator. If ``None``, initializes a generator with
            a random seed. Default: ``None``.

    Returns:
        (ndarray): a random stochastic matrix.

    """
    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if not isinstance(bistochastic, bool):
        raise_error(
            TypeError,
            f"bistochastic must be type bool, but it is type {type(bistochastic)}.",
        )

    if precision_tol is not None:
        if not isinstance(precision_tol, float):
            raise_error(
                TypeError,
                f"precision_tol must be type float, but it is type {type(precision_tol)}.",
            )
        if precision_tol < 0.0:
            raise_error(ValueError, f"precision_tol must be non-negative.")

    if max_iterations is not None:
        if not isinstance(max_iterations, int):
            raise_error(
                TypeError,
                f"max_iterations must be type int, but it is type {type(precision_tol)}.",
            )
        if max_iterations <= 0.0:
            raise_error(ValueError, f"max_iterations must be a positive int.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, f"seed must be either type int or numpy.random.Generator."
        )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    if precision_tol is None:
        precision_tol = PRECISION_TOL
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    matrix = local_state.random(size=(dims, dims))
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
            import warnings

            warnings.warn("Reached max iterations.", RuntimeWarning)
    else:
        matrix = matrix / np.outer(row_sum, [1] * dims)

    return matrix
