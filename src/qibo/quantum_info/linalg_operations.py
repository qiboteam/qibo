"""Module with common linear algebra operations for quantum information."""

import math

from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._linalg_operations import (
    _gram_schmidt_process,
    _lie_closure_matrix,
    _lie_closure_pauli_strings,
    _lie_closure_pauli_sums,
)


def commutator(operator_1: ArrayLike, operator_2: ArrayLike) -> ArrayLike:
    """Returns the commutator of ``operator_1`` and ``operator_2``.

    The commutator of two matrices :math:`A` and :math:`B` is given by

    .. math::
        [A, B] = A \\, B - B \\, A \\,.

    Args:
        operator_1 (ArrayLike): First operator.
        operator_2 (ArrayLike): Second operator.

    Returns:
        ArrayLike: Commutator of ``operator_1`` and ``operator_2``.
    """
    if (
        (len(operator_1.shape) >= 3)
        or (len(operator_1) == 0)
        or (len(operator_1.shape) == 2 and operator_1.shape[0] != operator_1.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_1`` must have shape (k,k), but have shape {operator_1.shape}.",
        )

    if (
        (len(operator_2.shape) >= 3)
        or (len(operator_2) == 0)
        or (len(operator_2.shape) == 2 and operator_2.shape[0] != operator_2.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_2`` must have shape (k,k), but have shape {operator_2.shape}.",
        )

    if operator_1.shape != operator_2.shape:
        raise_error(
            TypeError,
            "``operator_1`` and ``operator_2`` must have the same shape, "
            + f"but {operator_1.shape} != {operator_2.shape}",
        )

    return operator_1 @ operator_2 - operator_2 @ operator_1


def anticommutator(operator_1: ArrayLike, operator_2: ArrayLike) -> ArrayLike:
    """Returns the anticommutator of ``operator_1`` and ``operator_2``.

    The anticommutator of two matrices :math:`A` and :math:`B` is given by

    .. math::
        \\{A, B\\} = A \\, B + B \\, A \\,.

    Args:
        operator_1 (ArrayLike): First operator.
        operator_2 (ArrayLike): Second operator.

    Returns:
        ArrayLike: Anticommutator of ``operator_1`` and ``operator_2``.
    """
    if (
        (len(operator_1.shape) >= 3)
        or (len(operator_1) == 0)
        or (len(operator_1.shape) == 2 and operator_1.shape[0] != operator_1.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_1`` must have shape (k,k), but have shape {operator_1.shape}.",
        )

    if (
        (len(operator_2.shape) >= 3)
        or (len(operator_2) == 0)
        or (len(operator_2.shape) == 2 and operator_2.shape[0] != operator_2.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_2`` must have shape (k,k), but have shape {operator_2.shape}.",
        )

    if operator_1.shape != operator_2.shape:
        raise_error(
            TypeError,
            "``operator_1`` and ``operator_2`` must have the same shape, "
            + f"but {operator_1.shape} != {operator_2.shape}",
        )

    return operator_1 @ operator_2 + operator_2 @ operator_1


def partial_trace(
    state: ArrayLike,
    traced_qubits: list[int] | tuple[int, ...],
    backend: Backend | None = None,
) -> ArrayLike:
    """Return the density matrix resulting from tracing out ``traced_qubits`` from ``state``.

    Total number of qubits is inferred by the shape of ``state``.

    Args:
        state (ArrayLike): density matrix or statevector.
        traced_qubits (list[int] or tuple[int, ...]): indices of qubits to be traced out.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: Density matrix of the remaining qubit(s).
    """
    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"``state`` must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    backend = _check_backend(backend)

    return backend.partial_trace(state, traced_qubits)


def partial_transpose(
    operator: ArrayLike,
    partition: list[int] | tuple[int, ...],
    backend: Backend | None = None,
) -> ArrayLike:
    """Return matrix after the partial transposition of ``partition`` qubits in ``operator``.

    Given a :math:`n`-qubit operator :math:`O \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the partial transpose with respect to ``partition`` :math:`B` is given by

    .. math::
        \\begin{align}
        O^{T_{B}} &= \\sum_{jklm} \\, O_{lm}^{jk} \\, \\ketbra{j}{k} \\otimes
            \\left(\\ketbra{l}{m}\\right)^{T} \\\\
        &= \\sum_{jklm} \\, O_{lm}^{jk} \\, \\ketbra{j}{k} \\otimes \\ketbra{m}{l} \\\\
        &= \\sum_{jklm} \\, O_{ml}^{jk} \\, \\ketbra{j}{k} \\otimes \\ketbra{l}{m} \\, ,
        \\end{align}

    where the superscript :math:`T` indicates the transposition operation,
    and :math:`T_{B}` indicates transposition on ``partition`` :math:`B`.
    The total number of qubits is inferred by the shape of ``operator``.

    Args:
        operator (ArrayLike): :math:`1`- or :math:`2`-dimensional operator, or an array of
            :math:`1`- or :math:`2`-dimensional operators,
        partition (Union[List[int], Tuple[int, ...]]): indices of qubits to be transposed.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            it uses the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: Partially transposed operator(s) :math:`\\O^{T_{B}}`.
    """
    backend = _check_backend(backend)

    shape = operator.shape
    nstates = shape[0]
    dims = shape[-1]
    nqubits = math.log2(dims)

    if not nqubits.is_integer():
        raise_error(
            ValueError,
            "dimensions of ``state`` (or states in a batch) must be a power of 2.",
        )

    if (len(shape) > 3) or (nstates == 0) or (len(shape) == 2 and nstates != dims):
        raise_error(
            TypeError,
            "``operator`` must have dims either (k,), (k, k), (N, 1, k) or (N, k, k), "
            + f"but has dims {shape}.",
        )

    nqubits = int(nqubits)

    if len(shape) == 1:
        operator = backend.outer(operator, backend.conj(operator.T))
    elif len(shape) == 3 and shape[1] == 1:
        operator = backend.einsum(
            "aij,akl->aijkl", operator, backend.conj(operator)
        ).reshape(nstates, dims, dims)

    new_shape = list(range(2 * nqubits + 1))
    for ind in partition:
        ind += 1
        new_shape[ind] = ind + nqubits
        new_shape[ind + nqubits] = ind
    new_shape = tuple(new_shape)

    reshaped = backend.reshape(operator, [-1] + [2] * (2 * nqubits))
    reshaped = backend.transpose(reshaped, new_shape)

    final_shape = (dims, dims)
    if len(operator.shape) == 3:
        final_shape = (nstates,) + final_shape

    return backend.reshape(reshaped, final_shape)


def matrix_exponentiation(
    matrix: ArrayLike,
    phase: complex | None = None,
    eigenvectors: ArrayLike | None = None,
    eigenvalues: ArrayLike | None = None,
    backend: Backend | None = None,
) -> ArrayLike:
    """Calculates the exponential of a matrix.

    Given a ``matrix`` :math:`H` and a ``phase`` :math:`\\theta`,
    it returns the exponential of the form

    .. math::
        \\exp\\left(\\theta \\, H \\right) \\, .

    If the ``eigenvectors`` and ``eigenvalues`` are given, the matrix diagonalization
    is used for the exponentiation.

    Args:
        matrix (ArrayLike): matrix to be exponentiated.
        phase (float or int or complex): phase that multiplies the matrix.
            If ``None``, defaults to :math:`1`. Defaults to ``None``.
        eigenvectors (ArrayLike, optional): _if not ``None``, eigenvectors are used
            to calculate ``matrix`` exponentiation as part of diagonalization.
            Must be used together with ``eigenvalues``. Defaults to ``None``.
        eigenvalues (ArrayLike, optional): if not ``None``, eigenvalues are used
            to calculate ``matrix`` exponentiation as part of diagonalization.
            Must be used together with ``eigenvectors``. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: matrix exponential of :math:`-i \\, \\theta \\, H`.
    """
    backend = _check_backend(backend)

    return backend.matrix_exp(matrix, phase, eigenvectors, eigenvalues)


def matrix_logarithm(
    matrix: ArrayLike,
    base: float = 2,
    eigenvectors: ArrayLike | None = None,
    eigenvalues: ArrayLike | None = None,
    backend: Backend | None = None,
) -> ArrayLike:
    """Calculates the logarithm of a matrix.

    Given a ``matrix`` :math:`A` and a log base :math:`b`, it returns the logarithm of the form

    .. math::
        \\log_{b}\\left(\\theta \\, H \\right) \\, .

    If the ``eigenvectors`` and ``eigenvalues`` are given, the matrix diagonalization
    is used for the calculation.

    Args:
        matrix (ArrayLike): matrix to be logarithmed.
        eigenvectors (ArrayLike, optional): _if not ``None``, eigenvectors are used
            to calculate the ``matrix`` logarithm as part of diagonalization.
            Must be used together with ``eigenvalues``. Defaults to ``None``.
        eigenvalues (ArrayLike, optional): if not ``None``, eigenvalues are used
            to calculate the ``matrix`` logarithm as part of diagonalization.
            Must be used together with ``eigenvectors``. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: Matrix logarithm :math:`\\log_{b}(H)`.
    """
    backend = _check_backend(backend)

    return backend.matrix_log(matrix, base, eigenvectors, eigenvalues)


def matrix_power(
    matrix: ArrayLike,
    power: float,
    precision_singularity: float = 1e-14,
    backend: Backend | None = None,
) -> ArrayLike:
    """Given a ``matrix`` :math:`A` and power :math:`\\alpha`, calculate :math:`A^{\\alpha}`.

    Args:
        matrix (ArrayLike): matrix whose power to calculate.
        power (float or int): power to raise ``matrix`` to.
        precision_singularity (float, optional): If determinant of ``matrix`` is smaller than
            ``precision_singularity``, then matrix is considered to be singular.
            Used when ``power`` is negative.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: matrix power :math:`A^{\\alpha}`.
    """
    backend = _check_backend(backend)

    return backend.matrix_power(matrix, power, precision_singularity)


def matrix_sqrt(matrix: ArrayLike, backend: Backend | None = None) -> ArrayLike:
    """Given a ``matrix`` :math:`A`, calculate :math:`A^{1/2}`.

    Args:
        matrix (ArrayLike): matrix whose power to calculate.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: Matrix power :math:`A^{1/2}`.
    """
    return matrix_power(matrix, power=0.5, backend=backend)


def singular_value_decomposition(
    matrix: ArrayLike, backend: Backend | None = None
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Calculate the Singular Value Decomposition (SVD) of ``matrix``.

    Given an :math:`M \\times N` complex matrix :math:`A`, its SVD is given by

    .. math:
        A = U \\, S \\, V^{\\dagger} \\, ,

    where :math:`U` and :math:`V` are, respectively, an :math:`M \\times M`
    and an :math:`N \\times N` complex unitary matrices, and :math:`S` is an
    :math:`M \\times N` diagonal matrix with the singular values of :math:`A`.

    Args:
        matrix (ArrayLike): matrix whose SVD to calculate.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike, ArrayLike, ArrayLike: Singular value decomposition of :math:`A`, i.e.
        :math:`U`, :math:`S`, and :math:`V^{\\dagger}`, in that order.
    """
    backend = _check_backend(backend)

    return backend.singular_value_decomposition(matrix)


def schmidt_decomposition(
    state: ArrayLike,
    partition: list[int] | tuple[int, ...],
    backend: Backend | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return the Schmidt decomposition of a :math:`n`-qubit bipartite pure quantum ``state``.

    Given a bipartite pure state :math:`\\ket{\\psi}\\in\\mathcal{H}_{A}\\otimes\\mathcal{H}_{B}`,
    its Schmidt decomposition is given by

    .. math::
        \\ket{\\psi} = \\sum_{k = 1}^{\\min\\{a, \\, b\\}} \\, c_{k} \\,
            \\ket{\\phi_{k}} \\otimes \\ket{\\nu_{k}} \\, ,

    with :math:`a` and :math:`b` being the respective cardinalities of :math:`\\mathcal{H}_{A}`
    and :math:`\\mathcal{H}_{B}`, and :math:`\\{\\phi_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}
    \\subset \\mathcal{H}_{A}` and :math:`\\{\\nu_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}
    \\subset \\mathcal{H}_{B}` being orthonormal sets. The coefficients
    :math:`\\{c_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}` are real, non-negative, and unique
    up to re-ordering.

    The decomposition is calculated using :func:`qibo.quantum_info.singular_value_decomposition`,
    resulting in

    .. math::
        \\ketbra{\\psi}{\\psi} = U \\, S \\, V^{\\dagger} \\, ,

    where :math:`U` is an :math:`a \\times a` unitary matrix, :math:`V` is an :math:`b \\times b`
    unitary matrix, and :math:`S` is an :math:`a \\times b` positive semidefinite diagonal matrix
    that contains the singular values of :math:`\\ketbra{\\psi}{\\psi}`.

    Args:
        state (ArrayLike): stevector or density matrix.
        partition (Union[List[int], Tuple[int, ...]]): indices of qubits in one of the two
            partitions. The other partition is inferred as the remaining qubits.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ArrayLike, ArrayLike, ArrayLike: Respectively, the matrices :math:`U`, :math:`S`,
        and :math:`V^{\\dagger}`.
    """
    backend = _check_backend(backend)

    nqubits = math.log2(state.shape[-1])
    if not nqubits.is_integer():
        raise_error(ValueError, "dimensions of ``state`` must be a power of 2.")

    nqubits = int(nqubits)
    partition_2 = partition.__class__(set(range(nqubits)) ^ set(partition))

    tensor = backend.reshape(state, [2] * nqubits)
    tensor = backend.transpose(tensor, partition + partition_2)
    tensor = backend.reshape(tensor, (2 ** len(partition), -1))

    return singular_value_decomposition(tensor, backend=backend)


def lanczos(
    matrix: ArrayLike,
    steps: int | None = None,
    initial_vector: ArrayLike | None = None,
    precision_tol: float = 1e-8,
    seed: int | None = None,
    backend: Backend | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Lanczos iterative method to tridiagonalize a Hermitian matrix.

    Given a :math:`N \\times N` Hermitian matrix :math:`H` and a number of iterations
    :math:`m \\leq N`, the Lanczos algorithm outputs a :math:`N \\times m` orthonormal matrix
    :math:`U` and a :math:`m \\times m` tridiagonal real symmetric matrix
    :math:`T = U^{\\dagger} \\, H \\, U`. If :math:`m = N`, then :math:`U` is an unitary matrix.
    The eigenvalues of :math:`T` and :math:`H` coincide, while :math:`U \\ket{\\mathbf{x}}`
    are the eigenvectors of :math:`H`, with :math:`\\ket{\\mathbf{x}}` being the
    eigenvectors of :math:`T`.

    This reduces the problem of diagonalization of :math:`H` to constructing
    the matrix :math:`U` and diagonalizing :math:`T`.

    With :math:`\\|\\cdot\\|_{2}` being the Euclidean norm, the algorithm goes as follows:

    1. Generate random :math:`\\ket{v_{1}} \\in \\mathbb{C}^{N}` such that :math:`\\|\\ket{v_{1}}\\|_{2} = 1`
    2. :math:`\\ket{\\omega_{1}^{\\prime}} = H \\ket{v_{1}}`
    3. :math:`\\alpha_{1} = \\braket{\\omega_{1}^{\\prime} | v_{1}}`
    4. :math:`\\ket{\\omega_{1}} = H \\ket{v_{1}} - \\alpha_{1} \\ket{v_{1}}`
    5. For :math:`j = 2, \\dots, m - 1`:
        1. :math:`\\beta_{j} = \\|\\omega_{j-1}\\|_{2}`
        2. :math:`\\ket{v_{j}} = \\ket{\\omega_{j-1}} \\, / \\, \\beta_{j}` If :math:`\\beta_{j} \\neq 0` else generate random :math:`\\ket{v_{j}}` such that :math:`\\ket{v_{j}} \\perp \\{\\ket{v_{j^{\\prime}}}\\}_{j^{\\prime} \\in [1, j-1]}`
        3. :math:`\\ket{\\omega_{j}^{\\prime}} = H \\ket{v_{j}}`
        4. :math:`\\alpha_{j} = \\braket{\\omega_{j}^{\\prime} | v_{j}}`
        5. :math:`\\ket{\\omega_{j}} = \\ket{\\omega_{j}^{\\prime}} - \\alpha_{j} \\ket{v_{j}} - \\beta_{j} \\ket{v_{j-1}}`

    The columns of the orthogonal matrix :math:`U` are the *Lanczos vectors* :math:`\\{\\ket{v_{j}}\\}_{j\\in[1, m]}`.

    Args:
        matrix (ArrayLike): square Hermitian matrix to be tridiagonalized.
        steps (int, optional): number of iterations :math:`m`. If ``None``,
            defaults to the size of ``matrix``. Defaults to ``None``.
        initial_vector (ArrayLike, optional): vector to be used as the initial Lanczos vector
            :math:`\\ket{v_{1}}`. If ``None``, array is uniformly sampled.
            Defaults to ``None``.
        precision_tol (float, optional): precision threshold such that for :math:`\\beta_{j}`
            smaller than ``precision_tol``, it is considered to be zero.
        seed (int or :class:`numpy.random.Generator`, optional): Seed for the initial random vector
            :math:`\\ket{v_{1}}` Either a generator of random numbers or a fixed seed to initialize
            a generator. If ``None``, initializes a generator with a random seed.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (ArrayLike, ArrayLike): Tridiagonal matrix and the orthogonal matrix
        of Lanczos vectors, respectively.

    References:
        1.  Lanczos, C. *An iteration method for the solution of the eigenvalue problem of linear
        differential and integral operators*, Journal of Research of the National Bureau of
        Standards. 45 (4): 255–282 (1950).
    """
    from qibo.quantum_info.random_ensembles import (
        random_statevector,
    )

    backend = _check_backend(backend)
    backend.set_seed(seed)

    dims = matrix.shape[0]

    if steps is None:
        steps = dims

    vector = (
        random_statevector(dims, seed=seed, backend=backend)
        if initial_vector is None
        else initial_vector
    )

    omega_prime = matrix @ vector
    alpha = backend.conj(omega_prime.T) @ vector
    omega = omega_prime - alpha * vector

    lanczos_vectors = [vector]
    for _ in range(steps - 1):
        norm = backend.vector_norm(omega)
        if norm > precision_tol:
            vector = omega / norm
        else:  # pragma: no cover
            # this part is tested separatedly
            vector = random_statevector(dims, seed=seed, backend=backend)
            vector = _gram_schmidt_process(
                vector, backend.cast(lanczos_vectors).T, backend=backend
            )

        lanczos_vectors.append(vector)

        omega_prime = matrix @ vector
        alpha = backend.conj(omega_prime.T) @ vector
        omega = omega_prime - alpha * vector - norm * lanczos_vectors[-2]

    lanczos_vectors = backend.cast(lanczos_vectors)
    triadiagonal = backend.conj(lanczos_vectors) @ matrix
    lanczos_vectors = lanczos_vectors.T
    triadiagonal = triadiagonal @ lanczos_vectors

    return triadiagonal, lanczos_vectors


def lie_closure(
    generators: list[str] | list[dict[str, float]] | list[ArrayLike],
    max_iterations: int = 10000,
    tol: float = 1e-10,
    backend: Backend | None = None,
) -> list[str] | list[dict[str, float]] | list[ArrayLike]:
    """Compute the dynamical Lie algebra (DLA) generated by a set of operators.

    The generators are either matrices, Pauli strings (e.g. ``"XXI"``), or real linear
    combinations of Pauli strings given as dictionaries (e.g. ``{"XX": 1.0, "ZI": 0.5}``).
    Given generators :math:`\\{G_{j}\\}_{j}`, each either Hermitian or skew-Hermitian,
    the DLA is the real span of all nested commutators
    :math:`[G_{j}, [G_{k}, \\dots, [G_{l}, G_{m}]]]` of the generators.
    Following the physics convention, the algebra is :math:`\\{i \\, G_{\\alpha}\\}_{\\alpha}`
    and the returned operators :math:`G_{\\alpha}` are Hermitian.
    Commutators are added round by round: round :math:`r` contains the commutators of the operators
    added in round :math:`r - 1` with the original generators, i.e. nesting depth :math:`r`.

    The commutator of two Pauli strings is either zero or proportional to a Pauli string.
    Hence, Pauli generators are handled on the phase-space (tableau) representation of Paulis,
    in which the product of two Paulis is the XOR of their bit vectors :math:`(x | z)`,
    and they anticommute if and only if their symplectic inner product is :math:`1`.
    This avoids building :math:`2^{n} \\times 2^{n}` matrices. If all generators are Pauli strings,
    the DLA is itself spanned by Pauli strings. Otherwise, operators are stored as
    coefficient vectors over the Pauli strings encountered, and commutators are computed
    term by term.

    Args:
        generators (list[ArrayLike] or list[str] or list[dict]): either square matrices of
            identical shape, each Hermitian or skew-Hermitian, or Pauli strings of identical
            length composed of ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``, or dictionaries
            mapping such Pauli strings to real coefficients. Pauli strings and
            dictionaries can be mixed, but not combined with matrices.
        max_iterations (int, optional): maximum nesting depth of commutators.
            If reached before the algebra is closed, a warning is logged.
            Defaults to :math:`10000`.
        tol (float, optional): threshold on the norm below which a candidate
            is considered linearly dependent on the current basis (and, for dictionaries,
            below which a coefficient is set to zero). Ignored if all generators
            are Pauli strings, which are handled exactly. If an operator is found to be
            independent with a residual norm close to ``tol``, it may be numerical noise,
            and a warning is logged. This can happen for dense, ill-conditioned
            combinations of generators, for which a larger ``tol`` may be needed.
            Defaults to :math:`10^{-10}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike or list[str] or list[dict]: For matrices, array of shape ``(dim, d, d)`` with a
        Hermitian basis of the DLA, orthonormal with respect to the Hilbert-Schmidt inner product
        :math:`\\text{Tr}(A^{\\dagger} \\, B)`. For Pauli strings only, the list of
        Pauli strings (all with positive sign) that span the DLA. Otherwise, a list of
        dictionaries, each mapping Pauli strings to real coefficients with unit Euclidean norm
        (orthonormal basis).

    Example:
        Transverse-field Ising model on two qubits, whose DLA has dimension :math:`6`:

        .. code-block:: python

            from qibo import matrices
            from qibo.backends import NumpyBackend
            from qibo.quantum_info import lie_closure

            backend = NumpyBackend()
            I, X, Z = matrices.I, matrices.X, matrices.Z
            generators = [
                backend.kron(X, X),
                backend.kron(Z, I),
                backend.kron(I, Z),
            ]

            dla = lie_closure(generators, backend=backend)
            print(dla.shape)  # (6, 4, 4)

        The same algebra from Pauli strings:

        .. code-block:: python

            from qibo.quantum_info import lie_closure

            dla = lie_closure(["XX", "ZI", "IZ"])
            print(dla)  # ['XX', 'ZI', 'IZ', 'YX', 'XY', 'YY']

        Heisenberg chain on four qubits, with generators that are sums of Pauli strings:

        .. code-block:: python

            from qibo.quantum_info import lie_closure

            generators = [
                {"XXII": 1.0, "YYII": 1.0, "ZZII": 1.0},
                {"IXXI": 1.0, "IYYI": 1.0, "IZZI": 1.0},
                {"IIXX": 1.0, "IIYY": 1.0, "IIZZ": 1.0},
            ]

            dla = lie_closure(generators)
            print(len(dla))  # 12

    References:
        1.  M. Larocca *et al.*, *Diagnosing barren plateaus with tools from quantum optimal
        control*, `Quantum 6, 824 (2022) <https://doi.org/10.22331/q-2022-09-29-824>`_.
        2.  S. Aaronson and D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
        `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_.
    """
    backend = _check_backend(backend)

    nb_strings = sum(isinstance(gen, str) for gen in generators)
    nb_paulis = sum(isinstance(gen, (str, dict)) for gen in generators)

    if nb_paulis not in (0, len(generators)):
        raise_error(TypeError, "``generators`` cannot mix Pauli objects and matrices.")

    if nb_paulis == 0:
        return _lie_closure_matrix(generators, max_iterations, tol, backend)

    strings = [
        string
        for gen in generators
        for string in ([gen] if isinstance(gen, str) else gen)
    ]
    if len(strings) == 0 or any(
        len(string) != len(strings[0]) or not set(string) <= set("IXYZ")
        for string in strings
    ):
        raise_error(
            ValueError,
            "Pauli strings must be non-empty, have the same length and "
            + "contain only ``I``, ``X``, ``Y`` and ``Z``.",
        )

    if any(
        complex(coeff).imag != 0
        for gen in generators
        if isinstance(gen, dict)
        for coeff in gen.values()
    ):
        raise_error(ValueError, "Coefficients of Pauli sums must be real.")

    if nb_strings == nb_paulis:
        return _lie_closure_pauli_strings(generators, max_iterations, backend)

    return _lie_closure_pauli_sums(generators, max_iterations, tol, backend)
