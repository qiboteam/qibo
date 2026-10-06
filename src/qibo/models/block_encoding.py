from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.lcu import lcu_circuit


def block_encoding_circuit(
    matrix: ArrayLike,
    method: str = "dense",
    alpha: float | None = None,
    backend: Backend = None,
) -> tuple[Circuit, float, int]:
    """Creates a block encoding of a matrix.

    Let :math:`A` be ``matrix``, a square matrix that is padded with zeros up to
    dimension :math:`2^{n}` if needed, with :math:`n \\ge 1`. The circuit acts on
    :math:`a` auxiliary qubits, followed by :math:`n` system qubits, and the block of
    its unitary in which the auxiliary qubits start and end in :math:`|0\\rangle` is
    :math:`A / \\alpha`, where :math:`\\alpha` is a normalization. Both methods
    have a cost that grows exponentially with :math:`n`.

    - ``method="dense"``: :math:`a = 1`. With :math:`B = A / \\alpha`, whose singular
      values are at most one, the circuit is a single gate for the unitary dilation

      .. math::
          U = \\begin{pmatrix} B & \\sqrt{\\mathbb{1} - B B^{\\dagger}} \\\\
          \\sqrt{\\mathbb{1} - B^{\\dagger} B} & -B^{\\dagger} \\end{pmatrix},

      where :math:`\\mathbb{1}` is the identity and :math:`B^{\\dagger}` the conjugate
      transpose of :math:`B`. The matrix square roots are computed from the singular
      value decomposition of :math:`B`.
    - ``method="lcu"``: the matrix is decomposed as a sum of Pauli strings,
      :math:`A = \\sum_{i} c_{i} P_{i}`, with complex coefficients :math:`c_{i}`, and
      the circuit is the one of :func:`qibo.models.lcu.lcu_circuit`. The normalization
      is :math:`\\alpha = \\sum_{i} |c_{i}|`, and :math:`a = \\max(1, \\lceil \\log_{2} L
      \\rceil)`, with :math:`L` the number of non-zero coefficients.

    Args:
        matrix (ArrayLike): square matrix :math:`A`.
        method (str, optional): ``"dense"`` for the unitary dilation or ``"lcu"`` for the
            linear combination of unitaries of the Pauli strings. Defaults to ``"dense"``.
        alpha (float, optional): normalization :math:`\\alpha` if ``method="dense"``,
            which has to be at least the spectral norm of ``matrix``. If ``None``, it is
            the spectral norm if this is larger than one, and one otherwise. It is not
            used by ``method="lcu"``, for which it has to be ``None``. Defaults to
            ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: circuit on :math:`a + n` qubits, with the auxiliary qubits at the first
        qubits, the normalization :math:`\\alpha`, and the number :math:`a` of auxiliary
        qubits. The circuit and the number of auxiliary qubits are the arguments of
        :func:`qibo.models.qsvt.qsvt_circuit`.

    Example:
        Encode :math:`X / 2`, with :math:`X` the Pauli-:math:`X` matrix. The
        amplitudes with the auxiliary qubits in :math:`|0\\rangle` are those of
        :math:`|1\\rangle / 2` for the dense method, and those of :math:`|1\\rangle`
        for the LCU method, whose normalization is :math:`1 / 2`.

        .. testcode::

            from qibo import get_backend
            from qibo.models.block_encoding import block_encoding_circuit

            backend = get_backend()

            matrix = 0.5 * backend.matrices.X

            circuit, alpha, nauxiliary = block_encoding_circuit(matrix)
            state = circuit().state()[:2]
            print(backend.allclose(state, backend.cast([0.0, 0.5])), alpha, nauxiliary)

            circuit, alpha, nauxiliary = block_encoding_circuit(matrix, method="lcu")
            state = circuit().state()[:2]
            print(backend.allclose(state, backend.cast([0.0, 1.0])), alpha, nauxiliary)

        .. testoutput::

            True 1.0 1
            True 0.5 1

    References:
        1. A. Gilyén, Y. Su, G. H. Low, and N. Wiebe, *Quantum singular value
        transformation and beyond: exponential improvements for quantum matrix
        arithmetics*, `Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
        of Computing (STOC '19), 193-204 (2019)
        <https://doi.org/10.1145/3313276.3316366>`_.
    """
    backend = _check_backend(backend)

    if method not in ("dense", "lcu"):
        raise_error(ValueError, f"Unknown ``method`` {method}. Use 'dense' or 'lcu'.")

    matrix = backend.cast(matrix, dtype="complex128")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_error(
            ValueError,
            f"``matrix`` must be square, but has shape {tuple(matrix.shape)}.",
        )

    if method == "lcu" and alpha is not None:
        raise_error(ValueError, "``alpha`` is not used if ``method='lcu'``.")

    norm = float(backend.matrix_norm(matrix, order=2))
    if alpha is None:
        alpha = max(norm, 1.0)
    elif alpha <= 0.0 or norm > alpha * (1.0 + 1e-12):
        raise_error(
            ValueError,
            f"``alpha`` must be positive and at least the spectral norm {norm} "
            + f"of ``matrix``, but is {alpha}.",
        )

    # zero-padding up to the next power of two
    dim = matrix.shape[0]
    nqubits = max((dim - 1).bit_length(), 1)
    padding = 2**nqubits - dim
    if padding > 0:
        matrix = backend.concatenate(
            (matrix, backend.zeros((dim, padding), dtype="complex128")), axis=1
        )
        matrix = backend.concatenate(
            (matrix, backend.zeros((padding, 2**nqubits), dtype="complex128")), axis=0
        )

    if method == "lcu":
        from qibo.quantum_info import pauli_basis  # circular import

        # coefficients of the Pauli strings, whose index in base four has the Pauli
        # matrix I, X, Y, Z of the first qubit as its most significant digit
        basis = pauli_basis(nqubits, vectorize=True, order="row", backend=backend)
        coefficients = backend.matmul(
            backend.conj(basis), backend.reshape(matrix, (-1,))
        ) / (2**nqubits)

        magnitudes = backend.abs(coefficients)
        terms = backend.nonzero(magnitudes > 1e-12 * backend.max(magnitudes))[0]
        if len(terms) == 0:
            raise_error(ValueError, "``matrix`` must have a non-zero element.")

        unitaries = []
        for term in terms:
            unitary = Circuit(nqubits)
            for qubit in range(nqubits):
                pauli = (int(term) // 4 ** (nqubits - 1 - qubit)) % 4
                if pauli > 0:
                    unitary.add((gates.X, gates.Y, gates.Z)[pauli - 1](qubit))
            unitaries.append(unitary)

        return lcu_circuit(unitaries, coefficients[terms], backend=backend)

    matrix = matrix / alpha

    # sqrt(1 - B B^dagger) = W sqrt(1 - s^2) W^dagger and
    # sqrt(1 - B^dagger B) = V sqrt(1 - s^2) V^dagger, with B = W diag(s) V^dagger
    left, singular_values, right = backend.singular_value_decomposition(matrix)
    roots = backend.sqrt(backend.maximum(1.0 - singular_values**2, 0.0))
    left_root = backend.matmul(left * roots, backend.dagger(left))
    right_root = backend.matmul(backend.dagger(right) * roots, right)

    dilation = backend.concatenate(
        (
            backend.concatenate((matrix, left_root), axis=1),
            backend.concatenate((right_root, -backend.dagger(matrix)), axis=1),
        ),
        axis=0,
    )

    circuit = Circuit(nqubits + 1)
    circuit.add(gates.Unitary(dilation, *range(nqubits + 1)))

    return circuit, float(alpha), 1
