import math
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms.block_encoding import block_encoding_circuit
from qibo.models.algorithms.qaa import _check_circuits
from qibo.models.algorithms.qft import QFT
from qibo.models.algorithms.qsvt import qsvt_circuit, qsvt_phases
from qibo.models.circuit import Circuit
from qibo.models.encodings import binary_encoder


def HHL(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    vector: ArrayLike | Circuit,
    nclock: int | None = None,
    condition_number: float | None = None,
    method: str = "original",
    epsilon: float = 0.1,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates the circuit of the Harrow-Hassidim-Lloyd (HHL) algorithm for linear systems.

    Let :math:`A` be a Hermitian matrix of size :math:`N \\times N`, with :math:`N = 2^{n}`, and
    let :math:`\\ket{b}` be the state with the normalized entries of a vector :math:`b`. The
    circuit prepares a state whose input register, of :math:`n` qubits, is close to
    :math:`A^{-1} \\ket{b}`, which is the solution :math:`\\ket{x}` of the linear system
    :math:`A x = b`, when the remaining qubits are measured in a given outcome. The probability
    of this outcome is close to :math:`C^{2} \\lVert A^{-1} b \\rVert^{2} / \\lVert b
    \\rVert^{2}`, where :math:`C` is the second output, which gives the norm of the solution.
    Let :math:`\\lVert A \\rVert` be the largest absolute value of the eigenvalues of :math:`A`,
    and let :math:`\\kappa` be the ratio of the largest and the smallest ones, *i.e.* its
    condition number. The matrix is exponentiated or block-encoded exactly, which is inefficient
    in the number of qubits :math:`n`, and it is only meant for small systems. There are two
    methods:

    - ``method="original"``: the algorithm of Ref. [1], with phase estimation. The circuit acts
      on the flag, followed by the ``nclock`` qubits of the clock register, with
      :math:`T = 2^{\\texttt{nclock}}` states, and the :math:`n` qubits of the input register,
      where :math:`\\ket{b}` is loaded. The outcome is the flag in :math:`\\ket{1}`, and the
      clock register is then back to :math:`\\ket{0}` up to the error of the estimation. The
      constant is :math:`C = \\lVert A \\rVert / \\kappa`. The circuit is composed of the
      following steps:

      1. the loading of :math:`\\ket{b}`, and of the state
         :math:`\\ket{\\Psi_{0}} = \\sqrt{2 / T} \\sum_{\\tau = 0}^{T - 1}
         \\sin(\\pi (\\tau + 1 / 2) / T) \\ket{\\tau}` in the clock register;
      2. the controlled evolution :math:`\\sum_{\\tau} \\ket{\\tau}\\bra{\\tau} \\otimes
         e^{i A t_{0} \\tau / T}`, with :math:`t_{0} = \\pi T / (2 \\lVert A \\rVert)`;
      3. the inverse :func:`qibo.models.QFT` of the clock register, which then has the estimate
         :math:`\\tilde{\\lambda}_{k} = 2 \\pi k / t_{0}` of an eigenvalue of :math:`A`, where the
         integer :math:`k` is in two's complement, so that the eigenvalues can be negative;
      4. the rotation of the flag controlled by the clock register, which is a single
         :class:`qibo.gates.Multiplexor` gate,
         :math:`\\ket{0} \\mapsto \\sqrt{1 - C^{2} / \\tilde{\\lambda}_{k}^{2}} \\ket{0} +
         (C / \\tilde{\\lambda}_{k}) \\ket{1}`. The amplitude is at most one, and
         the clock values that are smaller than :math:`C` by more than the resolution
         :math:`2 \\pi / t_{0}` of the clock are discarded, which includes :math:`k = 0`;
      5. the inverse of the steps 3, 2 and the preparation of :math:`\\ket{\\Psi_{0}}`, which
         takes the clock register back to :math:`\\ket{0}` up to the error of the estimation.

      The relative error of the eigenvalues is of the order of :math:`4 \\kappa / T`, and,
      therefore, it needs :math:`T \\gtrsim 4 \\kappa / \\epsilon` for an error
      :math:`\\epsilon`.

    - ``method="qsvt"``: the algorithm of Ref. [2], with a quantum singular value transformation
      (QSVT), and the polynomial approximation of the inverse of Ref. [3]. The circuit is the
      :func:`qibo.models.qsvt_circuit` of the dense :func:`qibo.models.block_encoding_circuit` of
      :math:`A / \\lVert A \\rVert`. It acts on the signal qubit, followed by the auxiliary qubit
      of the block encoding, and the :math:`n` qubits of the input register, where
      :math:`\\ket{b}` is loaded. The outcome is the first two qubits in :math:`\\ket{0}`. The
      odd polynomial is :math:`P(x) = g(x) / s`, where :math:`g` is the Chebyshev series of
      :math:`(1 - (1 - x^{2})^{m}) / x`, with :math:`m = \\lceil \\kappa^{2} \\ln(\\kappa /
      \\epsilon) \\rceil`, truncated at the degree :math:`d = 2 j + 1`, with
      :math:`j = \\lceil \\sqrt{m \\ln(4 m / \\epsilon)} \\rceil` or at most :math:`m - 1`,
      and :math:`s` is the maximum of :math:`|g|` in :math:`[-1, 1]`. Then,
      :math:`g(x)` is :math:`2 \\epsilon`-close to :math:`1 / x` for
      :math:`|x| \\in [1 / \\kappa, 1]`, and the constant is :math:`C = \\lVert A \\rVert / s`.
      The circuit queries the block encoding :math:`d = O(\\kappa \\log(\\kappa / \\epsilon))`
      times.

    Example:
        Solve :math:`A x = b` with :math:`A = [[1, -1/3], [-1/3, 1]]` and :math:`b = [0, 1]`,
        whose normalized solution is :math:`[1, 3] / \\sqrt{10}`.

        .. testcode::

            import numpy as np

            from qibo.models import HHL

            matrix = np.array([[1.0, -1 / 3], [-1 / 3, 1.0]])
            vector = np.array([0.0, 1.0])

            nclock = 6
            circuit, constant = HHL(matrix, vector, nclock)

            # the flag is the first qubit, and the clock register is the second one
            state = circuit().state().reshape(2, 2**nclock, 2)
            solution = state[1, 0]

            print(np.round(np.abs(solution) ** 2 / np.sum(np.abs(solution) ** 2), 2))

            circuit, constant = HHL(matrix, vector, method="qsvt", epsilon=0.05)

            # the signal and the auxiliary qubits are the first two qubits
            state = circuit().state().reshape(2, 2, 2)
            solution = state[0, 0]

            print(np.round(np.abs(solution) ** 2 / np.sum(np.abs(solution) ** 2), 2))

        .. testoutput::

            [0.1 0.9]
            [0.1 0.9]

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix :math:`A`, or Hamiltonian with its matrix. Its size is
            :math:`2^{n} \\times 2^{n}`, with :math:`n \\geq 1`.
        vector (ArrayLike or :class:`qibo.models.circuit.Circuit`): vector :math:`b` of the
            :math:`2^{n}` entries, which is normalized, or circuit of :math:`n` qubits that
            prepares :math:`\\ket{b}` from the all-zero state. A circuit cannot contain
            measurements.
        nclock (int, optional): number of qubits of the clock register, which is at least two.
            It is only used, and required, by ``method="original"``. Defaults to ``None``.
        condition_number (float, optional): bound :math:`\\kappa \\geq 1` of the ratio of the
            largest and the smallest absolute values of the eigenvalues of :math:`A`. If it is
            smaller than the true condition number, the eigenvalues that are smaller than
            :math:`\\lVert A \\rVert / \\kappa` are not inverted, which is the pseudoinverse
            of the rest of the matrix. They are discarded by ``method="original"``, and
            suppressed by ``method="qsvt"``. If ``None``, it is computed from the eigenvalues of
            :math:`A`, which are not zero. Defaults to ``None``.
        method (str, optional): either ``"original"`` or ``"qsvt"``. Defaults to
            ``"original"``.
        epsilon (float, optional): precision :math:`\\epsilon \\in (0, 1/2)` of the
            approximation of the inverse. It is only used by ``method="qsvt"``.
            Defaults to ``0.1``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: Circuit of the algorithm, and the constant :math:`C`.

    References:
        1. A. W. Harrow, A. Hassidim, and S. Lloyd, *Quantum algorithm for linear systems of
        equations*, `Phys. Rev. Lett. 103, 150502 (2009)
        <https://doi.org/10.1103/PhysRevLett.103.150502>`_.
        2. A. Gilyén, Y. Su, G. H. Low, and N. Wiebe, *Quantum singular value
        transformation and beyond: exponential improvements for quantum matrix
        arithmetics*, `Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
        of Computing (STOC '19), 193-204 (2019)
        <https://doi.org/10.1145/3313276.3316366>`_.
        3. A. M. Childs, R. Kothari, and R. D. Somma, *Quantum algorithm for systems of linear
        equations with exponentially improved dependence on precision*, `SIAM J. Comput. 46,
        1920 (2017) <https://doi.org/10.1137/16M1087072>`_.
    """
    if method not in ("original", "qsvt"):
        raise_error(
            ValueError, f"Unknown ``method`` {method}. Use 'original' or 'qsvt'."
        )

    if method == "original":
        return _hhl_original(
            matrix, vector, nclock, condition_number, backend, **kwargs
        )

    return _hhl_qsvt(matrix, vector, epsilon, condition_number, backend, **kwargs)


def _hhl_inputs(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    vector: ArrayLike | Circuit,
    condition_number: float | None,
    backend: Backend = None,
) -> tuple[ArrayLike, int, Circuit, ArrayLike, ArrayLike, float, float]:
    """Checks the matrix, the vector and the condition number of the HHL algorithm.

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix of size :math:`2^{n} \\times 2^{n}`.
        vector (ArrayLike or :class:`qibo.models.circuit.Circuit`): vector of :math:`2^{n}`
            entries, or circuit of :math:`n` qubits that prepares its normalized state.
        condition_number (float, optional): bound of the condition number of ``matrix``.
            If ``None``, it is computed from its eigenvalues.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple: matrix, number :math:`n` of qubits, circuit that loads the vector, eigenvalues and
        eigenvectors of the matrix, largest absolute value of the eigenvalues, and condition
        number.
    """
    backend = _check_backend(backend)

    if isinstance(matrix, (Hamiltonian, SymbolicHamiltonian)):
        matrix = matrix.matrix
    matrix = backend.cast(matrix, dtype=backend.complex128)

    dimension = matrix.shape[0]
    if (
        len(matrix.shape) != 2
        or matrix.shape[1] != dimension
        or dimension < 2
        or dimension & (dimension - 1) != 0
    ):
        raise_error(
            ValueError,
            "matrix must be a square matrix of size 2^n with n >= 1, "
            + f"but its shape is {tuple(matrix.shape)}.",
        )
    nqubits = dimension.bit_length() - 1

    hermitian = backend.matrix_norm(
        matrix - backend.conj(backend.transpose(matrix)), order="fro"
    )
    if hermitian > 1e-8 * backend.matrix_norm(matrix, order="fro"):
        raise_error(ValueError, "matrix must be Hermitian.")

    if isinstance(vector, Circuit):
        if _check_circuits(vector=vector) != nqubits:
            raise_error(
                ValueError,
                f"vector must have {nqubits} qubits, but it has {vector.nqubits}.",
            )
        loading = vector
    else:
        vector = backend.cast(vector, dtype=backend.complex128)
        if tuple(vector.shape) != (dimension,) or backend.sum(backend.abs(vector)) == 0:
            raise_error(
                ValueError,
                f"vector must be a non-zero vector of {dimension} entries.",
            )
        loading = binary_encoder(nqubits, data=vector, backend=backend)

    eigenvalues, eigenvectors = backend.eigh(matrix)
    norm = float(backend.max(backend.abs(eigenvalues)))

    if condition_number is None:
        smallest = float(backend.min(backend.abs(eigenvalues)))
        if smallest <= 1e-12 * norm:
            raise_error(
                ValueError,
                "matrix is not invertible, and then condition_number must be given.",
            )
        condition_number = norm / smallest
    elif not condition_number >= 1:
        raise_error(
            ValueError,
            f"condition_number must be at least one, but it is {condition_number}.",
        )

    return matrix, nqubits, loading, eigenvalues, eigenvectors, norm, condition_number


def _hhl_original(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    vector: ArrayLike | Circuit,
    nclock: int,
    condition_number: float | None = None,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates the circuit of the HHL algorithm with phase estimation.

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix of size :math:`2^{n} \\times 2^{n}`.
        vector (ArrayLike or :class:`qibo.models.circuit.Circuit`): vector of :math:`2^{n}`
            entries, or circuit of :math:`n` qubits that prepares its normalized state.
        nclock (int): number of qubits of the clock register, which is at least two.
        condition_number (float, optional): bound of the condition number of ``matrix``.
            If ``None``, it is computed from its eigenvalues. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: circuit of the algorithm, and the constant :math:`C`.
    """
    backend = _check_backend(backend)

    if isinstance(nclock, bool) or not isinstance(nclock, Integral) or nclock < 2:
        raise_error(
            ValueError,
            f"nclock must be an integer of at least two, but it is {nclock}.",
        )

    matrix, nqubits, loading, eigenvalues, eigenvectors, norm, condition_number = (
        _hhl_inputs(matrix, vector, condition_number, backend)
    )

    constant = norm / condition_number
    nstates = 2**nclock
    time = math.pi * nstates / (2 * norm)
    resolution = 2 * math.pi / time

    flag = 0
    clock = list(range(1, nclock + 1))
    system = list(range(nclock + 1, nclock + nqubits + 1))

    # the eigenvalues are estimated in the clock register
    window = math.sqrt(2 / nstates) * backend.sin(
        math.pi * (backend.arange(nstates, dtype=backend.float64) + 0.5) / nstates
    )
    estimation = Circuit(nqubits + nclock + 1)
    estimation.add(
        binary_encoder(nclock, data=window, backend=backend).on_qubits(*clock)
    )
    for position, qubit in enumerate(clock):
        evolution = backend.matrix_exp(
            matrix,
            1j * time * 2 ** (nclock - 1 - position) / nstates,
            eigenvectors,
            eigenvalues,
        )
        estimation.add(gates.Unitary(evolution, *system).controlled_by(qubit))
    estimation.add(QFT(nclock).invert().on_qubits(*clock))

    # rotation of the flag, controlled by the integer k of the clock register
    angles = []
    for state in range(nstates):
        eigenvalue = 2 * math.pi * (state - nstates * (state >= nstates // 2)) / time
        if eigenvalue == 0 or abs(eigenvalue) < constant - resolution:
            angles.append(0.0)
            continue

        ratio = min(1.0, constant / abs(eigenvalue)) * (1 if eigenvalue > 0 else -1)
        angles.append(2 * float(backend.arcsin(ratio)))

    circuit = Circuit(nqubits + nclock + 1, **kwargs)
    circuit.add(loading.on_qubits(*system))
    circuit.add(estimation.on_qubits(*range(nqubits + nclock + 1)))
    circuit.add(
        gates.Multiplexor(clock, flag, backend.cast(angles, dtype=backend.float64))
    )
    circuit.add(estimation.invert().on_qubits(*range(nqubits + nclock + 1)))

    return circuit, constant


def _hhl_qsvt(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    vector: ArrayLike | Circuit,
    epsilon: float,
    condition_number: float | None = None,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates the circuit of the HHL algorithm with quantum singular value transformation.

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix of size :math:`2^{n} \\times 2^{n}`.
        vector (ArrayLike or :class:`qibo.models.circuit.Circuit`): vector of :math:`2^{n}`
            entries, or circuit of :math:`n` qubits that prepares its normalized state.
        epsilon (float): precision of the approximation of the inverse, in :math:`(0, 1/2)`.
        condition_number (float, optional): bound of the condition number of ``matrix``.
            If ``None``, it is computed from its eigenvalues. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: circuit of the algorithm, and the constant :math:`C`.
    """
    backend = _check_backend(backend)

    if not 0 < epsilon < 0.5:
        raise_error(ValueError, f"epsilon must be in (0, 1/2), but it is {epsilon}.")

    matrix, nqubits, loading, _, _, norm, condition_number = _hhl_inputs(
        matrix, vector, condition_number, backend
    )

    # Chebyshev series of the odd polynomial (1 - (1 - x^2)^m) / x, which is 2 * epsilon close to
    # 1 / x in [1 / kappa, 1], truncated at the degree 2 * cutoff + 1
    exponent = int(
        backend.ceil(
            condition_number**2
            * backend.log(
                backend.cast(condition_number / epsilon, dtype=backend.float64)
            )
        )
    )
    cutoff = min(
        exponent - 1,
        int(
            backend.ceil(
                backend.sqrt(
                    exponent
                    * backend.log(
                        backend.cast(4 * exponent / epsilon, dtype=backend.float64)
                    )
                )
            )
        ),
    )
    degree = 2 * cutoff + 1

    # the sum of the binomial coefficients above the center, which loses one term at each step
    coefficients = [0.0] * (degree + 1)
    tail = (4**exponent - math.comb(2 * exponent, exponent)) // 2
    for index in range(cutoff + 1):
        coefficients[2 * index + 1] = 4 * (-1) ** index * tail / 4**exponent
        tail -= math.comb(2 * exponent, exponent + index + 1)
    coefficients = backend.cast(coefficients, dtype=backend.float64)

    # the polynomial is divided by its maximum to be bounded by one
    nsamples = 64 * (degree + 1)
    angles = backend.reshape(
        math.pi * backend.arange(nsamples, dtype=backend.float64) / nsamples, (-1, 1)
    )
    values = backend.matmul(
        backend.cos(angles * backend.arange(degree + 1, dtype=backend.float64)),
        coefficients,
    )
    peak = float(backend.max(backend.abs(values)))
    phases = qsvt_phases(coefficients / peak, backend=backend)

    block, alpha, nauxiliary = block_encoding_circuit(
        matrix, method="dense", alpha=norm, backend=backend
    )
    transformation = qsvt_circuit(block, phases, nauxiliary)

    circuit = Circuit(transformation.nqubits, **kwargs)
    circuit.add(
        loading.on_qubits(
            *range(transformation.nqubits - nqubits, transformation.nqubits)
        )
    )
    circuit.add(transformation.on_qubits(*range(transformation.nqubits)))

    return circuit, alpha / peak
