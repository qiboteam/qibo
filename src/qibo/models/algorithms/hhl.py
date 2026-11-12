import math
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms.qaa import _check_circuits
from qibo.models.algorithms.qft import QFT
from qibo.models.circuit import Circuit
from qibo.models.encodings import binary_encoder


def HHL(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    vector: ArrayLike | Circuit,
    nclock: int,
    condition_number: float | None = None,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates the circuit of the Harrow-Hassidim-Lloyd (HHL) algorithm for linear systems.

    Let :math:`A` be a Hermitian matrix of size :math:`N \\times N`, with :math:`N = 2^{n}`, and
    let :math:`\\ket{b}` be the state with the normalized entries of a vector :math:`b`. The
    circuit prepares a state whose component with the first qubit, the flag, in :math:`\\ket{1}`
    is proportional to :math:`A^{-1} \\ket{b}`, which is the solution :math:`\\ket{x}` of the
    linear system :math:`A x = b`. The circuit acts on the flag, followed by the ``nclock``
    qubits of the clock register, with :math:`T = 2^{\\texttt{nclock}}` states, and the
    :math:`n` qubits of the input register, where :math:`\\ket{b}` is loaded. It is composed of
    the following steps, where :math:`\\lVert A \\rVert` is the largest absolute value of the
    eigenvalues of :math:`A`:

    1. the loading of :math:`\\ket{b}`, and of the state
       :math:`\\ket{\\Psi_{0}} = \\sqrt{2 / T} \\sum_{\\tau = 0}^{T - 1}
       \\sin(\\pi (\\tau + 1 / 2) / T) \\ket{\\tau}` in the clock register;
    2. the controlled evolution :math:`\\sum_{\\tau} \\ket{\\tau}\\bra{\\tau} \\otimes
       e^{i A t_{0} \\tau / T}`, with :math:`t_{0} = \\pi T / (2 \\lVert A \\rVert)`;
    3. the inverse :func:`qibo.models.QFT` of the clock register, which then has the estimate
       :math:`\\tilde{\\lambda}_{k} = 2 \\pi k / t_{0}` of an eigenvalue of :math:`A`, where the
       integer :math:`k` is in two's complement, so that the eigenvalues can be negative;
    4. the rotation of the flag controlled by the clock register,
       :math:`\\ket{0} \\mapsto \\sqrt{1 - C^{2} / \\tilde{\\lambda}_{k}^{2}} \\ket{0} +
       (C / \\tilde{\\lambda}_{k}) \\ket{1}`, where :math:`C = \\lVert A \\rVert / \\kappa`, and
       :math:`\\kappa` is the condition number of :math:`A`. The amplitude is at most one, and
       the clock values that are smaller than :math:`C` by more than the resolution
       :math:`2 \\pi / t_{0}` of the clock are discarded, which includes :math:`k = 0`;
    5. the inverse of the steps 3, 2 and the preparation of :math:`\\ket{\\Psi_{0}}`, which
       takes the clock register back to :math:`\\ket{0}` up to the error of the estimation.

    The state of the input register, if the flag is measured in :math:`\\ket{1}`, is close to
    :math:`\\ket{x}`. The probability of this outcome is close to
    :math:`C^{2} \\lVert A^{-1} b \\rVert^{2} / \\lVert b \\rVert^{2}`, which gives the norm of
    the solution. The relative error of the eigenvalues is of the order of :math:`4 \\kappa / T`,
    and, therefore, it needs :math:`T \\gtrsim 4 \\kappa / \\epsilon` for an error
    :math:`\\epsilon`. The matrix is exponentiated exactly, which is inefficient in the number
    of qubits :math:`n`, and it is only meant for small systems.

    Example:
        Solve :math:`A x = b` with :math:`A = [[1, -1/3], [-1/3, 1]]` and :math:`b = [0, 1]`,
        whose normalized solution is :math:`[1, 3] / \\sqrt{10}`.

        .. testcode::

            import numpy as np

            from qibo.models import hhl

            matrix = np.array([[1.0, -1 / 3], [-1 / 3, 1.0]])
            vector = np.array([0.0, 1.0])

            nclock = 6
            circuit, constant = hhl(matrix, vector, nclock)

            # the flag is the first qubit, and the clock register is the second one
            state = circuit().state().reshape(2, 2**nclock, 2)
            solution = state[1, 0]

            print(np.round(np.abs(solution) ** 2 / np.sum(np.abs(solution) ** 2), 2))

        .. testoutput::

            [0.1 0.9]

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix :math:`A`, or Hamiltonian with its matrix. Its size is
            :math:`2^{n} \\times 2^{n}`, with :math:`n \\geq 1`.
        vector (ArrayLike or :class:`qibo.models.circuit.Circuit`): vector :math:`b` of the
            :math:`2^{n}` entries, which is normalized, or circuit of :math:`n` qubits that
            prepares :math:`\\ket{b}` from the all-zero state. A circuit cannot contain
            measurements.
        nclock (int): number of qubits of the clock register, which is at least two.
        condition_number (float, optional): bound :math:`\\kappa \\geq 1` of the ratio of the
            largest and the smallest absolute values of the eigenvalues of :math:`A`. If it is
            smaller than the true condition number, only the eigenvalues that are larger than
            :math:`\\lVert A \\rVert / \\kappa` are inverted, and the solution is the
            pseudoinverse of that part. If ``None``, it is computed from the eigenvalues of
            :math:`A`, which are not zero. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: circuit of the algorithm, and the constant :math:`C`.

    References:
        1. A. W. Harrow, A. Hassidim, and S. Lloyd, *Quantum algorithm for linear systems of
        equations*, `Phys. Rev. Lett. 103, 150502 (2009)
        <https://doi.org/10.1103/PhysRevLett.103.150502>`_.
    """
    backend = _check_backend(backend)

    if isinstance(nclock, bool) or not isinstance(nclock, Integral) or nclock < 2:
        raise_error(
            ValueError,
            f"nclock must be an integer of at least two, but it is {nclock}.",
        )

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
    rotation = Circuit(nqubits + nclock + 1)
    for state in range(nstates):
        eigenvalue = 2 * math.pi * (state - nstates * (state >= nstates // 2)) / time
        if eigenvalue == 0 or abs(eigenvalue) < constant - resolution:
            continue

        ratio = min(1.0, constant / abs(eigenvalue)) * (1 if eigenvalue > 0 else -1)
        angle = 2 * float(backend.arcsin(ratio))
        flipped = [
            qubit
            for qubit, bit in zip(clock, format(state, f"0{nclock}b"))
            if bit == "0"
        ]
        rotation.add(gates.X(qubit) for qubit in flipped)
        rotation.add(gates.RY(flag, theta=angle).controlled_by(*clock))
        rotation.add(gates.X(qubit) for qubit in flipped)

    circuit = Circuit(nqubits + nclock + 1, **kwargs)
    circuit.add(loading.on_qubits(*system))
    circuit.add(estimation.on_qubits(*range(nqubits + nclock + 1)))
    circuit.add(rotation.on_qubits(*range(nqubits + nclock + 1)))
    circuit.add(estimation.invert().on_qubits(*range(nqubits + nclock + 1)))

    return circuit, constant
