import cmath
import functools
import math

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates, matrices
from qibo.backends import Backend, _check_backend, _numpy_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.quantum_info.linalg_operations import (
    cosine_sin_decomposition,
    schur_decomposition,
)
from qibo.transpiler._exceptions import DecompositionError

magic_basis = np.array(
    [[1, -1j, 0, 0], [0, 0, 1, -1j], [0, 0, -1, -1j], [1, 1j, 0, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [1, -1, 0, 0]]
) / np.sqrt(2)


# Euler bases of
# :func:`qibo.transpiler.decompositions.unitary_decompositions.single_qubit_decomposition`
# and :class:`qibo.transpiler.optimizer.Optimize1qGatesDecomposition`. Each name maps
# to the names of the gates that the basis needs.
_EULER_BASES = {
    "U3": ("u3",),
    "U321": ("u1", "u2", "u3"),
    "ZYZ": ("rz", "ry"),
    "ZXZ": ("rz", "rx"),
    "XZX": ("rx", "rz"),
    "XYX": ("rx", "ry"),
    "ZSX": ("rz", "sx"),
    "ZSXX": ("rz", "sx", "x"),
}

# Single-qubit gates without parameters and the angle of the rotations of the gates
# with an angle. Each rotation is given by its axis and by the extra arguments of the gate.
_FIXED_GATES = (
    gates.H,
    gates.S,
    gates.SDG,
    gates.SX,
    gates.SXDG,
    gates.T,
    gates.TDG,
    gates.X,
    gates.Y,
    gates.Z,
)
_ROTATIONS = {
    gates.RX: (((1.0, 0.0, 0.0), ()),),
    gates.RY: (((0.0, 1.0, 0.0), ()),),
    gates.RZ: (((0.0, 0.0, 1.0), ()),),
    gates.PRX: (
        ((1.0, 0.0, 0.0), (0.0,)),
        ((0.0, 1.0, 0.0), (math.pi / 2,)),
    ),
}


def u3_decomposition(
    unitary: ArrayLike, backend: Backend
) -> tuple[float, float, float]:
    """Decomposes arbitrary one-qubit gates to U3.

    The :class:`qibo.gates.gates.U3` gate is equal to the unitary divided by the square
    root of its determinant, so it implements the unitary up to a global phase, which
    is half the phase of the determinant of the unitary.

    Args:
        unitary (ArrayLike): Unitary :math:`2 \\times 2` matrix to be decomposed.
        backend (:class:`qibo.backends.abstract.Backend`): Backend to be used in the
            calculation.

    Returns:
        Tuple[float, float, float]: Parameters of :class:`qibo.gates.gates.U3` gate.
    """
    unitary = backend.cast(unitary)
    # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
    su2 = unitary / backend.sqrt(backend.det(unitary))
    theta = 2 * backend.arctan2(backend.abs(su2[1, 0]), backend.abs(su2[0, 0]))
    plus = backend.angle(su2[1, 1])
    minus = backend.angle(su2[1, 0])
    phi = plus + minus
    lam = plus - minus
    # explicit conversion to float to avoid issue on GPU
    return float(theta), float(phi), float(lam)


def calculate_psi(
    unitary: ArrayLike,
    magic_basis: ArrayLike = magic_basis,
    weight: float = math.sqrt(2),
    backend: Backend | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Solves the eigenvalue problem of :math:`U^{T} U`.

    See step (1) of Appendix A in arXiv:quant-ph/0011050.

    Args:
        unitary (ArrayLike): Unitary matrix of the gate we are decomposing
            in the computational basis.
        magic_basis (ArrayLike, optional): basis in which to solve the eigenvalue problem.
            Defaults to ``magic basis``.
        weight (float, optional): The matrix :math:`M = U^{T} U` written in the magic basis
            is symmetric and unitary, so its real and imaginary parts have the same
            eigenvectors. They are found by diagonalizing the real matrix
            :math:`\\text{Re}(M) + w \\, \\text{Im}(M)`, where :math:`w` is ``weight``.
            Two different eigenvalues :math:`e^{i \\varphi_{1}}` and
            :math:`e^{i \\varphi_{2}}` of :math:`M` become equal in this matrix if
            :math:`\\tan((\\varphi_{1} + \\varphi_{2}) / 2) = w`, and the eigenvectors are
            then not reliable. The default avoids this for angles that are multiples
            of :math:`\\pi / 4`, which are common in circuits made of Clifford and
            :math:`T` gates. Defaults to :math:`\\sqrt{2}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Eigenvectors in the computational basis
        and eigenvalues of :math:`U^{T} U`.
    """
    backend = _check_backend(backend)

    magic_basis = backend.cast(magic_basis)
    unitary = backend.cast(unitary)
    # write unitary in magic basis
    u_magic = backend.conj(magic_basis).T @ unitary @ magic_basis
    # construct and diagonalize UT_U
    ut_u = u_magic.T @ u_magic
    ut_u_real = backend.real(ut_u) + weight * backend.imag(ut_u)
    if backend.__class__.__name__ not in ("PyTorchBackend", "TensorflowBackend"):
        ut_u_real = backend.round(ut_u_real, decimals=15)

    _eigvals_real, psi_magic = backend.eigenvectors(ut_u_real, hermitian=True)
    # compute full eigvals as <psi|ut_u|psi>, as eigvals_real is only real
    eigvals = backend.sum(backend.conj(psi_magic) * (ut_u @ psi_magic), 0)
    # orthogonalize eigenvectors in the case of degeneracy (Gram-Schmidt)
    psi_magic, _ = backend.qr(psi_magic)
    # write psi in computational basis
    psi = magic_basis @ psi_magic
    return psi, eigvals


def calculate_single_qubit_unitaries(
    psi: ArrayLike, backend: Backend | None = None
) -> tuple[ArrayLike, ArrayLike]:
    """Calculates local unitaries that map a maximally entangled basis to the magic basis.

    See Lemma 1 of Appendix A in Ref. [1]. In the magic basis, the local unitaries are the
    real orthogonal matrices with unit determinant. The local unitary that maps the
    :math:`k`-th state of ``psi``, which is real in the magic basis, to the magic basis
    state :math:`\\pi(k)`, with :math:`\\pi = (0, 1, 3, 2)`, is then given by the transpose
    of that real matrix with its rows permuted, up to the sign of one row that fixes the
    determinant. It is factorized into :math:`U_{A} \\otimes U_{B}` with the singular value
    decomposition of its realignment. The states are mapped up to a phase, and the unitaries
    are defined up to a phase :math:`e^{i \\phi}` for :math:`U_{A}` and :math:`e^{-i \\phi}`
    for :math:`U_{B}`.

    Args:
        psi (ArrayLike): Maximally entangled two-qubit states that define a basis.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be used
            in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Local unitaries UA and UB that map the given
        basis to the magic basis.

    References:
        1. B. Kraus and J. I. Cirac, *Optimal creation of entanglement using a two-qubit gate*,
        `Phys. Rev. A 63, 062309 (2001) <https://doi.org/10.1103/PhysRevA.63.062309>`_.

    """
    backend = _check_backend(backend)
    magic = backend.cast(magic_basis)

    psi_magic = backend.dagger(magic) @ psi
    if (
        backend.real(backend.matrix_norm(backend.imag(psi_magic))) > 1e-6
    ):  # pragma: no cover
        raise_error(NotImplementedError, "Given state is not real in the magic basis.")

    orthogonal = backend.real(psi_magic).T[[0, 1, 3, 2]]
    if backend.det(orthogonal) < 0:
        orthogonal = (
            backend.cast([[-1], [1], [1], [1]], dtype=orthogonal.dtype) * orthogonal
        )
    product = (
        magic @ backend.cast(orthogonal, dtype=magic.dtype) @ backend.dagger(magic)
    )

    left, singular, right = backend.singular_value_decomposition(
        backend.reshape(
            backend.transpose(backend.reshape(product, (2, 2, 2, 2)), (0, 2, 1, 3)),
            (4, 4),
        )
    )
    factor = math.sqrt(singular[0])
    return factor * left[:, 0].reshape(2, 2), factor * right[0].reshape(2, 2)


def calculate_diagonal(
    unitary: ArrayLike,
    ua: ArrayLike,
    ub: ArrayLike,
    va: ArrayLike,
    vb: ArrayLike,
    backend: Backend,
) -> tuple[ArrayLike, ...]:
    """Calculates Ud matrix that can be written as exp(-iH).

    See Eq. (A1) in arXiv:quant-ph/0011050.
    Ud is diagonal in the magic and Bell basis.
    Also returns local unitaries that modify Ud so: pi/4 >= hx >= hy >= |hz|
    """
    # normalize U_A, U_B, V_A, V_B so that detU_d = 1
    # this is required so that sum(lambdas) = 0
    # and Ud can be written as exp(-iH)
    det = backend.det(unitary) ** (1 / 16)
    ua *= det
    ub *= det
    va *= det
    vb *= det
    dag = lambda u: backend.conj(u).T
    u_dagger = dag(
        backend.kron(
            ua,
            ub,
        )
    )
    v_dagger = dag(backend.kron(va, vb))
    ud = u_dagger @ unitary @ v_dagger

    lambdas = to_bell_diagonal(ud, backend=backend)

    # We permute to ensure we will have hx >= hy >= hz in the end
    hx, hy, hz = calculate_h_vector(lambdas, backend)

    # 1. force coefficients to be in pi/4 >= ... >= 0 interval, using pi/2 periodicity and
    # pi/4 symmetry, and 2. permute them to ensure the ordering
    fit = lambda alpha: min(
        alpha % (math.pi / 2), math.pi / 2 - (alpha % (math.pi / 2))
    )
    alphas = (hx, hy, hz)
    permutation = sorted(range(3), key=lambda k: fit(alphas[k]), reverse=True)

    # The permutation is a change of basis by Clifford unitaries, which are the same for
    # both qubits, and are applied to the left of the local unitaries and, inverted,
    # to the right of them. Only the unitaries of the second qubit differ afterwards.
    left_a, right_a = backend.matrices.I(), backend.matrices.I()
    if permutation[0] != 0:
        clifford = (None, backend.matrices.S, backend.matrices.H)[permutation[0]]
        left_a, right_a = left_a @ clifford, dag(clifford) @ right_a
    if not (permutation[1] == 1 or permutation[2] == 2):
        clifford = backend.matrices.S @ backend.matrices.H @ backend.matrices.S
        left_a, right_a = left_a @ clifford, dag(clifford) @ right_a
    left_b, right_b = left_a, right_a

    # 3. find local corrections to enforce, as possible, conditions on h
    paulis = (backend.matrices.X, backend.matrices.Y, backend.matrices.Z)
    for i, k in enumerate(permutation):
        alpha, pauli = alphas[k], paulis[i]
        if i < 2:
            if alpha < 0:
                alpha += math.pi / 2
                left_a = left_a @ (1j * pauli)
                left_b = left_b @ pauli
            if (
                alpha > math.pi / 4
            ):  # can't conjugate so sign of z alternates to compensate
                # Swap alpha_i and alpha_z sign
                other = paulis[(i + 1) % 2]
                left_b = left_b @ other
                right_b = other @ right_b
                # Add pi/2 to alpha_i
                left_a = left_a @ (1j * pauli)
                left_b = left_b @ pauli
        elif abs(alpha) > math.pi / 4:
            left_a = left_a @ ((1j if alpha < 0 else -1j) * pauli)
            left_b = left_b @ pauli

    # 4. apply corrections
    ua = ua @ left_a
    ub = ub @ left_b
    va = right_a @ va
    vb = right_b @ vb
    ud = backend.matmul(
        backend.kron(dag(left_a), dag(left_b)),
        backend.matmul(ud, backend.kron(dag(right_a), dag(right_b))),
    )
    return ua, ub, ud, va, vb


def magic_decomposition(
    unitary: ArrayLike,
    backend: Backend | None = None,
    weight: float = math.sqrt(2),
) -> tuple[ArrayLike, ...]:
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050.

    The argument ``weight`` is defined in
    :func:`qibo.transpiler.decompositions.unitary_decompositions.calculate_psi`.
    """
    backend = _check_backend(backend)
    unitary = backend.cast(unitary)
    psi, eigvals = calculate_psi(unitary, backend=backend, weight=weight)
    psi_tilde = backend.conj(backend.sqrt(eigvals)) * backend.matmul(unitary, psi)
    va, vb = calculate_single_qubit_unitaries(psi, backend=backend)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde, backend=backend)
    dag = lambda U: backend.dagger(U)
    ua, ub = dag(ua_dagger), dag(ub_dagger)
    return calculate_diagonal(unitary, ua, ub, va, vb, backend=backend)


def to_bell_diagonal(
    ud: ArrayLike, bell_basis: ArrayLike = bell_basis, backend: Backend | None = None
) -> ArrayLike | None:
    """Transforms a matrix to the Bell basis and checks if it is diagonal."""
    backend = _check_backend(backend)

    ud = backend.cast(ud)
    bell_basis = backend.cast(bell_basis)

    ud_bell = backend.conj(bell_basis).T @ ud @ bell_basis
    ud_diag = backend.diag(ud_bell)

    if not backend.allclose(
        backend.diag(ud_diag), ud_bell, atol=1e-6, rtol=1e-6
    ):  # pragma: no cover
        return None

    uprod = backend.prod(ud_diag)

    if not backend.allclose(uprod, 1.0, atol=1e-6, rtol=1e-6):  # pragma: no cover
        return None

    return ud_diag


def calculate_h_vector(
    ud_diag: ArrayLike, backend: Backend
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Finds h parameters corresponding to exp(-iH).

    See Eq. (4)-(5) in arXiv:quant-ph/0307177.
    """
    lambdas = -backend.angle(ud_diag)
    hx = (lambdas[0] + lambdas[2]) / 2.0
    hy = (lambdas[1] + lambdas[2]) / 2.0
    hz = (lambdas[0] + lambdas[1]) / 2.0
    return hx, hy, hz


def cnot_decomposition(
    q0: int, q1: int, hx: float, hy: float, hz: float, backend: Backend
) -> list[Gate]:
    """Performs decomposition (6) from arXiv:quant-ph/0307177."""
    h = backend.matrices.H
    u3 = -1j * h
    # use corrected version from PRA paper (not arXiv)
    u2 = -u3 @ gates.RX(0, 2 * hx - math.pi / 2).matrix(backend)
    # add an extra exp(-i pi / 4) global phase to get exact match
    v2 = cmath.exp(-1j * math.pi / 4) * gates.RZ(0, 2 * hz).matrix(backend)
    v3 = gates.RZ(0, -2 * hy).matrix(backend)
    w = backend.cast((matrices.I - 1j * matrices.X) / math.sqrt(2))
    # change CNOT to CZ using Hadamard gates
    return [
        gates.H(q1),
        gates.CZ(q0, q1),
        gates.Unitary(u2, q0),
        gates.Unitary(h @ v2 @ h, q1),
        gates.CZ(q0, q1),
        gates.Unitary(u3, q0),
        gates.Unitary(h @ v3 @ h, q1),
        gates.CZ(q0, q1),
        gates.Unitary(w, q0),
        gates.Unitary(backend.conj(w).T @ h, q1),
    ]


def cnot_decomposition_light(
    q0: int, q1: int, hx: float, hy: float, backend: Backend
) -> list[Gate]:
    """Performs decomposition (24) from arXiv:quant-ph/0307177."""
    h = backend.matrices.H
    w = (backend.matrices.I(2) - 1j * backend.matrices.X) / math.sqrt(2)
    u2 = gates.RX(0, 2 * hx).matrix(backend)
    v2 = gates.RZ(0, -2 * hy).matrix(backend)
    # change CNOT to CZ using Hadamard gates
    return [
        gates.Unitary(backend.conj(w).T, q0),
        gates.Unitary(h @ w, q1),
        gates.CZ(q0, q1),
        gates.Unitary(u2, q0),
        gates.Unitary(h @ v2 @ h, q1),
        gates.CZ(q0, q1),
        gates.Unitary(w, q0),
        gates.Unitary(backend.conj(w).T @ h, q1),
    ]


def quantum_shannon_decomposition(
    qubits: tuple[int, ...], unitary: ArrayLike, backend: Backend | None = None
) -> list[Gate]:
    """Quantum Shannon decomposition of a unitary matrix acting on several qubits.

    The cosine-sine decomposition of an :math:`n`-qubit unitary :math:`U`, whose blocks
    are defined by the first qubit, is

    .. math::
        U = (A_{1} \\oplus A_{2}) \\, \\begin{pmatrix} C & -S \\\\ S & C \\end{pmatrix} \\,
        (B_{1} \\oplus B_{2}) \\, ,

    where the middle matrix is a multiplexed :class:`qibo.gates.gates.RY` rotation of
    the first qubit, controlled by the other :math:`n - 1` qubits. Each block-diagonal
    matrix is decomposed as

    .. math::
        X_{1} \\oplus X_{2} = (\\mathbb{I} \\otimes V) \\, (D \\oplus D^{\\dagger}) \\,
        (\\mathbb{I} \\otimes W) \\, ,

    with :math:`V D^{2} V^{\\dagger} = X_{1} X_{2}^{\\dagger}` and
    :math:`W = D V^{\\dagger} X_{2}`, where :math:`D \\oplus D^{\\dagger}` is a
    multiplexed :class:`qibo.gates.gates.RZ` rotation of the first qubit. The result
    has four unitaries on the last :math:`n - 1` qubits and three multiplexed
    rotations, *i.e.* :class:`qibo.gates.gates.Multiplexor` gates. Decomposing the
    unitaries in the same way until they act on two qubits uses
    :math:`(9 / 16) \\, 4^{n} - (3 / 2) \\, 2^{n}` two-qubit gates, if the unitaries
    on two qubits are decomposed with three of them. It does not use auxiliary qubits.

    Args:
        qubits (tuple[int, ...]): Ids of the qubits that ``unitary`` acts on, with at
            least two. The first one is the most significant bit of the indices of
            ``unitary``.
        unitary (ArrayLike): Unitary matrix of shape :math:`(2^{n}, 2^{n})`, with
            :math:`n` the number of ``qubits``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use
            for the matrices. If ``None``, defaults to the global backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates, in order of application, that
        implement ``unitary``: four :class:`qibo.gates.gates.Unitary` gates on
        ``qubits[1:]`` and three :class:`qibo.gates.gates.Multiplexor` gates. Each
        of them acts on less qubits than ``unitary``, so they can be decomposed in turn.

    Raises:
        ValueError: If there are less than two ``qubits``, or ``unitary`` does not have
            the shape :math:`(2^{n}, 2^{n})`.

    References:
        1. V. V. Shende, S. S. Bullock, and I. L. Markov,
        *Synthesis of quantum-logic circuits*,
        `IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst. 25, 1000 (2006)
        <https://doi.org/10.1109/TCAD.2005.855930>`_.
    """
    backend = _check_backend(backend)

    if len(qubits) < 2 or unitary.shape != (2 ** len(qubits),) * 2:
        raise_error(
            ValueError,
            f"``unitary`` must have shape {(2 ** len(qubits),) * 2} for {len(qubits)} "
            + f"qubits, and there must be at least two, got shape {unitary.shape}.",
        )

    half = len(unitary) // 2
    left, middle, right = cosine_sin_decomposition(unitary, backend=backend)
    angles = 2 * backend.arctan2(
        backend.real(backend.diag(middle[half:, :half])),
        backend.real(backend.diag(middle[:half, :half])),
    )
    return [
        *_block_diagonal_decomposition(
            qubits, right[:half, :half], right[half:, half:], backend=backend
        ),
        gates.Multiplexor(qubits[1:], qubits[0], angles, axis="y"),
        *_block_diagonal_decomposition(
            qubits, left[:half, :half], left[half:, half:], backend=backend
        ),
    ]


def single_qubit_decomposition(
    unitary: ArrayLike,
    qubit: int,
    gate_classes: tuple[type[Gate], ...],
    atol: float = 1e-12,
    backend: Backend | None = None,
) -> list[Gate]:
    """Decomposes a single-qubit unitary into the given single-qubit gates.

    If the gates contain an Euler basis of
    :class:`qibo.transpiler.optimizer.Optimize1qGatesDecomposition`, such as
    :class:`qibo.gates.RZ` and :class:`qibo.gates.SX`, the shortest decomposition in the
    Euler bases is returned. Otherwise, two rotations with free angles about orthogonal
    axes, :math:`R_{a}(\\alpha) R_{b}(\\beta) R_{a}(\\gamma)`, are used, where
    :math:`R_{n}(\\varphi)` is the rotation by the angle :math:`\\varphi` about the axis
    :math:`n`. The rotations are those of :class:`qibo.gates.RX`,
    :class:`qibo.gates.RY`, :class:`qibo.gates.RZ` and :class:`qibo.gates.PRX`, and
    their axes can be rotated by the gates without parameters among
    :class:`qibo.gates.H`, :class:`qibo.gates.S`, :class:`qibo.gates.SDG`,
    :class:`qibo.gates.SX`, :class:`qibo.gates.SXDG`, :class:`qibo.gates.T`,
    :class:`qibo.gates.TDG`, :class:`qibo.gates.X`, :class:`qibo.gates.Y` and
    :class:`qibo.gates.Z`, which are conjugated with the rotation. These gates are
    equal to the unitary up to a global phase, which is added as a
    :class:`qibo.gates.GlobalPhase` gate at the end if it is not zero.

    Args:
        unitary (ArrayLike): Unitary :math:`2 \\times 2` matrix to be decomposed.
        qubit (int): Qubit the gates act on.
        gate_classes (tuple[type[:class:`qibo.gates.abstract.Gate`], ...]): Single-qubit
            gates that can be used.
        atol (float, optional): Tolerance to decide that an angle is zero.
            Defaults to :math:`10^{-12}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use
            for calculations. If ``None``, defaults to the global backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that implement the unitary,
        followed by a :class:`qibo.gates.GlobalPhase` gate if the gates implement
        the unitary only up to a global phase.

    Raises:
        DecompositionError: If the gates have no Euler basis and no two rotations about
            orthogonal axes.
    """
    backend = _check_backend(backend)
    unitary = backend.cast(unitary)

    names = {gate_class.__name__.lower() for gate_class in gate_classes}
    bases = [name for name, gates_ in _EULER_BASES.items() if set(gates_) <= names]
    if bases:
        hadamard = gates.H(0).matrix(backend)
        angles = u3_decomposition(unitary, backend)
        angles_x = u3_decomposition(hadamard @ unitary @ hadamard, backend)
        decomposition = min(
            (
                _euler_sequence(name, qubit, angles, angles_x, atol, backend)
                for name in bases
            ),
            key=len,
        )
    else:
        frames = _euler_frames(tuple(gate_classes))
        if frames is None:
            raise_error(
                DecompositionError,
                "The single-qubit native gates must include rotations about two "
                + "orthogonal axes with free angles, which can be rotated by native "
                + "gates without parameters. U3 and GPI2 can be used as well.",
            )

        frame = backend.cast(frames[0], dtype=unitary.dtype)
        theta, phi, lam = u3_decomposition(
            backend.conj(frame).T @ unitary @ frame, backend
        )

        # If the middle rotation is the identity, the two outer rotations are merged.
        if abs(math.remainder(theta, 2 * math.pi)) < atol:
            rotations = [(lam + phi, frames[1])]
        else:
            rotations = [(lam, frames[1]), (theta, frames[2]), (phi, frames[1])]

        decomposition = []
        for angle, (gate_class, extra, word, inverse_word) in rotations:
            if abs(math.remainder(angle, 2 * math.pi)) > atol:
                decomposition.extend(gate(qubit) for gate in inverse_word)
                decomposition.append(gate_class(qubit, angle, *extra))
                decomposition.extend(gate(qubit) for gate in word)

    # The gates implement the unitary up to a global phase, which is the phase of the
    # overlap between the unitary and the product of the gates.
    product = backend.identity(2, dtype=unitary.dtype)
    for gate in decomposition:
        product = gate.matrix(backend) @ product
    phase = float(backend.angle(backend.sum(backend.conj(product) * unitary)))
    if abs(math.remainder(phase, 2 * math.pi)) > atol:
        decomposition.append(gates.GlobalPhase(phase % (2 * math.pi), trainable=False))

    return decomposition


def two_qubit_decomposition(
    q0: int,
    q1: int,
    unitary: ArrayLike,
    threshold: float = 1e-6,
    weight: float = math.sqrt(2),
    backend: Backend | None = None,
) -> list[Gate]:
    """Performs two qubit unitary gate decomposition.

    The decomposition has at most three :class:`qibo.gates.CZ` gates [1]. It has two or
    fewer if the unitary needs fewer, which is decided from its Weyl coordinates, see
    :func:`qibo.transpiler.decompositions.unitary_decompositions.weyl_decomposition`: a
    product of two single-qubit unitaries uses no :class:`qibo.gates.CZ` gate, and a
    unitary that is locally equivalent to :class:`qibo.gates.CZ` uses one.

    Args:
        q0 (int): index of the first qubit.
        q1 (int): index of the second qubit.
        unitary (ndarray): Unitary :math:`4 \\times 4` to be decomposed.
        threshold (float): Tolerance of the decomposition, which is of the order of the
            error that neglecting a Weyl coordinate causes, see
            :func:`qibo.transpiler.decompositions.unitary_decompositions.weyl_decomposition`.
        weight (float, optional): Weight of the imaginary part in the matrix that is
            diagonalized to find the local gates, see
            :func:`qibo.transpiler.decompositions.unitary_decompositions.calculate_psi`.
            Defaults to :math:`\\sqrt{2}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list: gates implementing the decomposition

    References:
        1. F. Vatan and C. Williams,
        *Optimal quantum circuits for general two-qubit gates*,
        `Phys. Rev. A 69, 032315 (2004) <https://doi.org/10.1103/PhysRevA.69.032315>`_.
    """
    backend = _check_backend(backend)

    # Handle identity case efficiently
    if backend.allclose(unitary, backend.identity(4)):
        return []

    cz_count, (hx, hy, hz), (u4, v4, ud, u1, v1) = weyl_decomposition(
        unitary, threshold=threshold, weight=weight, backend=backend
    )

    if cz_count < 2:
        # The interaction part of the unitary is then a product of Pauli rotations by
        # multiples of pi / 2, which are local, and, if a CZ gate is needed, one rotation
        # by an odd multiple of pi / 4 about an axis P. The latter is exp(i pi / 4 PP) =
        # exp(i pi / 4) (K x K) CZ (S^dagger K^dagger x S^dagger K^dagger), with
        # K Z K^dagger = P. Without it, the interaction is itself a product.
        local = ud
        if cz_count == 1:
            hadamard = backend.matrices.H
            phase = backend.matrices.S
            axis = max(
                range(3),
                key=lambda k: abs(math.remainder((hx, hy, hz)[k], math.pi / 2)),
            )
            pauli, frame = (
                (backend.matrices.X, hadamard),
                (backend.matrices.Y, phase @ hadamard),
                (backend.matrices.Z, backend.matrices.I(2)),
            )[axis]
            entangler = (
                backend.matrices.I(4) + 1j * backend.kron(pauli, pauli)
            ) / math.sqrt(2)
            local = ud @ backend.dagger(entangler)

        first, coefficients, second = backend.singular_value_decomposition(
            backend.reshape(
                backend.transpose(backend.reshape(local, (2, 2, 2, 2)), (0, 2, 1, 3)),
                (4, 4),
            )
        )
        factor = math.sqrt(coefficients[0])
        first = factor * first[:, 0].reshape(2, 2)
        second = factor * second[0].reshape(2, 2)
        if cz_count == 0:
            return [
                gates.Unitary(u4 @ first @ u1, q0),
                gates.Unitary(v4 @ second @ v1, q1),
            ]

        before = backend.dagger(phase) @ backend.dagger(frame)
        after = cmath.exp(1j * math.pi / 4) * frame
        return [
            gates.Unitary(before @ u1, q0),
            gates.Unitary(before @ v1, q1),
            gates.CZ(q0, q1),
            gates.Unitary(u4 @ first @ after, q0),
            gates.Unitary(v4 @ second @ frame, q1),
        ]

    if cz_count == 2:
        # Theorem 2 decomposition (2 CNOTs) for the hz = 0 (modulo pi) case.
        gatelist = cnot_decomposition_light(q0, q1, hx, hy, backend=backend)

        # Combine with initial and final local unitaries
        g0, g1 = gatelist[:2]
        gatelist[0] = gates.Unitary(backend.cast(g0.parameters[0]) @ u1, q0)
        gatelist[1] = gates.Unitary(backend.cast(g1.parameters[0]) @ v1, q1)

        g0, g1 = gatelist[-2:]
        # A shift of ``hz`` by pi multiplies the unitary by a global phase of -1.
        sign = (-1) ** round(hz / math.pi)
        gatelist[-2] = gates.Unitary(sign * u4 @ g0.parameters[0], q0)
        gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

        return gatelist

    # Theorem 1 decomposition (3 CNOTs) for the hz != 0 case.
    cnot_dec = cnot_decomposition(q0, q1, hx, hy, hz, backend=backend)

    # Combine with initial and final local unitaries
    gatelist = [
        gates.Unitary(u1, q0),
        gates.Unitary(backend.matrices.H @ v1, q1),
    ]
    gatelist.extend(cnot_dec[1:])
    g0, g1 = gatelist[-2:]
    gatelist[-2] = gates.Unitary(u4 @ g0.parameters[0], q0)
    gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

    return gatelist


def weyl_decomposition(
    unitary: ArrayLike,
    threshold: float = 1e-6,
    weight: float = math.sqrt(2),
    backend: Backend | None = None,
) -> tuple[int, tuple[float, float, float], tuple[ArrayLike, ...]]:
    """Decomposes a two-qubit unitary into local unitaries and an interaction.

    The unitary :math:`U` is written as

    .. math::
        U = \\left(U_{4} \\otimes V_{4}\\right) \\,
        e^{-i \\left(h_{x} \\, XX + h_{y} \\, YY + h_{z} \\, ZZ\\right)} \\,
        \\left(U_{1} \\otimes V_{1}\\right) \\, ,

    where the interaction is the :math:`U_{d}` of
    :func:`qibo.transpiler.decompositions.unitary_decompositions.magic_decomposition`, and
    :math:`h_{x}`, :math:`h_{y}` and :math:`h_{z}` are its Weyl coordinates [1]. Local
    unitaries change a Weyl coordinate by multiples of :math:`\\pi / 2` only, and
    permute them, so only these coordinates modulo :math:`\\pi / 2` matter for the number
    of :class:`qibo.gates.CZ` gates that the unitary needs [2]:

    - none if all the coordinates are multiples of :math:`\\pi / 2`, *i.e.* the
      unitary is a product of single-qubit unitaries;
    - one if a coordinate is an odd multiple of :math:`\\pi / 4` and the others are
      multiples of :math:`\\pi / 2`, *i.e.* the unitary is locally equivalent to
      :class:`qibo.gates.CZ`;
    - two if :math:`h_{z}` is a multiple of :math:`\\pi`, which the decomposition ensures
      for any unitary with a coordinate that is a multiple of :math:`\\pi / 2`;
    - three otherwise.

    A coordinate is taken to be a multiple of :math:`\\pi / 2` if its distance to it is
    below ``threshold / 2`` for the first two cases, as it is about half of the
    operator-Schmidt coefficient that it causes, and below ``threshold`` for the third.
    Neglecting a coordinate then causes an error of the order of ``threshold``.

    Args:
        unitary (ArrayLike): Unitary :math:`4 \\times 4` to be decomposed.
        threshold (float, optional): Tolerance for taking a Weyl coordinate to be
            a multiple of :math:`\\pi / 2`. Defaults to :math:`10^{-6}`.
        weight (float, optional): Weight of the imaginary part in the matrix that is
            diagonalized to find the local gates, see
            :func:`qibo.transpiler.decompositions.unitary_decompositions.calculate_psi`.
            Defaults to :math:`\\sqrt{2}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: Number of :class:`qibo.gates.CZ` gates that the unitary needs, its Weyl
        coordinates :math:`(h_{x}, h_{y}, h_{z})`, and the output of
        :func:`qibo.transpiler.decompositions.unitary_decompositions.magic_decomposition`,
        *i.e.* :math:`U_{4}`, :math:`V_{4}`, :math:`U_{d}`, :math:`U_{1}` and :math:`V_{1}`.

    References:
        1. J. Zhang, J. Vala, S. Sastry and K. B. Whaley,
        *Geometric theory of nonlocal two-qubit operations*,
        `Phys. Rev. A 67, 042313 (2003) <https://doi.org/10.1103/PhysRevA.67.042313>`_.

        2. V. V. Shende, I. L. Markov and S. S. Bullock,
        *Minimal universal two-qubit controlled-NOT-based circuits*,
        `Phys. Rev. A 69, 062321 (2004) <https://doi.org/10.1103/PhysRevA.69.062321>`_.
    """
    backend = _check_backend(backend)

    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend, weight=weight)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    coordinates = tuple(float(h) for h in calculate_h_vector(ud_diag, backend=backend))

    # The distances of the coordinates to the multiples of pi / 2, in decreasing order
    first, second, _ = sorted(
        (abs(math.remainder(h, math.pi / 2)) for h in coordinates), reverse=True
    )
    if first < threshold / 2:
        cz_count = 0
    elif abs(first - math.pi / 4) < threshold / 2 and second < threshold / 2:
        cz_count = 1
    elif abs(math.remainder(coordinates[2], math.pi)) < threshold:
        cz_count = 2
    else:
        cz_count = 3

    return cz_count, coordinates, (u4, v4, ud, u1, v1)


def _block_diagonal_decomposition(
    qubits: tuple[int, ...],
    first_block: ArrayLike,
    second_block: ArrayLike,
    backend: Backend | None = None,
) -> list[Gate]:
    """Decompose a block-diagonal unitary, see :func:`quantum_shannon_decomposition`.

    Args:
        qubits (tuple[int, ...]): Qubits that the block-diagonal unitary acts on, where
            the first one selects the block.
        first_block (ArrayLike): Block for the first qubit in the state :math:`\\ket{0}`.
        second_block (ArrayLike): Block for the first qubit in the state :math:`\\ket{1}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use for the
            matrices.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: A unitary and a multiplexed
        :class:`qibo.gates.gates.RZ` rotation, followed by a unitary, in order of
        application.
    """
    backend = _check_backend(backend)

    schur_form, eigenvectors = schur_decomposition(
        first_block @ backend.dagger(second_block), backend=backend
    )
    eigenphases = backend.angle(backend.diag(schur_form))
    second_unitary = backend.exp(0.5j * eigenphases)[:, None] * (
        backend.dagger(eigenvectors) @ second_block
    )
    return [
        gates.Unitary(second_unitary, *qubits[1:]),
        gates.Multiplexor(qubits[1:], qubits[0], -eigenphases, axis="z"),
        gates.Unitary(eigenvectors, *qubits[1:]),
    ]


@functools.cache
def _euler_frames(gate_classes: tuple[type[Gate], ...]) -> tuple | None:
    """Finds two rotations about orthogonal axes for :func:`single_qubit_decomposition`.

    Rotations of the given gates are also considered when their axes are rotated by a
    sequence of the gates without parameters, :math:`G R_{n}(\\varphi) G^{\\dagger}`,
    where :math:`G` is the unitary of the sequence. Sequences with at most four gates
    are used. The rotations with the lowest total number of gates are chosen.

    Args:
        gate_classes (tuple[type[:class:`qibo.gates.abstract.Gate`], ...]): Single-qubit
            gates that can be used.

    Returns:
        tuple | None: ``None`` if there are no such rotations. Otherwise, the unitary that
        takes the axes to the :math:`z` and :math:`y` axes, followed by the rotation about
        the first axis and about the second axis, each given by the gate class, its extra
        arguments, and the gate classes of :math:`G` and of :math:`G^{\\dagger}`.
    """
    backend = _numpy_backend()
    paulis = (matrices.X, matrices.Y, matrices.Z)

    def to_rotation(unitary):
        return np.array(
            [
                [np.trace(p @ unitary @ q @ unitary.conj().T).real / 2 for q in paulis]
                for p in paulis
            ]
        )

    free = [
        (np.array(axis), gate_class, extra)
        for gate_class in gate_classes
        for axis, extra in _ROTATIONS.get(gate_class, ())
    ]
    # Rotations of the gates without parameters, and the number of times that each
    # of them is applied to get the identity (up to a global phase).
    fixed = {}
    for gate_class in gate_classes:
        if gate_class in _FIXED_GATES:
            matrix = backend.to_numpy(gate_class(0).matrix(backend))
            power = matrix
            for order in range(1, 17):
                if abs(np.trace(power)) > 2 - 1e-9:
                    fixed[gate_class] = (to_rotation(matrix), order)
                    break
                power = power @ matrix

    # Sequences of the gates without parameters, without repeated rotations.
    elements = {(): np.eye(3)}
    keys = {tuple(np.round(np.eye(3), 8).ravel() + 0.0)}
    frontier = [()]
    for _ in range(4):
        new_frontier = []
        for word in frontier:
            for gate_class, (rotation, _) in fixed.items():
                new_rotation = rotation @ elements[word]
                key = tuple(np.round(new_rotation, 8).ravel() + 0.0)
                if key not in keys:
                    keys.add(key)
                    elements[word + (gate_class,)] = new_rotation
                    new_frontier.append(word + (gate_class,))
        frontier = new_frontier

    candidates = [
        (
            len(word) + sum(fixed[gate_class][1] - 1 for gate_class in word),
            word,
            rotation @ axis,
            gate_class,
            extra,
        )
        for word, rotation in elements.items()
        for axis, gate_class, extra in free
    ]
    candidates.sort(key=lambda candidate: candidate[0])
    best = None
    for i, first in enumerate(candidates):
        if best is not None and 3 * first[0] >= best[0]:
            break
        for second in candidates[i + 1 :]:
            if abs(first[2] @ second[2]) < 1e-9:
                # The first rotation is used twice, so it is the cheaper one.
                total = 2 * first[0] + second[0]
                if best is None or total < best[0]:
                    best = (total, first, second)
                break

    if best is None:
        return None

    # The unitary of the rotation that takes the z axis to the first axis, and the
    # y axis to the second axis, obtained from the quaternion of the rotation.
    z_axis, y_axis = best[1][2], best[2][2]
    rotation = np.column_stack([np.cross(y_axis, z_axis), y_axis, z_axis])
    trace = np.trace(rotation)
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1)
        quaternion = np.array(
            [
                scale / 4,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2 * math.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        quaternion = np.zeros(4)
        quaternion[0] = (rotation[k, j] - rotation[j, k]) / scale
        quaternion[1 + i] = scale / 4
        quaternion[1 + j] = (rotation[j, i] + rotation[i, j]) / scale
        quaternion[1 + k] = (rotation[k, i] + rotation[i, k]) / scale
    frame = quaternion[0] * np.eye(2) - 1j * sum(
        component * pauli for component, pauli in zip(quaternion[1:], paulis)
    )

    axes = []
    for _, word, _, gate_class, extra in best[1:]:
        inverse_word = tuple(
            inverse_gate
            for gate in reversed(word)
            for inverse_gate in (gate,) * (fixed[gate][1] - 1)
        )
        axes.append((gate_class, extra, word, inverse_word))

    return (frame, *axes)


def _euler_sequence(
    name: str,
    qubit: int,
    angles: tuple[float, float, float],
    angles_x: tuple[float, float, float],
    atol: float,
    backend: Backend,
) -> list[Gate]:
    """Gates of a unitary in the Euler basis ``name``, up to a global phase.

    Args:
        name (str): Name of the Euler basis, a key of ``_EULER_BASES``.
        qubit (int): Qubit the gates act on.
        angles (tuple[float, float, float]): Parameters of the :class:`qibo.gates.U3`
            gate equal to the unitary, up to a global phase.
        angles_x (tuple[float, float, float]): The same for the unitary conjugated with
            :class:`qibo.gates.H`, which swaps the :math:`X` and :math:`Z` axes. It is
            used by the bases that start with a rotation about the :math:`X` axis.
        atol (float): Tolerance to decide that an angle is zero, or that
            :math:`\\theta` is :math:`\\pi / 2` or :math:`\\pi`.
        backend (:class:`qibo.backends.abstract.Backend`): Backend to use for calculations.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates of the basis. Rotations by a
        multiple of :math:`2 \\pi` are left out.
    """
    theta, phi, lam = angles
    theta_x, phi_x, lam_x = angles_x
    # Each step is a gate class followed by its parameters.
    if name == "U3":
        identity_angles = backend.abs(theta) <= atol and (
            abs(math.remainder(phi + lam, 2 * math.pi)) <= atol
        )
        steps = [] if identity_angles else [(gates.U3, theta, phi, lam)]
    elif name == "U321" and backend.abs(theta) <= atol:
        steps = [(gates.U1, phi + lam)]
    elif name == "U321" and backend.abs(theta - math.pi / 2) <= atol:
        steps = [(gates.U2, phi, lam)]
    elif name == "U321":
        steps = [(gates.U3, theta, phi, lam)]
    elif name in ("ZYZ", "ZXZ") and backend.abs(theta) <= atol:
        steps = [(gates.RZ, phi + lam)]
    elif name in ("XYX", "XZX") and backend.abs(theta_x) <= atol:
        steps = [(gates.RX, phi_x + lam_x)]
    elif name == "ZYZ":
        steps = [(gates.RZ, lam), (gates.RY, theta), (gates.RZ, phi)]
    elif name == "ZXZ":
        steps = [
            (gates.RZ, lam - math.pi / 2),
            (gates.RX, theta),
            (gates.RZ, phi + math.pi / 2),
        ]
    elif name == "XYX":
        steps = [
            (gates.RX, lam_x),
            (gates.RY, -theta_x),
            (gates.RX, phi_x),
        ]
    elif name == "XZX":
        steps = [
            (gates.RX, lam_x - math.pi / 2),
            (gates.RZ, theta_x),
            (gates.RX, phi_x + math.pi / 2),
        ]
    elif backend.abs(theta) <= atol:
        # ZSX and ZSXX bases, with theta equal to zero.
        steps = [(gates.RZ, phi + lam)]
    elif name == "ZSXX" and backend.abs(theta - math.pi) <= atol:
        steps = [(gates.RZ, lam + math.pi), (gates.X,), (gates.RZ, phi)]
    elif backend.abs(theta - math.pi / 2) <= atol:
        steps = [
            (gates.RZ, lam - math.pi / 2),
            (gates.SX,),
            (gates.RZ, phi + math.pi / 2),
        ]
    else:
        steps = [
            (gates.RZ, lam),
            (gates.SX,),
            (gates.RZ, theta + math.pi),
            (gates.SX,),
            (gates.RZ, phi + math.pi),
        ]

    # Rotations by a multiple of 2 pi are left out.
    sequence = [
        step[0](qubit, *step[1:])
        for step in steps
        if len(step) != 2 or abs(math.remainder(step[1], 2 * math.pi)) > atol
    ]

    return sequence
