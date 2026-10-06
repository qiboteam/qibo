import functools
import math

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates, matrices
from qibo.backends import Backend, _check_backend, _numpy_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.quantum_info.linalg_operations import schmidt_decomposition
from qibo.transpiler._exceptions import DecompositionError

magic_basis = np.array(
    [[1, -1j, 0, 0], [0, 0, 1, -1j], [0, 0, -1, -1j], [1, 1j, 0, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [1, -1, 0, 0]]
) / np.sqrt(2)


# Euler bases of :func:`qibo.transpiler.unitary_decompositions.single_qubit_decomposition`
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

    Args:
        unitary (ArrayLike): Unitary :math:`2 \\times 2` matrix to be decomposed.

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
    """Calculates local unitaries that maps a maximally entangled basis to the magic basis.

    See Lemma 1 of Appendix A in Ref. [1].

    Args:
        psi (ArrayLike): Maximally entangled two-qubit states that define a basis.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Local unitaries UA and UB that map the given
        basis to the magic basis.

    References:
        1. B. Kraus and J. I. Cirac, *Optimal creation of entanglement using a two-qubit gate*,
        `Phys. Rev. A 63, 062309 (2001) <https://doi.org/10.1103/PhysRevA.63.062309 >`_.

    """
    psi_magic = backend.matmul(backend.conj(backend.cast(magic_basis)).T, psi)
    if (
        backend.real(backend.matrix_norm(backend.imag(psi_magic))) > 1e-6
    ):  # pragma: no cover
        raise_error(NotImplementedError, "Given state is not real in the magic basis.")
    psi_bar = backend.cast(psi.T, copy=True)

    # find e and f by inverting (A3), (A4)
    ef = (psi_bar[0] + 1j * psi_bar[1]) / np.sqrt(2)
    e_f_ = (psi_bar[0] - 1j * psi_bar[1]) / np.sqrt(2)
    e, _, f = schmidt_decomposition(ef, [0], backend=backend)
    e, f = e[:, 0], f[0]
    e_, _, f_ = schmidt_decomposition(e_f_, [0], backend=backend)
    e_, f_ = e_[:, 0], f_[0]
    # find exp(1j * delta) using (A5a)
    ef_ = backend.kron(e, f_)
    phase = 1j * np.sqrt(2) * backend.sum(backend.conj(ef_) * psi_bar[2])
    v0 = backend.cast(np.asarray([1, 0]))
    v1 = backend.cast(np.asarray([0, 1]))
    # construct unitaries UA, UB using (A6a), (A6b)
    ua = backend.outer(v0, backend.conj(e)) + phase * backend.outer(
        v1, backend.conj(e_)
    )
    ub = backend.outer(v0, backend.conj(f)) + backend.conj(phase) * backend.outer(
        v1, backend.conj(f_)
    )
    return ua, ub


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

    # 1. force coefficients to be in pi/4 >= ... >= 0 interval, using pi/2 periodicity and pi/4 symmetry
    fit = lambda alpha: min(alpha % (np.pi / 2), np.pi / 2 - (alpha % (np.pi / 2)))

    # 2. permute to ensure ordering:
    alphas_ordered = sorted(
        [[hx, 0], [hy, 1], [hz, 2]], key=lambda x: fit(x[0]), reverse=True
    )

    H = backend.matrices.H
    S = backend.matrices.S

    correction = {
        "left_A": backend.matrices.I(),
        "left_B": backend.matrices.I(),
        "right_A": backend.matrices.I(),
        "right_B": backend.matrices.I(),
    }

    permutation = [x[1] for x in alphas_ordered]

    if permutation[0] == 1:
        correction["left_A"] @= S
        correction["left_B"] @= S
        correction["right_A"] = dag(S) @ correction["right_A"]
        correction["right_B"] = dag(S) @ correction["right_B"]

    elif permutation[0] == 2:
        correction["left_A"] = correction["left_A"] @ H
        correction["left_B"] = correction["left_B"] @ H
        correction["right_A"] = dag(H) @ correction["right_A"]
        correction["right_B"] = dag(H) @ correction["right_B"]

    if not (permutation[1] == 1 or permutation[2] == 2):
        correction["left_A"] = correction["left_A"] @ (S @ H @ S)
        correction["left_B"] = correction["left_B"] @ (S @ H @ S)
        correction["right_A"] = dag(S @ H @ S) @ correction["right_A"]
        correction["right_B"] = dag(S @ H @ S) @ correction["right_B"]

    # 3. find local corrections to enforce, as possible, conditions on h
    paulis = ["X", "Y", "Z"]
    for i, (alpha, _) in enumerate(alphas_ordered):
        if i < 2:
            if alpha < 0:
                alpha += np.pi / 2
                p_corr = getattr(backend.matrices, paulis[i])
                correction["left_A"] = correction["left_A"] @ (1j * p_corr)
                correction["left_B"] = correction["left_B"] @ p_corr
            if (
                alpha > np.pi / 4
            ):  # can't conjugate so sign of z alternates to compensate
                # Swap alpha_i and alpha_z sign
                correction["left_B"] = correction["left_B"] @ getattr(
                    backend.matrices, paulis[(i + 1) % 2]
                )
                correction["right_B"] = (
                    getattr(backend.matrices, paulis[(i + 1) % 2])
                    @ correction["right_B"]
                )
                # Add pi/2 to alpha_i
                correction["left_A"] = correction["left_A"] @ (
                    1j * getattr(backend.matrices, paulis[i])
                )
                correction["left_B"] = correction["left_B"] @ getattr(
                    backend.matrices, paulis[i]
                )
        elif abs(alpha) > np.pi / 4:
            correction["left_A"] = correction["left_A"] @ (
                (1j if alpha < 0 else -1j) * getattr(backend.matrices, paulis[i])
            )
            correction["left_B"] = correction["left_B"] @ getattr(
                backend.matrices, paulis[i]
            )

    # 4. apply corrections
    ua = ua @ correction["left_A"]
    ub = ub @ correction["left_B"]
    va = correction["right_A"] @ va
    vb = correction["right_B"] @ vb
    ud = backend.matmul(
        backend.kron(dag(correction["left_A"]), dag(correction["left_B"])),
        backend.matmul(
            ud,
            backend.kron(dag(correction["right_A"]), dag(correction["right_B"])),
        ),
    )
    return ua, ub, ud, va, vb


def magic_decomposition(
    unitary: ArrayLike,
    backend: Backend | None = None,
    weight: float = math.sqrt(2),
) -> tuple[ArrayLike, ...]:
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050.

    The argument ``weight`` is defined in :func:`qibo.transpiler.unitary_decompositions.calculate_psi`.
    """
    backend = _check_backend(backend)
    unitary = backend.cast(unitary, dtype=unitary.dtype)
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
    u2 = -u3 @ gates.RX(0, 2 * hx - np.pi / 2).matrix(backend)
    # add an extra exp(-i pi / 4) global phase to get exact match
    v2 = np.exp(-1j * np.pi / 4) * gates.RZ(0, 2 * hz).matrix(backend)
    v3 = gates.RZ(0, -2 * hy).matrix(backend)
    w = backend.cast((matrices.I - 1j * matrices.X) / np.sqrt(2))
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
    :class:`qibo.gates.Z`, which are conjugated with the rotation. The decomposition
    is equal to the unitary up to a global phase.

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
        list[:class:`qibo.gates.abstract.Gate`]: Gates that implement the unitary.

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
        return min(
            (
                _euler_sequence(name, qubit, angles, angles_x, atol, backend)
                for name in bases
            ),
            key=len,
        )

    frames = _euler_frames(tuple(gate_classes))
    if frames is None:
        raise_error(
            DecompositionError,
            "The single-qubit native gates must include rotations about two orthogonal "
            + "axes with free angles, which can be rotated by native gates without "
            + "parameters. U3 and GPI2 can be used as well.",
        )

    frame = backend.cast(frames[0], dtype=unitary.dtype)
    theta, phi, lam = u3_decomposition(backend.conj(frame).T @ unitary @ frame, backend)

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

    Args:
        q0 (int): index of the first qubit.
        q1 (int): index of the second qubit.
        unitary (ndarray): Unitary :math:`4 \\times 4` to be decomposed.
        threshold (float): Threshold for determining if hz component is zero.
        weight (float, optional): Weight of the imaginary part in the matrix that is
            diagonalized to find the local gates, see
            :func:`qibo.transpiler.unitary_decompositions.calculate_psi`.
            Defaults to :math:`\\sqrt{2}`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list: gates implementing the decomposition
    """
    backend = _check_backend(backend)

    # Handle identity case efficiently
    if backend.allclose(unitary, backend.identity(4)):
        return []

    z_component = _get_z_component(unitary, weight, backend)
    if abs(z_component) < threshold:
        return _two_qubit_decomposition_without_z(q0, q1, unitary, weight, backend)
    return _two_qubit_decomposition_with_z(q0, q1, unitary, weight, backend)


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
            gate for gate in reversed(word) for gate in (gate,) * (fixed[gate][1] - 1)
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


def _get_z_component(unitary: ArrayLike, weight: float, backend: Backend) -> float:
    """Calculates the hz component from a unitary's magic decomposition."""
    _, _, ud, _, _ = magic_decomposition(unitary, backend=backend, weight=weight)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    _, _, hz = calculate_h_vector(ud_diag, backend=backend)
    return float(hz)


def _two_qubit_decomposition_without_z(
    q0: int, q1: int, unitary: ArrayLike, weight: float, backend: Backend
) -> list[Gate]:
    """Implements Theorem 2 decomposition (2 CNOTs) for hz=0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend, weight=weight)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    hx, hy, _ = calculate_h_vector(ud_diag, backend=backend)
    hx, hy = float(hx), float(hy)

    # Get light decomposition
    gatelist = cnot_decomposition_light(q0, q1, hx, hy, backend=backend)
    # Combine with initial and final local unitaries
    g0, g1 = gatelist[:2]
    gatelist[0] = gates.Unitary(backend.cast(g0.parameters[0]) @ u1, q0)
    gatelist[1] = gates.Unitary(backend.cast(g1.parameters[0]) @ v1, q1)

    g0, g1 = gatelist[-2:]
    gatelist[-2] = gates.Unitary(u4 @ g0.parameters[0], q0)
    gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

    return gatelist


def _two_qubit_decomposition_with_z(
    q0: int, q1: int, unitary: ArrayLike, weight: float, backend: Backend
) -> list[Gate]:
    """Implements Theorem 1 decomposition (3 CNOTs) for hz≠0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend, weight=weight)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    hx, hy, hz = calculate_h_vector(ud_diag, backend=backend)
    hx, hy, hz = float(hx), float(hy), float(hz)

    # Get full decomposition
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
