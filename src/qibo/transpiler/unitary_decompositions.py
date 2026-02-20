import math
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates, matrices
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.quantum_info.linalg_operations import schmidt_decomposition

magic_basis = np.array(
    [[1, -1j, 0, 0], [0, 0, 1, -1j], [0, 0, -1, -1j], [1, 1j, 0, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [1, -1, 0, 0]]
) / np.sqrt(2)


def u3_decomposition(
    unitary: ArrayLike, backend: Backend
) -> Tuple[float, float, float]:
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
    unitary: ArrayLike, backend: Backend, magic_basis: ArrayLike = magic_basis
) -> Tuple[ArrayLike, ArrayLike]:
    """Solves the eigenvalue problem of :math:`U^{T} U`.

    See step (1) of Appendix A in arXiv:quant-ph/0011050.

    Args:
        unitary (ArrayLike): Unitary matrix of the gate we are decomposing
            in the computational basis.
        magic_basis (ArrayLike, optional): basis in which to solve the eigenvalue problem.
            Defaults to ``magic basis``.
        backend (:class:`qibo.backends.abstract.Backend`): Backend to use for calculations.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Eigenvectors in the computational basis
        and eigenvalues of :math:`U^{T} U`.
    """
    magic_basis = backend.cast(magic_basis)
    unitary = backend.cast(unitary)
    # write unitary in magic basis
    u_magic = backend.conj(magic_basis).T @ unitary @ magic_basis
    # construct and diagonalize UT_U
    ut_u = u_magic.T @ u_magic
    ut_u_real = backend.real(ut_u) + backend.imag(ut_u)
    if backend.__class__.__name__ not in ("PyTorchBackend", "TensorflowBackend"):
        ut_u_real = backend.round(ut_u_real, decimals=15)

    eigvals_real, psi_magic = backend.eigenvectors(ut_u_real, hermitian=True)
    # compute full eigvals as <psi|ut_u|psi>, as eigvals_real is only real
    eigvals = backend.sum(backend.conj(psi_magic) * (ut_u @ psi_magic), 0)
    # orthogonalize eigenvectors in the case of degeneracy (Gram-Schmidt)
    psi_magic, _ = backend.qr(psi_magic)
    # write psi in computational basis
    psi = magic_basis @ psi_magic
    return psi, eigvals


def calculate_single_qubit_unitaries(
    psi: ArrayLike, backend: Optional[Backend] = None
) -> Tuple[ArrayLike, ArrayLike]:
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
) -> Tuple[ArrayLike, ...]:
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

    lambdas = to_bell_diagonal(ud, backend)

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
    unitary: ArrayLike, backend: Optional[Backend] = None
) -> Tuple[ArrayLike, ...]:
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050."""

    unitary = backend.cast(unitary, dtype=unitary.dtype)
    psi, eigvals = calculate_psi(unitary, backend=backend)
    psi_tilde = backend.conj(backend.sqrt(eigvals)) * backend.matmul(unitary, psi)
    va, vb = calculate_single_qubit_unitaries(psi, backend=backend)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde, backend=backend)
    dag = lambda U: backend.transpose(backend.conj(U), (1, 0))
    ua, ub = dag(ua_dagger), dag(ub_dagger)
    return calculate_diagonal(unitary, ua, ub, va, vb, backend=backend)


def to_bell_diagonal(
    ud: ArrayLike, bell_basis: ArrayLike = bell_basis, backend: Optional[Backend] = None
) -> Union[ArrayLike, None]:
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
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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
) -> List[Gate]:
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
) -> List[Gate]:
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


def two_qubit_decomposition(
    q0: int, q1: int, unitary: ArrayLike, backend: Backend, threshold: float = 1e-6
) -> List[Gate]:
    """Performs two qubit unitary gate decomposition.

    Args:
        q0 (int): index of the first qubit.
        q1 (int): index of the second qubit.
        unitary (ndarray): Unitary :math:`4 \\times 4` to be decomposed.
        backend (:class:`qibo.backends.Backend`): Backend to use for calculations.
        threshold (float): Threshold for determining if hz component is zero.

    Returns:
        list: gates implementing the decomposition
    """
    # Handle identity case efficiently
    if backend.allclose(unitary, backend.identity(4)):
        return []

    z_component = _get_z_component(unitary, backend)
    if abs(z_component) < threshold:
        return _two_qubit_decomposition_without_z(q0, q1, unitary, backend)
    return _two_qubit_decomposition_with_z(q0, q1, unitary, backend)


def _get_z_component(unitary: ArrayLike, backend: Backend) -> float:
    """Calculates the hz component from a unitary's magic decomposition."""
    _, _, ud, _, _ = magic_decomposition(unitary, backend=backend)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    _, _, hz = calculate_h_vector(ud_diag, backend=backend)
    return float(hz)


def _two_qubit_decomposition_without_z(
    q0: int, q1: int, unitary: ArrayLike, backend: Backend
) -> List[Gate]:
    """Implements Theorem 2 decomposition (2 CNOTs) for hz=0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
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
    q0: int, q1: int, unitary: ArrayLike, backend: Backend
) -> List[Gate]:
    """Implements Theorem 1 decomposition (3 CNOTs) for hz≠0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
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
