import numpy as np

from qibo import gates, matrices
from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error

magic_basis = np.array(
    [[1, -1j, 0, 0], [0, 0, 1, -1j], [0, 0, -1, -1j], [1, 1j, 0, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [1, -1, 0, 0]]
) / np.sqrt(2)

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def u3_decomposition(unitary):
    """Decomposes arbitrary one-qubit gates to U3.

    Args:
        unitary (ndarray): Unitary :math:`2 \\times 2` matrix to be decomposed.

    Returns:
        (float, float, float): parameters of U3 gate.
    """
    # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
    su2 = unitary / np.sqrt(np.linalg.det(unitary))
    theta = 2 * np.arctan2(abs(su2[1, 0]), abs(su2[0, 0]))
    plus = np.angle(su2[1, 1])
    minus = np.angle(su2[1, 0])
    phi = plus + minus
    lam = plus - minus

    return theta, phi, lam


def calculate_psi(unitary, magic_basis=magic_basis, backend=None):
    """Solves the eigenvalue problem of :math:`U^{T} U`.

    See step (1) of Appendix A in arXiv:quant-ph/0011050.

    Args:
        unitary (ndarray): Unitary matrix of the gate we are decomposing
            in the computational basis.
        magic_basis (ndarray, optional): basis in which to solve the eigenvalue problem.
            Defaults to ``magic basis``.
        backend (:class:`qibo.backends.abstract.Backend`): Backend to use for calculations.

    Returns:
        ndarray: Eigenvectors in the computational basis and eigenvalues of :math:`U^{T} U`.
    """
    backend = _check_backend(backend)

    if backend.__class__.__name__ in [
        "CupyBackend",
        "CuQuantumBackend",
    ]:  # pragma: no cover
        raise_error(
            NotImplementedError,
            f"{backend.__class__.__name__} does not support `linalg.eig.`",
        )

    magic_basis = backend.cast(magic_basis)
    unitary = backend.cast(unitary)
    # write unitary in magic basis
    u_magic = (
        backend.np.transpose(backend.np.conj(magic_basis), (1, 0))
        @ unitary
        @ magic_basis
    )
    # construct and diagonalize UT_U
    ut_u = backend.np.transpose(u_magic, (1, 0)) @ u_magic
    if backend.__class__.__name__ != "PyTorchBackend":
        # eig seems to have a different behavior based on backend/hardware,
        # use np.round to increase precision seems to fix the issue
        eigvals, psi_magic = np.linalg.eig(np.round(ut_u, decimals=20))
    else:
        eigvals, psi_magic = backend.calculate_eigenvectors(ut_u, hermitian=False)
    # orthogonalize eigenvectors in the case of degeneracy (Gram-Schmidt)
    psi_magic, _ = backend.np.linalg.qr(psi_magic)
    # write psi in computational basis
    psi = backend.np.matmul(magic_basis, psi_magic)
    return psi, eigvals


def schmidt_decompose(state, backend=None):
    """Decomposes a two-qubit product state to its single-qubit parts.

    Args:
        state (ndarray): product state to be decomposed.

    Returns:
        (ndarray, ndarray): decomposition

    """
    backend = _check_backend(backend)
    # tf.linalg.svd has a different behaviour
    if backend.__class__.__name__ == "TensorflowBackend":
        u, d, v = np.linalg.svd(backend.np.reshape(state, (2, 2)))
    else:
        u, d, v = backend.np.linalg.svd(backend.np.reshape(state, (2, 2)))
    if not np.allclose(backend.to_numpy(d), [1, 0]):  # pragma: no cover
        raise_error(
            ValueError,
            f"Unexpected singular values: {d}\nCan only decompose product states.",
        )
    return u[:, 0], v[0]


def calculate_single_qubit_unitaries(psi, backend=None):
    """Calculates local unitaries that maps a maximally entangled basis to the magic basis.

    See Lemma 1 of Appendix A in arXiv:quant-ph/0011050.

    Args:
        psi (ndarray): Maximally entangled two-qubit states that define a basis.

    Returns:
        (ndarray, ndarray): Local unitaries UA and UB that map the given basis to the magic basis.
    """
    backend = _check_backend(backend)
    psi_magic = backend.np.matmul(backend.np.conj(backend.cast(magic_basis)).T, psi)
    if (
        backend.np.real(
            backend.calculate_norm_density_matrix(backend.np.imag(psi_magic))
        )
        > PRECISION_TOL
    ):  # pragma: no cover
        raise_error(NotImplementedError, "Given state is not real in the magic basis.")
    psi_bar = backend.cast(psi.T, copy=True)

    # find e and f by inverting (A3), (A4)
    ef = (psi_bar[0] + 1j * psi_bar[1]) / np.sqrt(2)
    e_f_ = (psi_bar[0] - 1j * psi_bar[1]) / np.sqrt(2)
    e, f = schmidt_decompose(ef, backend=backend)
    e_, f_ = schmidt_decompose(e_f_, backend=backend)
    # find exp(1j * delta) using (A5a)
    ef_ = backend.np.kron(e, f_)
    phase = (
        1j
        * np.sqrt(2)
        * backend.np.sum(backend.np.multiply(backend.np.conj(ef_), psi_bar[2]))
    )
    v0 = backend.cast(np.asarray([1, 0]))
    v1 = backend.cast(np.asarray([0, 1]))
    # construct unitaries UA, UB using (A6a), (A6b)
    ua = backend.np.tensordot(v0, backend.np.conj(e), 0) + phase * backend.np.tensordot(
        v1, backend.np.conj(e_), 0
    )
    ub = backend.np.tensordot(v0, backend.np.conj(f), 0) + backend.np.conj(
        phase
    ) * backend.np.tensordot(v1, backend.np.conj(f_), 0)
    return ua, ub


def calculate_diagonal(unitary, ua, ub, va, vb, backend=None):
    """Calculates Ud matrix that can be written as exp(-iH).

    See Eq. (A1) in arXiv:quant-ph/0011050.
    Ud is diagonal in the magic and Bell basis.
    """
    backend = _check_backend(backend)
    # normalize U_A, U_B, V_A, V_B so that detU_d = 1
    # this is required so that sum(lambdas) = 0
    # and Ud can be written as exp(-iH)
    if backend.__class__.__name__ == "TensorflowBackend":
        det = np.linalg.det(unitary) ** (1 / 16)
    else:
        det = backend.np.linalg.det(unitary) ** (1 / 16)
    ua *= det
    ub *= det
    va *= det
    vb *= det
    u_dagger = backend.np.transpose(
        backend.np.conj(
            backend.np.kron(
                ua,
                ub,
            )
        ),
        (1, 0),
    )
    v_dagger = backend.np.transpose(backend.np.conj(backend.np.kron(va, vb)), (1, 0))
    ud = u_dagger @ unitary @ v_dagger
    return ua, ub, ud, va, vb


def magic_decomposition(unitary, backend=None):
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050."""
    backend = _check_backend(backend)
    unitary = backend.cast(unitary)
    psi, eigvals = calculate_psi(unitary, backend=backend)
    psi_tilde = backend.np.conj(backend.np.sqrt(eigvals)) * backend.np.matmul(
        unitary, psi
    )
    va, vb = calculate_single_qubit_unitaries(psi, backend=backend)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde, backend=backend)
    ua, ub = backend.np.transpose(
        backend.np.conj(ua_dagger), (1, 0)
    ), backend.np.transpose(backend.np.conj(ub_dagger), (1, 0))
    return calculate_diagonal(unitary, ua, ub, va, vb, backend=backend)


def to_bell_diagonal(ud, bell_basis=bell_basis, backend=None):
    """Transforms a matrix to the Bell basis and checks if it is diagonal."""
    backend = _check_backend(backend)

    ud = backend.cast(ud)
    bell_basis = backend.cast(bell_basis)

    ud_bell = (
        backend.np.transpose(backend.np.conj(bell_basis), (1, 0)) @ ud @ bell_basis
    )
    ud_diag = backend.np.diag(ud_bell)
    if not backend.np.allclose(backend.np.diag(ud_diag), ud_bell):  # pragma: no cover
        return None
    uprod = backend.np.prod(ud_diag)
    if not np.allclose(backend.to_numpy(uprod), 1):  # pragma: no cover
        return None
    return ud_diag


def calculate_h_vector(ud_diag, backend=None):
    """Finds h parameters corresponding to exp(-iH).

    See Eq. (4)-(5) in arXiv:quant-ph/0307177.
    """
    backend = _check_backend(backend)
    lambdas = -backend.np.angle(ud_diag)
    hx = (lambdas[0] + lambdas[2]) / 2.0
    hy = (lambdas[1] + lambdas[2]) / 2.0
    hz = (lambdas[0] + lambdas[1]) / 2.0
    return hx, hy, hz


def cnot_decomposition(q0, q1, hx, hy, hz, backend=None):
    """Performs decomposition (6) from arXiv:quant-ph/0307177."""
    backend = _check_backend(backend)
    h = backend.cast(H)
    u3 = backend.cast(-1j * matrices.H)
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
        gates.Unitary(backend.np.conj(w).T @ h, q1),
    ]


def cnot_decomposition_light(q0, q1, hx, hy, backend=None):
    """Performs decomposition (24) from arXiv:quant-ph/0307177."""
    backend = _check_backend(backend)
    h = backend.cast(H)
    w = backend.cast((matrices.I - 1j * matrices.X) / np.sqrt(2))
    u2 = gates.RX(0, 2 * hx).matrix(backend)
    v2 = gates.RZ(0, -2 * hy).matrix(backend)
    # change CNOT to CZ using Hadamard gates
    return [
        gates.Unitary(backend.np.conj(w).T, q0),
        gates.Unitary(h @ w, q1),
        gates.CZ(q0, q1),
        gates.Unitary(u2, q0),
        gates.Unitary(h @ v2 @ h, q1),
        gates.CZ(q0, q1),
        gates.Unitary(w, q0),
        gates.Unitary(backend.np.conj(w).T @ h, q1),
    ]


def two_qubit_decomposition(q0, q1, unitary, backend=None):
    """Performs two qubit unitary gate decomposition (24) from arXiv:quant-ph/0307177.

    Args:
        q0 (int): index of the first qubit.
        q1 (int): index of the second qubit.
        unitary (ndarray): Unitary :math:`4 \\times 4` to be decomposed.

    Returns:
        (list): gates implementing decomposition (24) from arXiv:quant-ph/0307177
    """
    backend = _check_backend(backend)

    ud_diag = to_bell_diagonal(unitary, backend=backend)
    ud = None
    if ud_diag is None:
        u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
        ud_diag = to_bell_diagonal(ud, backend=backend)

    hx, hy, hz = calculate_h_vector(ud_diag, backend=backend)
    hx, hy, hz = float(hx), float(hy), float(hz)
    if np.allclose([hx, hy, hz], [0, 0, 0]):
        u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
        gatelist = [gates.Unitary(u4 @ u1, q0), gates.Unitary(v4 @ v1, q1)]
    elif np.allclose(hz, 0):
        gatelist = cnot_decomposition_light(q0, q1, hx, hy, backend=backend)
        if ud is None:
            return gatelist
        g0, g1 = gatelist[:2]
        gatelist[0] = gates.Unitary(backend.cast(g0.parameters[0]) @ u1, q0)
        gatelist[1] = gates.Unitary(backend.cast(g1.parameters[0]) @ v1, q1)

        g0, g1 = gatelist[-2:]
        gatelist[-2] = gates.Unitary(u4 @ g0.parameters[0], q0)
        gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

    else:
        cnot_dec = cnot_decomposition(q0, q1, hx, hy, hz, backend=backend)
        if ud is None:
            return cnot_dec

        gatelist = [
            gates.Unitary(u1, q0),
            gates.Unitary(backend.cast(H) @ v1, q1),
        ]
        gatelist.extend(cnot_dec[1:])
        g0, g1 = gatelist[-2:]
        gatelist[-2] = gates.Unitary(u4 @ g0.parameters[0], q0)
        gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

    return gatelist
