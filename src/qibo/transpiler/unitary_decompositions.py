import numpy as np

from qibo import gates, matrices
from qibo.config import raise_error
from qibo.quantum_info.linalg_operations import schmidt_decomposition

magic_basis = np.array(
    [[1, -1j, 0, 0], [0, 0, 1, -1j], [0, 0, -1, -1j], [1, 1j, 0, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [1, -1, 0, 0]]
) / np.sqrt(2)

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def u3_decomposition(unitary, backend):
    """Decomposes arbitrary one-qubit gates to U3.

    Args:
        unitary (ndarray): Unitary :math:`2 \\times 2` matrix to be decomposed.

    Returns:
        (float, float, float): parameters of U3 gate.
    """
    unitary = backend.cast(unitary)
    # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
    su2 = unitary / backend.np.sqrt(backend.np.linalg.det(unitary))
    theta = 2 * backend.np.arctan2(backend.np.abs(su2[1, 0]), backend.np.abs(su2[0, 0]))
    plus = backend.np.angle(su2[1, 1])
    minus = backend.np.angle(su2[1, 0])
    phi = plus + minus
    lam = plus - minus
    # explicit conversion to float to avoid issue on GPU
    return float(theta), float(phi), float(lam)


def calculate_psi(unitary, backend, magic_basis=magic_basis):
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
    ut_u_real = backend.np.real(ut_u) + backend.np.imag(ut_u)
    if backend.__class__.__name__ not in ("PyTorchBackend", "TensorflowBackend"):
        ut_u_real = np.round(ut_u_real, decimals=15)

    eigvals_real, psi_magic = backend.calculate_eigenvectors(ut_u_real, hermitian=True)
    # compute full eigvals as <psi|ut_u|psi>, as eigvals_real is only real
    eigvals = backend.np.sum(backend.np.conj(psi_magic) * (ut_u @ psi_magic), 0)

    print("psi_magic: ", psi_magic)
    # We permute to ensure we will have hx >= hy >= hz in the end
    angles = -np.angle(np.sqrt(eigvals))
    print("Eigvals psi: ", eigvals)
    print("conj sqrt Eigvals psi: ", backend.np.conj(backend.np.sqrt(eigvals)))
    print("uncorrected angles: ", angles)
    angles -= np.sum(angles)/4  # take into account det(Ud) = 1 normalization

    print("corrected angles: ", angles)
    
    # 1. force coefficients to be in pi/4 >= ... >= 0 interval, using pi/2 periodicity and pi/4 symmetry
    fit = lambda alpha: min(alpha%(np.pi/2), np.pi/2 - (alpha%(np.pi/2)))
    alphay = (angles[1] + angles[2]) / 2.0
    alphax = (angles[0] + angles[2]) / 2.0
    alphaz = (angles[0] + angles[1]) / 2.0

    print("Unsorted alphas", alphax, alphay ,alphaz)

    # 2. permute to ensure ordering:  x <-> y == 1 <-> 0 / x <--> z == 1 <--> 3 / y <--> z == 0 <--> 3
    alphas_ordered = sorted( [[alphax, 1], [alphay, 0], [alphaz, 2]], key=lambda x: fit(x[0]),  reverse=True)

    permutation = [x[1] for x in alphas_ordered]
    print("Permutation : ", permutation)
    eigvals[[1,0,2]] = eigvals[permutation]
    psi_magic[:,[1,0,2]] = psi_magic[:, permutation]

    print("Sorted alphas", *(x[0] for x in alphas_ordered))
    print("Sorted and fit alphas", *(fit(x[0]) for x in alphas_ordered))

    # From construction we may only have pi/2 >= alpha >= -pi/2
    correction = {"left_A": matrices.I.copy(), "left_B": matrices.I.copy(), "right_B": matrices.I.copy()}
    paulis = ['X', 'Y', 'Z']
    for i, (alpha, _) in enumerate(alphas_ordered):
        print(f"i: {i}, alpha: {alpha}")
        if i < 2: 
            if alpha < 0:
                print("corrected")
                alpha += np.pi/2
                correction["left_A"] = correction["left_A"] @ (1j * getattr(matrices, paulis[i]))
                correction["left_B"] = correction["left_B"] @ getattr(matrices, paulis[i])
            if alpha > np.pi/4: # can't conjugate so sign of z alternates to compensate
                print("Swaping signs")
                # Swap alpha_i and alpha_z sign
                correction["left_B"]  = correction["left_B"] @ getattr(matrices, paulis[(i+1)%2])
                correction["right_B"] = getattr(matrices, paulis[(i+1)%2]) @ correction["right_B"] 
                # Add pi/2 to alpha_i
                correction["left_A"]  = correction["left_A"] @ (1j * getattr(matrices, paulis[i]))
                correction["left_B"]  = correction["left_B"] @ getattr(matrices, paulis[i])
        elif abs(alpha) > np.pi/4:
            print("Z change")
            correction["left_A"] = correction["left_A"] @ ((1j if alpha < 0 else -1j) * getattr(matrices, paulis[i]))
            correction["left_B"] = correction["left_B"] @ getattr(matrices, paulis[i])
                
    # orthogonalize eigenvectors in the case of degeneracy (Gram-Schmidt)
    psi_magic, _ = backend.np.linalg.qr(psi_magic)
    # write psi in computational basis
    psi = backend.np.matmul(magic_basis, psi_magic)
    print()
    return psi, eigvals, correction


def calculate_single_qubit_unitaries(psi, backend=None):
    """Calculates local unitaries that maps a maximally entangled basis to the magic basis.

    See Lemma 1 of Appendix A in arXiv:quant-ph/0011050.

    Args:
        psi (ndarray): Maximally entangled two-qubit states that define a basis.

    Returns:
        (ndarray, ndarray): Local unitaries UA and UB that map the given basis to the magic basis.
    """
    psi_magic = backend.np.matmul(backend.np.conj(backend.cast(magic_basis)).T, psi)
    if (
        backend.np.real(backend.calculate_matrix_norm(backend.np.imag(psi_magic)))
        > 1e-6
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


def calculate_diagonal(unitary, ua, ub, va, vb, backend):
    """Calculates Ud matrix that can be written as exp(-iH).

    See Eq. (A1) in arXiv:quant-ph/0011050.
    Ud is diagonal in the magic and Bell basis.
    """
    # normalize U_A, U_B, V_A, V_B so that detU_d = 1
    # this is required so that sum(lambdas) = 0
    # and Ud can be written as exp(-iH)
    if backend.__class__.__name__ == "TensorflowBackend":  # pragma: no cover
        det = np.linalg.det(unitary) ** (1 / 16)
    else:
        det = backend.np.linalg.det(unitary) ** (1 / 16)
    print("\n Inside Calc diagonal")
    print(f"BEF: ua {backend.np.linalg.det(ua)}, ub: {backend.np.linalg.det(ub)}, va: {backend.np.linalg.det(va)}, vb: {backend.np.linalg.det(vb)}, U {backend.np.linalg.det(unitary)}")
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
    print()
    return ua, ub, ud, va, vb


def magic_decomposition(unitary, backend=None):
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050."""

    unitary = backend.cast(unitary)
    psi, eigvals, correction = calculate_psi(unitary, backend=backend)
    print("EIGVALS IN MAGIC: ", backend.np.conj(backend.np.sqrt(eigvals)))
    print("angles of eigvals n magic: ", np.angle(backend.np.conj(backend.np.sqrt(eigvals))))
    psi_tilde = backend.np.conj(backend.np.sqrt(eigvals)) * backend.np.matmul(
        unitary, psi
    )
    va, vb = calculate_single_qubit_unitaries(psi, backend=backend)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde, backend=backend)
    dag = lambda U: backend.np.transpose(backend.np.conj(U), (1, 0))
    ua, ub = dag(ua_dagger), dag(ub_dagger)
    ua, ub, ud, va, vb = calculate_diagonal(unitary, ua, ub, va, vb, backend=backend)
    print("Ud before correction", to_bell_diagonal(ud, qibo.get_backend()))
    calculate_h_vector(to_bell_diagonal(ud, qibo.get_backend()), qibo.get_backend())
    ub = backend.np.matmul(ub, correction["right_B"])
    va = backend.np.matmul(correction["left_A"], va)
    vb = backend.np.matmul(correction["left_B"], vb)
    ud = backend.np.matmul(backend.np.kron(dag(correction["left_A"]), dag(correction["left_B"])),
                           backend.np.matmul(ud, backend.np.kron(backend.cast(matrices.I), dag(correction["right_B"]))))
    print("Ud after correction", to_bell_diagonal(ud, qibo.get_backend()))
    calculate_h_vector(to_bell_diagonal(ud, qibo.get_backend()), qibo.get_backend())
    return ua, ub, ud, va, vb

def to_bell_diagonal(ud, backend, bell_basis=bell_basis):
    """Transforms a matrix to the Bell basis and checks if it is diagonal."""
    ud = backend.cast(ud)
    bell_basis = backend.cast(bell_basis)

    ud_bell = (
        backend.np.transpose(backend.np.conj(bell_basis), (1, 0)) @ ud @ bell_basis
    )
    ud_diag = backend.np.diag(ud_bell)
    if not backend.np.allclose(
        backend.np.diag(ud_diag), ud_bell, atol=1e-6, rtol=1e-6
    ):  # pragma: no cover
        return None
    uprod = backend.to_numpy(backend.np.prod(ud_diag))
    if not np.allclose(uprod, 1.0, atol=1e-6, rtol=1e-6):  # pragma: no cover
        return None
    return ud_diag


def calculate_h_vector(ud_diag, backend):
    """Finds h parameters corresponding to exp(-iH).

    See Eq. (4)-(5) in arXiv:quant-ph/0307177.
    """
    lambdas = -backend.np.angle(ud_diag)
    hx = (lambdas[0] + lambdas[2]) / 2.0
    hy = (lambdas[1] + lambdas[2]) / 2.0
    hz = (lambdas[0] + lambdas[1]) / 2.0
    print(f"Prod of ud_diag: {np.prod(ud_diag)}")
    print("Lambdas in h_vec calc: ", lambdas)
    print(f"H in cv: hx: {hx}, hy: {hy}, hz: {hz}",)
    return hx, hy, hz


def cnot_decomposition(q0, q1, hx, hy, hz, backend):
    """Performs decomposition (6) from arXiv:quant-ph/0307177."""
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


def cnot_decomposition_light(q0, q1, hx, hy, backend):
    """Performs decomposition (24) from arXiv:quant-ph/0307177."""
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


def _get_z_component(unitary, backend):
    """Calculates the hz component from a unitary's magic decomposition."""
    ud_diag = to_bell_diagonal(unitary, backend=backend)
    if ud_diag is None:
        _, _, ud, _, _ = magic_decomposition(unitary, backend=backend)
        ud_diag = to_bell_diagonal(ud, backend=backend)
    _, _, hz = calculate_h_vector(ud_diag, backend=backend)
    return float(hz)


def _two_qubit_decomposition_without_z(q0, q1, unitary, backend):
    """Implements Theorem 2 decomposition (2 CNOTs) for hz=0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    hx, hy, _ = calculate_h_vector(ud_diag, backend=backend)
    hx, hy = float(hx), float(hy)

    print("Hs light:", hx, hy, _)

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


def _two_qubit_decomposition_with_z(q0, q1, unitary, backend):
    """Implements Theorem 1 decomposition (3 CNOTs) for hz≠0 case."""
    # Get magic decomposition
    u4, v4, ud, u1, v1 = magic_decomposition(unitary, backend=backend)
    ud_diag = to_bell_diagonal(ud, backend=backend)
    hx, hy, hz = calculate_h_vector(ud_diag, backend=backend)
    hx, hy, hz = float(hx), float(hy), float(hz)
    print("HS: ", hx, hy, hz)

    # Get full decomposition
    cnot_dec = cnot_decomposition(q0, q1, hx, hy, hz, backend=backend)

    # Combine with initial and final local unitaries
    gatelist = [
        gates.Unitary(u1, q0),
        gates.Unitary(backend.cast(H) @ v1, q1),
    ]
    gatelist.extend(cnot_dec[1:])
    g0, g1 = gatelist[-2:]
    gatelist[-2] = gates.Unitary(u4 @ g0.parameters[0], q0)
    gatelist[-1] = gates.Unitary(v4 @ g1.parameters[0], q1)

    return gatelist


def two_qubit_decomposition(q0, q1, unitary, backend, threshold=1e-6):
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
    if backend.np.allclose(
        unitary, backend.identity_density_matrix(nqubits=2, normalize=False)
    ):
        return []

    z_component = _get_z_component(unitary, backend)
    if abs(z_component) < threshold:
        return _two_qubit_decomposition_without_z(q0, q1, unitary, backend)
    return _two_qubit_decomposition_with_z(q0, q1, unitary, backend)

if __name__ == "__main__":
    import qibo
    from qibo import Circuit, gates
    import numpy as np
    from qibo.quantum_info.random_ensembles import random_unitary
    q0 = 0
    q1 = 1
    circ = Circuit(2)
    circ.add(gates.iSWAP(q0,q1))

    backend = qibo.get_backend()

    unitary = circ.unitary()

    #unitary = random_unitary(4, seed=10)

    circ2 = Circuit(2)

    #print("Desired unitary (CNOT):\n", unitary)
    decomp = two_qubit_decomposition(q0, q1, unitary, backend)
    circ2.add(decomp)
    reconstructed_unitary = circ2.unitary()
    #print("Decomposed unitary: ", reconstructed_unitary)

    print("ALLCLOSE?", np.allclose(unitary, reconstructed_unitary))

    #print("decomposition:", decomp)

    print("AllClose without phase: ", np.allclose(unitary/unitary[0,0], reconstructed_unitary/reconstructed_unitary[0,0]))


    """print("\n MANUAL::")
    print("Unitary in magic basis: \n", np.transpose(np.conj(magic_basis), (1, 0))
        @ unitary
        @ magic_basis)"""