import math

import numpy as np
import pytest
from scipy.linalg import expm

from qibo import Circuit, gates, matrices
from qibo.quantum_info.linalg_operations import partial_trace
from qibo.quantum_info.metrics import purity
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler import Optimize1qGatesDecomposition
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.unitary_decompositions import (
    bell_basis,
    calculate_h_vector,
    calculate_psi,
    calculate_single_qubit_unitaries,
    cnot_decomposition,
    cnot_decomposition_light,
    magic_basis,
    magic_decomposition,
    single_qubit_decomposition,
    to_bell_diagonal,
    two_qubit_decomposition,
)


def bell_unitary(hx, hy, hz, backend):
    ham = (
        hx * backend.cast(np.kron(matrices.X, matrices.X))
        + hy * backend.cast(np.kron(matrices.Y, matrices.Y))
        + hz * backend.cast(np.kron(matrices.Z, matrices.Z))
    )
    return backend.cast(expm(-1j * backend.to_numpy(ham)))


def assert_single_qubits(backend, psi, ua, ub):
    """Assert UA, UB map the maximally entangled basis ``psi`` to the magic basis."""
    uaub = backend.to_numpy(backend.kron(ua, ub))
    psi = backend.to_numpy(psi)
    for i, j in zip(range(4), [0, 1, 3, 2]):
        final_state = np.matmul(uaub, psi[:, i])
        target_state = magic_basis[:, j]
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        backend.assert_allclose(fidelity, 1)


def test_u3_decomposition(backend):
    theta, phi, lam = 0.1, 0.2, 0.3
    u3_matrix = gates.U3(0, theta, phi, lam).matrix(backend)

    rz1 = gates.RZ(0, phi).matrix(backend)
    rz2 = gates.RZ(0, theta).matrix(backend)
    rz3 = gates.RZ(0, lam).matrix(backend)
    rx1 = gates.RX(0, -np.pi / 2).matrix(backend)
    rx2 = gates.RX(0, np.pi / 2).matrix(backend)

    target_matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3

    backend.assert_allclose(u3_matrix, target_matrix)


@pytest.mark.parametrize("seed", [None, 10])
def test_eigenbasis_entanglement(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)

    """Check that the eigenvectors of UT_U are maximally entangled."""
    states, eigvals = calculate_psi(unitary, backend=backend)
    eigvals = backend.cast(eigvals, dtype=eigvals.dtype)

    backend.assert_allclose(backend.abs(eigvals), np.ones(4))

    for state in states.T:
        state = partial_trace(state, [1], backend=backend)
        backend.assert_allclose(purity(state, backend=backend), 0.5)


@pytest.mark.parametrize("seed", [None, 10])
def test_v_decomposition(backend, seed):
    """Check that V_A V_B |psi_k> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    psi, _ = calculate_psi(unitary, backend=backend)
    va, vb = calculate_single_qubit_unitaries(psi, backend=backend)
    assert_single_qubits(backend, psi, va, vb)


@pytest.mark.parametrize("seed", [None, 10])
def test_u_decomposition(backend, seed):
    r"""Check that U_A\dagger U_B\dagger |psi_k tilde> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    psi, eigvals = calculate_psi(unitary, backend=backend)
    psi_tilde = backend.conj(backend.sqrt(eigvals)) * backend.matmul(unitary, psi)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde, backend=backend)
    assert_single_qubits(backend, psi_tilde, ua_dagger, ub_dagger)


@pytest.mark.parametrize("seed", [None, 10])
def test_ud_eigenvalues(backend, seed):
    """Check that U_d is diagonal in the Bell basis."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    ua, ub, ud, va, vb = magic_decomposition(unitary, backend=backend)
    # Check kron
    unitary_recon = backend.kron(ua, ub) @ ud @ backend.kron(va, vb)
    backend.assert_allclose(unitary_recon, unitary)

    ud_bell = backend.dagger(backend.cast(bell_basis)) @ ud @ backend.cast(bell_basis)
    ud_diag = backend.diag(ud_bell)

    backend.assert_allclose(backend.diag(ud_diag), ud_bell, atol=1e-6, rtol=1e-6)
    backend.assert_allclose(backend.prod(ud_diag), 1, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed", [None, 10])
def test_calculate_h_vector(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)
    _, _, ud, _, _ = magic_decomposition(unitary, backend=backend)
    ud_diag = to_bell_diagonal(ud, backend=backend)

    assert ud_diag is not None

    hx, hy, hz = calculate_h_vector(ud_diag, backend=backend)
    target_matrix = bell_unitary(hx, hy, hz, backend)

    backend.assert_allclose(ud, target_matrix, atol=1e-6, rtol=1e-6)


def test_cnot_decomposition(backend):
    hx, hy, hz = np.random.random(3)
    target_matrix = bell_unitary(hx, hy, hz, backend)

    circuit = Circuit(2)
    circuit.add(cnot_decomposition(0, 1, hx, hy, hz, backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, target_matrix, atol=1e-6, rtol=1e-6)


def test_cnot_decomposition_light(backend):
    hx, hy = np.random.random(2)
    target_matrix = bell_unitary(hx, hy, 0, backend)

    circuit = Circuit(2)
    circuit.add(cnot_decomposition_light(0, 1, hx, hy, backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, target_matrix, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed", [None, 10])
def test_two_qubit_decomposition(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)

    circuit = Circuit(2)
    circuit.add(two_qubit_decomposition(0, 1, unitary, backend=backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, unitary, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("gate_name", ["CNOT", "CZ", "SWAP", "iSWAP", "fSim", "I"])
def test_two_qubit_decomposition_common_gates(backend, gate_name):
    """Test general two-qubit decomposition on some common gates."""
    gate = (
        gates.fSim(0, 1, theta=0.1, phi=0.2)
        if gate_name == "fSim"
        else getattr(gates, gate_name)(0, 1)
    )
    matrix = gate.matrix(backend)

    circuit = Circuit(2)
    circuit.add(two_qubit_decomposition(0, 1, matrix, backend=backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, matrix, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("hz_zero", [False, True])
def test_two_qubit_decomposition_bell_unitary(backend, hz_zero):
    hx, hy, hz = (2 * np.random.random(3) - 1) * np.pi
    if hz_zero:
        hz = 0
    unitary = backend.cast(bell_unitary(hx, hy, hz, backend))

    circuit = Circuit(2)
    circuit.add(two_qubit_decomposition(0, 1, unitary, backend=backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, unitary, atol=1e-6, rtol=1e-6)


def test_two_qubit_decomposition_no_entanglement(backend):
    """Test two-qubit decomposition on unitary that creates no entanglement."""
    matrix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix = backend.cast(matrix, dtype=matrix.dtype)

    circuit = Circuit(2)
    circuit.add(two_qubit_decomposition(0, 1, matrix, backend=backend))
    final_matrix = circuit.unitary(backend)

    backend.assert_allclose(final_matrix, matrix, atol=1e-6, rtol=1e-6)


def test_two_qubit_decomposition_structured_angles(backend):
    """Test decomposition of a unitary whose eigenvalue angles are multiples of pi/4."""
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.T(1))
    matrix = circuit.unitary(backend)

    decomposition = Circuit(2)
    decomposition.add(two_qubit_decomposition(0, 1, matrix, backend=backend))

    backend.assert_allclose(decomposition.unitary(backend), matrix, atol=1e-6)


@pytest.mark.parametrize("weight", [math.sqrt(2), math.pi, 0.3])
def test_two_qubit_decomposition_weight(backend, weight):
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.T(1))
    matrix = circuit.unitary(backend)

    decomposition = Circuit(2)
    decomposition.add(
        two_qubit_decomposition(0, 1, matrix, backend=backend, weight=weight)
    )

    backend.assert_allclose(decomposition.unitary(backend), matrix, atol=1e-6)


def test_calculate_psi_weight(backend):
    unitary = random_unitary(4, backend=backend)
    for weight in (math.sqrt(2), 2.0):
        states, eigvals = calculate_psi(unitary, backend=backend, weight=weight)
        magic = backend.cast(magic_basis)
        u_magic = backend.conj(magic).T @ unitary @ magic
        backend.assert_allclose(
            u_magic.T @ u_magic @ backend.conj(magic).T @ states,
            backend.conj(magic).T @ states * eigvals,
            atol=1e-8,
        )


@pytest.mark.parametrize(
    "gate_classes",
    [
        (gates.RX, gates.RZ),
        (gates.RY, gates.RZ),
        (gates.RX, gates.RY),
        (gates.RX, gates.RY, gates.RZ),
        (gates.RZ, gates.H),
        (gates.RX, gates.H),
        (gates.RZ, gates.SX),
        (gates.RZ, gates.SXDG),
        (gates.RX, gates.S),
        (gates.RX, gates.T),
        (gates.RZ, gates.H, gates.T),
        (gates.PRX,),
    ],
)
def test_single_qubit_decomposition(backend, gate_classes):
    for seed in range(10):
        unitary = random_unitary(2, seed=seed, backend=backend)
        decomposition = single_qubit_decomposition(
            unitary, 1, gate_classes, backend=backend
        )
        assert all(type(gate) in gate_classes for gate in decomposition)
        assert all(gate.qubits == (1,) for gate in decomposition)

        circuit = Circuit(2)
        circuit.add(decomposition)
        result = backend.to_numpy(circuit.unitary(backend))[:2, :2]
        target = backend.to_numpy(unitary)
        overlap = np.abs(np.trace(target.conj().T @ result)) / 2
        backend.assert_allclose(overlap, 1.0, atol=1e-8)


@pytest.mark.parametrize(
    "gate_classes",
    [(gates.RX, gates.RZ), (gates.RZ, gates.H), (gates.RZ, gates.SX), (gates.PRX,)],
)
def test_single_qubit_decomposition_special_unitaries(backend, gate_classes):
    # The identity needs no gates, and the Pauli gates are special cases of the rotations
    identity = backend.matrices.I(2)
    assert single_qubit_decomposition(identity, 0, gate_classes, backend=backend) == []
    for gate in (gates.X(0), gates.Y(0), gates.Z(0), gates.H(0), gates.T(0)):
        circuit = Circuit(1)
        circuit.add(
            single_qubit_decomposition(
                gate.matrix(backend), 0, gate_classes, backend=backend
            )
        )
        result = backend.to_numpy(circuit.unitary(backend))
        target = backend.to_numpy(gate.matrix(backend))
        overlap = np.abs(np.trace(target.conj().T @ result)) / 2
        backend.assert_allclose(overlap, 1.0, atol=1e-8)


@pytest.mark.parametrize(
    "gate_classes",
    [
        (),
        (gates.RZ,),
        (gates.RX,),
        (gates.RZ, gates.X),
        (gates.RZ, gates.T),
        (gates.H, gates.T),
    ],
)
def test_single_qubit_decomposition_error(backend, gate_classes):
    with pytest.raises(DecompositionError):
        single_qubit_decomposition(backend.matrices.X, 0, gate_classes, backend=backend)


@pytest.mark.parametrize(
    "gate_classes,maximum",
    [
        ((gates.RZ, gates.RY), 3),
        ((gates.RZ, gates.RX), 3),
        ((gates.RX, gates.RY), 3),
        ((gates.RZ, gates.SX), 5),
        ((gates.RZ, gates.SX, gates.X), 5),
        ((gates.U3,), 1),
    ],
)
def test_single_qubit_decomposition_euler_bases(backend, gate_classes, maximum):
    """The Euler bases are shared with the single-qubit optimizer, which gives the same gates."""
    names = [gate_class.__name__.lower() for gate_class in gate_classes]
    for seed in range(10):
        unitary = random_unitary(2, seed=seed, backend=backend)
        decomposition = single_qubit_decomposition(
            unitary, 0, gate_classes, backend=backend
        )
        assert len(decomposition) <= maximum

        circuit = Circuit(1)
        circuit.add(gates.Unitary(unitary, 0))
        optimized = Optimize1qGatesDecomposition(basis=names)(circuit, backend=backend)
        assert [gate.name for gate in optimized.queue] == [
            gate.name for gate in decomposition
        ]
        for gate, optimized_gate in zip(decomposition, optimized.queue):
            backend.assert_allclose(gate.parameters, optimized_gate.parameters)
