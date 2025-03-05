"""Test gates defined in `qibo/gates/gates.py`."""

import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.parameter import Parameter
from qibo.quantum_info import random_hermitian, random_statevector, random_unitary
from qibo.transpiler.decompositions import standard_decompositions


def apply_gates(backend, gatelist, nqubits=None, initial_state=None):
    if initial_state is None:
        state = backend.zero_state(nqubits)
    else:
        state = backend.cast(initial_state, dtype=initial_state.dtype, copy=True)
        if nqubits is None:
            nqubits = int(np.log2(len(state)))
        else:  # pragma: no cover
            assert nqubits == int(np.log2(len(state)))

    for gate in gatelist:
        state = backend.apply_gate(gate, state, nqubits)

    return state


def test_h(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1)], nqubits=2)
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.H(0).qasm_label == "h"
    assert gates.H(0).clifford
    assert gates.H(0).unitary


def test_x(backend):
    final_state = apply_gates(backend, [gates.X(0)], nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[2] = 1.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.X(0).qasm_label == "x"
    assert gates.X(0).clifford
    assert gates.X(0).unitary


def test_y(backend):
    final_state = apply_gates(backend, [gates.Y(1)], nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[1] = 1j
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.Y(0).qasm_label == "y"
    assert gates.Y(0).clifford
    assert gates.Y(0).unitary


def test_z(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.Z(0)], nqubits=2)
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.Z(0).qasm_label == "z"
    assert gates.Z(0).clifford
    assert gates.Z(0).unitary


def test_sx(backend):
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.SX(0)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.SX(0).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2.0
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, backend=backend)
    np_final_state_decompose = backend.to_numpy(final_state_decompose)
    np_obs = backend.to_numpy(observable)
    np_target_state = backend.to_numpy(target_state)
    backend.assert_allclose(
        np.conj(np_final_state_decompose).T @ np_obs @ np_final_state_decompose,
        np.conj(np_target_state).T @ np_obs @ np_target_state,
        atol=1e-6,
    )

    assert gates.SX(0).qasm_label == "sx"
    assert gates.SX(0).clifford
    assert gates.SX(0).unitary


def test_sxdg(backend):
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.SXDG(0)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.SXDG(0).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, backend=backend)
    np_final_state_decompose = backend.to_numpy(final_state_decompose)
    np_obs = backend.to_numpy(observable)
    np_target_state = backend.to_numpy(target_state)
    backend.assert_allclose(
        np.conj(np_final_state_decompose).T @ np_obs @ np_final_state_decompose,
        np.conj(np_target_state).T @ np_obs @ np_target_state,
        atol=1e-6,
    )

    assert gates.SXDG(0).qasm_label == "sxdg"
    assert gates.SXDG(0).clifford
    assert gates.SXDG(0).unitary


def test_s(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.S(1)], nqubits=2)
    target_state = np.array([0.5, 0.5j, 0.5, 0.5j])
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.S(0).qasm_label == "s"
    assert gates.S(0).clifford
    assert gates.S(0).unitary


def test_sdg(backend):
    final_state = apply_gates(
        backend, [gates.H(0), gates.H(1), gates.SDG(1)], nqubits=2
    )
    target_state = np.array([0.5, -0.5j, 0.5, -0.5j])
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.SDG(0).qasm_label == "sdg"
    assert gates.SDG(0).clifford
    assert gates.SDG(0).unitary


def test_t(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.T(1)], nqubits=2)
    target_state = np.array([0.5, (1 + 1j) / np.sqrt(8), 0.5, (1 + 1j) / np.sqrt(8)])
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.T(0).qasm_label == "t"
    assert not gates.T(0).clifford
    assert gates.T(0).unitary


def test_tdg(backend):
    final_state = apply_gates(
        backend, [gates.H(0), gates.H(1), gates.TDG(1)], nqubits=2
    )
    target_state = np.array([0.5, (1 - 1j) / np.sqrt(8), 0.5, (1 - 1j) / np.sqrt(8)])
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.TDG(0).qasm_label == "tdg"
    assert not gates.TDG(0).clifford
    assert gates.TDG(0).unitary


def test_identity(backend):
    gatelist = [gates.H(0), gates.H(1), gates.I(0), gates.I(1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    gatelist = [gates.H(0), gates.H(1), gates.I(0, 1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.I(0).qasm_label == "id"
    assert gates.I(0).clifford
    assert gates.I(0).unitary


def test_align(backend):
    with pytest.raises(TypeError):
        gates.Align(0, delay="0.1")
    with pytest.raises(ValueError):
        gates.Align(0, delay=-1)

    nqubits = 1

    gate = gates.Align(0, 0)
    gate_list = [gates.H(0), gate]

    final_state = apply_gates(backend, gate_list, nqubits=nqubits)
    target_state = backend.plus_state(nqubits)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    gate_matrix = gate.matrix(backend)
    identity = backend.identity_density_matrix(nqubits, normalize=False)
    backend.assert_allclose(gate_matrix, identity, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gate.qasm_label

    assert not gates.Align(0, delay=0).clifford
    assert not gates.Align(0, delay=0).unitary


# :class:`qibo.core.cgates.M` is tested seperately in `test_measurement_gate.py`


@pytest.mark.parametrize("theta", [np.random.rand(), np.pi / 2.0, -np.pi / 2.0, np.pi])
def test_rx(backend, theta):
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.RX(0, theta=theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag], [-1j * phase.imag, phase.real]])
    gate = backend.cast(gate)
    target_state = gate @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RX(0, theta=theta).qasm_label == "rx"
    assert gates.RX(0, theta=theta).unitary
    if (theta % (np.pi / 2.0)).is_integer():
        assert gates.RX(0, theta=theta).clifford
    else:
        assert not gates.RX(0, theta=theta).clifford

    # test Parameter
    assert (
        gates.RX(
            0,
            theta=Parameter(
                lambda x, th1: 10 * th1 + x, trainable=[0.2], features=[40]
            ),
        ).init_kwargs["theta"]
        == 42
    )


@pytest.mark.parametrize("theta", [np.random.rand(), np.pi / 2.0, -np.pi / 2.0, np.pi])
def test_ry(backend, theta):
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.RY(0, theta=theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag], [phase.imag, phase.real]])
    gate = backend.cast(gate)
    target_state = gate @ backend.cast(initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RY(0, theta=theta).qasm_label == "ry"
    assert gates.RY(0, theta=theta).unitary
    if (theta % (np.pi / 2.0)).is_integer():
        assert gates.RY(0, theta=theta).clifford
    else:
        assert not gates.RY(0, theta=theta).clifford


@pytest.mark.parametrize("apply_x", [True, False])
@pytest.mark.parametrize("theta", [np.random.rand(), np.pi / 2.0, -np.pi / 2.0, np.pi])
def test_rz(backend, theta, apply_x):
    nqubits = 1
    if apply_x:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.RZ(0, theta))
    final_state = apply_gates(backend, gatelist, nqubits=nqubits)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    p = int(apply_x)
    target_state[p] = np.exp((2 * p - 1) * 1j * theta / 2.0)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RZ(0, theta).qasm_label == "rz"
    assert gates.RZ(0, theta=theta).unitary
    if (theta % (np.pi / 2)).is_integer():
        assert gates.RZ(0, theta=theta).clifford
    else:
        assert not gates.RZ(0, theta=theta).clifford


def test_prx(backend):
    theta = 0.52
    phi = 0.24

    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.PRX(0, theta, phi)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.PRX(0, theta, phi).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    exponent1 = -1.0j * np.exp(-1.0j * phi)
    exponent2 = -1.0j * np.exp(1.0j * phi)
    matrix = np.array([[cos, exponent1 * sin], [exponent2 * sin, cos]])
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state)
    backend.assert_allclose(final_state_decompose, target_state)

    assert gates.PRX(0, theta, phi).qasm_label == "prx"
    assert not gates.PRX(0, theta, phi).clifford
    assert gates.PRX(0, theta, phi).unitary


def test_gpi(backend):
    phi = 0.1234
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gates.GPI(0, phi)], initial_state=initial_state)

    phase = np.exp(1.0j * phi)
    matrix = np.array([[0, np.conj(phase)], [phase, 0]])
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.GPI(0, phi).qasm_label == "gpi"
    assert not gates.GPI(0, phi).clifford
    assert gates.GPI(0, phi).unitary


def test_gpi2(backend):
    phi = 0.1234
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend, [gates.GPI2(0, phi)], initial_state=initial_state
    )

    phase = np.exp(1.0j * phi)
    matrix = np.array([[1, -1.0j * np.conj(phase)], [-1.0j * phase, 1]]) / np.sqrt(2)
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.GPI2(0, phi).qasm_label == "gpi2"
    assert not gates.GPI2(0, phi).clifford
    assert gates.GPI2(0, phi).unitary


def test_u1(backend):
    theta = 0.1234
    final_state = apply_gates(backend, [gates.X(0), gates.U1(0, theta)], nqubits=1)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[1] = np.exp(1j * theta)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.U1(0, theta).qasm_label == "u1"
    assert not gates.U1(0, theta).clifford
    assert gates.U1(0, theta).unitary


def test_u2(backend):
    phi = 0.1234
    lam = 0.4321
    nqubits = 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend, [gates.U2(0, phi, lam)], initial_state=initial_state
    )
    matrix = np.array(
        [
            [np.exp(-1j * (phi + lam) / 2), -np.exp(-1j * (phi - lam) / 2)],
            [np.exp(1j * (phi - lam) / 2), np.exp(1j * (phi + lam) / 2)],
        ]
    )
    target_state = np.matmul(matrix, backend.to_numpy(initial_state)) / np.sqrt(2)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.U2(0, phi, lam).qasm_label == "u2"
    assert not gates.U2(0, phi, lam).clifford
    assert gates.U2(0, phi, lam).unitary


@pytest.mark.parametrize("seed_observable", list(range(1, 10 + 1)))
@pytest.mark.parametrize("seed_state", list(range(1, 10 + 1)))
def test_u3(backend, seed_state, seed_observable):
    nqubits = 1
    theta, phi, lam = np.random.rand(3)

    initial_state = random_statevector(2**nqubits, seed=seed_state, backend=backend)
    final_state = apply_gates(
        backend, [gates.U3(0, theta, phi, lam)], initial_state=initial_state
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.U3(0, theta, phi, lam).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    cost, sint = np.cos(theta / 2), np.sin(theta / 2)
    ep = np.exp(1j * (phi + lam) / 2)
    em = np.exp(1j * (phi - lam) / 2)

    matrix = np.array([[ep.conj() * cost, -em.conj() * sint], [em * sint, ep * cost]])
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, seed=seed_observable, backend=backend)
    backend.assert_allclose(
        backend.cast(backend.np.conj(final_state_decompose).T)
        @ observable
        @ final_state_decompose,
        backend.cast(backend.np.conj(target_state).T)
        @ observable
        @ backend.cast(target_state),
        atol=1e-6,
    )
    assert gates.U3(0, theta, phi, lam).qasm_label == "u3"
    assert not gates.U3(0, theta, phi, lam).clifford
    assert gates.U3(0, theta, phi, lam).unitary


@pytest.mark.parametrize("seed_state", list(range(1, 10 + 1)))
def test_u1q(backend, seed_state):
    nqubits = 1
    theta, phi = np.random.rand(2)

    initial_state = random_statevector(2**nqubits, seed=seed_state, backend=backend)
    final_state = apply_gates(
        backend, [gates.U1q(0, theta, phi)], initial_state=initial_state
    )
    cost, sint = np.cos(theta / 2), np.sin(theta / 2)

    matrix = np.array(
        [[cost, -1j * np.exp(-1j * phi) * sint], [-1j * np.exp(1j * phi) * sint, cost]]
    )
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert not gates.U1q(0, theta, phi).clifford
    assert gates.U1q(0, theta, phi).unitary


@pytest.mark.parametrize("applyx", [False, True])
def test_cnot(backend, applyx):
    if applyx:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.CNOT(0, 1))
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[3 * int(applyx)] = 1.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.CNOT(0, 1).qasm_label == "cx"
    assert gates.CNOT(0, 1).clifford
    assert gates.CNOT(0, 1).unitary


@pytest.mark.parametrize("seed_observable", list(range(1, 10 + 1)))
@pytest.mark.parametrize("seed_state", list(range(1, 10 + 1)))
def test_cy(backend, seed_state, seed_observable):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, seed=seed_state, backend=backend)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ]
    )
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.CY(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    gate = gates.CY(0, 1)

    final_state = apply_gates(backend, [gate], initial_state=initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, seed=seed_observable, backend=backend)
    backend.assert_allclose(
        backend.cast(backend.np.conj(final_state_decompose).T)
        @ observable
        @ final_state_decompose,
        backend.cast(backend.np.conj(target_state).T)
        @ observable
        @ backend.cast(target_state),
        atol=1e-6,
    )

    assert gate.name == "cy"
    assert gate.qasm_label == "cy"
    assert gate.clifford
    assert gate.unitary


@pytest.mark.parametrize("seed_observable", list(range(1, 10 + 1)))
@pytest.mark.parametrize("seed_state", list(range(1, 10 + 1)))
def test_cz(backend, seed_state, seed_observable):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, seed=seed_state, backend=backend)
    matrix = np.eye(4)
    matrix[3, 3] = -1
    matrix = backend.cast(matrix)

    target_state = backend.np.matmul(matrix, initial_state)
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.CZ(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    gate = gates.CZ(0, 1)

    final_state = apply_gates(backend, [gate], initial_state=initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, seed=seed_observable, backend=backend)
    backend.assert_allclose(
        backend.cast(backend.np.conj(final_state_decompose).T)
        @ observable
        @ final_state_decompose,
        backend.cast(backend.np.conj(target_state).T)
        @ observable
        @ backend.cast(target_state),
        atol=1e-6,
    )

    assert gate.name == "cz"
    assert gate.qasm_label == "cz"
    assert gate.clifford
    assert gate.unitary


def test_csx(backend):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.CSX(0, 1)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.CSX(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, (1 + 1j) / 2, (1 - 1j) / 2],
            [0, 0, (1 - 1j) / 2, (1 + 1j) / 2],
        ]
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    assert gates.CSX(0, 1).qasm_label == "csx"
    assert not gates.CSX(0, 1).clifford
    assert gates.CSX(0, 1).unitary


def test_csxdg(backend):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.CSXDG(0, 1)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.CSXDG(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, (1 - 1j) / 2, (1 + 1j) / 2],
            [0, 0, (1 + 1j) / 2, (1 - 1j) / 2],
        ]
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    assert gates.CSXDG(0, 1).qasm_label == "csxdg"
    assert not gates.CSXDG(0, 1).clifford
    assert gates.CSXDG(0, 1).unitary


@pytest.mark.parametrize(
    "name,params",
    [
        ("CRX", {"theta": 0.1}),
        ("CRX", {"theta": np.random.randint(-5, 5) * np.pi / 2}),
        ("CRY", {"theta": 0.2}),
        ("CRY", {"theta": np.random.randint(-5, 5) * np.pi / 2}),
        ("CRZ", {"theta": 0.3}),
        ("CRZ", {"theta": np.random.randint(-5, 5) * np.pi / 2}),
        ("CU1", {"theta": 0.1}),
        ("CU2", {"phi": 0.1, "lam": 0.2}),
        ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
    ],
)
def test_cun(backend, name, params):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)

    gate = getattr(gates, name)(0, 1, **params)

    if name == "CRY":
        decomposition = gate.decompose()

    assert gate.unitary

    if name != "CU2":
        assert gate.qasm_label == gate.name
    else:
        with pytest.raises(NotImplementedError):
            gate.qasm_label

    if name in ["CRX", "CRY", "CRZ"]:
        theta = params["theta"]
        if (theta % (np.pi / 2)).is_integer():
            assert gate.clifford
        else:
            assert not gate.clifford

    final_state = apply_gates(backend, [gate], initial_state=initial_state)

    _matrix = gate.matrix(backend)
    gate = backend.cast(_matrix, dtype=_matrix.dtype)

    target_state = backend.np.matmul(gate, initial_state)

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    if name == "CRY":
        matrix = Circuit(2)
        matrix.add(decomposition)
        matrix = matrix.unitary(backend=backend)
        backend.assert_allclose(matrix, _matrix, atol=1e-10)


def test_swap(backend):
    final_state = apply_gates(backend, [gates.X(1), gates.SWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[2] = 1.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.SWAP(0, 1).qasm_label == "swap"
    assert gates.SWAP(0, 1).clifford
    assert gates.SWAP(0, 1).unitary


def test_iswap(backend):
    final_state = apply_gates(backend, [gates.X(1), gates.iSWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[2] = 1.0j
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.iSWAP(0, 1).qasm_label == "iswap"
    assert gates.iSWAP(0, 1).clifford
    assert gates.iSWAP(0, 1).unitary


def test_siswap(backend):
    final_state = apply_gates(backend, [gates.X(1), gates.SiSWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[1] = 1.0 / np.sqrt(2)
    target_state[2] = 1.0j / np.sqrt(2)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert not gates.SiSWAP(0, 1).clifford
    assert gates.SiSWAP(0, 1).unitary


def test_fswap(backend):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.FSWAP(0, 1)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.FSWAP(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    assert gates.FSWAP(0, 1).qasm_label == "fswap"
    assert gates.FSWAP(0, 1).clifford
    assert gates.FSWAP(0, 1).unitary


def test_multiple_swap(backend):
    gatelist = [gates.X(0), gates.X(2), gates.SWAP(0, 1), gates.SWAP(2, 3)]
    final_state = apply_gates(backend, gatelist, nqubits=4)
    gatelist = [gates.X(1), gates.X(3)]
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_fsim(backend):
    theta = 0.1234
    phi = 0.4321
    gatelist = [gates.H(0), gates.H(1), gates.fSim(0, 1, theta, phi)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2.0
    rotation = np.array(
        [[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]]
    )
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    matrix = backend.cast(matrix)
    target_state = backend.np.matmul(matrix, backend.cast(target_state))

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.fSim(0, 1, theta, phi).qasm_label

    assert not gates.fSim(0, 1, theta, phi).clifford
    assert gates.fSim(0, 1, theta, phi).unitary


def test_sycamore(backend):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.SYC(0, 1)],
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1 + 0j, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, np.exp(-1j * np.pi / 6)],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.SYC(0, 1).qasm_label

    assert not gates.SYC(0, 1).clifford
    assert gates.SYC(0, 1).unitary


def test_generalized_fsim(backend):
    phi = np.random.random()
    rotation = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    gatelist = [gates.H(0), gates.H(1), gates.H(2)]
    gatelist.append(gates.GeneralizedfSim(1, 2, rotation, phi))
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.ones(len(final_state), dtype=complex) / np.sqrt(8)
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    # matrix = backend.cast(matrix, dtype=matrix.dtype)
    target_state[:4] = np.matmul(matrix, target_state[:4])
    target_state[4:] = np.matmul(matrix, target_state[4:])
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gatelist[-1].qasm_label

    assert not gates.GeneralizedfSim(0, 1, rotation, phi).clifford
    assert gates.GeneralizedfSim(0, 1, rotation, phi).unitary


def test_generalized_fsim_parameter_setter(backend):
    phi = np.random.random()
    matrix = np.random.random((2, 2))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    backend.assert_allclose(gate.parameters[0], matrix, atol=1e-6)

    assert gate.parameters[1] == phi

    matrix = np.random.random((4, 4))

    with pytest.raises(ValueError):
        gates.GeneralizedfSim(0, 1, matrix, phi)

    with pytest.raises(NotImplementedError):
        gate.qasm_label

    assert not gate.clifford
    assert gate.unitary


def test_rxx(backend):
    theta = 0.1234
    final_state = apply_gates(
        backend, [gates.H(0), gates.H(1), gates.RXX(0, 1, theta=theta)], nqubits=2
    )
    phase = np.exp(1j * theta / 2.0)
    gate = np.array(
        [
            [phase.real, 0, 0, -1j * phase.imag],
            [0, phase.real, -1j * phase.imag, 0],
            [0, -1j * phase.imag, phase.real, 0],
            [-1j * phase.imag, 0, 0, phase.real],
        ]
    )
    target_state = gate.dot(np.ones(4)) / 2.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RXX(0, 1, theta=theta).qasm_label == "rxx"
    assert not gates.RXX(0, 1, theta).clifford
    assert gates.RXX(0, 1, theta).unitary


def test_ryy(backend):
    theta = 0.1234
    final_state = apply_gates(
        backend, [gates.H(0), gates.H(1), gates.RYY(0, 1, theta=theta)], nqubits=2
    )
    phase = np.exp(1j * theta / 2.0)
    gate = np.array(
        [
            [phase.real, 0, 0, 1j * phase.imag],
            [0, phase.real, -1j * phase.imag, 0],
            [0, -1j * phase.imag, phase.real, 0],
            [1j * phase.imag, 0, 0, phase.real],
        ]
    )
    target_state = gate.dot(np.ones(4)) / 2.0
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RYY(0, 1, theta=theta).qasm_label == "ryy"
    assert not gates.RYY(0, 1, theta).clifford
    assert gates.RYY(0, 1, theta).unitary


def test_rzz(backend):
    theta = 0.1234
    final_state = apply_gates(
        backend, [gates.X(0), gates.X(1), gates.RZZ(0, 1, theta=theta)], nqubits=2
    )
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[3] = np.exp(-1j * theta / 2.0)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.RZZ(0, 1, theta=theta).qasm_label == "rzz"
    assert not gates.RZZ(0, 1, theta).clifford
    assert gates.RZZ(0, 1, theta).unitary


def test_rzx(backend):
    theta = 0.1234
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.RZX(0, 1, theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.RZX(0, 1, theta).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    cos, sin = np.cos(theta / 2), np.sin(theta / 2)
    matrix = np.array(
        [
            [cos, -1j * sin, 0, 0],
            [-1j * sin, cos, 0, 0],
            [0, 0, cos, 1j * sin],
            [0, 0, 1j * sin, cos],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.RZX(0, 1, theta).qasm_label

    assert not gates.RZX(0, 1, theta).clifford
    assert gates.RZX(0, 1, theta).unitary


def test_rxxyy(backend):
    theta = 0.1234
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.RXXYY(0, 1, theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.RXXYY(0, 1, theta).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    cos, sin = np.cos(theta / 2), np.sin(theta / 2)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, cos, -1j * sin, 0],
            [0, -1j * sin, cos, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)
    target_state = matrix @ initial_state

    observable = random_hermitian(2**nqubits, backend=backend)

    backend.assert_allclose(final_state, target_state, atol=1e-6)
    # testing random expectation value due to global phase difference
    backend.assert_allclose(
        backend.cast(backend.np.conj(final_state_decompose).T)
        @ observable
        @ final_state_decompose,
        backend.cast(backend.np.conj(target_state).T) @ observable @ target_state,
        atol=1e-6,
    )

    with pytest.raises(NotImplementedError):
        gates.RXXYY(0, 1, theta).qasm_label

    assert not gates.RXXYY(0, 1, theta).clifford
    assert gates.RXXYY(0, 1, theta).unitary


def test_ms(backend):
    phi0 = 0.1234
    phi1 = 0.4321
    theta = np.pi / 2

    with pytest.raises(ValueError):
        gates.MS(0, 1, phi0=phi0, phi1=phi1, theta=np.pi)

    final_state = apply_gates(
        backend,
        [gates.H(0), gates.H(1), gates.MS(0, 1, phi0=phi0, phi1=phi1)],
        nqubits=2,
    )
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2.0
    plus = np.exp(1.0j * (phi0 + phi1))
    minus = np.exp(1.0j * (phi0 - phi1))

    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[3, 0] = -1.0j * plus
    matrix[0, 3] = -1.0j * np.conj(plus)
    matrix[2, 1] = -1.0j * minus
    matrix[1, 2] = -1.0j * np.conj(minus)
    matrix /= np.sqrt(2)
    matrix = backend.cast(matrix)
    target_state = backend.np.matmul(matrix, backend.cast(target_state))

    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gates.MS(0, 1, phi0, phi1, theta).qasm_label == "ms"
    assert not gates.MS(0, 1, phi0, phi1).clifford
    assert gates.MS(0, 1, phi0, phi1).unitary


def test_givens(backend):
    theta = 0.1234
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.GIVENS(0, 1, theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.GIVENS(0, 1, theta).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.GIVENS(0, 1, theta).qasm_label

    assert not gates.GIVENS(0, 1, theta).clifford
    assert gates.GIVENS(0, 1, theta).unitary


def test_rbs(backend):
    theta = 0.1234
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.RBS(0, 1, theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.RBS(0, 1, theta).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), np.sin(theta), 0],
            [0, -np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    backend.assert_allclose(final_state_decompose, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.RBS(0, 1, theta).qasm_label

    assert not gates.RBS(0, 1, theta).clifford
    assert gates.RBS(0, 1, theta).unitary


def test_ecr(backend):
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.ECR(0, 1)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.ECR(0, 1).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [0, 0, 1, 1j],
            [0, 0, 1j, 1],
            [1, -1j, 0, 0],
            [-1j, 1, 0, 0],
        ],
        dtype=np.complex128,
    ) / np.sqrt(2)
    matrix = backend.cast(matrix)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    # testing random expectation value due to global phase difference
    observable = random_hermitian(2**nqubits, backend=backend)
    backend.assert_allclose(
        backend.cast(backend.np.conj(final_state_decompose).T)
        @ observable
        @ final_state_decompose,
        backend.cast(backend.np.conj(target_state).T) @ observable @ target_state,
        atol=1e-6,
    )

    with pytest.raises(NotImplementedError):
        gates.ECR(0, 1).qasm_label

    assert gates.ECR(0, 1).clifford
    assert gates.ECR(0, 1).unitary


@pytest.mark.parametrize("applyx", [False, True])
def test_toffoli(backend, applyx):
    if applyx:
        gatelist = [gates.X(0), gates.X(1), gates.TOFFOLI(0, 1, 2)]
    else:
        gatelist = [gates.X(1), gates.TOFFOLI(0, 1, 2)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    if applyx:
        target_state[-1] = 1
    else:
        target_state[2] = 1
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    assert gatelist[-1].qasm_label == "ccx"
    assert not gates.TOFFOLI(0, 1, 2).clifford
    assert gates.TOFFOLI(0, 1, 2).unitary

    # test decomposition
    decomposition = Circuit(3)
    decomposition.add(standard_decompositions(gates.TOFFOLI(0, 1, 2)))
    decomposition = decomposition.unitary(backend)

    backend.assert_allclose(decomposition, backend.cast(matrices.TOFFOLI), atol=1e-10)


def test_ccz(backend):
    nqubits = 3
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.CCZ(0, 1, 2)],
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix, dtype=matrix.dtype)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state)

    assert gates.CCZ(0, 1, 2).qasm_label == "ccz"
    assert not gates.CCZ(0, 1, 2).clifford
    assert gates.CCZ(0, 1, 2).unitary

    # test decomposition
    decomposition = Circuit(3)
    decomposition.add(gates.CCZ(0, 1, 2).decompose())
    decomposition = decomposition.unitary(backend)

    backend.assert_allclose(decomposition, backend.cast(matrices.CCZ), atol=1e-10)


def test_deutsch(backend):
    theta = 0.1234
    nqubits = 3
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.DEUTSCH(0, 1, 2, theta)],
        nqubits=nqubits,
        initial_state=initial_state,
    )

    sin, cos = np.sin(theta), np.cos(theta)
    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1j * cos, sin],
            [0, 0, 0, 0, 0, 0, sin, 1j * cos],
        ],
        dtype=np.complex128,
    )
    matrix = backend.cast(matrix)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.DEUTSCH(0, 1, 2, theta).qasm_label

    assert not gates.DEUTSCH(0, 1, 2, theta).clifford
    assert gates.DEUTSCH(0, 1, 2, theta).unitary


def test_generalized_rbs(backend):
    theta, phi = 0.1234, 0.4321
    qubits_in, qubits_out = [0, 3], [1, 2]
    nqubits = len(qubits_in + qubits_out)
    integer_in = "".join(["1" if k in qubits_in else "0" for k in range(nqubits)])
    integer_out = "".join(["1" if k in qubits_out else "0" for k in range(nqubits)])
    integer_in, integer_out = int(integer_in, 2), int(integer_out, 2)

    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend,
        [gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi)],
        nqubits=nqubits,
        initial_state=initial_state,
    )
    # test decomposition
    final_state_decompose = apply_gates(
        backend,
        gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi).decompose(),
        nqubits=nqubits,
        initial_state=initial_state,
    )

    matrix = np.eye(2**nqubits, dtype=complex)
    exp, sin, cos = np.exp(1j * phi), np.sin(theta), np.cos(theta)
    matrix[integer_in, integer_in] = exp * cos
    matrix[integer_in, integer_out] = -exp * sin
    matrix[integer_out, integer_in] = np.conj(exp) * sin
    matrix[integer_out, integer_out] = np.conj(exp) * cos
    matrix = backend.cast(matrix, dtype=matrix.dtype)

    target_state = matrix @ initial_state
    backend.assert_allclose(final_state, target_state)
    backend.assert_allclose(final_state_decompose, target_state)

    with pytest.raises(NotImplementedError):
        gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi).qasm_label

    assert not gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi).clifford
    assert gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi).unitary


@pytest.mark.parametrize("seed", [10])
def test_generalized_rbs_apply(backend, seed):
    rng = np.random.default_rng(seed)

    nqubits = 4
    dims = 2**nqubits
    theta, phi = 2 * np.pi * rng.random(2)

    qubit_ids = rng.choice(np.arange(0, nqubits), size=nqubits - 1, replace=False)
    qubits_in, qubits_out = qubit_ids[:1], qubit_ids[1:]

    gate = gates.GeneralizedRBS(qubits_in, qubits_out, theta, phi)
    matrix = Circuit(nqubits)
    matrix.add(gate)
    matrix = matrix.unitary(backend=backend)

    state = random_statevector(dims, seed=rng, backend=backend)
    target = matrix @ state

    state = gate.apply(backend, state, nqubits)

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("nqubits", [2, 3])
def test_unitary(backend, nqubits):
    initial_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)
    matrix = np.random.random(2 * (2 ** (nqubits - 1),))
    target_state = np.kron(np.eye(2), matrix).dot(initial_state)
    gatelist = [gates.H(i) for i in range(nqubits)]
    gate = gates.Unitary(matrix, *range(1, nqubits), name="random")
    gatelist.append(gate)
    final_state = apply_gates(backend, gatelist, nqubits=nqubits)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_unitary_initialization(backend):

    matrix = np.random.random((4, 4))
    gate = gates.Unitary(matrix, 0, 1)
    backend.assert_allclose(gate.parameters[0], matrix, atol=1e-6)

    with pytest.raises(NotImplementedError):
        gates.Unitary(matrix, 0, 1).qasm_label

    assert not gates.Unitary(matrix, 0, 1).clifford
    assert not gates.Unitary(matrix, 0, 1, check_unitary=False).unitary
    assert gates.Unitary(random_unitary(2, backend=backend), 0).unitary


def test_unitary_common_gates(backend):
    target_state = apply_gates(backend, [gates.X(0), gates.H(1)], nqubits=2)
    gatelist = [
        gates.Unitary(backend.cast([[0.0, 1.0], [1.0, 0.0]]), 0),
        gates.Unitary(backend.cast([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2), 1),
    ]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    thetax = 0.1234
    thetay = 0.4321
    gatelist = [gates.RX(0, theta=thetax), gates.RY(1, theta=thetay), gates.CNOT(0, 1)]
    target_state = apply_gates(backend, gatelist, nqubits=2)

    rx = np.array(
        [
            [np.cos(thetax / 2), -1j * np.sin(thetax / 2)],
            [-1j * np.sin(thetax / 2), np.cos(thetax / 2)],
        ]
    )
    ry = np.array(
        [
            [np.cos(thetay / 2), -np.sin(thetay / 2)],
            [np.sin(thetay / 2), np.cos(thetay / 2)],
        ]
    )
    cnot = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    gatelist = [gates.Unitary(rx, 0), gates.Unitary(ry, 1), gates.Unitary(cnot, 0, 1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_unitary_multiqubit(backend):
    gatelist = [gates.H(i) for i in range(4)]
    gatelist.append(gates.CNOT(0, 1))
    gatelist.append(gates.CNOT(2, 3))
    gatelist.extend(gates.X(i) for i in range(4))

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    x = np.array([[0, 1], [1, 0]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    matrix = np.kron(np.kron(x, x), np.kron(x, x))
    matrix = matrix @ np.kron(cnot, cnot)
    matrix = matrix @ np.kron(np.kron(h, h), np.kron(h, h))
    unitary = gates.Unitary(matrix, 0, 1, 2, 3)
    final_state = apply_gates(backend, [unitary], nqubits=4)
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


############################# Test ``controlled_by`` #############################


def test_controlled_x(backend):
    gatelist = [
        gates.X(0),
        gates.X(1),
        gates.X(2),
        gates.X(3).controlled_by(0, 1, 2),
        gates.X(0),
        gates.X(2),
    ]
    final_state = apply_gates(backend, gatelist, nqubits=4)
    gatelist = [gates.X(1), gates.X(3)]
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_x_vs_cnot(backend):
    gatelist = [gates.X(0), gates.X(2).controlled_by(0)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.CNOT(0, 2)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_x_vs_toffoli(backend):
    gatelist = [gates.X(0), gates.X(2), gates.X(1).controlled_by(0, 2)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.X(2), gates.TOFFOLI(0, 2, 1)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_rx(backend, applyx):
    theta = 0.1234
    gatelist = [gates.X(0)]
    if applyx:
        gatelist.append(gates.X(1))
    gatelist.append(gates.RX(2, theta).controlled_by(0, 1))
    gatelist.append(gates.X(0))
    final_state = apply_gates(backend, gatelist, nqubits=3)

    gatelist = []
    if applyx:
        gatelist.extend([gates.X(1), gates.RX(2, theta)])
    target_state = apply_gates(backend, gatelist, nqubits=3)

    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_u1(backend):
    theta = 0.1234
    gatelist = [gates.X(i) for i in range(3)]
    gatelist.append(gates.U1(2, theta).controlled_by(0, 1))
    gatelist.append(gates.X(0))
    gatelist.append(gates.X(1))
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.zeros_like(backend.to_numpy(final_state))
    target_state[1] = np.exp(1j * theta)
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    gate = gates.U1(0, theta).controlled_by(1)
    assert gate.__class__.__name__ == "CU1"


def test_controlled_u2(backend):
    phi = 0.1234
    lam = 0.4321
    gatelist = [gates.X(0), gates.X(1)]
    gatelist.append(gates.U2(2, phi, lam).controlled_by(0, 1))
    gatelist.extend([gates.X(0), gates.X(1)])
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.X(1), gates.U2(2, phi, lam), gates.X(0), gates.X(1)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    # for coverage
    gate = gates.CU2(0, 1, phi, lam)
    assert gate.parameters == (phi, lam)


def test_controlled_u3(backend):
    theta, phi, lam = 0.1, 0.1234, 0.4321
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(
        backend, [gates.U3(1, theta, phi, lam).controlled_by(0)], 2, initial_state
    )
    target_state = apply_gates(
        backend, [gates.CU3(0, 1, theta, phi, lam)], 2, initial_state
    )
    backend.assert_allclose(final_state, target_state, atol=1e-6)
    # for coverage
    gate = gates.U3(0, theta, phi, lam)
    assert gate.parameters == (theta, phi, lam)


@pytest.mark.parametrize("applyx", [False, True])
@pytest.mark.parametrize("free_qubit", [False, True])
def test_controlled_swap(backend, applyx, free_qubit):
    f = int(free_qubit)
    gatelist = []
    if applyx:
        gatelist.append(gates.X(0))
    gatelist.extend(
        [
            gates.RX(1 + f, theta=0.1234),
            gates.RY(2 + f, theta=0.4321),
            gates.SWAP(1 + f, 2 + f).controlled_by(0),
        ]
    )
    final_state = apply_gates(backend, gatelist, 3 + f)
    gatelist = [gates.RX(1 + f, theta=0.1234), gates.RY(2 + f, theta=0.4321)]
    if applyx:
        gatelist.extend([gates.X(0), gates.SWAP(1 + f, 2 + f)])
    target_state = apply_gates(backend, gatelist, 3 + f)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_swap_double(backend, applyx):
    gatelist = [gates.X(0)]
    if applyx:
        gatelist.append(gates.X(3))
    gatelist.append(gates.RX(1, theta=0.1234))
    gatelist.append(gates.RY(2, theta=0.4321))
    gatelist.append(gates.SWAP(1, 2).controlled_by(0, 3))
    gatelist.append(gates.X(0))
    final_state = apply_gates(backend, gatelist, 4)

    gatelist = [gates.RX(1, theta=0.1234), gates.RY(2, theta=0.4321)]
    if applyx:
        gatelist.extend([gates.X(3), gates.SWAP(1, 2)])
    target_state = apply_gates(backend, gatelist, 4)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_fsim(backend):
    theta, phi = 0.1234, 0.4321
    gatelist = [gates.H(i) for i in range(6)]
    gatelist.append(gates.fSim(5, 3, theta, phi).controlled_by(0, 2, 1))
    final_state = apply_gates(backend, gatelist, 6)

    target_state = np.ones(len(final_state), dtype=complex) / np.sqrt(2**6)
    rotation = np.array(
        [[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]]
    )
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    ids = [56, 57, 60, 61]
    target_state[ids] = np.matmul(matrix, target_state[ids])
    ids = [58, 59, 62, 63]
    target_state[ids] = np.matmul(matrix, target_state[ids])
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_unitary(backend):
    matrix = random_unitary(2**1, backend=backend)
    gatelist = [gates.H(0), gates.H(1), gates.Unitary(matrix, 1).controlled_by(0)]
    final_state = apply_gates(backend, gatelist, 2)
    target_state = np.ones(len(final_state), dtype=complex) / 2.0
    target_state[2:] = np.matmul(backend.to_numpy(matrix), target_state[2:])
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

    matrix = random_unitary(2**2, backend=backend)
    gatelist = [gates.H(i) for i in range(4)]
    gatelist.append(gates.Unitary(matrix, 1, 3).controlled_by(0, 2))
    final_state = apply_gates(backend, gatelist, 4)
    target_state = np.ones(len(final_state), dtype=complex) / 4.0
    ids = [10, 11, 14, 15]
    target_state[ids] = np.matmul(backend.to_numpy(matrix), target_state[ids])
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


###############################################################################

################################# Test dagger #################################
GATES = [
    ("H", (0,)),
    ("X", (0,)),
    ("Y", (0,)),
    ("Z", (0,)),
    ("SX", (0,)),
    ("SXDG", (0,)),
    ("S", (0,)),
    ("SDG", (0,)),
    ("T", (0,)),
    ("TDG", (0,)),
    ("RX", (0, 0.1)),
    ("RY", (0, 0.2)),
    ("RZ", (0, 0.3)),
    ("PRX", (0, 0.1, 0.2)),
    ("GPI", (0, 0.1)),
    ("GPI2", (0, 0.2)),
    ("U1", (0, 0.1)),
    ("U2", (0, 0.2, 0.3)),
    ("U3", (0, 0.1, 0.2, 0.3)),
    ("U1q", (0, 0.1, 0.2)),
    ("CNOT", (0, 1)),
    ("CZ", (0, 1)),
    ("CSX", (0, 1)),
    ("CSXDG", (0, 1)),
    ("CRX", (0, 1, 0.1)),
    ("CRZ", (0, 1, 0.3)),
    ("CU1", (0, 1, 0.1)),
    ("CU2", (0, 1, 0.2, 0.3)),
    ("CU3", (0, 1, 0.1, 0.2, 0.3)),
    ("fSim", (0, 1, 0.1, 0.2)),
    ("SYC", (0, 1)),
    ("RXX", (0, 1, 0.1)),
    ("RYY", (0, 1, 0.2)),
    ("RZZ", (0, 1, 0.3)),
    ("RZX", (0, 1, 0.4)),
    ("RXXYY", (0, 1, 0.5)),
    ("MS", (0, 1, 0.1, 0.2, 0.3)),
    ("GIVENS", (0, 1, 0.1)),
    ("RBS", (0, 1, 0.2)),
    ("ECR", (0, 1)),
    ("SiSWAP", (0, 1)),
    ("SiSWAPDG", (0, 1)),
]


@pytest.mark.parametrize("gate,args", GATES)
def test_dagger(backend, gate, args):
    gate = getattr(gates, gate)(*args)
    nqubits = len(gate.qubits)
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gate, gate.dagger()], nqubits, initial_state)
    backend.assert_allclose(final_state, initial_state, atol=1e-6)


GATES = [
    ("H", (3,)),
    ("X", (3,)),
    ("Y", (3,)),
    ("S", (3,)),
    ("SDG", (3,)),
    ("T", (3,)),
    ("TDG", (3,)),
    ("RX", (3, 0.1)),
    ("U1", (3, 0.1)),
    ("U3", (3, 0.1, 0.2, 0.3)),
]


@pytest.mark.parametrize("gate,args", GATES)
def test_controlled_dagger(backend, gate, args):
    gate = getattr(gates, gate)(*args).controlled_by(0, 1, 2)
    nqubits = 4
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gate, gate.dagger()], 4, initial_state)
    backend.assert_allclose(final_state, initial_state, atol=1e-6)


@pytest.mark.parametrize("gate_1,gate_2", [("S", "SDG"), ("T", "TDG")])
@pytest.mark.parametrize("qubit", (0, 2, 4))
def test_dagger_consistency(backend, gate_1, gate_2, qubit):
    gate_1 = getattr(gates, gate_1)(qubit)
    gate_2 = getattr(gates, gate_2)(qubit)
    initial_state = random_statevector(2 ** (qubit + 1), backend=backend)
    final_state = apply_gates(backend, [gate_1, gate_2], qubit + 1, initial_state)
    backend.assert_allclose(final_state, initial_state, atol=1e-6)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_dagger(backend, nqubits):
    matrix = np.random.random((2**nqubits, 2**nqubits))
    matrix = backend.cast(matrix)
    gate = gates.Unitary(matrix, *range(nqubits))
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gate, gate.dagger()], nqubits, initial_state)
    target_state = backend.np.matmul(matrix, initial_state)
    target_state = backend.np.matmul(backend.np.conj(matrix).T, target_state)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_controlled_unitary_dagger(backend):
    from scipy.linalg import expm

    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    matrix = backend.cast(matrix)
    gate = gates.Unitary(matrix, 0).controlled_by(1, 2, 3, 4)
    nqubits = 5
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gate, gate.dagger()], 5, initial_state)
    backend.assert_allclose(final_state, initial_state, atol=1e-6)


def test_generalizedfsim_dagger(backend):
    from scipy.linalg import expm

    phi = 0.2
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    nqubits = 2
    initial_state = random_statevector(2**nqubits, backend=backend)
    final_state = apply_gates(backend, [gate, gate.dagger()], 2, initial_state)
    backend.assert_allclose(final_state, initial_state, atol=1e-6)


###############################################################################

############################# Test basis rotation #############################


def test_gate_basis_rotation(backend):
    gate = gates.X(0).basis_rotation()
    assert isinstance(gate, gates.H)
    gate = gates.Y(0).basis_rotation()
    assert isinstance(gate, gates.Unitary)
    target_matrix = np.array([[1, -1j], [1j, -1]]) / np.sqrt(2)
    backend.assert_allclose(gate.matrix(backend), target_matrix)
    with pytest.raises(NotImplementedError):
        gates.RX(0, np.pi / 2).basis_rotation()


###############################################################################

########################### Test gate decomposition ###########################


@pytest.mark.parametrize(
    ("target", "controls", "free"),
    [
        (0, (1,), ()),
        (2, (0, 1), ()),
        (3, (0, 1, 4), (2, 5)),
        (7, (0, 1, 2, 3, 4), (5, 6)),
        (5, (0, 2, 4, 6, 7), (1, 3)),
        (8, (0, 2, 4, 6, 9), (3, 5, 7)),
    ],
)
@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_execution(backend, target, controls, free, use_toffolis):
    """Check that applying the decomposition is equivalent to applying the multi-control gate."""
    gate = gates.X(target).controlled_by(*controls)
    nqubits = max((target,) + controls + free) + 1
    initial_state = random_statevector(2**nqubits, backend=backend)
    target_state = backend.apply_gate(gate, backend.np.copy(initial_state), nqubits)
    dgates = gate.decompose(*free, use_toffolis=use_toffolis)
    final_state = backend.np.copy(initial_state)
    for gate in dgates:
        final_state = backend.apply_gate(gate, final_state, nqubits)
    backend.assert_allclose(final_state, target_state, atol=1e-6)


###############################################################################

####################### Test Clifford updates #################################


@pytest.mark.parametrize(
    "gate",
    [
        gates.RX(0, 0),
        gates.RY(0, 0),
        gates.RZ(0, 0),
        gates.CRX(0, 1, 0),
        gates.CRY(0, 1, 0),
        gates.CRZ(0, 1, 0),
    ],
)
def test_clifford_condition_update(backend, gate):
    """Test clifford condition update if setting new angle into the rotations."""
    assert gate.clifford
    gate.parameters = 0.5
    assert not gate.clifford
    gate.parameters = np.pi
    assert gate.clifford


###############################################################################
