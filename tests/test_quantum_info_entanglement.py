import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.config import PRECISION_TOL
from qibo.quantum_info.entanglement import (
    concurrence,
    entanglement_fidelity,
    entanglement_of_formation,
    entangling_capability,
    meyer_wallach_entanglement,
    negativity,
)
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector


@pytest.mark.parametrize("check_purity", [True, False])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_concurrence_and_formation(backend, bipartition, base, check_purity):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = concurrence(
            state, bipartition=bipartition, check_purity=check_purity, backend=backend
        )
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = concurrence(
            state, bipartition=bipartition, check_purity="True", backend=backend
        )

    if check_purity is True:
        with pytest.raises(NotImplementedError):
            state = backend.identity_density_matrix(2, normalize=False)
            test = concurrence(state, bipartition=bipartition, backend=backend)

    nqubits = 2
    dim = 2**nqubits
    state = random_statevector(dim, backend=backend)
    concur = concurrence(
        state, bipartition=bipartition, check_purity=check_purity, backend=backend
    )
    ent_form = entanglement_of_formation(
        state,
        bipartition=bipartition,
        base=base,
        check_purity=check_purity,
        backend=backend,
    )
    backend.assert_allclose(0.0 <= concur <= np.sqrt(2), True)
    backend.assert_allclose(0.0 <= ent_form <= 1.0, True)

    state = backend.np.kron(
        random_density_matrix(2, pure=True, backend=backend),
        random_density_matrix(2, pure=True, backend=backend),
    )
    concur = concurrence(state, bipartition, check_purity=check_purity, backend=backend)
    ent_form = entanglement_of_formation(
        state,
        bipartition=bipartition,
        base=base,
        check_purity=check_purity,
        backend=backend,
    )
    backend.assert_allclose(concur, 0.0, atol=10 * PRECISION_TOL)
    backend.assert_allclose(ent_form, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("p", [1 / 5, 1 / 3 + 0.01, 1.0])
def test_negativity(backend, p):
    # werner state
    zero, one = np.array([1, 0]), np.array([0, 1])
    psi = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)
    psi = np.outer(psi, psi.T)
    psi = backend.cast(psi)
    state = p * psi + (1 - p) * backend.identity_density_matrix(2, normalize=True)

    neg = negativity(state, [0], backend=backend)

    if p == 1 / 5:
        target = 0.0
    elif p == 1.0:
        target = 1 / 2
    else:
        target = 3 / 400

    backend.assert_allclose(neg, target, atol=1e-10)


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("nqubits", [4, 6])
@pytest.mark.parametrize("channel", [gates.DepolarizingChannel])
def test_entanglement_fidelity(backend, channel, nqubits, check_hermitian):
    with pytest.raises(TypeError):
        test = entanglement_fidelity(
            channel, nqubits=[0], check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        test = entanglement_fidelity(
            channel, nqubits=0, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3, 2)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_fidelity(
            channel,
            nqubits,
            state=state,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    with pytest.raises(TypeError):
        state = random_statevector(2, backend=backend)
        test = entanglement_fidelity(
            channel, nqubits, state=state, check_hermitian="False", backend=backend
        )

    channel = channel([0, 1], 0.5)

    # test on maximally entangled state
    ent_fid = entanglement_fidelity(
        channel, nqubits=nqubits, check_hermitian=check_hermitian, backend=backend
    )
    backend.assert_allclose(ent_fid, 0.625, atol=PRECISION_TOL)

    # test with a state vector
    state = backend.plus_state(nqubits)
    ent_fid = entanglement_fidelity(
        channel,
        nqubits=nqubits,
        state=state,
        check_hermitian=check_hermitian,
        backend=backend,
    )
    backend.assert_allclose(ent_fid, 0.625, atol=PRECISION_TOL)

    # test on maximally mixed state
    state = backend.identity_density_matrix(nqubits)
    ent_fid = entanglement_fidelity(
        channel,
        nqubits=nqubits,
        state=state,
        check_hermitian=check_hermitian,
        backend=backend,
    )
    backend.assert_allclose(ent_fid, 1.0, atol=PRECISION_TOL)


def test_meyer_wallach_entanglement(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3, 2).astype(complex)
        state = backend.cast(state, dtype=state.dtype)
        test = meyer_wallach_entanglement(state, backend=backend)

    nqubits = 2

    circuit1 = Circuit(nqubits)
    circuit1.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))
    state1 = backend.execute_circuit(circuit1).state()

    circuit2 = Circuit(nqubits)
    circuit2.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))
    circuit2.add(gates.CNOT(0, 1))
    state2 = backend.execute_circuit(circuit2).state()

    backend.assert_allclose(
        meyer_wallach_entanglement(state1, backend=backend), 0.0, atol=1e-6, rtol=1e-6
    )

    backend.assert_allclose(
        meyer_wallach_entanglement(state2, backend=backend), 1.0, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_entangling_capability(backend, seed):
    with pytest.raises(TypeError):
        circuit = Circuit(1)
        samples = 0.5
        entangling_capability(circuit, samples, seed=seed, backend=backend)
    with pytest.raises(TypeError):
        circuit = Circuit(1)
        entangling_capability(circuit, samples=10, seed="10", backend=backend)

    nqubits = 2
    samples = 100

    c1 = Circuit(nqubits)
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    c1.add(gates.CNOT(0, 1))
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    ent_mw1 = entangling_capability(c1, samples, seed=seed, backend=backend)

    c2 = Circuit(nqubits)
    c2.add(gates.H(0))
    c2.add(gates.CNOT(0, 1))
    c2.add(gates.RX(0, 0, trainable=True))
    ent_mw2 = entangling_capability(c2, samples, seed=seed, backend=backend)

    c3 = Circuit(nqubits)
    ent_mw3 = entangling_capability(c3, samples, seed=seed, backend=backend)

    backend.assert_allclose(ent_mw3 < ent_mw1 < ent_mw2, True)
