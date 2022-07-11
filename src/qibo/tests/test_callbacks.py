"""Test methods defined in `qibo/core/callbacks.py`."""
import pytest
import numpy as np
from qibo import gates, callbacks
from qibo.models import Circuit, AdiabaticEvolution


# Absolute testing tolerance for the cases of zero entanglement entropy
_atol = 1e-8


def test_abstract_callback_properties():
    callback = callbacks.Callback()
    callback.nqubits = 5
    callback.append(1)
    callback.extend([2, 3])
    assert callback.nqubits == 5
    assert callback.results == [1, 2, 3]


def test_creating_callbacks():
    callback = callbacks.EntanglementEntropy()
    callback = callbacks.EntanglementEntropy([1, 2], compute_spectrum=True)
    callback = callbacks.Norm()
    callback = callbacks.Overlap(0)
    callback = callbacks.Energy("test")
    callback = callbacks.Gap()
    callback = callbacks.Gap(2)
    with pytest.raises(ValueError):
        callback = callbacks.Gap("test")
    with pytest.raises(TypeError):
        callback = callbacks.Gap(1.0)


def test_getitem_bad_indexing(backend):
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    final_state = backend.execute_circuit(c)
    entropy[0]
    with pytest.raises(IndexError):
        entropy[1]
    with pytest.raises(IndexError):
        entropy["a"]


def test_entropy_product_state(backend):
    """Check that the |++> state has zero entropy."""
    entropy = callbacks.EntanglementEntropy()
    entropy.nqubits = 2
    state = np.ones(4) / 2.0
    result = entropy.apply(backend, state)
    backend.assert_allclose(result, 0, atol=_atol)


def test_entropy_singlet_state(backend):
    """Check that the singlet state has maximum entropy."""
    entropy = callbacks.EntanglementEntropy([0])
    entropy.nqubits = 2
    state = np.zeros(4)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)
    result = entropy.apply(backend, state)
    backend.assert_allclose(result, 1.0)


def test_entropy_random_state(backend):
    """Check that entropy calculation agrees with numpy."""
    from qibo.config import EIGVAL_CUTOFF
    # Generate a random positive and hermitian density matrix
    rho = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    rho = rho + rho.conj().T
    _, u = np.linalg.eigh(rho)
    s = 5 * np.random.random(8)
    s = s / s.sum()
    rho = u.dot(np.diag(s)).dot(u.conj().T)

    result, spectrum = backend.entanglement_entropy(rho)
    target = - (s * np.log2(s)).sum()
    backend.assert_allclose(result, target)

    target_eigvals = np.linalg.eigvalsh(rho)
    masked_eigvals = target_eigvals[np.where(target_eigvals > EIGVAL_CUTOFF)]
    target_spectrum = - np.log(masked_eigvals)
    backend.assert_allclose(spectrum, target_spectrum)


def test_entropy_switch_partition(backend):
    """Check that partition is switched to the largest counterpart."""
    entropy = callbacks.EntanglementEntropy([0])
    entropy.nqubits = 5
    # Prepare ghz state of 5 qubits
    state = np.zeros(2 ** 5)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)
    result = entropy.apply(backend, state)
    backend.assert_allclose(result, 1.0)


def test_entropy_numerical(backend):
    """Check that entropy calculation does not fail for tiny eigenvalues."""
    eigvals = np.array([-1e-10, -1e-15, -2e-17, -1e-18, -5e-60, 1e-48, 4e-32,
                        5e-14, 1e-14, 9.9e-13, 9e-13, 5e-13, 1e-13, 1e-12,
                        1e-11, 1e-10, 1e-9, 1e-7, 1, 4, 10])
    rho = np.diag(eigvals)
    result, _ = backend.entanglement_entropy(rho)
    mask = eigvals > 0
    target = - (eigvals[mask] * np.log2(eigvals[mask])).sum()
    backend.assert_allclose(result, target)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_entropy_in_circuit(backend, density_matrix):
    """Check that entropy calculation works in circuit."""
    entropy = callbacks.EntanglementEntropy([0], compute_spectrum=True)
    c = Circuit(2, density_matrix=density_matrix)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = backend.execute_circuit(c)

    target = [0, 0, 1.0]
    values = [backend.to_numpy(x) for x in entropy[:]]
    backend.assert_allclose(values, target, atol=_atol)

    target_spectrum = [0, 0, np.log(2), np.log(2)]
    entropy_spectrum = np.concatenate(entropy.spectrum).ravel().tolist()
    backend.assert_allclose(entropy_spectrum, target_spectrum, atol=_atol)


@pytest.mark.parametrize("gateconf,target_entropy",
                         [(["H", "CNOT", "entropy"], [1.0]),
                          (["H", "entropy", "CNOT"], [0.0]),
                          (["entropy", "H", "CNOT"], [0.0]),
                          (["entropy", "H", "CNOT", "entropy"], [0.0, 1.0]),
                          (["H", "entropy", "CNOT", "entropy"], [0.0, 1.0]),
                          (["entropy", "H", "entropy", "CNOT"], [0.0, 0.0])])
def test_entropy_in_distributed_circuit(backend, accelerators, gateconf, target_entropy):
    """Check that various entropy configurations work in distributed circuit."""
    target_c = Circuit(4)
    target_c.add([gates.H(0), gates.CNOT(0, 1)])
    target_state = backend.execute_circuit(target_c)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(4, accelerators)
    for gate in gateconf:
        if gate == "H":
            c.add(gates.H(0))
        elif gate == "CNOT":
            c.add(gates.CNOT(0, 1))
        elif gate == "entropy":
            c.add(gates.CallbackGate(entropy))
    final_state = backend.execute_circuit(c)
    backend.assert_allclose(final_state, target_state)
    values = [backend.to_numpy(x) for x in entropy[:]]
    backend.assert_allclose(values, target_entropy, atol=_atol)


def test_entropy_multiple_executions(backend, accelerators):
    """Check entropy calculation when the callback is used in multiple executions."""
    target_c = Circuit(4)
    target_c.add([gates.RY(0, 0.1234), gates.CNOT(0, 1)])
    target_state = backend.execute_circuit(target_c)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(4, accelerators)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = backend.execute_circuit(c)
    backend.assert_allclose(state, target_state)

    target_c = Circuit(4)
    target_c.add([gates.RY(0, 0.4321), gates.CNOT(0, 1)])
    target_state = backend.execute_circuit(target_c)

    c = Circuit(4, accelerators)
    c.add(gates.RY(0, 0.4321))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = backend.execute_circuit(c)
    backend.assert_allclose(state, target_state)

    def target_entropy(t):
        cos = np.cos(t / 2.0) ** 2
        sin = np.sin(t / 2.0) ** 2
        return - cos * np.log2(cos) - sin * np.log2(sin)

    target = [0, target_entropy(0.1234), 0, target_entropy(0.4321)]
    values = [backend.to_numpy(x) for x in entropy[:]]
    backend.assert_allclose(values, target, atol=_atol)

    c = Circuit(8, accelerators)
    with pytest.raises(RuntimeError):
        c.add(gates.CallbackGate(entropy))
        state = backend.execute_circuit(c)


def test_entropy_large_circuit(backend, accelerators):
    """Check that entropy calculation works for variational like circuit."""
    thetas = np.pi * np.random.random((3, 8))
    target_entropy = callbacks.EntanglementEntropy([0, 2, 4, 5])
    target_entropy.nqubits = 8
    c1 = Circuit(8)
    c1.add((gates.RY(i, thetas[0, i]) for i in range(8)))
    c1.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    state1 = backend.execute_circuit(c1).state()
    e1 = target_entropy.apply(backend, state1)

    c2 = Circuit(8)
    c2.add((gates.RY(i, thetas[1, i]) for i in range(8)))
    c2.add((gates.CZ(i, i + 1) for i in range(1, 7, 2)))
    c2.add(gates.CZ(0, 7))
    state2 = backend.execute_circuit(c1 + c2).state()
    e2 = target_entropy.apply(backend, state2)

    c3 = Circuit(8)
    c3.add((gates.RY(i, thetas[2, i]) for i in range(8)))
    c3.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    state3 = backend.execute_circuit(c1 + c2 + c3).state()
    e3 = target_entropy.apply(backend, state3)

    entropy = callbacks.EntanglementEntropy([0, 2, 4, 5])
    c = Circuit(8, accelerators)
    c.add(gates.CallbackGate(entropy))
    c.add((gates.RY(i, thetas[0, i]) for i in range(8)))
    c.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    c.add(gates.CallbackGate(entropy))
    c.add((gates.RY(i, thetas[1, i]) for i in range(8)))
    c.add((gates.CZ(i, i + 1) for i in range(1, 7, 2)))
    c.add(gates.CZ(0, 7))
    c.add(gates.CallbackGate(entropy))
    c.add((gates.RY(i, thetas[2, i]) for i in range(8)))
    c.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    c.add(gates.CallbackGate(entropy))
    state = backend.execute_circuit(c)

    backend.assert_allclose(state3, state)
    values = [backend.to_numpy(x) for x in entropy[:]]
    targets = [backend.to_numpy(x) for x in [0, e1, e2, e3]]
    backend.assert_allclose(values, targets)


def test_entropy_density_matrix(backend):
    from qibo.tests.utils import random_density_matrix
    rho = random_density_matrix(4)
    # this rho is not always positive. Make rho positive for this application
    _, u = np.linalg.eigh(rho)
    rho = u.dot(np.diag(5 * np.random.random(u.shape[0]))).dot(u.conj().T)
    # this is a positive rho

    entropy = callbacks.EntanglementEntropy([1, 3])
    entropy.nqubits = 4
    final_ent = entropy.apply_density_matrix(backend, rho)

    rho = rho.reshape(8 * (2,))
    reduced_rho = np.einsum("abcdafch->bdfh", rho).reshape((4, 4))
    eigvals = np.linalg.eigvalsh(reduced_rho).real
    # assert that all eigenvalues are non-negative
    assert (eigvals >= 0).prod()
    mask = eigvals > 0
    target_ent = - (eigvals[mask] * np.log2(eigvals[mask])).sum()
    backend.assert_allclose(final_ent, target_ent)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_state_callback(backend, density_matrix, copy):
    statec = callbacks.State(copy=copy)
    c = Circuit(2, density_matrix=density_matrix)
    c.add(gates.H(0))
    c.add(gates.CallbackGate(statec))
    c.add(gates.H(1))
    c.add(gates.CallbackGate(statec))
    final_state = backend.execute_circuit(c)

    target_state0 = np.array([1, 0, 1, 0]) / np.sqrt(2)
    target_state1 = np.ones(4) / 2.0
    if not copy and backend.name == "qibojit":
        # when copy is disabled in the callback and in-place updates are used
        target_state0 = target_state1
    if density_matrix:
        target_state0 = np.tensordot(target_state0, target_state0, axes=0)
        target_state1 = np.tensordot(target_state1, target_state1, axes=0)
    backend.assert_allclose(statec[0], target_state0)
    backend.assert_allclose(statec[1], target_state1)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_norm(backend, density_matrix):
    norm = callbacks.Norm()
    if density_matrix:
        norm.nqubits = 1
        state = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
        target_norm = np.trace(state)
        final_norm = norm.apply_density_matrix(backend, state)
    else:
        norm.nqubits = 2
        state = np.random.random(4) + 1j * np.random.random(4)
        target_norm = np.sqrt((np.abs(state) ** 2).sum())
        final_norm = norm.apply(backend, state)

    backend.assert_allclose(final_norm, target_norm)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_overlap(backend, density_matrix):
    state0 = np.random.random(4) + 1j * np.random.random(4)
    state1 = np.random.random(4) + 1j * np.random.random(4)
    overlap = callbacks.Overlap(state0)
    overlap.nqubits = 2
    if density_matrix:
        with pytest.raises(NotImplementedError):
            overlap.apply_density_matrix(backend, state1)
    else:
        final_overlap = overlap.apply(backend, state1)
        target_overlap = np.abs((state0.conj() * state1).sum())
        backend.assert_allclose(final_overlap, target_overlap)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_energy(backend, density_matrix):
    from qibo import hamiltonians
    ham = hamiltonians.TFIM(4, h=1.0, backend=backend)
    energy = callbacks.Energy(ham)
    matrix = backend.to_numpy(ham.matrix)
    if density_matrix:
        from qibo.tests.utils import random_density_matrix
        state = random_density_matrix(4)
        target_energy = np.trace(matrix.dot(state))
        final_energy = energy.apply_density_matrix(backend, state)
    else:
        from qibo.tests.utils import random_state
        state = random_state(4)
        target_energy = state.conj().dot(matrix.dot(state))
        final_energy = energy.apply(backend, state)
    backend.assert_allclose(final_energy, target_energy)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("check_degenerate", [False, True])
def test_gap(backend, dense, check_degenerate):
    from qibo import hamiltonians
    h0 = hamiltonians.X(4, dense=dense, backend=backend)
    if check_degenerate:
        # use h=0 to make this Hamiltonian degenerate
        h1 = hamiltonians.TFIM(4, h=0, dense=dense, backend=backend)
    else:
        h1 = hamiltonians.TFIM(4, h=1, dense=dense, backend=backend)

    ham = lambda t: (1 - t) * h0.matrix + t * h1.matrix
    targets = {"ground": [], "excited": [], "gap": []}
    for t in np.linspace(0, 1, 11):
        eigvals = np.real(np.linalg.eigvalsh(ham(t)))
        targets["ground"].append(eigvals[0])
        targets["excited"].append(eigvals[1])
        targets["gap"].append(eigvals[1] - eigvals[0])
    if check_degenerate:
        targets["gap"][-1] = eigvals[3] - eigvals[0]

    gap = callbacks.Gap(check_degenerate=check_degenerate)
    ground = callbacks.Gap(0)
    excited = callbacks.Gap(1)
    evolution = AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-1,
                                   callbacks=[gap, ground, excited])
    final_state = evolution(final_time=1.0)
    targets = {k: np.stack(v) for k, v in targets.items()}

    values = {
        "ground": np.array([backend.to_numpy(x) for x in ground]),
        "excited": np.array([backend.to_numpy(x) for x in excited]),
        "gap": np.array([backend.to_numpy(x) for x in gap])
    }
    for k, v in values.items():
        backend.assert_allclose(v, targets.get(k))


def test_gap_errors():
    """Check errors in gap callback instantiation."""
    # invalid string ``mode``
    with pytest.raises(ValueError):
        gap = callbacks.Gap("test")
    # invalid ``mode`` type
    with pytest.raises(TypeError):
        gap = callbacks.Gap([])

    gap = callbacks.Gap()
    # call before setting evolution model
    with pytest.raises(RuntimeError):
        gap.apply(None, np.ones(4))
    # not implemented for density matrices
    with pytest.raises(NotImplementedError):
        gap.apply_density_matrix(None, np.zeros(8))
