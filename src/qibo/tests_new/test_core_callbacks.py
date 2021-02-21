"""Test methods defined in `qibo/core/callbacks.py`."""
import pytest
import numpy as np
import qibo
from qibo.models import Circuit, AdiabaticEvolution
from qibo import gates, callbacks
from qibo.config import EIGVAL_CUTOFF


# Absolute testing tolerance for the cases of zero entanglement entropy
_atol = 1e-8


def test_getitem_bad_indexing(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    final_state = c()
    entropy[0]
    with pytest.raises(IndexError):
        entropy[1]
    with pytest.raises(IndexError):
        entropy["a"]
    qibo.set_backend(original_backend)


def test_entropy_product_state(backend):
    """Check that the |++> state has zero entropy."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy()
    state = np.ones(4) / 2.0

    result = entropy(state)
    np.testing.assert_allclose(result, 0, atol=_atol)
    qibo.set_backend(original_backend)


def test_entropy_singlet_state(backend):
    """Check that the singlet state has maximum entropy."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import K
    entropy = callbacks.EntanglementEntropy([0])
    state = np.zeros(4)
    state[0], state[-1] = 1, 1
    state = K.cast(state / np.sqrt(2))
    result = entropy(state)
    np.testing.assert_allclose(result, 1.0)
    qibo.set_backend(original_backend)


def test_entropy_bad_state_type(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy([0])
    with pytest.raises(TypeError):
        _ = entropy("test")
    qibo.set_backend(original_backend)


def test_entropy_random_state(backend):
    """Check that entropy calculation agrees with numpy."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    # Generate a random positive and hermitian density matrix
    rho = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    rho = rho + rho.conj().T
    _, u = np.linalg.eigh(rho)
    s = 5 * np.random.random(8)
    s = s / s.sum()
    rho = u.dot(np.diag(s)).dot(u.conj().T)

    callback = callbacks.EntanglementEntropy(compute_spectrum=True)
    result = callback.entropy(rho)
    target = - (s * np.log2(s)).sum()
    np.testing.assert_allclose(result, target)

    ref_eigvals = np.linalg.eigvalsh(rho)
    masked_eigvals = ref_eigvals[np.where(ref_eigvals > EIGVAL_CUTOFF)]
    ref_spectrum = - np.log(masked_eigvals)
    np.testing.assert_allclose(callback.spectrum[0], ref_spectrum)
    qibo.set_backend(original_backend)


def test_entropy_switch_partition(backend):
    """Check that partition is switched to the largest counterpart."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy([0])
    # Prepare ghz state of 5 qubits
    state = np.zeros(2 ** 5)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)

    result = entropy(state)
    np.testing.assert_allclose(result, 1.0)
    qibo.set_backend(original_backend)


def test_entropy_numerical(backend):
    """Check that entropy calculation does not fail for tiny eigenvalues."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import K
    eigvals = np.array([-1e-10, -1e-15, -2e-17, -1e-18, -5e-60, 1e-48, 4e-32,
                        5e-14, 1e-14, 9.9e-13, 9e-13, 5e-13, 1e-13, 1e-12,
                        1e-11, 1e-10, 1e-9, 1e-7, 1, 4, 10])
    rho = K.cast(np.diag(eigvals))
    callback = callbacks.EntanglementEntropy()
    result = callback.entropy(rho)

    mask = eigvals > 0
    target = - (eigvals[mask] * np.log2(eigvals[mask])).sum()

    np.testing.assert_allclose(result, target)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_entropy_in_circuit(backend, density_matrix):
    """Check that entropy calculation works in circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy([0], compute_spectrum=True)
    c = Circuit(2, density_matrix=density_matrix)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()

    target = [0, 0, 1.0]
    np.testing.assert_allclose(entropy[:], target, atol=_atol)

    target_spectrum = [0, 0, np.log(2), np.log(2)]
    entropy_spectrum = np.concatenate(entropy.spectrum).ravel().tolist()
    np.testing.assert_allclose(entropy_spectrum, target_spectrum, atol=_atol)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("gateconf,target_entropy",
                         [(["H", "CNOT", "entropy"], [1.0]),
                          (["H", "entropy", "CNOT"], [0.0]),
                          (["entropy", "H", "CNOT"], [0.0]),
                          (["entropy", "H", "CNOT", "entropy"], [0.0, 1.0]),
                          (["H", "entropy", "CNOT", "entropy"], [0.0, 1.0]),
                          (["entropy", "H", "entropy", "CNOT"], [0.0, 0.0])])
def test_entropy_in_distributed_circuit(backend, accelerators, gateconf, target_entropy):
    """Check that various entropy configurations work in distributed circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_c = Circuit(4)
    target_c.add([gates.H(0), gates.CNOT(0, 1)])
    target_state = target_c()

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(4, accelerators)
    for gate in gateconf:
        if gate == "H":
            c.add(gates.H(0))
        elif gate == "CNOT":
            c.add(gates.CNOT(0, 1))
        elif gate == "entropy":
            c.add(gates.CallbackGate(entropy))
    final_state = c()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:], target_entropy, atol=_atol)
    qibo.set_backend(original_backend)


def test_entropy_in_compiled_circuit(backend):
    """Check that entropy calculation works when circuit is compiled."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    if backend == "custom":
        with pytest.raises(RuntimeError):
            c.compile()
    else:
        c.compile()
        final_state = c()
        np.testing.assert_allclose(entropy[:], [0, 0, 1.0], atol=_atol)
    qibo.set_backend(original_backend)


def test_entropy_multiple_executions(backend, accelerators):
    """Check entropy calculation when the callback is used in multiple executions."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_c = Circuit(4)
    target_c.add([gates.RY(0, 0.1234), gates.CNOT(0, 1)])
    target_state = target_c()

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(4, accelerators)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()
    np.testing.assert_allclose(state, target_state)

    target_c = Circuit(4)
    target_c.add([gates.RY(0, 0.4321), gates.CNOT(0, 1)])
    target_state = target_c()

    c = Circuit(4, accelerators)
    c.add(gates.RY(0, 0.4321))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()
    np.testing.assert_allclose(state, target_state)

    def target_entropy(t):
        cos = np.cos(t / 2.0) ** 2
        sin = np.sin(t / 2.0) ** 2
        return - cos * np.log2(cos) - sin * np.log2(sin)

    target = [0, target_entropy(0.1234), 0, target_entropy(0.4321)]
    np.testing.assert_allclose(entropy[:], target, atol=_atol)
    qibo.set_backend(original_backend)


def test_entropy_large_circuit(backend, accelerators):
    """Check that entropy calculation works for variational like circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.pi * np.random.random((3, 8))
    target_entropy = callbacks.EntanglementEntropy([0, 2, 4, 5])
    c1 = Circuit(8)
    c1.add((gates.RY(i, thetas[0, i]) for i in range(8)))
    c1.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    state1 = c1()
    e1 = target_entropy(state1)

    c2 = Circuit(8)
    c2.add((gates.RY(i, thetas[1, i]) for i in range(8)))
    c2.add((gates.CZ(i, i + 1) for i in range(1, 7, 2)))
    c2.add(gates.CZ(0, 7))
    state2 = (c1 + c2)()
    e2 = target_entropy(state2)

    c3 = Circuit(8)
    c3.add((gates.RY(i, thetas[2, i]) for i in range(8)))
    c3.add((gates.CZ(i, i + 1) for i in range(0, 7, 2)))
    state3 = (c1 + c2 + c3)()
    e3 = target_entropy(state3)

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
    state = c()

    np.testing.assert_allclose(state3, state)
    np.testing.assert_allclose(entropy[:], [0, e1, e2, e3])
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_norm(backend, density_matrix):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    norm = callbacks.Norm()

    if density_matrix:
        norm.density_matrix = True
        state = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
        target_norm = np.trace(state)
    else:
        state = np.random.random(4) + 1j * np.random.random(4)
        target_norm = np.sqrt((np.abs(state) ** 2).sum())

    np.testing.assert_allclose(norm(state), target_norm)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_overlap(backend, density_matrix):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    state0 = np.random.random(4) + 1j * np.random.random(4)
    state1 = np.random.random(4) + 1j * np.random.random(4)
    overlap = callbacks.Overlap(state0)
    if density_matrix:
        overlap.density_matrix = True
        with pytest.raises(NotImplementedError):
            overlap(state1)
    else:
        target_overlap = np.abs((state0.conj() * state1).sum())
        np.testing.assert_allclose(overlap(state1), target_overlap)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_energy(backend, density_matrix):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import hamiltonians
    ham = hamiltonians.TFIM(4, h=1.0)
    energy = callbacks.Energy(ham)
    matrix = np.array(ham.matrix)
    if density_matrix:
        energy.density_matrix = True
        state = np.random.random((16, 16)) + 1j * np.random.random((16, 16))
        target_energy = np.trace(matrix.dot(state))
    else:
        state = np.random.random(16) + 1j * np.random.random(16)
        target_energy = state.conj().dot(matrix.dot(state))
    np.testing.assert_allclose(energy(state), target_energy)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("trotter", [False, True])
@pytest.mark.parametrize("check_degenerate", [False, True])
def test_gap(backend, trotter, check_degenerate):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import hamiltonians
    h0 = hamiltonians.X(4, trotter=trotter)
    if check_degenerate:
        # use h=0 to make this Hamiltonian degenerate
        h1 = hamiltonians.TFIM(4, h=0, trotter=trotter)
    else:
        h1 = hamiltonians.TFIM(4, h=1, trotter=trotter)

    ham = lambda t: (1 - t) * h0.matrix + t * h1.matrix
    targets = {"ground": [], "excited": [], "gap": []}
    for t in np.linspace(0, 1, 11):
        eigvals = np.linalg.eigvalsh(ham(t)).real
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
    np.testing.assert_allclose(ground[:], targets["ground"])
    np.testing.assert_allclose(excited[:], targets["excited"])
    np.testing.assert_allclose(gap[:], targets["gap"])
    qibo.set_backend(original_backend)


def test_gap_errors():
    """Check errors in gap callback instantiation."""
    # invalid string ``mode``
    with pytest.raises(ValueError):
        gap = callbacks.Gap("test")
    # invalid ``mode`` type
    with pytest.raises(TypeError):
        gap = callbacks.Gap([])

    gap = callbacks.Gap()
    # invalid evolution model type
    with pytest.raises(TypeError):
        gap.evolution = "test"
    # call before setting evolution model
    with pytest.raises(ValueError):
        gap(np.ones(4))
    # not implemented for density matrices
    gap.density_matrix = True
    with pytest.raises(NotImplementedError):
        gap(np.zeros(8))
    # for coverage
    _ = gap.density_matrix
    gap.density_matrix = False
