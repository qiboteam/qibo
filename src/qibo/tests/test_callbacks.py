"""
Testing tensorflow callbacks.
"""
import pytest
import numpy as np
from qibo.models import Circuit, AdiabaticEvolution
from qibo import gates, callbacks

# Absolute testing tolerance for the cases of zero entanglement entropy
_atol = 1e-8


def test_entropy_product_state():
    """Check that the |++> state has zero entropy."""
    entropy = callbacks.EntanglementEntropy()
    state = np.ones(4) / 2.0

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 0, atol=_atol)


def test_entropy_singlet_state():
    """Check that the singlet state has maximum entropy."""
    entropy = callbacks.EntanglementEntropy([0])
    state = np.zeros(4)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)
    # Pass the state as `tf.Tensor` to test this functionality as well
    import tensorflow as tf
    from qibo.config import DTYPES
    state = tf.convert_to_tensor(state, dtype=DTYPES.get('DTYPECPX'))

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 1.0)


def test_entropy_random_state():
    """Check that entropy calculation agrees with numpy."""
    # Generate a random positive and hermitian density matrix
    rho = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    rho = rho + rho.conj().T
    _, u = np.linalg.eigh(rho)
    s = 5 * np.random.random(8)
    s = s / s.sum()
    rho = u.dot(np.diag(s)).dot(u.conj().T)

    result = callbacks.EntanglementEntropy._entropy(rho).numpy()
    target = - (s * np.log2(s)).sum()
    np.testing.assert_allclose(result, target)


def test_entropy_switch_partition():
    """Check that partition is switched to the largest counterpart."""
    entropy = callbacks.EntanglementEntropy([0])
    # Prepare ghz state of 5 qubits
    state = np.zeros(2 ** 5)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 1.0)


def test_state_invalid_type():
    """Check that ``TypeError`` is raised for bad state type."""
    entropy = callbacks.EntanglementEntropy([0])
    # Prepare ghz state of 5 qubits
    with pytest.raises(TypeError):
        result = entropy([0, 1, 0, 0])


def test_entropy_numerical():
    """Check that entropy calculation does not fail for tiny eigenvalues."""
    import tensorflow as tf
    from qibo.config import DTYPES
    eigvals = np.array([-1e-10, -1e-15, -2e-17, -1e-18, -5e-60, 1e-48, 4e-32,
                        5e-14, 1e-14, 9.9e-13, 9e-13, 5e-13, 1e-13, 1e-12,
                        1e-11, 1e-10, 1e-9, 1e-7, 1, 4, 10])
    rho = tf.convert_to_tensor(np.diag(eigvals), dtype=DTYPES.get('DTYPECPX'))
    result = callbacks.EntanglementEntropy._entropy(rho).numpy()

    mask = eigvals > 0
    target = - (eigvals[mask] * np.log2(eigvals[mask])).sum()

    np.testing.assert_allclose(result, target)


@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2}])
def test_entropy_in_circuit(accelerators):
    """Check that entropy calculation works in circuit."""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()

    target = [0, 0, 1.0]
    np.testing.assert_allclose(entropy[:].numpy(), target, atol=_atol)


def test_entropy_in_distributed_circuit():
    """Check that various entropy configurations work in distributed circuit."""
    target_c = Circuit(2)
    target_c.add([gates.H(0), gates.CNOT(0, 1)])
    target_state = target_c().numpy()
    accelerators = {"/GPU:0": 1, "/GPU:1": 1}

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.CNOT(0, 1), gates.CallbackGate(entropy)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [1.0], atol=_atol)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.CallbackGate(entropy), gates.CNOT(0, 1)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [0.0], atol=_atol)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.CallbackGate(entropy), gates.H(0), gates.CNOT(0, 1)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [0.0], atol=_atol)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.CallbackGate(entropy), gates.H(0),
           gates.CNOT(0, 1), gates.CallbackGate(entropy)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [0, 1.0], atol=_atol)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.CallbackGate(entropy),
           gates.CNOT(0, 1), gates.CallbackGate(entropy)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [0, 1.0], atol=_atol)

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2, accelerators)
    c.add([gates.CallbackGate(entropy), gates.H(0),
           gates.CallbackGate(entropy), gates.CNOT(0, 1)])
    final_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)
    np.testing.assert_allclose(entropy[:].numpy(), [0, 0], atol=_atol)


def test_entropy_in_compiled_circuit():
    """Check that entropy calculation works when circuit is compiled."""
    import qibo
    original_backend = qibo.get_backend()
    qibo.set_backend("matmuleinsum")
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    c.compile()
    state = c()
    qibo.set_backend("custom")

    target = [0, 0, 1.0]
    np.testing.assert_allclose(entropy[:].numpy(), target, atol=_atol)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2}])
def test_entropy_multiple_executions(accelerators):
    """Check entropy calculation when the callback is used in multiple executions."""
    entropy = callbacks.EntanglementEntropy([0])

    target_c = Circuit(2)
    target_c.add([gates.RY(0, 0.1234), gates.CNOT(0, 1)])
    target_state = target_c().numpy()

    c = Circuit(2, accelerators)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()
    np.testing.assert_allclose(state.numpy(), target_state)

    target_c = Circuit(2)
    target_c.add([gates.RY(0, 0.4321), gates.CNOT(0, 1)])
    target_state = target_c().numpy()

    c = Circuit(2, accelerators)
    c.add(gates.RY(0, 0.4321))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()
    np.testing.assert_allclose(state.numpy(), target_state)

    def target_entropy(t):
        cos = np.cos(t / 2.0) ** 2
        sin = np.sin(t / 2.0) ** 2
        return - cos * np.log2(cos) - sin * np.log2(sin)

    target = [0, target_entropy(0.1234), 0, target_entropy(0.4321)]
    np.testing.assert_allclose(entropy[:].numpy(), target, atol=_atol)


@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2},
                                          {"/GPU:0": 2, "/GPU:1": 2},
                                          {"/GPU:0": 3, "/GPU:1": 1,
                                           "/GPU:2": 4}])
def test_entropy_large_circuit(accelerators):
    """Check that entropy calculation works for variational like circuit."""
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

    np.testing.assert_allclose(state3.numpy(), state.numpy())
    np.testing.assert_allclose(entropy[:].numpy(), [0, e1, e2, e3])


def test_entropy_bad_indexing():
    """Check exceptions in ``Callback.__getitem__``."""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    state = c()

    entropy[0]
    with pytest.raises(IndexError):
        entropy[1]
    with pytest.raises(IndexError):
        entropy["a"]


def test_norm():
    """Check norm callback for state vectors and density matrices."""
    norm = callbacks.Norm()

    state = np.random.random(4) + 1j * np.random.random(4)
    target_norm = np.sqrt((np.abs(state) ** 2).sum())
    np.testing.assert_allclose(norm(state), target_norm)

    state = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    target_norm = np.trace(state)
    np.testing.assert_allclose(norm(state, True), target_norm)


def test_overlap():
    state0 = np.random.random(4) + 1j * np.random.random(4)
    overlap = callbacks.Overlap(state0)

    state1 = np.random.random(4) + 1j * np.random.random(4)
    target_overlap = np.abs((state0.conj() * state1).sum())
    np.testing.assert_allclose(overlap(state1), target_overlap)

    with pytest.raises(NotImplementedError):
        overlap(state1, is_density_matrix=True)


def test_energy():
    """Check energy callback for state vectors and density matrices."""
    from qibo import hamiltonians
    ham = hamiltonians.TFIM(4, h=1.0)
    energy = callbacks.Energy(ham)
    matrix = np.array(ham.matrix)

    state = np.random.random(16) + 1j * np.random.random(16)
    target_energy = state.conj().dot(matrix.dot(state))
    np.testing.assert_allclose(energy(state), target_energy)

    state = np.random.random((16, 16)) + 1j * np.random.random((16, 16))
    target_energy = np.trace(matrix.dot(state))
    np.testing.assert_allclose(energy(state, True), target_energy)


@pytest.mark.parametrize("trotter", [False, True])
def test_gap(trotter):
    """Check gap callback for adiabatic evolution model."""
    from qibo import hamiltonians
    h0 = hamiltonians.X(3, trotter=trotter)
    h1 = hamiltonians.TFIM(3, h=1.0, trotter=trotter)

    ham = lambda t: ((1 - t) * h0.matrix + t * h1.matrix).numpy()
    targets = {"ground": [], "excited": [], "gap": []}
    for t in np.linspace(0, 1, 11):
        eigvals = np.linalg.eigvalsh(ham(t)).real
        targets["ground"].append(eigvals[0])
        targets["excited"].append(eigvals[1])
        targets["gap"].append(eigvals[1] - eigvals[0])

    gap = callbacks.Gap()
    ground = callbacks.Gap(0)
    excited = callbacks.Gap(1)
    evolution = AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-1,
                                   callbacks=[gap, ground, excited])
    final_state = evolution(final_time=1.0)

    np.testing.assert_allclose(ground[:], targets["ground"])
    np.testing.assert_allclose(excited[:], targets["excited"])
    np.testing.assert_allclose(gap[:], targets["gap"])
    # check not implemented for density matrices
    with pytest.raises(NotImplementedError):
        gap(np.zeros(8), is_density_matrix=True)


def test_gap_errors():
    """Check errors in gap callback instantiation."""
    # invalid string ``mode``
    with pytest.raises(ValueError):
        gap = callbacks.Gap("test")
    # invalid ``mode`` type
    with pytest.raises(TypeError):
        gap = callbacks.Gap([])
    # invalid evolution model type
    with pytest.raises(TypeError):
        gap = callbacks.Gap()
        gap.evolution = "test"
    # call before setting evolution model
    with pytest.raises(ValueError):
        gap = callbacks.Gap()
        gap(np.ones(4))
