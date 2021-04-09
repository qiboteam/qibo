import numpy as np
import pytest
import qibo
from qibo import models, gates, callbacks
from qibo.tests import utils


_atol = 1e-8


def test_circuit_dm(backend):
    """Check passing density matrix as initial state to a circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    initial_rho = utils.random_density_matrix(3)

    c = models.Circuit(3, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    c.add(gates.H(2))
    final_rho = c(np.copy(initial_rho))

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]])
    m1 = np.kron(np.kron(h, h), np.eye(2))
    m2 = np.kron(cnot, np.eye(2))
    m3 = np.kron(np.eye(4), h)
    target_rho = m1.dot(initial_rho).dot(m1.T.conj())
    target_rho = m2.dot(target_rho).dot(m2.T.conj())
    target_rho = m3.dot(target_rho).dot(m3.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_density_matrix_circuit_initial_state(backend):
    """Check that circuit transforms state vector initial state to density matrix."""
    import tensorflow as tf
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_psi = utils.random_numpy_state(3)
    c = models.Circuit(3, density_matrix=True)
    final_rho = c(np.copy(initial_psi))
    target_rho = np.outer(initial_psi, initial_psi.conj())
    np.testing.assert_allclose(final_rho, target_rho)

    initial_psi = tf.cast(initial_psi, dtype=final_rho.dtype)
    final_rho = c(initial_psi)
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_bitflip_noise(backend):
    """Test `gates.PauliNoiseChannel` on random initial density matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = utils.random_density_matrix(2)

    c = models.Circuit(2, density_matrix=True)
    c.add(gates.PauliNoiseChannel(1, px=0.3))
    final_rho = c(np.copy(initial_rho)).numpy()

    c = models.Circuit(2, density_matrix=True)
    c.add(gates.X(1))
    target_rho = 0.3 * c(np.copy(initial_rho)).numpy()
    target_rho += 0.7 * initial_rho.reshape(target_rho.shape)

    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_multiple_noise(backend):
    """Test `gates.NoiseChnanel` with multiple noise probabilities."""
    from qibo import matrices
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.PauliNoiseChannel(0, px=0.5, pz=0.3))
    c.add(gates.PauliNoiseChannel(1, py=0.1, pz=0.3))
    final_rho = c().numpy()

    psi = np.ones(4) / 2
    rho = np.outer(psi, psi.conj())
    m1 = np.kron(matrices.X, matrices.I)
    m2 = np.kron(matrices.Z, matrices.I)
    rho = 0.2 * rho + 0.5 * m1.dot(rho.dot(m1)) + 0.3 * m2.dot(rho.dot(m2))
    m1 = np.kron(matrices.I, matrices.Y)
    m2 = np.kron(matrices.I, matrices.Z)
    rho = 0.6 * rho + 0.1 * m1.dot(rho.dot(m1)) + 0.3 * m2.dot(rho.dot(m2))
    np.testing.assert_allclose(final_rho, rho)
    qibo.set_backend(original_backend)


def test_circuit_reexecution(backend):
    """Test re-executing a circuit with `gates.NoiseChnanel`."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.PauliNoiseChannel(0, px=0.5))
    c.add(gates.PauliNoiseChannel(1, pz=0.3))
    final_rho = c().numpy()
    final_rho2 = c().numpy()
    np.testing.assert_allclose(final_rho, final_rho2)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("tfmatrices", [False, True])
@pytest.mark.parametrize("oncircuit", [False, True])
def test_general_channel(backend, tfmatrices, oncircuit):
    """Test `gates.KrausChannel`."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = utils.random_density_matrix(2)

    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 0, 1], [0, 0, 1, 0]])
    if tfmatrices:
        from qibo import K
        a1, a2 = K.cast(a1), K.cast(a2)

    gate = gates.KrausChannel([((1,), a1), ((0, 1), a2)])
    assert gate.target_qubits == (0, 1)
    if oncircuit:
        c = models.Circuit(2, density_matrix=True)
        c.add(gate)
        final_rho = c(np.copy(initial_rho))
    else:
        final_rho = gate(np.copy(initial_rho))

    m1 = np.kron(np.eye(2), np.array(a1))
    m2 = np.array(a2)
    target_rho = (m1.dot(initial_rho).dot(m1.conj().T) +
                  m2.dot(initial_rho).dot(m2.conj().T))
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_controlled_by_channel():
    """Test that attempting to control channels raises error."""
    c = models.Circuit(2, density_matrix=True)
    with pytest.raises(ValueError):
        c.add(gates.PauliNoiseChannel(0, px=0.5).controlled_by(1))

    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                                  [0, 0, 1, 0]])
    config = [((1,), a1), ((0, 1), a2)]
    with pytest.raises(ValueError):
        gates.KrausChannel(config).controlled_by(1)


def test_krauss_channel_errors(backend):
    """Test errors raised by `gates.KrausChannel`."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    # bad Kraus matrix shape
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        gate = gates.KrausChannel([((0, 1), a1)])
    # Using KrausChannel on state vectors
    channel = gates.KrausChannel([((0,), np.eye(2))])
    with pytest.raises(ValueError):
        channel.state_vector_call(np.random.random(4))
    # Attempt to construct unitary for KrausChannel
    with pytest.raises(ValueError):
        channel.construct_unitary()
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("density_matrix", [True, False])
def test_unitary_channel(backend, density_matrix):
    """Test creating `gates.UnitaryChannel` from matrices and errors."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    a1 = np.array([[0, 1], [1, 0]])
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probs = [0.4, 0.3]
    matrices = [((0,), a1), ((2, 3), a2)]
    if density_matrix:
        initial_state = utils.random_density_matrix(4)
    else:
        initial_state = utils.random_numpy_state(4)
    c = models.Circuit(4, density_matrix=density_matrix)
    c.add(gates.UnitaryChannel(probs, matrices, seed=123))
    final_state = c(initial_state, nshots=20).numpy()

    eye = np.eye(2, dtype=final_state.dtype)
    ma1 = np.kron(np.kron(a1, eye), np.kron(eye, eye))
    ma2 = np.kron(np.kron(eye, eye), a2)
    if density_matrix:
        # use density matrices
        target_state = (0.3 * initial_state +
                        0.4 * ma1.dot(initial_state.dot(ma1)) +
                        0.3 * ma2.dot(initial_state.dot(ma2)))
    else:
        # sample unitary channel
        target_state = []
        np.random.seed(123)
        for _ in range(20):
            temp_state = np.copy(initial_state)
            if np.random.random() < 0.4:
                temp_state = ma1.dot(temp_state)
            if np.random.random() < 0.3:
                temp_state = ma2.dot(temp_state)
            target_state.append(np.copy(temp_state))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_unitary_channel_errors():
    """Check errors raised by ``gates.UnitaryChannel``."""
    a1 = np.array([[0, 1], [1, 0]])
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probs = [0.4, 0.3]
    matrices = [((0,), a1), ((2, 3), a2)]
    # Invalid probability length
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([0.1, 0.3, 0.2], matrices)
    # Probability > 1
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([1.1, 0.2], matrices)
    # Probability sum = 0
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([0.0, 0.0], matrices)


def test_circuit_with_noise_gates():
    """Check that ``circuit.with_noise()`` adds the proper noise channels."""
    c = models.Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    assert noisy_c.depth == 4
    assert noisy_c.ngates == 7
    for i in [1, 3, 5, 6]:
        assert noisy_c.queue[i].__class__.__name__ == "PauliNoiseChannel"


def test_circuit_with_noise_execution(backend):
    """Check ``circuit.with_noise()`` execution."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    target_c = models.Circuit(2, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(gates.PauliNoiseChannel(0, 0.1, 0.2, 0.3))
    target_c.add(gates.H(1))
    target_c.add(gates.PauliNoiseChannel(1, 0.1, 0.2, 0.3))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


def test_circuit_with_noise_with_measurements(backend):
    """Check ``circuit.with_noise() when using measurement noise."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    c.add(gates.M(0))
    noisy_c = c.with_noise(3 * (0.1,))

    target_c = models.Circuit(2, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(gates.PauliNoiseChannel(0, 0.1, 0.1, 0.1))
    target_c.add(gates.H(1))
    target_c.add(gates.PauliNoiseChannel(1, 0.1, 0.1, 0.1))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


def test_circuit_with_noise_noise_map(backend):
    """Check ``circuit.with_noise() when giving noise map."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    noise_map = {0: (0.1, 0.2, 0.1), 1: (0.2, 0.3, 0.0),
                 2: (0.0, 0.0, 0.0)}

    c = models.Circuit(3, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.X(2)])
    c.add(gates.M(2))
    noisy_c = c.with_noise(noise_map)

    target_c = models.Circuit(3, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(gates.PauliNoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.H(1))
    target_c.add(gates.PauliNoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.X(2))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


def test_circuit_with_noise_noise_map_exceptions():
    """Check that proper exceptions are raised when noise map is invalid."""
    c = models.Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3))
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1), 1: (0.3, 0.1)})
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1)})
    with pytest.raises(TypeError):
        noisy_c = c.with_noise({0, 1})


def test_circuit_with_noise_exception():
    """Check that calling ``with_noise`` in a noisy circuit raises error."""
    c = models.Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.PauliNoiseChannel(0, px=0.2)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3, 0.0))


def test_density_matrix_measurement(backend):
    """Check measurement gate on density matrices."""
    from qibo.tests_new.test_measurement_gate import assert_result
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = np.zeros(4)
    state[2] = 1
    rho = np.outer(state, state.conj())
    mgate = gates.M(0, 1)
    mgate.density_matrix = True
    result = mgate(rho, nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(result,
                  decimal_samples=2 * np.ones((100,)),
                  binary_samples=target_binary_samples,
                  decimal_frequencies={2: 100},
                  binary_frequencies={"10": 100})
    qibo.set_backend(original_backend)


def test_density_matrix_circuit_measurement(backend):
    """Check measurement gate on density matrices using circuit."""
    from qibo.tests_new.test_measurement_gate import assert_result
    from qibo.tests_new.test_measurement_gate_registers import assert_register_result
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = np.zeros(16)
    state[0] = 1
    init_rho = np.outer(state, state.conj())

    c = models.Circuit(4, density_matrix=True)
    c.add(gates.X(1))
    c.add(gates.X(3))
    c.add(gates.M(0, 1, register_name="A"))
    c.add(gates.M(3, 2, register_name="B"))
    result = c(init_rho, nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    assert_result(result,
                  decimal_samples=6 * np.ones((100,)),
                  binary_samples=target_binary_samples,
                  decimal_frequencies={6: 100},
                  binary_frequencies={"0110": 100})

    target = {}
    target["decimal_samples"] = {"A": np.ones((100,)),
                                 "B": 2 * np.ones((100,))}
    target["binary_samples"] = {"A": np.zeros((100, 2)),
                                "B": np.zeros((100, 2))}
    target["binary_samples"]["A"][:, 1] = 1
    target["binary_samples"]["B"][:, 0] = 1
    target["decimal_frequencies"] = {"A": {1: 100}, "B": {2: 100}}
    target["binary_frequencies"] = {"A": {"01": 100}, "B": {"10": 100}}
    assert_register_result(result, **target)
    qibo.set_backend(original_backend)


def test_reset_channel(backend):
    """Check ``gates.ResetChannel`` on a 3-qubit random density matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = utils.random_density_matrix(3)
    c = models.Circuit(3, density_matrix=True)
    c.add(gates.ResetChannel(0, p0=0.2, p1=0.2))
    final_rho = c(np.copy(initial_rho))

    dtype = initial_rho.dtype
    collapsed_rho = np.copy(initial_rho).reshape(6 * (2,))
    collapsed_rho[0, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho[1, :, :, 0, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho[1, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho = collapsed_rho.reshape((8, 8))
    collapsed_rho /= np.trace(collapsed_rho)
    mx = np.kron(np.array([[0, 1], [1, 0]]), np.eye(4))
    flipped_rho = mx.dot(collapsed_rho.dot(mx))
    target_rho = 0.6 * initial_rho + 0.2 * (collapsed_rho + flipped_rho)
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)])
def test_thermal_relaxation_channel(backend, t1, t2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = utils.random_density_matrix(3)
    c = models.Circuit(3, density_matrix=True)
    gate = gates.ThermalRelaxationChannel(0, t1, t2, time=time,
        excited_population=excpop)
    c.add(gate)
    final_rho = c(np.copy(initial_rho))

    exp, p0, p1 = gate.calculate_probabilities(t1, t2, time, excpop)
    if t2 > t1:
        matrix = np.diag([1 - p1, p0, p1, 1 - p0])
        matrix[0, -1], matrix[-1, 0] = exp, exp
        matrix = matrix.reshape(4 * (2,))
        # Apply matrix using Eq. (3.28) from arXiv:1111.6950
        target_rho = np.copy(initial_rho).reshape(6 * (2,))
        target_rho = np.einsum("abcd,aJKcjk->bJKdjk", matrix, target_rho)
        target_rho = target_rho.reshape(initial_rho.shape)
    else:
        pz = exp
        pi = 1 - pz - p0 - p1
        dtype = initial_rho.dtype
        collapsed_rho = np.copy(initial_rho).reshape(6 * (2,))
        collapsed_rho[0, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho[1, :, :, 0, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho[1, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho = collapsed_rho.reshape((8, 8))
        collapsed_rho /= np.trace(collapsed_rho)
        mx = np.kron(np.array([[0, 1], [1, 0]]), np.eye(4))
        mz = np.kron(np.array([[1, 0], [0, -1]]), np.eye(4))
        z_rho = mz.dot(initial_rho.dot(mz))
        flipped_rho = mx.dot(collapsed_rho.dot(mx))
        target_rho = (pi * initial_rho + pz * z_rho + p0 * collapsed_rho +
                      p1 * flipped_rho)
    np.testing.assert_allclose(final_rho, target_rho)
    # Try to apply to state vector if t1 < t2
    if t1 < t2:
        with pytest.raises(ValueError):
            gate.state_vector_call(initial_rho) # pylint: disable=no-member
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(1.0, 0.5, 1.5, 1.5), (1.0, 0.5, -0.5, 0.5),
                          (1.0, -0.5, 1.5, 0.5), (-1.0, 0.5, 1.5, 0.5),
                          (1.0, 3.0, 1.5, 0.5)])
def test_thermal_relaxation_channel_errors(backend, t1, t2, time, excpop):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    with pytest.raises(ValueError):
        gate = gates.ThermalRelaxationChannel(
            0, t1, t2, time, excited_population=excpop)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(nqubits)
    c = models.Circuit(nqubits, density_matrix=True)
    c.add((gates.RY(i, t) for i, t in enumerate(theta)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_state = c()
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    c = models.Circuit(nqubits, density_matrix=True)
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ, theta))
    final_state = c()
    np.testing.assert_allclose(target_state, final_state)
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ, theta)
    gate.density_matrix = True
    final_state = gate(c.get_initial_state())
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


def test_entanglement_entropy(backend):
    """Check that entanglement entropy calculation works for density matrices."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    rho = utils.random_density_matrix(4)
    # this rho is not always positive. Make rho positive for this application
    _, u = np.linalg.eigh(rho)
    rho = u.dot(np.diag(5 * np.random.random(u.shape[0]))).dot(u.conj().T)
    # this is a positive rho

    entropy = callbacks.EntanglementEntropy([1, 3])
    entropy.density_matrix = True
    final_ent = entropy(rho)

    rho = rho.reshape(8 * (2,))
    reduced_rho = np.einsum("abcdafch->bdfh", rho).reshape((4, 4))
    eigvals = np.linalg.eigvalsh(reduced_rho).real
    # assert that all eigenvalues are non-negative
    assert (eigvals >= 0).prod()
    mask = eigvals > 0
    target_ent = - (eigvals[mask] * np.log2(eigvals[mask])).sum()
    np.testing.assert_allclose(final_ent, target_ent)
    qibo.set_backend(original_backend)


def test_density_matrix_circuit_errors():
    """Check errors of circuits that simulate density matrices."""
    # Attempt to distribute density matrix circuit
    with pytest.raises(NotImplementedError):
        c = models.Circuit(5, accelerators={"/GPU:0": 2}, density_matrix=True)
    # Attempt to add Kraus channel to non-density matrix circuit
    c = models.Circuit(5)
    with pytest.raises(ValueError):
        c.add(gates.KrausChannel([((0,), np.eye(2))]))
