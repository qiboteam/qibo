"""Test channels defined in `qibo/gates.py`."""
import pytest
import numpy as np
from qibo import gates
from qibo.tests.utils import random_density_matrix


def test_general_channel(backend):
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 0, 1], [0, 0, 1, 0]])
    initial_rho = random_density_matrix(2)
    channel = gates.KrausChannel([((1,), a1), ((0, 1), a2)])
    assert channel.target_qubits == (0, 1)
    final_rho = backend.apply_channel_density_matrix(channel, np.copy(initial_rho), 2)
    m1 = np.kron(np.eye(2), backend.to_numpy(a1))
    m2 = backend.to_numpy(a2)
    target_rho = (m1.dot(initial_rho).dot(m1.conj().T) +
                  m2.dot(initial_rho).dot(m2.conj().T))
    backend.assert_allclose(final_rho, target_rho)


def test_krauss_channel_errors():
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        gate = gates.KrausChannel([((0, 1), a1)])


def test_controlled_by_channel_error():
    with pytest.raises(ValueError):
        gates.PauliNoiseChannel(0, px=0.5).controlled_by(1)

    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                                  [0, 0, 1, 0]])
    config = [((1,), a1), ((0, 1), a2)]
    with pytest.raises(ValueError):
        gates.KrausChannel(config).controlled_by(1)


def test_unitary_channel(backend):
    a1 = np.array([[0, 1], [1, 0]])
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probs = [0.4, 0.3]
    matrices = [((0,), a1), ((2, 3), a2)]
    initial_state = random_density_matrix(4)
    channel = gates.UnitaryChannel(probs, matrices)
    final_state = backend.apply_channel_density_matrix(channel, np.copy(initial_state), 4)

    eye = np.eye(2)
    ma1 = np.kron(np.kron(a1, eye), np.kron(eye, eye))
    ma2 = np.kron(np.kron(eye, eye), a2)
    target_state = (0.3 * initial_state
                    + 0.4 * ma1.dot(initial_state.dot(ma1))
                    + 0.3 * ma2.dot(initial_state.dot(ma2)))
    backend.assert_allclose(final_state, target_state)


@pytest.mark.skip
@pytest.mark.parametrize("precision", ["double", "single"])
def test_unitary_channel_probability_tolerance(backend, precision):
    """Create ``UnitaryChannel`` with probability sum within tolerance (see #562)."""
    import qibo
    original_precision = qibo.get_precision()
    qibo.set_precision(precision)
    nqubits = 2
    param = 0.006
    num_terms = 2 ** (2 * nqubits)
    max_param = num_terms / (num_terms - 1)
    prob_identity = 1 - param / max_param
    prob_pauli = param / num_terms
    probs = [prob_identity] + [prob_pauli] * (num_terms - 1)
    if precision == "double":
        probs = np.array(probs, dtype="float64")
    else:
        probs = np.array(probs, dtype="float32")
    matrices = len(probs) * [((0, 1), np.random.random((4, 4)))]
    gate = gates.UnitaryChannel(probs, matrices)
    qibo.set_precision(original_precision)


@pytest.mark.skip
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
    # Probability sum < 0
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([0.0, 0.0], matrices)


def test_pauli_noise_channel(backend):
    initial_rho = random_density_matrix(2)
    channel = gates.PauliNoiseChannel(1, px=0.3)
    final_rho = backend.apply_channel_density_matrix(channel, np.copy(initial_rho), 2)
    gate = gates.X(1)
    target_rho = backend.apply_gate_density_matrix(gate, np.copy(initial_rho), 2)
    target_rho = 0.3 * backend.to_numpy(target_rho)
    target_rho += 0.7 * initial_rho
    backend.assert_allclose(final_rho, target_rho)


@pytest.mark.skip
def test_reset_channel(backend):
    initial_rho = random_density_matrix(3)
    gate = gates.ResetChannel(0, p0=0.2, p1=0.2)
    gate.density_matrix = True
    final_rho = gate(np.copy(initial_rho))

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
    backend.assert_allclose(final_rho, target_rho)


@pytest.mark.skip
@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)])
def test_thermal_relaxation_channel(backend, t1, t2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    initial_rho = random_density_matrix(3)
    gate = gates.ThermalRelaxationChannel(0, t1, t2, time=time,
        excited_population=excpop)
    gate.density_matrix = True
    final_rho = gate(np.copy(initial_rho)) # pylint: disable=E1102

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
    backend.assert_allclose(final_rho, target_rho)
    # Try to apply to state vector if t1 < t2
    if t1 < t2:
        with pytest.raises(ValueError):
            gate._state_vector_call(initial_rho) # pylint: disable=no-member


@pytest.mark.skip
@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(1.0, 0.5, 1.5, 1.5), (1.0, 0.5, -0.5, 0.5),
                          (1.0, -0.5, 1.5, 0.5), (-1.0, 0.5, 1.5, 0.5),
                          (1.0, 3.0, 1.5, 0.5)])
def test_thermal_relaxation_channel_errors(backend, t1, t2, time, excpop):
    with pytest.raises(ValueError):
        gate = gates.ThermalRelaxationChannel(
            0, t1, t2, time, excited_population=excpop)

