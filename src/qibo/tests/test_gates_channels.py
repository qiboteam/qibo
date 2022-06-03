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


def test_unitary_channel_probability_tolerance(backend):
    """Create ``UnitaryChannel`` with probability sum within tolerance (see #562)."""
    nqubits = 2
    param = 0.006
    num_terms = 2 ** (2 * nqubits)
    max_param = num_terms / (num_terms - 1)
    prob_identity = 1 - param / max_param
    prob_pauli = param / num_terms
    probs = [prob_identity] + [prob_pauli] * (num_terms - 1)
    probs = np.array(probs, dtype="float64")
    matrices = len(probs) * [((0, 1), np.random.random((4, 4)))]
    gate = gates.UnitaryChannel(probs, matrices)


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


def test_reset_channel(backend):
    initial_rho = random_density_matrix(3)
    gate = gates.ResetChannel(0, p0=0.2, p1=0.2)
    final_rho = backend.reset_error_density_matrix(gate, np.copy(initial_rho), 3)

    trace = backend.to_numpy(backend.partial_trace_density_matrix(initial_rho, (0,), 3))
    trace = np.reshape(trace, 4 * (2,))
    zeros = np.tensordot(trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0)
    ones = np.tensordot(trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0)
    zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
    ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])
    target_rho = 0.6 * initial_rho + 0.2 * np.reshape(zeros + ones, initial_rho.shape)
    backend.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)])
def test_thermal_relaxation_channel(backend, t1, t2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    initial_rho = random_density_matrix(3)
    gate = gates.ThermalRelaxationChannel(0, t1, t2, time=time, excited_population=excpop)
    final_rho = gate.apply_density_matrix(backend, np.copy(initial_rho), 3)

    if t2 > t1:
        p0, p1, exp = gate.coefficients
        matrix = np.diag([1 - p1, p0, p1, 1 - p0])
        matrix[0, -1], matrix[-1, 0] = exp, exp
        matrix = matrix.reshape(4 * (2,))
        # Apply matrix using Eq. (3.28) from arXiv:1111.6950
        target_rho = np.copy(initial_rho).reshape(6 * (2,))
        target_rho = np.einsum("abcd,aJKcjk->bJKdjk", matrix, target_rho)
        target_rho = target_rho.reshape(initial_rho.shape)
    else:
        p0, p1, pz = gate.coefficients
        mz = np.kron(np.array([[1, 0], [0, -1]]), np.eye(4))
        z_rho = mz.dot(initial_rho.dot(mz))

        trace = backend.to_numpy(backend.partial_trace_density_matrix(initial_rho, (0,), 3))
        trace = np.reshape(trace, 4 * (2,))
        zeros = np.tensordot(trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0)
        ones = np.tensordot(trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0)
        zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
        ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])

        pi = 1 - p0 - p1 - pz
        target_rho = pi * initial_rho + pz * z_rho 
        target_rho += np.reshape(p0 * zeros + p1 * ones, initial_rho.shape)

    backend.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(1.0, 0.5, 1.5, 1.5), (1.0, 0.5, -0.5, 0.5),
                          (1.0, -0.5, 1.5, 0.5), (-1.0, 0.5, 1.5, 0.5),
                          (1.0, 3.0, 1.5, 0.5)])
def test_thermal_relaxation_channel_errors(backend, t1, t2, time, excpop):
    with pytest.raises(ValueError):
        gate = gates.ThermalRelaxationChannel(
            0, t1, t2, time, excited_population=excpop)
