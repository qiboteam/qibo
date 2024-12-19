"""Test channels defined in `qibo/gates.py`."""

import numpy as np
import pytest

from qibo import gates, matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import (
    partial_trace,
    random_density_matrix,
    random_statevector,
    random_stochastic_matrix,
)


def test_general_channel(backend):
    """"""
    a_1 = backend.cast(np.sqrt(0.4) * matrices.X)
    a_2 = backend.cast(
        np.sqrt(0.6)
        * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    )
    initial_state = random_density_matrix(2**2, backend=backend)
    m_1 = backend.np.kron(backend.identity_density_matrix(1, normalize=False), a_1)
    m_1 = backend.cast(m_1, dtype=m_1.dtype)
    m_2 = backend.cast(a_2, dtype=a_2.dtype)
    target_state = backend.np.matmul(
        backend.np.matmul(m_1, initial_state),
        backend.np.transpose(backend.np.conj(m_1), (1, 0)),
    )
    target_state = target_state + backend.np.matmul(
        backend.np.matmul(m_2, initial_state),
        backend.np.transpose(backend.np.conj(m_2), (1, 0)),
    )

    channel1 = gates.KrausChannel([(1,), (0, 1)], [a_1, a_2])
    assert channel1.target_qubits == (0, 1)
    final_state = backend.apply_channel_density_matrix(
        channel1, backend.np.copy(initial_state), 2
    )
    backend.assert_allclose(final_state, target_state)

    a_1 = gates.Unitary(a_1, 1)
    a_2 = gates.Unitary(a_2, 0, 1)
    channel2 = gates.KrausChannel([(1,), (0, 1)], [a_1, a_2])
    assert channel2.target_qubits == (0, 1)
    final_state = backend.apply_channel_density_matrix(
        channel2, backend.np.copy(initial_state), 2
    )
    backend.assert_allclose(final_state, target_state)

    with pytest.raises(NotImplementedError):
        channel1.on_qubits({})
    with pytest.raises(NotImplementedError):
        state = random_statevector(2**2, backend=backend)
        channel1.apply(backend, state, 2)
    with pytest.raises(NotImplementedError):
        channel1.matrix(backend)


def test_controlled_by_channel_error():
    """"""
    with pytest.raises(ValueError):
        gates.PauliNoiseChannel(0, [("X", 0.5)]).controlled_by(1)

    a_1 = np.sqrt(0.4) * matrices.X
    a_2 = np.sqrt(0.6) * np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    config = ([(1,), (0, 1)], [a_1, a_2])
    with pytest.raises(ValueError):
        gates.KrausChannel(*config).controlled_by(1)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
def test_kraus_channel(backend, pauli_order):
    """"""
    a_1 = np.sqrt(0.4) * matrices.X
    a_2 = np.sqrt(0.6) * matrices.Z

    with pytest.raises(TypeError):
        gates.KrausChannel("0", [a_1])
    with pytest.raises(TypeError):
        gates.KrausChannel([0, 1], [a_1, a_2])
    with pytest.raises(ValueError):
        gates.KrausChannel((0, 1), [a_1])

    test_superop = np.array(
        [
            [0.6 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.4 + 0.0j],
            [0.0 + 0.0j, -0.6 + 0.0j, 0.4 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.4 + 0.0j, -0.6 + 0.0j, 0.0 + 0.0j],
            [0.4 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.6 + 0.0j],
        ]
    )
    test_choi = backend.cast(
        np.reshape(test_superop, [2] * 4).swapaxes(0, 3).reshape([4, 4])
    )

    pauli_elements = {"I": 2.0, "X": -0.4, "Y": -2.0, "Z": 0.4}
    test_pauli = backend.cast(np.diag([pauli_elements[p] for p in pauli_order]))

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)
    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    channel = gates.KrausChannel(0, [a_1, a_2])

    backend.assert_allclose(
        float(
            backend.calculate_matrix_norm(
                channel.to_liouville(backend=backend) - test_superop, order=2
            )
        )
        < PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        float(
            backend.calculate_matrix_norm(
                channel.to_choi(backend=backend) - test_choi, order=2
            )
        )
        < PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        float(
            backend.calculate_vector_norm(
                channel.to_pauli_liouville(pauli_order=pauli_order, backend=backend)
                - test_pauli
            )
        )
        < PRECISION_TOL,
        True,
    )

    gates.DepolarizingChannel((0, 1), 0.98).to_choi()


def test_unitary_channel(backend):
    """"""
    a_1 = backend.cast(matrices.X)
    a_2 = backend.cast(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    )

    qubits = [(0,), (2, 3)]
    probabilities = [0.4, 0.3]
    matrices_ = list(zip(probabilities, [a_1, a_2]))

    channel = gates.UnitaryChannel(qubits, matrices_)
    initial_state = random_density_matrix(2**4, backend=backend)
    final_state = backend.apply_channel_density_matrix(
        channel, backend.np.copy(initial_state), 4
    )

    eye = backend.identity_density_matrix(1, normalize=False)
    ma_1 = backend.np.kron(backend.np.kron(a_1, eye), backend.np.kron(eye, eye))
    ma_2 = backend.np.kron(backend.np.kron(eye, eye), a_2)
    ma_1 = backend.cast(ma_1, dtype=ma_1.dtype)
    ma_2 = backend.cast(ma_2, dtype=ma_2.dtype)
    target_state = (
        0.3 * initial_state
        + 0.4 * backend.np.matmul(ma_1, backend.np.matmul(initial_state, ma_1))
        + 0.3 * backend.np.matmul(ma_2, backend.np.matmul(initial_state, ma_2))
    )
    backend.assert_allclose(final_state, target_state)


def test_unitary_channel_probability_tolerance(backend):
    """Create ``UnitaryChannel`` with probability sum within tolerance (see #562)."""
    nqubits = 2
    param = 0.006
    num_terms = 2 ** (2 * nqubits)
    max_param = num_terms / (num_terms - 1)
    prob_identity = 1 - param / max_param
    prob_pauli = param / num_terms
    qubits = (0, 1)
    probs = [prob_identity] + [prob_pauli] * (num_terms - 1)
    probs = np.array(probs, dtype="float64")
    matrices_ = [(p, np.random.random((4, 4))) for p in probs]
    gates.UnitaryChannel(qubits, matrices_)

    probs = np.zeros_like(probs)
    matrices_ = [(p, np.random.random((4, 4))) for p in probs]
    identity_channel = gates.UnitaryChannel(qubits, matrices_)
    backend.assert_allclose(
        identity_channel.to_liouville(backend=backend), backend.np.eye(num_terms)
    )


def test_unitary_channel_errors():
    """Check errors raised by ``gates.UnitaryChannel``."""
    a_1 = matrices.X
    a_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    qubits = [(0,), (2, 3)]
    # Invalid ops
    with pytest.raises(TypeError):
        gates.UnitaryChannel(qubits, [a_1, (0.1, a_2)])
    # Invalid qubit length
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits + [(0,)], [(0.4, a_1), (0.3, a_2)])
    # Probability > 1
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits, [(0.4, a_1), (1.1, a_2)])
    # Probability sum > 1
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits, [(0.5, a_1), (0.6, a_2)])


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
def test_pauli_noise_channel(backend, pauli_order):
    """"""
    initial_state = random_density_matrix(2**2, backend=backend)
    qubits = (1,)
    channel = gates.PauliNoiseChannel(qubits, [("X", 0.3)])
    final_state = backend.apply_channel_density_matrix(
        channel, backend.np.copy(initial_state), 2
    )
    gate = gates.X(1)
    target_state = backend.apply_gate_density_matrix(
        gate, backend.np.copy(initial_state), 2
    )
    target_state = 0.3 * target_state + 0.7 * initial_state
    backend.assert_allclose(final_state, target_state)

    basis = ["X", "Y", "Z"]
    pnp = np.array([0.1, 0.02, 0.05])
    noise_elements = {
        "I": 1,
        "X": 1 - 2 * pnp[1] - 2 * pnp[2],
        "Y": 1 - 2 * pnp[0] - 2 * pnp[2],
        "Z": 1 - 2 * pnp[0] - 2 * pnp[1],
    }
    test_representation = np.diag([noise_elements[p] for p in pauli_order])

    liouville = gates.PauliNoiseChannel(0, list(zip(basis, pnp))).to_pauli_liouville(
        normalize=True, pauli_order=pauli_order, backend=backend
    )
    norm = float(
        backend.calculate_matrix_norm(
            backend.to_numpy(liouville) - test_representation, order=2
        )
    )

    assert norm < PRECISION_TOL


def test_depolarizing_channel_errors():
    """"""
    with pytest.raises(ValueError):
        gates.DepolarizingChannel(0, 1.5)
    with pytest.raises(ValueError):
        gates.DepolarizingChannel([0, 1], 0.1).to_choi(nqubits=1)


def test_depolarizing_channel(backend):
    """"""
    lam = 0.3
    initial_state = random_density_matrix(2**3, backend=backend)
    initial_state_r = partial_trace(initial_state, (2,), backend=backend)
    channel = gates.DepolarizingChannel((0, 1), lam)
    final_state = channel.apply_density_matrix(
        backend, backend.np.copy(initial_state), 3
    )
    final_state_r = partial_trace(final_state, (2,), backend=backend)
    target_state_r = (1 - lam) * initial_state_r + lam * backend.cast(
        np.identity(4)
    ) / 4
    backend.assert_allclose(final_state_r, target_state_r)


def test_amplitude_damping_channel(backend):
    """"""
    with pytest.raises(TypeError):
        gates.AmplitudeDampingChannel(0, "0.1")
    with pytest.raises(ValueError):
        gates.AmplitudeDampingChannel(0, 1.1)

    gamma = np.random.rand()
    kraus_0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    kraus_1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    kraus_0 = backend.cast(kraus_0, dtype=kraus_0.dtype)
    kraus_1 = backend.cast(kraus_1, dtype=kraus_1.dtype)

    channel = gates.AmplitudeDampingChannel(0, gamma)

    initial_state = random_density_matrix(2**1, backend=backend)
    final_state = channel.apply_density_matrix(
        backend, backend.np.copy(initial_state), 1
    )
    target_state = kraus_0 @ initial_state @ backend.np.transpose(
        backend.np.conj(kraus_0), (1, 0)
    ) + kraus_1 @ initial_state @ backend.np.transpose(backend.np.conj(kraus_1), (1, 0))

    backend.assert_allclose(final_state, target_state)


def test_phase_damping_channel(backend):
    """"""
    with pytest.raises(TypeError):
        gates.PhaseDampingChannel(0, "0.1")
    with pytest.raises(ValueError):
        gates.PhaseDampingChannel(0, 1.1)

    gamma = np.random.rand()
    kraus_0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    kraus_1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
    kraus_0 = backend.cast(kraus_0, dtype=kraus_0.dtype)
    kraus_1 = backend.cast(kraus_1, dtype=kraus_1.dtype)

    channel = gates.PhaseDampingChannel(0, gamma)

    initial_state = random_density_matrix(2**1, backend=backend)
    final_state = channel.apply_density_matrix(
        backend, backend.np.copy(initial_state), 1
    )
    target_state = kraus_0 @ initial_state @ backend.np.transpose(
        backend.np.conj(kraus_0), (1, 0)
    ) + kraus_1 @ initial_state @ backend.np.transpose(backend.np.conj(kraus_1), (1, 0))

    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(
    "t_1,t_2,time,excpop", [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)]
)
def test_thermal_relaxation_channel(backend, t_1, t_2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    initial_state = random_density_matrix(2**3, backend=backend)
    gate = gates.ThermalRelaxationChannel(0, [t_1, t_2, time, excpop])
    final_state = gate.apply_density_matrix(backend, backend.np.copy(initial_state), 3)

    if t_2 > t_1:
        p_0, p_1, exp = (
            gate.init_kwargs["p_0"],
            gate.init_kwargs["p_1"],
            gate.init_kwargs["e_t2"],
        )
        matrix = np.diag([1 - p_1, p_1, p_0, 1 - p_0])
        matrix[0, -1], matrix[-1, 0] = exp, exp
        matrix = matrix.reshape(4 * (2,))
        # Apply matrix using Eq. (3.28) from arXiv:1111.6950
        target_state = backend.np.copy(initial_state).reshape(6 * (2,))
        target_state = np.einsum(
            "abcd,aJKcjk->bJKdjk", matrix, backend.to_numpy(target_state)
        )
        target_state = target_state.reshape(initial_state.shape)
    else:
        p_0, p_1, p_z = (
            gate.init_kwargs["p_0"],
            gate.init_kwargs["p_1"],
            gate.init_kwargs["p_z"],
        )
        m_z = backend.np.kron(
            backend.cast(matrices.Z),
            backend.np.kron(backend.cast(matrices.I), backend.cast(matrices.I)),
        )
        m_z = backend.cast(m_z, dtype=m_z.dtype)
        z_rho = m_z @ initial_state @ m_z

        trace = backend.to_numpy(partial_trace(initial_state, (0,), backend=backend))
        trace = np.reshape(trace, 4 * (2,))
        zeros = np.tensordot(
            trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0
        )
        ones = np.tensordot(
            trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0
        )
        zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
        ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])

        zeros = backend.cast(zeros, dtype=zeros.dtype)
        ones = backend.cast(ones, dtype=ones.dtype)

        target_state = (1 - p_0 - p_1 - p_z) * initial_state + p_z * z_rho
        target_state += backend.np.reshape(
            p_0 * zeros + p_1 * ones, initial_state.shape
        )

    target_state = backend.cast(target_state, dtype=target_state.dtype)

    backend.assert_allclose(
        float(backend.calculate_matrix_norm(final_state - target_state, order=2))
        < PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize(
    "params",
    [
        [0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [1.0, 0.5, 1.5, 1.5],
        [1.0, 0.5, -0.5, 0.5],
        [1.0, -0.5, 1.5, 0.5],
        [-1.0, 0.5, 1.5, 0.5],
        [1.0, 3.0, 1.5, 0.5],
    ],
)
def test_thermal_relaxation_channel_errors(params):
    """"""
    with pytest.raises(ValueError):
        gates.ThermalRelaxationChannel(0, params)


def test_readout_error_channel(backend):
    """"""
    with pytest.raises(ValueError):
        gates.ReadoutErrorChannel(0, np.array([[1.1, 0], [0.5, 0.5]]))

    nqubits = 1
    dim = 2**nqubits

    rho = random_density_matrix(dim, seed=1, backend=backend)
    stochastic_noise = random_stochastic_matrix(dim, seed=1, backend=backend)

    probability_sum = gates.ReadoutErrorChannel(
        0, stochastic_noise
    ).apply_density_matrix(backend, rho, 1)
    probability_sum = np.diag(backend.to_numpy(probability_sum)).sum().real

    backend.assert_allclose(probability_sum - 1 < PRECISION_TOL, True)


def test_reset_channel(backend):
    """"""
    initial_state = random_density_matrix(2**3, backend=backend)
    gate = gates.ResetChannel(0, [0.2, 0.2])
    final_state = backend.reset_error_density_matrix(
        gate, backend.np.copy(initial_state), 3
    )

    trace = backend.to_numpy(partial_trace(initial_state, (0,), backend=backend))
    trace = np.reshape(trace, 4 * (2,))

    zeros = np.tensordot(trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0)
    ones = np.tensordot(trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0)
    zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
    ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])
    zeros = backend.cast(zeros, dtype=zeros.dtype)
    ones = backend.cast(ones, dtype=ones.dtype)

    target_state = 0.6 * initial_state + 0.2 * backend.np.reshape(
        zeros + ones, initial_state.shape
    )

    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("p_0,p_1", [(0, -0.1), (-0.1, 0), (0.5, 0.6), (0.8, 0.3)])
def test_reset_channel_errors(p_0, p_1):
    """"""
    with pytest.raises(ValueError):
        gates.ResetChannel(0, [p_0])
    with pytest.raises(ValueError):
        gates.ResetChannel(0, [p_0, p_1])
