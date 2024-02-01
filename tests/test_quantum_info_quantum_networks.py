"""Tests for quantum_info.quantum_networks submodule"""

import numpy as np
import pytest

from qibo import gates
from qibo.quantum_info.quantum_networks import QuantumNetwork
from qibo.quantum_info.random_ensembles import (
    random_density_matrix,
    random_gaussian_matrix,
    random_unitary,
)


def test_errors(backend):
    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)
    nqubits = len(channel.target_qubits)
    dims = 2**nqubits
    partition = (dims, dims)
    network = QuantumNetwork(
        channel.to_choi(backend=backend), partition, backend=backend
    )

    state = random_density_matrix(dims, backend=backend)
    network_state = QuantumNetwork(state, (1, 2), backend=backend)

    matrix = random_density_matrix(2**3, backend=backend)
    net = QuantumNetwork(matrix, (2,) * 3, backend=backend)

    with pytest.raises(TypeError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=True)

    with pytest.raises(ValueError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(1, "2"))

    with pytest.raises(ValueError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(-1, 2))

    with pytest.raises(ValueError):
        QuantumNetwork(
            channel.to_choi(backend=backend), partition=(1, 2), system_output=(1, 2, 3)
        )

    with pytest.raises(TypeError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(1, 2), pure="True")

    with pytest.raises(ValueError):
        network.hermitian(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        network.unital(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        network.causal(precision_tol=-1e-8)

    with pytest.raises(TypeError):
        network + 1

    with pytest.raises(ValueError):
        network + network_state

    with pytest.raises(TypeError):
        network * "1"

    with pytest.raises(TypeError):
        network / "1"

    network_2 = network.copy()
    with pytest.raises(ValueError):
        network_2.system_output = (False,)
        network += network_2

    # Multiplying QuantumNetwork with non-QuantumNetwork
    with pytest.raises(TypeError):
        network @ network.matrix(backend)

    # Linking QuantumNetwork with non-QuantumNetwork
    with pytest.raises(TypeError):
        network.link_product(network.matrix(backend))

    with pytest.raises(TypeError):
        network.link_product(network, subscripts=True)

    with pytest.raises(NotImplementedError):
        network.link_product(network, subscripts="jk,lm->no")

    with pytest.raises(NotImplementedError):
        net @ net

    with pytest.raises(ValueError):
        net @ network

    with pytest.raises(ValueError):
        QuantumNetwork(matrix, (1, 2), backend=backend)


def test_operational_logic(backend):
    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)
    nqubits = len(channel.target_qubits)
    dims = 2**nqubits
    partition = (dims, dims)
    network = QuantumNetwork(
        channel.to_choi(backend=backend), partition, backend=backend
    )

    state = random_density_matrix(dims, backend=backend)
    network_state_pure = QuantumNetwork(state, (2, 2), pure=True, backend=backend)

    # Sum with itself has to match multiplying by int
    backend.assert_allclose(
        (network + network).matrix(backend), (2 * network).matrix(backend)
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).matrix(backend),
        (2 * network_state_pure).to_full(),
    )

    # Sum with itself has to match multiplying by float
    backend.assert_allclose(
        (network + network).matrix(backend), (2.0 * network).matrix(backend)
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).matrix(backend),
        (2.0 * network_state_pure).to_full(),
    )

    # Multiplying and dividing by same scalar has to bring back to original network
    backend.assert_allclose(
        ((2.0 * network) / 2).matrix(backend), network.matrix(backend)
    )

    unitary = random_unitary(dims, backend=backend)
    network_unitary = QuantumNetwork(unitary, (dims, dims), pure=True, backend=backend)
    backend.assert_allclose(
        (network_unitary / 2).matrix(backend), unitary / np.sqrt(2), atol=1e-5
    )


def test_parameters(backend):
    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)

    nqubits = len(channel.target_qubits)
    dims = 2**nqubits
    partition = (dims, dims)

    network = QuantumNetwork(
        channel.to_choi(backend=backend), partition, backend=backend
    )

    backend.assert_allclose(network.matrix(backend=backend).shape, (2, 2, 2, 2))
    backend.assert_allclose(network.dims, 4)
    backend.assert_allclose(network.partition, partition)
    backend.assert_allclose(network.system_output, (False, True))

    assert network.causal()
    assert network.unital()
    assert network.hermitian()
    assert network.positive_semidefinite()
    assert network.channel()


def test_with_states(backend):
    nqubits = 1
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)
    network_state = QuantumNetwork(state, (1, 2), backend=backend)

    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)
    network_channel = QuantumNetwork(
        channel.to_choi(backend=backend), (dims, dims), backend=backend
    )

    state_output = channel.apply_density_matrix(backend, state, nqubits)
    state_output_network = network_channel.apply(state)
    state_output_link = network_state.link_product(
        network_channel, subscripts="ij,jk -> ik"
    )

    backend.assert_allclose(state_output_network, state_output)
    backend.assert_allclose(
        state_output_link.matrix(backend=backend).reshape((dims, dims)), state_output
    )

    assert network_state.hermitian()
    assert network_state.positive_semidefinite()


@pytest.mark.parametrize("subscript", ["jk,kl->jl", "jk,lj->lk"])
def test_with_unitaries(backend, subscript):
    nqubits = 2
    dims = 2**nqubits

    unitary_1 = random_unitary(dims, backend=backend)
    unitary_2 = random_unitary(dims, backend=backend)

    network_1 = QuantumNetwork(unitary_1, (dims, dims), pure=True, backend=backend)
    network_2 = QuantumNetwork(unitary_2, (dims, dims), pure=True, backend=backend)
    network_3 = QuantumNetwork(
        unitary_2 @ unitary_1, (dims, dims), pure=True, backend=backend
    )
    network_4 = QuantumNetwork(
        unitary_1 @ unitary_2, (dims, dims), pure=True, backend=backend
    )

    test = network_1.link_product(network_2, subscript).to_full(backend=backend)

    if subscript[1] == subscript[3]:
        backend.assert_allclose(test, network_3.to_full(backend), atol=1e-8)

        backend.assert_allclose(
            test, (network_1 @ network_2).to_full(backend=backend), atol=1e-8
        )

    if subscript[0] == subscript[4]:
        backend.assert_allclose(test, network_4.to_full(backend))

        backend.assert_allclose(test, (network_2 @ network_1).to_full(backend=backend))


def test_with_comb(backend):
    subscript = "jklm,lmno->jkno"
    partition = (2,) * 4
    sys_out = (False, True) * 2

    comb = random_density_matrix(2**4, backend=backend)
    comb_2 = random_density_matrix(2**4, backend=backend)

    comb_choi = QuantumNetwork(comb, partition, system_output=sys_out, backend=backend)
    comb_choi_2 = QuantumNetwork(
        comb_2, partition, system_output=sys_out, backend=backend
    )
    comb_choi_3 = QuantumNetwork(
        comb @ comb_2, partition, system_output=sys_out, backend=backend
    ).to_full(backend)

    test = comb_choi.link_product(comb_choi_2, subscript).to_full(backend)

    backend.assert_allclose(test, comb_choi_3, atol=1e-5)
    backend.assert_allclose(test, (comb_choi @ comb_choi_2).to_full(backend), atol=1e-5)


def test_apply(backend):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)
    unitary = random_unitary(dims, backend=backend)
    network = QuantumNetwork(unitary, (dims, dims), pure=True, backend=backend)

    applied = network.apply(state)
    target = unitary @ state @ np.transpose(np.conj(unitary))

    backend.assert_allclose(applied, target, atol=1e-8)


def test_non_hermitian_and_prints(backend):
    nqubits = 2
    dims = 2**nqubits

    matrix = random_gaussian_matrix(dims**2, backend=backend)
    network = QuantumNetwork(matrix, (dims, dims), pure=False, backend=backend)

    assert not network.hermitian()
    assert not network.causal()
    assert not network.positive_semidefinite()
    assert not network.channel()

    assert network.__str__() == "J[4 -> 4]"
