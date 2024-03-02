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

    comb_partition = (2,) * 4
    comb_sys_out = (False, True) * 2
    comb = random_density_matrix(2**4, backend=backend)
    comb_choi = QuantumNetwork(
        comb, comb_partition, system_input=comb_sys_out, backend=backend
    )

    with pytest.raises(TypeError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=True)

    with pytest.raises(ValueError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(1, "2"))

    with pytest.raises(ValueError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(-1, 2))

    with pytest.raises(ValueError):
        QuantumNetwork(
            channel.to_choi(backend=backend), partition=(1, 2), system_input=(1, 2, 3)
        )

    with pytest.raises(TypeError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=(1, 2), pure="True")

    with pytest.raises(ValueError):
        network.is_hermitian(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        network.is_unital(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        network.is_causal(precision_tol=-1e-8)

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
        network_2.system_input = (False,)
        network += network_2

    # Multiplying QuantumNetwork with non-QuantumNetwork
    with pytest.raises(TypeError):
        network @ network.operator(backend)

    # Linking QuantumNetwork with non-QuantumNetwork
    with pytest.raises(TypeError):
        network.link_product(network.operator(backend))

    with pytest.raises(TypeError):
        network.link_product(network, subscripts=True)

    with pytest.raises(NotImplementedError):
        network.link_product(network, subscripts="jk,lm->no")

    with pytest.raises(NotImplementedError):
        net @ net

    with pytest.raises(NotImplementedError):
        net @ network

    with pytest.raises(ValueError):
        network @ net

    with pytest.raises(ValueError):
        network @ QuantumNetwork(comb, (16, 16), pure=True, backend=backend)

    with pytest.raises(ValueError):
        comb_choi @ QuantumNetwork(comb, (16, 16), pure=True, backend=backend)

    with pytest.raises(ValueError):
        comb_choi @ net

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
        (network + network).operator(backend), (2 * network).operator(backend)
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).operator(backend),
        (2 * network_state_pure).full(),
    )

    # Sum with itself has to match multiplying by float
    backend.assert_allclose(
        (network + network).operator(backend), (2.0 * network).operator(backend)
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).operator(backend),
        (2.0 * network_state_pure).full(),
    )

    # Multiplying and dividing by same scalar has to bring back to original network
    backend.assert_allclose(
        ((2.0 * network) / 2).operator(backend), network.operator(backend)
    )

    unitary = random_unitary(dims, backend=backend)
    network_unitary = QuantumNetwork(unitary, (dims, dims), pure=True, backend=backend)
    backend.assert_allclose(
        (network_unitary / 2).operator(backend), unitary / np.sqrt(2), atol=1e-5
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

    backend.assert_allclose(network.operator(backend=backend).shape, (2, 2, 2, 2))
    backend.assert_allclose(network.dims, 4)
    backend.assert_allclose(network.partition, partition)
    backend.assert_allclose(network.system_input, (False, True))

    assert network.is_causal()
    assert network.is_unital()
    assert network.is_hermitian()
    assert network.is_positive_semidefinite()
    assert network.is_channel()


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
        state_output_link.operator(backend=backend).reshape((dims, dims)), state_output
    )

    assert network_state.is_hermitian()
    assert network_state.is_positive_semidefinite()


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

    test = network_1.link_product(network_2, subscript).full(backend=backend,update=True)

    if subscript[1] == subscript[3]:
        backend.assert_allclose(test, network_3.full(), atol=1e-8)

        backend.assert_allclose(
            test, (network_1 @ network_2).full(backend=backend), atol=1e-8
        )

    if subscript[0] == subscript[4]:
        backend.assert_allclose(test, network_4.full(backend))

        backend.assert_allclose(test, (network_2 @ network_1).full(backend=backend))


def test_with_comb(backend):
    subscript = "jklm,kl->jm"
    comb_partition = (2,) * 4
    channel_partition = (2,) * 2
    comb_sys_out = (False, True) * 2
    channel_sys_out = (False, True)

    comb = random_density_matrix(2**4, backend=backend)
    channel = random_density_matrix(2**2, backend=backend)

    comb_choi = QuantumNetwork(
        comb, comb_partition, system_input=comb_sys_out, backend=backend
    )
    channel_choi = QuantumNetwork(
        channel, channel_partition, system_input=channel_sys_out, backend=backend
    )

    test = comb_choi.link_product(channel_choi, subscript).full(backend,update=True)
    channel_choi2 = comb_choi @ channel_choi

    backend.assert_allclose(test, channel_choi2.full(backend), atol=1e-5)


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

    assert not network.is_hermitian()
    assert not network.is_causal()
    assert not network.is_positive_semidefinite()
    assert not network.is_channel()

    assert network.__str__() == "J[4 -> 4]"

def test_uility_func():
    old_shape = (0,10,1,11,2,12,3,13)
    test_ls = np.ones(old_shape)
    n = len(test_ls.shape) // 2

    system_input = (False, True, False, True)

    order2op = QuantumNetwork._order_tensor2operator(n , system_input)
    order2tensor = QuantumNetwork._order_operator2tensor(n , system_input)
    new_shape = test_ls.transpose(order2op).shape

    for i in range(n):
        if system_input[i]:
            assert (new_shape[i] - new_shape[i+n]) == 10
        else:
            assert (new_shape[i] - new_shape[i+n]) == -10
    
    assert tuple(test_ls.transpose(order2op).transpose(order2tensor).shape) == old_shape
