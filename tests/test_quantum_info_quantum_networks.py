"""Tests for quantum_info.quantum_networks submodule"""

import numpy as np
import pytest

from qibo import gates
from qibo.quantum_info.quantum_networks import (
    IdentityChannel,
    QuantumChannel,
    QuantumComb,
    QuantumNetwork,
    TraceOperation,
    link_product,
)
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
    network = QuantumNetwork.from_operator(
        channel.to_choi(backend=backend), partition, backend=backend
    )
    quantum_comb = QuantumComb.from_operator(
        channel.to_choi(backend=backend), partition, backend=backend
    )
    quantum_channel = QuantumChannel.from_operator(
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

    with pytest.raises(TypeError):
        QuantumNetwork(channel.to_choi(backend=backend), partition=1, pure=True)

    with pytest.raises(ValueError):
        network.is_hermitian(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        network.is_positive_semidefinite(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        quantum_comb.is_causal(precision_tol=-1e-8)

    with pytest.raises(ValueError):
        quantum_channel.is_unital(precision_tol=-1e-8)

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
        network @ network.operator(backend=backend)

    # Linking QuantumNetwork with non-QuantumNetwork
    with pytest.raises(TypeError):
        network.link_product(network.operator(backend=backend))

    with pytest.raises(TypeError):
        network.link_product(network, subscripts=True)

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

    with pytest.raises(ValueError):
        QuantumNetwork(matrix, (1, 1), pure=True, backend=backend)

    with pytest.raises(ValueError):
        QuantumNetwork.from_operator(matrix, (1, 2), pure=True, backend=backend)

    vec = np.random.rand(4)
    vec = backend.cast(vec, dtype=vec.dtype)
    vec = backend.cast(vec, dtype=vec.dtype)
    with pytest.raises(ValueError):
        QuantumNetwork.from_operator(vec, backend=backend)

    with pytest.raises(ValueError):
        QuantumComb.from_operator(vec, pure=True, backend=backend)

    with pytest.raises(ValueError):
        QuantumChannel(matrix, partition=(2, 2, 2), pure=True, backend=backend)

    with pytest.raises(TypeError):
        link_product(1, quantum_comb, backend=backend)

    with pytest.raises(TypeError):
        link_product("ij, i", quantum_comb, matrix, backend=backend)

    # raise warning
    link_product("ii", quantum_channel, backend=backend)
    link_product("ij, kj", network_state, quantum_channel, backend=backend)
    link_product("ij, jj", network_state, quantum_channel, backend=backend)
    link_product(
        "ij, jj, jj", network_state, quantum_channel, quantum_channel, backend=backend
    )


def test_class_methods(backend):
    matrix = random_density_matrix(2**2, backend=backend)
    with pytest.raises(ValueError):
        QuantumNetwork._operator_to_tensor(matrix, (3,))


def test_operational_logic(backend):
    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)
    nqubits = len(channel.target_qubits)
    dims = 2**nqubits
    partition = (dims, dims)
    network = QuantumNetwork.from_operator(
        channel.to_choi(backend=backend), partition, backend=backend
    )

    state = random_density_matrix(dims, backend=backend)
    network_state_pure = QuantumNetwork(state, (2, 2), pure=True, backend=backend)

    # Sum with itself has to match multiplying by int
    backend.assert_allclose(
        (network + network).operator(backend=backend),
        (2 * network).operator(backend=backend),
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).operator(backend=backend),
        (2 * network_state_pure).operator(full=True, backend=backend),
    )

    # Sum with itself has to match multiplying by float
    backend.assert_allclose(
        (network + network).operator(backend=backend),
        (2.0 * network).operator(backend=backend),
    )
    backend.assert_allclose(
        (network_state_pure + network_state_pure).operator(backend=backend),
        (2.0 * network_state_pure).operator(full=True, backend=backend),
    )

    # Multiplying and dividing by same scalar has to bring back to original network
    backend.assert_allclose(
        ((2.0 * network) / 2).operator(backend=backend),
        network.operator(backend=backend),
    )

    unitary = random_unitary(dims, backend=backend)
    network_unitary = QuantumNetwork(unitary, (dims, dims), pure=True, backend=backend)
    backend.assert_allclose(
        (network_unitary / 2).operator(backend=backend), unitary / np.sqrt(2), atol=1e-5
    )

    # Complex conjugate of a network has to match the complex conjugate of the operator
    backend.assert_allclose(
        network.conj().operator(backend=backend),
        backend.np.conj(network.operator(backend=backend)),
    )


def test_parameters(backend):
    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)

    nqubits = len(channel.target_qubits)
    dims = 2**nqubits
    partition = (dims, dims)

    choi = channel.to_choi(backend=backend)

    network = QuantumNetwork.from_operator(choi, partition, backend=backend)
    quantum_comb = QuantumComb.from_operator(choi, partition, backend=backend)
    quantum_channel = QuantumChannel.from_operator(
        choi, partition, backend=backend, inverse=True
    )

    rand = random_density_matrix(dims**2, backend=backend)
    non_channel = QuantumChannel.from_operator(
        rand, partition, backend=backend, inverse=True
    )

    backend.assert_allclose(network.operator(backend=backend).shape, (2, 2, 2, 2))
    backend.assert_allclose(network.dims, 4)
    backend.assert_allclose(network.partition, partition)
    backend.assert_allclose(network.system_input, (True, False))

    assert network.is_hermitian()
    assert network.is_positive_semidefinite()
    assert quantum_comb.is_causal()
    assert quantum_channel.is_unital()
    assert quantum_channel.is_channel()

    # Test non-unital and non_causal
    assert not non_channel.is_causal()
    assert not non_channel.is_unital()


def test_with_states(backend):
    nqubits = 1
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)
    network_state = QuantumChannel.from_operator(state, backend=backend)

    lamb = float(np.random.rand())
    channel = gates.DepolarizingChannel(0, lamb)
    network_channel = QuantumChannel.from_operator(
        channel.to_choi(backend=backend), (dims, dims), backend=backend, inverse=True
    )

    state_output = channel.apply_density_matrix(backend, state, nqubits)
    state_output_network = network_channel.apply(state)
    state_output_link = network_state.link_product("ij,jk -> ik", network_channel)

    backend.assert_allclose(state_output_network, state_output)
    backend.assert_allclose(state_output_link.matrix(backend=backend), state_output)

    assert network_state.is_hermitian()
    assert network_state.is_positive_semidefinite()


@pytest.mark.parametrize("subscript", ["jk,kl->jl", "jk,lj->lk"])
def test_with_unitaries(backend, subscript):
    nqubits = 2
    dims = 2**nqubits

    unitary_1 = random_unitary(dims, backend=backend)
    unitary_2 = random_unitary(dims, backend=backend)

    network_1 = QuantumComb.from_operator(
        unitary_1, (dims, dims), pure=True, backend=backend, inverse=True
    )
    network_2 = QuantumComb.from_operator(
        unitary_2, (dims, dims), pure=True, backend=backend, inverse=True
    )
    network_3 = QuantumComb.from_operator(
        unitary_2 @ unitary_1, (dims, dims), pure=True, backend=backend, inverse=True
    )
    network_4 = QuantumComb.from_operator(
        unitary_1 @ unitary_2, (dims, dims), pure=True, backend=backend, inverse=True
    )

    test = network_1.link_product(subscript, network_2).full(
        backend=backend, update=True
    )

    if subscript[1] == subscript[3]:
        backend.assert_allclose(
            test, network_3.full(backend=backend, update=True), atol=1e-8
        )

        backend.assert_allclose(
            test, (network_1 @ network_2).full(backend=backend, update=True), atol=1e-8
        )

    if subscript[0] == subscript[4]:
        backend.assert_allclose(test, network_4.full(backend))

        backend.assert_allclose(test, (network_2 @ network_1).full(backend=backend))

    # Check properties for pure states
    assert network_1.is_causal()
    assert network_1.is_hermitian()
    assert network_1.is_positive_semidefinite()


def test_with_comb(backend):
    subscript = "jklm,kl->jm"
    comb_partition = (2,) * 4
    channel_partition = (2,) * 2
    comb_sys_in = (False, True) * 2
    channel_sys_in = (False, True)

    rand_choi = random_density_matrix(4**2, backend=backend)
    unitary_1 = random_unitary(4, backend=backend)
    unitary_2 = random_unitary(4, backend=backend)
    non_channel = QuantumNetwork.from_operator(
        rand_choi,
        (2, 2, 2, 2),
        system_input=(True, True, False, False),
        backend=backend,
    )
    unitary_channel = QuantumNetwork.from_operator(
        unitary_1,
        (2, 2, 2, 2),
        system_input=(True, True, False, False),
        pure=True,
        backend=backend,
    )
    unitary_channel2 = QuantumNetwork.from_operator(
        unitary_2,
        (2, 2, 2, 2),
        system_input=(True, True, False, False),
        pure=True,
        backend=backend,
    )

    non_comb = link_product(
        "ij kl, km on -> jl mn", non_channel, unitary_channel, backend=backend
    )
    non_comb = QuantumComb(
        non_comb.full(backend=backend),
        (2, 2, 2, 2),
        system_input=(True, False, True, False),
        backend=backend,
    )
    two_comb = link_product(
        "ij kl, km on, i, o",
        unitary_channel,
        unitary_channel2,
        TraceOperation(2, backend=backend),
        TraceOperation(2, backend=backend),
        backend=backend,
    )
    two_comb = QuantumComb(
        two_comb.full(backend=backend),
        (2, 2, 2, 2),
        system_input=(True, False, True, False),
        backend=backend,
    )

    comb = random_density_matrix(2**4, backend=backend)
    channel = random_density_matrix(2**2, backend=backend)

    comb_choi = QuantumNetwork.from_operator(
        comb, comb_partition, system_input=comb_sys_in, backend=backend
    )
    channel_choi = QuantumNetwork.from_operator(
        channel, channel_partition, system_input=channel_sys_in, backend=backend
    )

    test = comb_choi.link_product(subscript, channel_choi).full(
        update=True, backend=backend
    )
    channel_choi2 = comb_choi @ channel_choi

    backend.assert_allclose(test, channel_choi2.full(backend), atol=1e-5)

    assert non_comb.is_hermitian()
    assert not non_comb.is_causal()

    assert two_comb.is_hermitian()
    assert two_comb.is_causal()


def test_apply(backend):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)
    unitary = random_unitary(dims, backend=backend)
    network = QuantumChannel.from_operator(
        unitary, (dims, dims), pure=True, backend=backend, inverse=True
    )

    applied = network.apply(state)
    target = unitary @ state @ backend.np.conj(unitary).T

    backend.assert_allclose(applied, target, atol=1e-8)


def test_non_hermitian_and_prints(backend):
    nqubits = 2
    dims = 2**nqubits

    matrix = random_gaussian_matrix(dims**2, backend=backend)
    network = QuantumNetwork.from_operator(
        matrix, (dims, dims), pure=False, backend=backend
    )

    assert not network.is_hermitian()
    assert not network.is_positive_semidefinite()

    assert network.__str__() == "J[┍4┑, ┕4┙]"


def test_uility_function():
    # _order_tensor2operator should convert
    # (a0,a1,b0,b1,...) to (a0,b0,..., a1,b1,...)
    old_shape = (0, 10, 1, 11, 2, 12, 3, 13)
    test_ls = np.ones(old_shape)
    n = len(test_ls.shape) // 2

    order2op = QuantumNetwork._order_tensor_to_operator(n)
    order2tensor = QuantumNetwork._order_operator_to_tensor(n)

    new_shape = test_ls.transpose(order2op).shape
    for i in range(n):
        assert (new_shape[i] - new_shape[i + n]) == -10

    assert tuple(test_ls.transpose(order2op).transpose(order2tensor).shape) == old_shape


def test_predefined(backend):
    tr_ch = TraceOperation(2, backend=backend)

    id_ch = IdentityChannel(2, backend=backend)
    id_mat = id_ch.matrix(backend=backend)

    backend.assert_allclose(
        id_mat,
        backend.cast(
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            dtype=id_mat.dtype,
        ),
        atol=1e-8,
    )

    traced = link_product("ij,j", id_ch, tr_ch, backend=backend)

    backend.assert_allclose(
        tr_ch.matrix(backend=backend), traced.matrix(backend=backend), atol=1e-8
    )


def test_default_construction(backend):
    vec = np.random.rand(4).reshape([4, 1])
    mat = np.random.rand(16).reshape([2, 2, 2, 2])
    tensor = np.random.rand(16).reshape([4, 4])
    vec = backend.cast(vec, dtype=vec.dtype)
    mat = backend.cast(mat, dtype=mat.dtype)
    tensor = backend.cast(tensor, dtype=tensor.dtype)
    vec = backend.cast(vec, dtype=vec.dtype)
    mat = backend.cast(mat, dtype=mat.dtype)
    tensor = backend.cast(tensor, dtype=tensor.dtype)
    network = QuantumNetwork.from_operator(vec, pure=True, backend=backend)
    assert network.partition == (4, 1)
    assert network.system_input == (True, False)
    comb1 = QuantumComb.from_operator(vec, (4, 1), pure=True, backend=backend)
    assert comb1.system_input == (True, False)
    comb2 = QuantumComb.from_operator(vec, pure=True, backend=backend)
    assert comb2.partition == (4, 1)
    assert comb2.system_input == (True, False)
    comb3 = QuantumComb.from_operator(mat, pure=False, backend=backend)
    assert comb3.partition == (2, 2)
    assert comb3.system_input == (True, False)
    comb3 = QuantumComb(vec, system_input=(True, True), pure=True, backend=backend)
    assert comb3.partition == (4, 1)
    assert comb3.system_input == (True, False)
    channel1 = QuantumChannel.from_operator(vec, pure=True, backend=backend)
    assert channel1.partition == (4, 1)
    assert channel1.system_input == (True, False)
    channel2 = QuantumChannel(
        vec, partition=4, system_input=True, pure=True, backend=backend
    )
    assert channel2.partition == (4, 1)
    assert channel2.system_input == (True, False)
    channel3 = QuantumChannel(vec, partition=4, pure=True, backend=backend)
    assert channel3.partition == (1, 4)
    assert channel3.system_input == (True, False)
    channel4 = QuantumChannel(
        vec, partition=4, system_input=False, pure=True, backend=backend
    )
    assert channel4.partition == (1, 4)
    assert channel4.system_input == (True, False)
    channel5 = QuantumChannel(tensor, pure=False, backend=backend)
    assert channel5.partition == (2, 2)
    assert channel5.system_input == (True, False)
