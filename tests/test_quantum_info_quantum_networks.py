"""Tests for quantum_info.quantum_networks submodule"""

import numpy as np
import pytest

from qibo import gates
from qibo.quantum_info.quantum_networks import QuantumNetwork


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

    assert network.is_causal()
    assert network.is_unital()
    assert network.is_hermitian()
    assert network.is_positive_semidefinite()
