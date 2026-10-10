"""Tests for the protected functions in ``qibo.quantum_info._superoperator_transformations``."""

import numpy as np
import pytest

from qibo import gates
from qibo.quantum_info import (
    kraus_to_choi,
    kraus_to_liouville,
    liouville_to_pauli,
    pauli_to_liouville,
)
from qibo.quantum_info._superoperator_transformations import (
    _check_pauli_order,
    _check_pauli_superoperator_shape,
    _reshuffling,
)


@pytest.mark.parametrize("pauli_order", ["IXY", "IXYZI", "IXYY", "ABCD", ""])
def test_check_pauli_order(pauli_order):
    with pytest.raises(ValueError, match="pauli_order has to contain 4 symbols"):
        _check_pauli_order(pauli_order)


@pytest.mark.parametrize("function", [liouville_to_pauli, pauli_to_liouville])
@pytest.mark.parametrize(
    "shape", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (4,), (4, 4, 4)]
)
def test_pauli_superoperator_invalid_shape(backend, function, shape):
    super_op = backend.cast(np.zeros(shape, dtype=complex))
    with pytest.raises((ValueError, TypeError)):
        function(super_op, backend=backend)


@pytest.mark.parametrize("nqubits", [1, 2, 5])
def test_check_pauli_superoperator_shape(backend, nqubits):
    super_op = backend.identity(4**nqubits, dtype=backend.complex128)

    assert _check_pauli_superoperator_shape(super_op, "super_op") == (
        2**nqubits,
        nqubits,
    )


@pytest.mark.parametrize(
    "shape", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (4,), (8, 8)]
)
def test_check_pauli_superoperator_shape_errors(backend, shape):
    # error messages should always be the explicit ``ValueError``
    with pytest.raises(ValueError, match="must be of shape"):
        _check_pauli_superoperator_shape(backend.cast(np.zeros(shape)), "super_op")


@pytest.mark.parametrize("function", [kraus_to_choi, kraus_to_liouville])
def test_empty_kraus_operators(backend, function):
    with pytest.raises(ValueError, match="at least one"):
        function([], backend=backend)


def test_reshuffling_system_order(backend):
    with pytest.raises(ValueError):
        _reshuffling(backend.identity(4, dtype=backend.complex128), "system", backend)

    # a valid channel is a smoke test of the supported orders
    channel = gates.AmplitudeDampingChannel(0, 0.3)
    for order in ("row", "column"):
        assert kraus_to_choi(channel, order=order, backend=backend).shape == (4, 4)
