"""Tests for the quantum adders defined in `qibo/models/adders.py`."""

from itertools import product

import pytest

from qibo import Circuit, gates
from qibo.models import VBE


def _add(backend, adder, nbits, a, b, **kwargs):
    # little-endian registers: the qubit k of each register holds the bit 2^k
    circuit = Circuit(adder.nqubits, **kwargs)
    circuit.add(gates.X(k) for k in range(nbits) if (a >> k) & 1)
    circuit.add(gates.X(nbits + k) for k in range(nbits) if (b >> k) & 1)
    circuit.add(adder)
    circuit.add(gates.M(*range(adder.nqubits)))

    frequencies = backend.execute_circuit(circuit, nshots=1).frequencies(binary=True)
    assert len(frequencies) == 1
    bitstring = next(iter(frequencies))

    to_int = lambda bits: int(bits[::-1], 2)
    return (
        to_int(bitstring[:nbits]),
        to_int(bitstring[nbits : 2 * nbits + 1]),
        to_int(bitstring[2 * nbits + 1 :]),
    )


@pytest.mark.parametrize("nbits", [1, 2, 3])
def test_VBE(backend, nbits):
    adder = VBE(nbits)
    assert adder.nqubits == 3 * nbits + 1

    for a, b in product(range(2**nbits), repeat=2):
        # a is unchanged, b is replaced by the sum, and the carry qubits are reset
        assert _add(backend, adder, nbits, a, b) == (a, a + b, 0)


@pytest.mark.parametrize("nbits", [5, 6])
def test_VBE_carry_propagation(backend, nbits):
    adder = VBE(nbits)
    largest = 2**nbits - 1

    for a, b in [(largest, 1), (1, largest), (largest, largest), (0, largest)]:
        assert _add(backend, adder, nbits, a, b) == (a, a + b, 0)


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_VBE_gates(nbits):
    adder = VBE(nbits)

    assert adder.gate_types[gates.TOFFOLI] == 4 * nbits - 2
    assert adder.gate_types[gates.CNOT] == 4 * nbits
    assert adder.ngates == 8 * nbits - 2


def test_VBE_density_matrix(backend):
    adder = VBE(2, density_matrix=True)
    assert adder.density_matrix

    assert _add(backend, adder, 2, 3, 2, density_matrix=True) == (3, 5, 0)


@pytest.mark.parametrize("nbits", [0, -1, 1.5, True, "2"])
def test_VBE_errors(nbits):
    with pytest.raises(ValueError):
        VBE(nbits)
