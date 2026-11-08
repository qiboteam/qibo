"""Tests for the quantum adders defined in `qibo/models/adders.py`."""

from itertools import product

import pytest

from qibo import Circuit, gates
from qibo.models import TTK, VBE, Adder, Cuccaro, Draper

# adders, and the number of auxiliary qubits that they use for the sum of nbits-bit integers
ADDERS = [
    (VBE, lambda nbits: nbits),
    (Cuccaro, lambda nbits: 1),
    (Draper, lambda nbits: 0),
    (TTK, lambda nbits: 0),
]


def _add(backend, adder, nbits, a, b, **kwargs):
    # little-endian registers: the qubit k of each register holds the bit 2^k
    circuit = Circuit(adder.nqubits, **kwargs)
    circuit.add(gates.X(k) for k in range(nbits) if (a >> k) & 1)
    circuit.add(gates.X(nbits + k) for k in range(nbits + 1) if (b >> k) & 1)
    circuit.add(adder)
    circuit.add(gates.M(*range(adder.nqubits)))

    frequencies = backend.execute_circuit(circuit, nshots=1).frequencies(binary=True)
    assert len(frequencies) == 1
    bitstring = next(iter(frequencies))

    to_int = lambda bits: int(bits[::-1] or "0", 2)
    return (
        to_int(bitstring[:nbits]),
        to_int(bitstring[nbits : 2 * nbits + 1]),
        to_int(bitstring[2 * nbits + 1 :]),
    )


@pytest.mark.parametrize("nbits", [1, 2, 3])
@pytest.mark.parametrize("adder, nauxiliary", ADDERS)
def test_adders(backend, adder, nauxiliary, nbits):
    circuit = adder(nbits)
    assert circuit.nqubits == 2 * nbits + 1 + nauxiliary(nbits)

    for a, b in product(range(2**nbits), range(2 ** (nbits + 1))):
        # a is unchanged, b is replaced by the sum modulo 2^(nbits + 1), and the auxiliary
        # qubits are reset
        assert _add(backend, circuit, nbits, a, b) == (a, (a + b) % 2 ** (nbits + 1), 0)


@pytest.mark.parametrize("nbits", [5, 6])
@pytest.mark.parametrize("adder, nauxiliary", ADDERS)
def test_adders_carry_propagation(backend, adder, nauxiliary, nbits):
    circuit = adder(nbits)
    largest = 2**nbits - 1

    for a, b in [(largest, 1), (1, largest), (largest, largest), (0, largest)]:
        assert _add(backend, circuit, nbits, a, b) == (a, a + b, 0)


@pytest.mark.parametrize("adder, nauxiliary", ADDERS)
def test_adders_density_matrix(backend, adder, nauxiliary):
    circuit = adder(2, density_matrix=True)
    assert circuit.density_matrix

    assert _add(backend, circuit, 2, 3, 2, density_matrix=True) == (3, 5, 0)


@pytest.mark.parametrize("nbits", [0, -1, 1.5, True, "2"])
@pytest.mark.parametrize("adder, nauxiliary", ADDERS)
def test_adders_errors(adder, nauxiliary, nbits):
    with pytest.raises(ValueError):
        adder(nbits)


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_Cuccaro_gates(nbits):
    circuit = Cuccaro(nbits)

    assert circuit.gate_types[gates.TOFFOLI] == 2 * nbits
    assert circuit.gate_types[gates.CNOT] == 4 * nbits + 1
    assert circuit.ngates == 6 * nbits + 1


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_Draper_gates(nbits):
    circuit = Draper(nbits)

    # two QFTs of nbits + 1 qubits without swaps, and the rotations that add a to b
    assert circuit.gate_types[gates.H] == 2 * (nbits + 1)
    assert circuit.gate_types[gates.CU1] == 3 * nbits * (nbits + 1) // 2 + nbits
    assert circuit.gate_types[gates.SWAP] == 0
    assert circuit.ngates == circuit.gate_types[gates.H] + circuit.gate_types[gates.CU1]


def test_TTK_half_adder():
    circuit = TTK(1)

    assert circuit.gate_types[gates.TOFFOLI] == 1
    assert circuit.gate_types[gates.CNOT] == 1
    assert circuit.ngates == 2


@pytest.mark.parametrize("nbits", [2, 3, 4, 8])
def test_TTK_gates(nbits):
    circuit = TTK(nbits)

    assert circuit.gate_types[gates.TOFFOLI] == 2 * nbits - 1
    assert circuit.gate_types[gates.CNOT] == 5 * nbits - 5
    assert circuit.ngates == 7 * nbits - 6


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_VBE_gates(nbits):
    circuit = VBE(nbits)

    assert circuit.gate_types[gates.TOFFOLI] == 4 * nbits - 2
    assert circuit.gate_types[gates.CNOT] == 4 * nbits
    assert circuit.ngates == 8 * nbits - 2


@pytest.mark.parametrize("method", ["Cuccaro", "Draper", "TTK", "VBE"])
def test_Adder(backend, method):
    nbits = 3
    methods = {"Cuccaro": Cuccaro, "Draper": Draper, "TTK": TTK, "VBE": VBE}

    circuit = Adder(nbits, method=method)

    assert circuit.raw == methods[method](nbits).raw
    assert _add(backend, circuit, nbits, 5, 6) == (5, 11, 0)


def test_Adder_default_and_kwargs():
    assert Adder(3).raw == Cuccaro(3).raw
    assert Adder(2, "TTK", density_matrix=True).density_matrix


@pytest.mark.parametrize("method", ["Gidney", "cuccaro", 1, None, ["VBE"]])
def test_Adder_errors(method):
    with pytest.raises(ValueError):
        Adder(2, method=method)

    with pytest.raises(ValueError):
        Adder(0)
