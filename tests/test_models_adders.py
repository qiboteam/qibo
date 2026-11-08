"""Tests for the quantum adders defined in `qibo/models/adders.py`."""

import math
import random
from itertools import product

import pytest

from qibo import Circuit, gates
from qibo.models import DKRS, TTK, VBE, Adder, Cuccaro, Draper, Gidney


def _dkrs_nauxiliary(nbits):
    # qubits of the sum, and the ones that hold the products of the propagate bits
    return nbits + 1 + nbits - nbits.bit_count() - (nbits.bit_length() - 1)


# adders, number of auxiliary qubits that they use for the sum of nbits-bit integers, and
# whether they add modulo 2^(nbits + 1) when the overflow qubit is not in the state 0
ADDERS = [
    (Cuccaro, lambda nbits: 1, True),
    (DKRS, _dkrs_nauxiliary, False),
    (Draper, lambda nbits: 0, True),
    (Gidney, lambda nbits: nbits - 1, False),
    (TTK, lambda nbits: 0, True),
    (VBE, lambda nbits: nbits, True),
]

# adders that only have X, CNOT, TOFFOLI and SWAP gates, so that they are classical circuits
CLASSICAL_ADDERS = [Cuccaro, DKRS, TTK, VBE]


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


def _classical_add(adder, nbits, a, b):
    bits = [0] * adder.nqubits
    for k in range(nbits):
        bits[k] = (a >> k) & 1
        bits[nbits + k] = (b >> k) & 1

    for gate in adder.queue:
        if gate.name == "swap":
            first, second = gate.target_qubits
            bits[first], bits[second] = bits[second], bits[first]
        elif all(bits[qubit] for qubit in gate.control_qubits):
            for qubit in gate.target_qubits:
                bits[qubit] ^= 1

    to_int = lambda register: sum(bit << k for k, bit in enumerate(register))
    return (
        to_int(bits[:nbits]),
        to_int(bits[nbits : 2 * nbits + 1]),
        to_int(bits[2 * nbits + 1 :]),
    )


@pytest.mark.parametrize("nbits", [1, 2, 3])
@pytest.mark.parametrize("adder, nauxiliary, modular", ADDERS)
def test_adders(backend, adder, nauxiliary, modular, nbits):
    circuit = adder(nbits)
    assert circuit.nqubits == 2 * nbits + 1 + nauxiliary(nbits)

    # the overflow qubit is only in the state 1 if the adder is modular
    for a, b in product(range(2**nbits), range(2 ** (nbits + int(modular)))):
        # a is unchanged, b is replaced by the sum modulo 2^(nbits + 1), and the auxiliary
        # qubits are reset
        assert _add(backend, circuit, nbits, a, b) == (a, (a + b) % 2 ** (nbits + 1), 0)


@pytest.mark.parametrize("adder, nauxiliary, modular", ADDERS)
def test_adders_superposition(backend, adder, nauxiliary, modular):
    nbits = 2
    circuit = Circuit(adder(nbits).nqubits, density_matrix=True)
    circuit.add(gates.H(qubit) for qubit in range(2 * nbits))
    circuit.add(adder(nbits))

    # uniform superposition of the states |a>|a + b>|0>
    target = backend.zeros(2**circuit.nqubits, dtype=backend.complex128)
    for a, b in product(range(2**nbits), repeat=2):
        bits = [(a >> k) & 1 for k in range(nbits)]
        bits.extend(((a + b) >> k) & 1 for k in range(nbits + 1))
        bits.extend([0] * nauxiliary(nbits))
        target[int("".join(map(str, bits)), 2)] = 1 / 2**nbits

    # the circuits with measurements end in a pure state, up to the outcomes, so that the
    # fidelity with the target state is one, regardless of the global phase
    state = backend.execute_circuit(circuit).state()
    fidelity = backend.real(backend.sum(backend.conj(target) * (state @ target)))
    backend.assert_allclose(fidelity, 1.0, atol=1e-8)


@pytest.mark.parametrize("nbits", [4, 5])
@pytest.mark.parametrize("adder, nauxiliary, modular", ADDERS)
def test_adders_carry_propagation(backend, adder, nauxiliary, modular, nbits):
    circuit = adder(nbits)
    largest = 2**nbits - 1

    for a, b in [(largest, 1), (1, largest), (largest, largest), (0, largest)]:
        assert _add(backend, circuit, nbits, a, b) == (a, a + b, 0)


@pytest.mark.parametrize("nbits", [4, 7, 8, 11, 16])
@pytest.mark.parametrize("adder", CLASSICAL_ADDERS)
def test_adders_classical(adder, nbits):
    circuit = adder(nbits)
    largest = 2**nbits - 1

    random.seed(nbits)
    pairs = [(largest, 1), (1, largest), (largest, largest), (0, largest)]
    pairs.extend(
        (random.randrange(2**nbits), random.randrange(2**nbits)) for _ in range(50)
    )
    for a, b in pairs:
        assert _classical_add(circuit, nbits, a, b) == (a, a + b, 0)


@pytest.mark.parametrize("adder, nauxiliary, modular", ADDERS)
def test_adders_density_matrix(backend, adder, nauxiliary, modular):
    circuit = adder(2, density_matrix=True)
    assert circuit.density_matrix

    assert _add(backend, circuit, 2, 3, 2, density_matrix=True) == (3, 5, 0)


@pytest.mark.parametrize("nbits", [0, -1, 1.5, True, "2"])
@pytest.mark.parametrize("adder, nauxiliary, modular", ADDERS)
def test_adders_errors(adder, nauxiliary, modular, nbits):
    with pytest.raises(ValueError):
        adder(nbits)


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_Cuccaro_gates(nbits):
    circuit = Cuccaro(nbits)

    assert circuit.gate_types[gates.TOFFOLI] == 2 * nbits
    assert circuit.gate_types[gates.CNOT] == 4 * nbits + 1
    assert circuit.ngates == 6 * nbits + 1


@pytest.mark.parametrize("nbits", range(1, 17))
def test_DKRS_nqubits(nbits):
    # 3n + 2 + m qubits, with m = n - w(n) - floor(log2(n)) and w(n) the number of ones of n
    nproducts = nbits - nbits.bit_count() - math.floor(math.log2(nbits))

    assert DKRS(nbits).nqubits == 3 * nbits + 2 + nproducts


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_Draper_gates(nbits):
    circuit = Draper(nbits)

    # two QFTs of nbits + 1 qubits without swaps, and the rotations that add a to b
    assert circuit.gate_types[gates.H] == 2 * (nbits + 1)
    assert circuit.gate_types[gates.CU1] == 3 * nbits * (nbits + 1) // 2 + nbits
    assert circuit.gate_types[gates.SWAP] == 0
    assert circuit.ngates == circuit.gate_types[gates.H] + circuit.gate_types[gates.CU1]


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_Gidney_gates(nbits):
    circuit = Gidney(nbits)

    # four T or T^dagger gates per temporary logical-AND, and the erasure of the carries,
    # except for the last one, with measurements and gates conditioned on their outcomes
    assert circuit.gate_types[gates.T] == 2 * nbits
    assert circuit.gate_types[gates.TDG] == 2 * nbits
    assert circuit.gate_types[gates.CU1] == nbits - 1
    assert circuit.gate_types[gates.RX] == nbits - 1
    assert circuit.gate_types[gates.M] == nbits - 1
    assert circuit.nqubits == 3 * nbits


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


@pytest.mark.parametrize(
    "method", ["Cuccaro", "DKRS", "Draper", "Gidney", "TTK", "VBE"]
)
def test_Adder(backend, method):
    nbits = 3
    methods = {
        "Cuccaro": Cuccaro,
        "DKRS": DKRS,
        "Draper": Draper,
        "Gidney": Gidney,
        "TTK": TTK,
        "VBE": VBE,
    }

    circuit = Adder(nbits, method=method)

    assert circuit.nqubits == methods[method](nbits).nqubits
    assert circuit.ngates == methods[method](nbits).ngates
    assert _add(backend, circuit, nbits, 5, 6) == (5, 11, 0)


def test_Adder_default_and_kwargs():
    assert Adder(3).raw == Cuccaro(3).raw
    assert Adder(2, "TTK", density_matrix=True).density_matrix


@pytest.mark.parametrize("method", ["Gidney ", "cuccaro", 1, None, ["VBE"]])
def test_Adder_errors(method):
    with pytest.raises(ValueError):
        Adder(2, method=method)

    with pytest.raises(ValueError):
        Adder(0)
