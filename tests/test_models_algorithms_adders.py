"""Tests for the quantum adders defined in `qibo/models/algorithms/adders.py`."""

import math
import random
from itertools import product

import pytest

from qibo import Circuit, gates
from qibo.models import DKRS, RV, TTK, VBE, Adder, Cuccaro, Draper, Gidney
from qibo.models.algorithms.adders import _mcx_gates, _mcx_ladder


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
    (RV, lambda nbits: 0, True),
    (TTK, lambda nbits: 0, True),
    (VBE, lambda nbits: nbits, True),
]

# adders that only have X, CNOT, TOFFOLI and SWAP gates, so that they are classical circuits
CLASSICAL_ADDERS = [Cuccaro, DKRS, RV, TTK, VBE]


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


def _apply_classical(gate_list, bits):
    # X, CNOT, TOFFOLI and SWAP gates, which map the states of the computational basis to others
    for gate in gate_list:
        if gate.name == "swap":
            first, second = gate.target_qubits
            bits[first], bits[second] = bits[second], bits[first]
        elif all(bits[qubit] for qubit in gate.control_qubits):
            for qubit in gate.target_qubits:
                bits[qubit] ^= 1

    return bits


def _classical_add(adder, nbits, a, b):
    bits = [0] * adder.nqubits
    for k in range(nbits):
        bits[k] = (a >> k) & 1
        bits[nbits + k] = (b >> k) & 1

    bits = _apply_classical(adder.queue, bits)

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


# the RV adder has gates with at least 20 controls from about 80 bits
@pytest.mark.parametrize("nbits", [4, 7, 8, 11, 16, 128])
@pytest.mark.parametrize("adder", CLASSICAL_ADDERS)
def test_adders_classical(adder, nbits):
    circuit = adder(nbits)
    largest = 2**nbits - 1

    # the integers do not fit in 64 bits for the largest circuits
    generator = random.Random(nbits)
    pairs = [(largest, 1), (1, largest), (largest, largest), (0, largest)]
    pairs.extend(
        (generator.getrandbits(nbits), generator.getrandbits(nbits)) for _ in range(50)
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


@pytest.mark.parametrize("nbits", [1, 2, 3, 8, 33])
def test_RV_gates(nbits):
    circuit = RV(nbits)

    assert circuit.nqubits == 2 * nbits + 1
    assert set(circuit.gate_types) <= {gates.X, gates.CNOT, gates.TOFFOLI}


def test_RV_half_adder():
    circuit = RV(1)

    assert circuit.gate_types[gates.TOFFOLI] == 1
    assert circuit.gate_types[gates.CNOT] == 1
    assert circuit.ngates == 2


def test_RV_depth():
    # the ladders have a logarithmic depth, instead of the linear depth of the other adder
    assert RV(256).depth < TTK(256).depth
    assert RV(1024).depth < TTK(1024).depth / 2


@pytest.mark.parametrize("consecutive", [True, False])
@pytest.mark.parametrize("ngates", range(1, 26))
def test_mcx_ladder(ngates, consecutive):
    generator = random.Random(ngates)
    if consecutive:
        # ladder of CNOT gates
        positions = list(range(1, ngates + 1))
    else:
        positions = sorted(generator.sample(range(1, 5 * ngates + 1), ngates))

    layers = _mcx_ladder(list(range(positions[-1] + 1)), positions)

    # the depth of Ref. [1], where the number of gates plus one is k
    k = ngates + 1
    assert len(layers) == k.bit_length() - 1 + (2 * k // 3).bit_length() - 1

    # the gates of each layer act on different qubits, so that they are parallel
    for layer in layers:
        acting = [qubit for controls, target in layer for qubit in (*controls, target)]
        assert len(acting) == len(set(acting))

    for _ in range(100):
        bits = [generator.getrandbits(1) for _ in range(positions[-1] + 1)]

        # each target is flipped if all of the qubits from the previous target to it are 1,
        # with the values that they have before the ladder
        expected = list(bits)
        previous = 0
        for position in positions:
            expected[position] ^= int(all(bits[previous:position]))
            previous = position

        for layer in layers:
            for controls, target in layer:
                if all(bits[qubit] for qubit in controls):
                    bits[target] ^= 1

        assert bits == expected


def test_mcx_ladder_without_gates():
    assert _mcx_ladder([0, 1], []) == []


@pytest.mark.parametrize("nqubits", [9, 11])
def test_mcx_gates(backend, nqubits):
    # gates with one, two and three controls on different qubits. With 9 qubits, there are
    # no qubits that the layer does not act on to be used as auxiliary qubits
    layer = [((0,), 1), ((2, 3), 4), ((5, 6, 7), 8)]
    decomposition = _mcx_gates(layer, nqubits, backend)

    assert {type(gate) for gate in decomposition} <= {
        gates.X,
        gates.CNOT,
        gates.TOFFOLI,
    }

    for state in range(2**nqubits):
        bits = [(state >> qubit) & 1 for qubit in range(nqubits)]

        expected = list(bits)
        for controls, target in layer:
            expected[target] ^= int(all(bits[qubit] for qubit in controls))

        assert _apply_classical(decomposition, bits) == expected


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
    "method", ["Cuccaro", "DKRS", "Draper", "Gidney", "RV", "TTK", "VBE"]
)
def test_Adder(backend, method):
    nbits = 3
    methods = {
        "Cuccaro": Cuccaro,
        "DKRS": DKRS,
        "Draper": Draper,
        "Gidney": Gidney,
        "RV": RV,
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
