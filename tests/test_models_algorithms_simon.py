"""Tests for Simon's algorithm defined in `qibo/models/algorithms/simon.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models import Simon, simon_secret_string


@pytest.mark.parametrize("string", ["1", "11", "101", "0110", "10011"])
def test_simon(backend, string):
    nbits = len(string)
    circuit = Simon(_oracle(string))
    assert circuit.nqubits == 2 * nbits

    backend.set_seed(nbits)
    result = backend.execute_circuit(circuit, nshots=20 * nbits)
    frequencies = result.frequencies()

    # all the outcomes are orthogonal to the hidden string
    for outcome in frequencies:
        assert sum(int(a) * int(b) for a, b in zip(outcome, string)) % 2 == 0

    assert simon_secret_string(result, nbits) == string
    assert simon_secret_string(frequencies) == string


def test_simon_errors():
    with pytest.raises(TypeError):
        Simon(gates.CNOT(0, 1))

    with pytest.raises(ValueError):
        Simon(Circuit(1))

    with pytest.raises(ValueError):
        Simon(Circuit(3))

    oracle = _oracle("11")
    oracle.add(gates.M(0))
    with pytest.raises(ValueError):
        Simon(oracle)


@pytest.mark.parametrize("measure", [False, True])
def test_simon_measure(measure):
    circuit = Simon(_oracle("101"), measure=measure, density_matrix=True)
    assert circuit.density_matrix
    assert len(circuit.measurements) == int(measure)
    assert circuit.nqubits == 6


def test_simon_one_to_one(backend):
    # the function is the identity, so the hidden string is zero
    circuit = Simon(_oracle("000"))
    result = backend.execute_circuit(circuit, nshots=100)
    assert simon_secret_string(result) == "000"


@pytest.mark.parametrize("nbits", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simon_random_oracle(backend, nbits, seed):
    backend.set_seed(100 * nbits + seed)

    # nonzero random hidden string
    string = "0" * nbits
    while "1" not in string:
        string = "".join(
            str(int(bit)) for bit in backend.random_integers(2, size=nbits)
        )

    # two-to-one function f'(x) = T f(x) + b, where f is the function of `_oracle`,
    # T is a random invertible linear map (CNOT network) acting on the output qubits and
    # b is a random bitstring (X gates); then f'(x) = f'(x XOR s) iff f(x) = f(x XOR s)
    oracle = _oracle(string)
    for _ in range(3 * nbits):
        control, target = (
            int(qubit) for qubit in backend.random_choice(nbits, size=2, replace=False)
        )
        oracle.add(gates.CNOT(nbits + control, nbits + target))
    oracle.add(
        gates.X(nbits + qubit)
        for qubit, bit in enumerate(backend.random_integers(2, size=nbits))
        if bit
    )

    # f'(x) = f'(y) if and only if y in {x, x XOR s}, where the qubit 0 is the most
    # significant bit of the integers
    secret = int(string, 2)
    values = []
    for x in range(2**nbits):
        circuit = Circuit(2 * nbits)
        circuit.add(
            gates.X(qubit) for qubit in range(nbits) if (x >> (nbits - 1 - qubit)) & 1
        )
        circuit.add(oracle)
        state = backend.execute_circuit(circuit).state()
        values.append(int(backend.argmax(backend.abs(state))) % 2**nbits)
    assert all(
        (values[x] == values[y]) == (y in (x, x ^ secret))
        for x in range(2**nbits)
        for y in range(2**nbits)
    )

    # seeded executions are reproducible
    outputs = []
    for _ in range(2):
        backend.set_seed(1234)
        result = backend.execute_circuit(Simon(oracle), nshots=30 * nbits)
        outputs.append((result.frequencies(), simon_secret_string(result, nbits)))
    assert outputs[0] == outputs[1]

    frequencies, secret_string = outputs[0]
    for outcome in frequencies:
        assert sum(int(a) * int(b) for a, b in zip(outcome, string)) % 2 == 0
    assert secret_string == string


def test_simon_secret_string_errors():
    with pytest.raises(TypeError):
        simon_secret_string("01")

    with pytest.raises(ValueError):
        simon_secret_string([])

    with pytest.raises(ValueError):
        simon_secret_string(["01", "2"])

    with pytest.raises(ValueError):
        simon_secret_string(["01", "011"])

    with pytest.raises(ValueError):
        simon_secret_string(["01"], nbits=3)

    with pytest.raises(ValueError):
        simon_secret_string(["01"], nbits=0)

    # only one independent equation for three bits: the secret is not unique
    with pytest.raises(ValueError):
        simon_secret_string(["001", "001", "000"])


def test_simon_secret_string_inputs():
    expected = "110"
    # y . s = 0 for s = 110: 001, 110, 111, 000
    assert simon_secret_string(["001", "110"]) == expected
    assert simon_secret_string([[0, 0, 1], (1, 1, 0)]) == expected
    assert simon_secret_string({"001": 4, "110": 1, "000": 2}) == expected
    assert simon_secret_string(["001", "110", "111", "000"], nbits=3) == expected
    # the second bitstring is eliminated from the first one (back substitution)
    assert simon_secret_string(["110", "010"]) == "001"
    assert simon_secret_string(["1"]) == "0"
    assert simon_secret_string(["0", "0"]) == "1"


def _oracle(string):
    # oracle of f(x) = x XOR (x_j * s), where j is the first nonzero bit of s;
    # it is one-to-one if s = 0, and two-to-one with f(x) = f(x XOR s) otherwise
    nbits = len(string)
    oracle = Circuit(2 * nbits)
    oracle.add(gates.CNOT(qubit, qubit + nbits) for qubit in range(nbits))
    if "1" in string:
        first = string.index("1")
        oracle.add(
            gates.CNOT(first, nbits + qubit)
            for qubit, bit in enumerate(string)
            if bit == "1"
        )
    return oracle
