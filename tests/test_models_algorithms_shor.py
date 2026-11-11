"""Tests for Shor's algorithm in `qibo/models/algorithms/shor.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models import Shor
from qibo.models.algorithms.shor import _modular_exponentiation


# bases that are coprime to the number, so that the order is found with the circuit, and for which
# the order is even and the power of the base to half of it is not -1 modulo the number
@pytest.mark.parametrize(
    "nvalue,base",
    [
        (15, 7),
        (15, 2),
        (15, 13),
        (15, 4),
        (21, 2),
        (21, 8),
        (33, 5),
        (33, 10),
        (35, 3),
        (35, 6),
    ],
)
def test_shor_base(backend, nvalue, base):
    backend.set_seed(nvalue + base)
    factors = Shor(nvalue, base=base).execute(ntrials=40, backend=backend)

    assert 1 < factors[0] <= factors[1] < nvalue
    assert factors[0] * factors[1] == nvalue


@pytest.mark.parametrize("nvalue", [15, 21])
def test_shor_random_bases(backend, nvalue):
    backend.set_seed(nvalue)
    factors = Shor(nvalue, seed=nvalue).execute(ntrials=40, backend=backend)

    assert factors[0] * factors[1] == nvalue
    assert 1 < factors[0] <= factors[1]


def test_shor_shortcuts(backend):
    # the base shares a divisor with the number, the number is even, or it is a prime power
    assert Shor(15, base=3).execute(ntrials=1, backend=backend) == (3, 5)
    assert Shor(10).execute(backend=backend) == (2, 5)
    for nvalue, factors in ((9, (3, 3)), (25, (5, 5)), (27, (3, 9)), (125, (5, 25))):
        assert Shor(nvalue).execute(backend=backend) == factors


def test_shor_failures(backend):
    # the order of 14 is 2, and 14 is -1 modulo 15, so that the factors are not found. The
    # measured outcomes are 0 and q / 2, and the first one does not give the order
    backend.set_seed(0)
    with pytest.raises(RuntimeError):
        Shor(15, base=14).execute(ntrials=30, backend=backend)

    # the order of 4 modulo 21 is odd
    backend.set_seed(1)
    with pytest.raises(RuntimeError):
        Shor(21, base=4).execute(ntrials=5, backend=backend)

    # a prime number does not have factors
    backend.set_seed(2)
    with pytest.raises(RuntimeError):
        Shor(13, seed=0).execute(ntrials=3, backend=backend)


@pytest.mark.parametrize("base", [7, 2])
def test_modular_exponentiation(backend, base):
    nvalue, nbits, ncounting = 15, 4, 3

    for value in range(2**ncounting):
        circuit = Circuit(ncounting + nbits)
        bits = format(value, f"0{ncounting}b")
        circuit.add(gates.X(qubit) for qubit, bit in enumerate(bits) if bit == "1")
        circuit.add(_modular_exponentiation(base, nvalue, nbits, ncounting))
        circuit.add(gates.M(*range(ncounting, ncounting + nbits)))

        frequencies = backend.execute_circuit(circuit, nshots=1).frequencies(
            binary=True
        )
        assert int(next(iter(frequencies)), 2) == pow(base, value, nvalue)


def test_shor_errors():
    for nvalue in (3, 0, -15, 15.5, True, "15"):
        with pytest.raises(ValueError):
            Shor(nvalue)

    for base in (1, 0, 15, 20, 2.5, True):
        with pytest.raises(ValueError):
            Shor(15, base=base)

    for ntrials in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            Shor(15).execute(ntrials=ntrials)


def test_shor_circuit(backend):
    shor = Shor(15, base=7)

    # the circuit is stored when it is executed, and it is the one of the last trial
    assert shor.circuit is None
    assert (shor.ncounting, shor.nbits) == (8, 4)

    backend.set_seed(0)
    assert shor.execute(ntrials=40, backend=backend) == (3, 5)
    circuit = shor.circuit
    assert circuit.nqubits == shor.ncounting + shor.nbits
    assert len(circuit.measurements) == 1
    assert circuit.measurements[0].qubits == tuple(range(shor.ncounting))

    # it is replaced by the circuit of the next execution
    assert shor.execute(ntrials=40, backend=backend) == (3, 5)
    assert shor.circuit is not circuit


def test_shor_no_circuit(backend):
    # the circuit is not needed, so that it is erased
    shor = Shor(15, base=7)
    backend.set_seed(1)
    shor.execute(ntrials=40, backend=backend)
    assert shor.circuit is not None

    shor.base = 3
    assert shor.execute(ntrials=40, backend=backend) == (3, 5)
    assert shor.circuit is None


def test_shor_seed(backend):
    # a generator of the backend, which continues in each execution, and an integer seed, which
    # starts again in each of them
    backend.set_seed(7)
    shor = Shor(21, seed=backend.default_rng(0))
    assert shor.execute(ntrials=40, backend=backend) == (3, 7)

    shor = Shor(21, seed=3)
    assert shor.seed == 3
    assert shor.execute(ntrials=40, backend=backend) == (3, 7)
    assert shor.execute(ntrials=40, backend=backend) == (3, 7)
