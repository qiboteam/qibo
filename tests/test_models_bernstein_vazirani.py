"""Tests for the Bernstein-Vazirani algorithm defined in `qibo/models/bernstein_vazirani.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models import BernsteinVazirani, DeutschJozsa


def _oracle(string, bias):
    # oracle of f(x) = s . x + bias modulo 2, with the output qubit as the last one
    nbits = len(string)
    oracle = Circuit(nbits + 1)
    oracle.add(gates.CNOT(qubit, nbits) for qubit, bit in enumerate(string) if bit)
    if bias:
        oracle.add(gates.X(nbits))
    return oracle


@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("nbits", [1, 2, 4, 6])
def test_bernstein_vazirani(backend, nbits, bias):
    backend.set_seed(nbits)
    string = [int(bit) for bit in backend.random_integers(2, size=nbits)]

    circuit = BernsteinVazirani(_oracle(string, bias))
    assert circuit.nqubits == nbits + 1

    frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
    assert frequencies == {"".join(map(str, string)): 20}


@pytest.mark.parametrize("measure", [False, True])
def test_bernstein_vazirani_measure(measure):
    oracle = _oracle([1, 0, 1], 0)

    circuit = BernsteinVazirani(oracle, measure=measure, density_matrix=True)
    assert circuit.density_matrix
    assert len(circuit.measurements) == int(measure)

    # the circuit is the one of the Deutsch-Jozsa algorithm
    reference = DeutschJozsa(oracle, measure=measure, density_matrix=True)
    assert circuit.raw == reference.raw


def test_bernstein_vazirani_errors():
    with pytest.raises(TypeError):
        BernsteinVazirani(gates.CNOT(0, 1))

    with pytest.raises(ValueError):
        BernsteinVazirani(Circuit(1))

    oracle = _oracle([1, 0, 1], 0)
    oracle.add(gates.M(0))
    with pytest.raises(ValueError):
        BernsteinVazirani(oracle)
