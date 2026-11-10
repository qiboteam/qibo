"""Tests for the Deutsch-Jozsa algorithm defined in `qibo/models/algorithms/deutsch_jozsa.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models import DeutschJozsa


def _constant_zero(nqubits):
    return Circuit(nqubits)


def _constant_one(nqubits):
    oracle = Circuit(nqubits)
    oracle.add(gates.X(nqubits - 1))
    return oracle


def _first_bit(nqubits):
    oracle = Circuit(nqubits)
    oracle.add(gates.CNOT(0, nqubits - 1))
    return oracle


def _not_first_bit(nqubits):
    oracle = _first_bit(nqubits)
    oracle.add(gates.X(nqubits - 1))
    return oracle


def _parity(nqubits):
    oracle = Circuit(nqubits)
    oracle.add(gates.CNOT(qubit, nqubits - 1) for qubit in range(nqubits - 1))
    return oracle


def _nonlinear(nqubits):
    # f(x) = x_0 + x_1 x_2 modulo 2, which is balanced
    oracle = Circuit(nqubits)
    oracle.add(gates.CNOT(0, nqubits - 1))
    oracle.add(gates.TOFFOLI(1, 2, nqubits - 1))
    return oracle


@pytest.mark.parametrize("nqubits", [2, 3, 6])
@pytest.mark.parametrize("build_oracle", [_constant_zero, _constant_one])
def test_deutsch_jozsa_constant(backend, build_oracle, nqubits):
    circuit = DeutschJozsa(build_oracle(nqubits), measure=False)
    assert circuit.nqubits == nqubits

    probabilities = backend.execute_circuit(circuit).probabilities(
        list(range(nqubits - 1))
    )
    backend.assert_allclose(probabilities[0], 1.0, atol=1e-10)


@pytest.mark.parametrize("nqubits", [2, 4, 6])
@pytest.mark.parametrize("build_oracle", [_first_bit, _not_first_bit, _parity])
def test_deutsch_jozsa_balanced(backend, build_oracle, nqubits):
    circuit = DeutschJozsa(build_oracle(nqubits), measure=False)

    probabilities = backend.execute_circuit(circuit).probabilities(
        list(range(nqubits - 1))
    )
    backend.assert_allclose(probabilities[0], 0.0, atol=1e-10)


@pytest.mark.parametrize("nqubits", [4, 6])
def test_deutsch_jozsa_balanced_nonlinear(backend, nqubits):
    circuit = DeutschJozsa(_nonlinear(nqubits), measure=False)

    probabilities = backend.execute_circuit(circuit).probabilities(
        list(range(nqubits - 1))
    )
    backend.assert_allclose(probabilities[0], 0.0, atol=1e-10)


@pytest.mark.parametrize("measure", [False, True])
def test_deutsch_jozsa_measure(backend, measure):
    circuit = DeutschJozsa(_first_bit(4), measure=measure, density_matrix=True)
    assert circuit.density_matrix
    assert len(circuit.measurements) == int(measure)

    if measure:
        frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
        assert frequencies == {"100": 20}


def test_deutsch_jozsa_errors():
    with pytest.raises(TypeError):
        DeutschJozsa(gates.CNOT(0, 1))

    with pytest.raises(ValueError):
        DeutschJozsa(Circuit(1))

    oracle = _first_bit(3)
    oracle.add(gates.M(0))
    with pytest.raises(ValueError):
        DeutschJozsa(oracle)
