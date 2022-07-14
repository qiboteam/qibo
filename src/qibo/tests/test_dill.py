import os
import pytest
import dill
import numpy as np


@pytest.mark.skip
def test_dill_circuit():
    from qibo import gates
    from qibo.models import Circuit
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    serial = dill.dumps(circuit)
    new_circuit = dill.loads(serial)
    assert type(new_circuit) == type(circuit)
    assert new_circuit.nqubits == circuit.nqubits
    assert new_circuit.to_qasm() == circuit.to_qasm()


@pytest.mark.skip
def test_dill_circuit_result(backend):
    from qibo.models import QFT
    circuit = QFT(4)
    result = backend.execute_circuit(circuit)
    serial = dill.dumps(result)
    new_result = dill.loads(serial)
    assert type(new_result) == type(result)
    assert str(new_result) == str(result)
    backend.assert_allclose(new_result.state(), result.state())


def test_dill_backend():
    from qibo.backends import GlobalBackend
    backend = GlobalBackend()
    serial = dill.dumps(backend)
    new_backend = dill.loads(serial)
    assert type(new_backend) == type(backend)
    assert new_backend.name == backend.name


def test_dill_hamiltonian(backend):
    from qibo.hamiltonians import XXZ, Hamiltonian
    matrix = np.random.random((4, 4))
    ham1 = Hamiltonian(2, matrix, backend=backend)
    ham2 = XXZ(3, backend=backend)
    serial1 = dill.dumps(ham1)
    serial2 = dill.dumps(ham2)
    nham1 = dill.loads(serial1)
    nham2 = dill.loads(serial2)
    assert type(nham1) == type(ham1)
    assert type(nham2) == type(ham2)
    backend.assert_allclose(nham1.matrix, nham1.matrix)
    backend.assert_allclose(nham2.matrix, nham2.matrix)


def test_dill_symbols():
    from qibo.symbols import Symbol, X
    matrix = np.random.random((2, 2))
    s = Symbol(0, matrix)
    x = X(1)
    ns = dill.loads(dill.dumps(s))
    nx = dill.loads(dill.dumps(x))
    assert type(ns) == type(s)
    assert type(nx) == type(x)
    np.testing.assert_allclose(nx.matrix, x.matrix)
    np.testing.assert_allclose(ns.matrix, s.matrix)
