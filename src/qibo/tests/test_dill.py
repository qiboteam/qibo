import pytest
import dill
import numpy as np


def test_dill_backends(backend):
    serial = dill.dumps(backend)
    new_backend = dill.loads(serial)
    assert type(new_backend) == type(backend)
    assert new_backend.name == backend.name
    assert new_backend.platform == backend.platform
    assert type(new_backend.matrices) == type(backend.matrices)


def test_dill_global_backend():
    from qibo.backends import GlobalBackend
    backend = GlobalBackend()
    serial = dill.dumps(backend)
    new_backend = dill.loads(serial)
    assert type(new_backend) == type(backend)
    assert new_backend.name == backend.name


def test_dill_circuit(accelerators):
    from qibo import gates
    from qibo.models import Circuit
    circuit = Circuit(5, accelerators=accelerators)
    circuit.add(gates.H(i) for i in range(5))
    serial = dill.dumps(circuit)
    new_circuit = dill.loads(serial)
    assert type(new_circuit) == type(circuit)
    assert new_circuit.nqubits == circuit.nqubits
    assert new_circuit.to_qasm() == circuit.to_qasm()


def test_dill_circuit_result(backend):
    from qibo.models import QFT
    circuit = QFT(4)
    result = backend.execute_circuit(circuit)
    serial = dill.dumps(result)
    new_result = dill.loads(serial)
    assert type(new_result) == type(result)
    assert str(new_result) == str(result)
    backend.assert_allclose(new_result.state(), result.state())


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


def test_dill_measurement_symbol(backend):
    from qibo import gates
    from qibo.models import Circuit
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    symbol = circuit.add(gates.M(0, collapse=True))
    result = backend.execute_circuit(circuit, nshots=1)
    nsymbol = dill.loads(dill.dumps(symbol))
    assert type(nsymbol) == type(symbol)
    backend.assert_allclose(nsymbol.outcome(), symbol.outcome())


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


def test_dill_symbolic_hamiltonian(backend):
    from qibo.symbols import X, Y, Z
    from qibo.hamiltonians import SymbolicHamiltonian
    form = X(0) * X(1) + Y(0) * Y(1) + Z(0) * Z(1)
    ham = SymbolicHamiltonian(form, backend=backend)
    serial = dill.dumps(ham)
    nham = dill.loads(serial)
    assert type(nham) == type(ham)
    backend.assert_allclose(nham.matrix, ham.matrix)


def test_dill_variational_models(backend):
    from qibo import gates
    from qibo.hamiltonians import TFIM
    from qibo.models import Circuit, VQE, QAOA
    ham = TFIM(4, backend=backend)
    circuit = Circuit(4)
    circuit.add(gates.RX(i, theta=0) for i in range(4))
    vqe = VQE(circuit, ham)
    qaoa = QAOA(ham)
    nvqe = dill.loads(dill.dumps(vqe))
    nqaoa = dill.loads(dill.dumps(qaoa))
    assert type(nvqe) == type(vqe)
    assert type(nqaoa) == type(qaoa)
