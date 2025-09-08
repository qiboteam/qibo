import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.quantum_info import random_density_matrix


def test_eager_execute(backend, accelerators):
    circuit = Circuit(4, accelerators)
    circuit.add(gates.H(i) for i in range(4))
    final_state = backend.execute_circuit(circuit)._state
    target_state = np.ones(16) / 4.0
    backend.assert_allclose(final_state, target_state)


def test_compiled_execute(backend):
    def create_circuit(theta=0.1234):
        circuit = Circuit(2)
        circuit.add(gates.X(0))
        circuit.add(gates.X(1))
        circuit.add(gates.CU1(0, 1, theta))
        return circuit

    # Try to compile circuit without gates
    empty_circuit = Circuit(2)
    with pytest.raises(RuntimeError):
        empty_circuit.compile()

    # Run eager circuit
    circuit_1 = create_circuit()
    r1 = backend.execute_circuit(circuit_1).state()

    # Run compiled circuit
    circuit_2 = create_circuit()
    circuit_2.compile(backend)
    r2 = circuit_2().state()
    np.testing.assert_allclose(backend.to_numpy(r1), backend.to_numpy(r2))


def test_compiling_twice_exception(backend):
    """Check that compiling a circuit a second time raises error."""
    circuit = Circuit(2)
    circuit.add([gates.H(0), gates.H(1)])
    circuit.compile()
    with pytest.raises(RuntimeError):
        circuit.compile()


@pytest.mark.linux
def test_memory_error(backend, accelerators):
    """Check that ``RuntimeError`` is raised if device runs out of memory."""
    circuit = Circuit(40, accelerators)
    circuit.add(gates.H(i) for i in range(0, 40, 5))
    with pytest.raises(RuntimeError):
        final_state = backend.execute_circuit(circuit)


def test_repeated_execute(backend, accelerators):
    if accelerators is not None:
        with pytest.raises(NotImplementedError):
            circuit = Circuit(4, accelerators, density_matrix=True)
    else:
        circuit = Circuit(4, accelerators, density_matrix=True)
        thetas = np.random.random(4)
        circuit.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        target_state = backend.execute_circuit(circuit).state()
        circuit.has_collapse = True
        if accelerators is not None:
            with pytest.raises(NotImplementedError):
                final_state = backend.execute_circuit(circuit, nshots=20)
        else:
            final_state = backend.execute_circuit(circuit, nshots=20).state()
            backend.assert_allclose(final_state, target_state)


def test_final_state_property(backend):
    """Check accessing final state using the circuit's property."""
    circuit = Circuit(2)
    circuit.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = circuit.final_state

    backend.execute_circuit(circuit)._state
    target_state = np.ones(4) / 2
    backend.assert_allclose(circuit.final_state, target_state)


def test_density_matrix_circuit(backend):
    initial_rho = random_density_matrix(2**3, backend=backend)

    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.H(2))
    final_rho = backend.execute_circuit(circuit, backend.copy(initial_rho)).state()

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    m1 = np.kron(np.kron(h, h), np.eye(2))
    m2 = np.kron(cnot, np.eye(2))
    m3 = np.kron(np.eye(4), h)

    target_rho = np.dot(
        m1, np.dot(backend.to_numpy(initial_rho), np.transpose(np.conj(m1)))
    )
    target_rho = np.dot(m2, np.dot(target_rho, np.transpose(np.conj(m2))))
    target_rho = np.dot(m3, np.dot(target_rho, np.transpose(np.conj(m3))))

    backend.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("density_matrix", [True, False])
def test_circuit_as_initial_state(backend, density_matrix):
    nqubits = 10
    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.X(i) for i in range(nqubits))

    circuit_1 = Circuit(nqubits, density_matrix=density_matrix)
    circuit_1.add(gates.H(i) for i in range(nqubits))

    actual_circuit = circuit_1 + circuit

    output = backend.execute_circuit(circuit, circuit_1).state()
    target = backend.execute_circuit(actual_circuit).state()

    backend.assert_allclose(target, output)


def test_initial_state_error(backend):
    nqubits = 10
    circuit = Circuit(nqubits)
    circuit.add(gates.X(i) for i in range(nqubits))

    circuit_1 = Circuit(nqubits, density_matrix=True)
    circuit_1.add(gates.H(i) for i in range(nqubits))

    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, circuit_1)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_initial_state_shape_error(backend, density_matrix):
    nqubits = 2
    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.X(i) for i in range(nqubits))

    initial_state = random_density_matrix(2, backend=backend)
    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, initial_state=initial_state)
