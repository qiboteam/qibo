import pytest

from qibo import Circuit, gates
from qibo.backends import CliffordBackend
from qibo.gates.special import Barrier, remove_barriers


def _bell_circuit(with_barriers: bool, density_matrix: bool = False) -> Circuit:
    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.H(0))
    if with_barriers:
        circuit.add(Barrier(0, 1))
    circuit.add(gates.CNOT(0, 1))
    if with_barriers:
        circuit.add(Barrier(*range(circuit.nqubits)))
    return circuit


@pytest.mark.parametrize("density_matrix", [False, True])
def test_barrier_does_not_change_state(backend, density_matrix):
    with_barriers = backend.execute_circuit(_bell_circuit(True, density_matrix))
    without_barriers = backend.execute_circuit(_bell_circuit(False, density_matrix))
    backend.assert_allclose(with_barriers.state(), without_barriers.state())


def test_barrier_unitary_ignores_barrier(backend):
    backend.assert_allclose(
        _bell_circuit(True).unitary(backend), _bell_circuit(False).unitary(backend)
    )


def test_barrier_before_measurement_keeps_measurement():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(Barrier(0, 1))
    circuit.add(gates.M(0, 1))
    assert not circuit.has_collapse
    assert len(circuit.measurements) == 1


def test_barrier_after_measurement_makes_it_collapse():
    circuit = Circuit(2)
    circuit.add(gates.M(0, 1))
    circuit.add(Barrier(0, 1))
    assert circuit.has_collapse
    assert len(circuit.measurements) == 0


def test_barrier_clifford_backend():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(Barrier(0, 1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1))
    frequencies = (
        CliffordBackend(platform="numpy")
        .execute_circuit(circuit, nshots=100)
        .frequencies()
    )
    assert set(frequencies) <= {"00", "11"}


def test_barrier_draw():
    expected = "0: ─H─░─o─░─\n1: ───░─X─░─\n2: ───────░─"
    assert _bell_circuit(True).diagram() == expected


def test_barrier_qasm_export():
    qasm = _bell_circuit(True).to_qasm()
    assert "barrier q[0],q[1];" in qasm
    assert "barrier q[0],q[1],q[2];" in qasm
    assert qasm.index("h q[0];") < qasm.index("barrier q[0],q[1];")


def test_barrier_invert_reverses_order():
    inverted = _bell_circuit(True).invert()
    assert [gate.name for gate in inverted.queue] == ["barrier", "cx", "barrier", "h"]
    assert inverted.queue[0].qubits == (0, 1, 2)


def test_barrier_blocks_fusion():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(Barrier(0))
    circuit.add(gates.H(0))
    assert [gate.name for gate in circuit.fuse().queue] == ["h", "barrier", "h"]

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    assert len(circuit.fuse().queue) == 1


def test_barrier_never_commutes():
    assert not Barrier(0).commutes(gates.X(1))
    assert not gates.X(1).commutes(Barrier(0))


def test_barrier_on_qubits():
    barrier = Barrier(0, 1).on_qubits({0: 3, 1: 2})
    assert isinstance(barrier, Barrier)
    assert barrier.qubits == (3, 2)

    subroutine = Circuit(2)
    subroutine.add(gates.H(0))
    subroutine.add(Barrier(0, 1))
    circuit = Circuit(4)
    circuit.add(subroutine.on_qubits(2, 3))
    assert circuit.queue[1].qubits == (2, 3)


def test_barrier_raw():
    raw = Barrier(0, 2).raw
    assert raw["_class"] == "Barrier"
    assert raw["init_args"] == [0, 2]


def test_barrier_errors():
    with pytest.raises(ValueError):
        Barrier()
    with pytest.raises(ValueError):
        Barrier(1, 1)
    with pytest.raises(ValueError):
        Circuit(2).add(Barrier(0, 5))
    with pytest.raises(KeyError):
        Barrier(0, 1).on_qubits({0: 2})


def test_remove_barriers():
    circuit = _bell_circuit(True)
    circuit.add(gates.M(0, 1))
    cleaned = remove_barriers(circuit)
    assert [gate.name for gate in cleaned.queue] == ["h", "cx", "measure"]
    assert len(cleaned.measurements) == 1
    # the input circuit is left untouched
    assert circuit.gate_names["barrier"] == 2


def test_remove_barriers_removes_barrier_class_defined_in_qibo():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.Barrier(0, 1))
    circuit.add(Barrier(0, 1))
    assert [gate.name for gate in remove_barriers(circuit).queue] == ["h"]
