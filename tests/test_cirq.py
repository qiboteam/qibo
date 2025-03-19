"""Test that Qibo gate execution agrees with Cirq."""

# import cirq
import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.backends import NumpyBackend
from qibo.models import QFT
from qibo.quantum_info import random_statevector, random_unitary

numpy_backend = NumpyBackend()

pytest.skip(allow_module_level=True)


def random_active_qubits(nqubits, nmin=None, nactive=None):
    """Generates random list of target and control qubits."""
    all_qubits = np.arange(nqubits)
    np.random.shuffle(all_qubits)
    if nactive is None:
        nactive = np.random.randint(nmin + 1, nqubits)
    return list(all_qubits[:nactive])


def execute_cirq(cirq_gates, nqubits, initial_state=None):
    """Executes a Cirq circuit with the given list of gates."""
    c = cirq.Circuit()
    q = [cirq.LineQubit(i) for i in range(nqubits)]
    # apply identity gates to all qubits so that they become part of the circuit
    c.append([cirq.I(qi) for qi in q])
    for gate, targets in cirq_gates:
        c.append(gate(*[q[i] for i in targets]))
    result = cirq.Simulator().simulate(
        c, initial_state=initial_state
    )  # pylint: disable=no-member
    depth = len(cirq.Circuit(c.all_operations()))
    return result.final_state_vector, depth - 1


def assert_gates_equivalent(
    backend, qibo_gate, cirq_gates, nqubits, ndevices=None, atol=1e-7
):
    """Asserts that QIBO and Cirq gates have equivalent action on a random
    state.

    Args:
        qibo_gate: QIBO gate.
        cirq_gates: List of tuples (cirq gate, target qubit IDs).
        nqubits: Total number of qubits in the circuit.
        atol: Absolute tolerance in state vector comparsion.
    """
    initial_state = random_statevector(2**nqubits, backend=numpy_backend)
    copy = numpy_backend.cast(initial_state, copy=True)
    target_state, target_depth = execute_cirq(cirq_gates, nqubits, copy)
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    accelerators = None
    if ndevices is not None:
        accelerators = {"/GPU:0": ndevices}

    c = Circuit(nqubits, accelerators)
    c.add(qibo_gate)
    assert c.depth == target_depth
    if accelerators and not backend.supports_multigpu:
        with pytest.raises(NotImplementedError):
            final_state = backend.execute_circuit(
                c, backend.cast(initial_state, copy=True)
            ).state()
    else:
        final_state = backend.execute_circuit(
            c, backend.cast(initial_state, copy=True)
        ).state()
        backend.assert_allclose(final_state, target_state, atol=atol)


def assert_cirq_gates_equivalent(qibo_gate, cirq_gate):
    """Asserts that qibo gate is equivalent to cirq gate.

    Checks that:
        * Gate type agrees.
        * Target and control qubits agree.
        * Parameter (if applicable) agrees.
    Cirq gate parameters are extracted by parsing the gate string.
    """
    import re

    # Fix for cirq >= 0.15.0
    chars = iter(str(cirq_gate))
    clean_gate = []
    open = False
    for char in chars:
        if char == "q":
            open = True
            next(chars)
        elif open and char == ")":
            open = False
        else:
            clean_gate.append(char)

    pieces = [x for x in re.split("[()]", "".join(clean_gate)) if x]
    if len(pieces) == 2:
        gatename, targets = pieces
        theta = None
    elif len(pieces) == 3:
        gatename, theta, targets = pieces
    else:  # pragma: no cover
        # case not tested because it fails
        raise RuntimeError(f"Cirq gate parsing failed with {pieces}.")

    qubits = list(int(x) for x in targets.replace(" ", "").split(","))
    targets = (qubits.pop(),)
    controls = set(qubits)

    qibo_to_cirq = {"CNOT": "CNOT", "RY": "Ry", "TOFFOLI": "TOFFOLI"}
    assert qibo_to_cirq[qibo_gate.__class__.__name__] == gatename
    assert qibo_gate.target_qubits == targets
    assert set(qibo_gate.control_qubits) == controls
    if theta is not None:
        if "π" in theta:
            theta = np.pi * float(theta.replace("π", ""))
        else:  # pragma: no cover
            # case doesn't happen in tests (could remove)
            theta = float(theta)
        np.testing.assert_allclose(theta, qibo_gate.parameters)


@pytest.mark.parametrize(
    ("target", "controls", "free"),
    [
        (0, (1,), ()),
        (2, (0, 1), ()),
        (3, (0, 1, 4), (2, 5)),
        (7, (0, 1, 2, 3, 4), (5, 6)),
    ],
)
def test_x_decompose_with_cirq(target, controls, free):
    """Check that decomposition of multi-control ``X`` agrees with Cirq."""
    gate = gates.X(target).controlled_by(*controls)
    qibo_decomp = gate.decompose(*free, use_toffolis=False)

    # Calculate the decomposition using Cirq.
    nqubits = max((target,) + controls + free) + 1
    qubits = [cirq.LineQubit(i) for i in range(nqubits)]
    controls = [qubits[i] for i in controls]
    free = [qubits[i] for i in free]
    cirq_decomp = cirq.decompose_multi_controlled_x(controls, qubits[target], free)
    assert len(qibo_decomp) == len(cirq_decomp)
    for qibo_gate, cirq_gate in zip(qibo_decomp, cirq_decomp):
        assert_cirq_gates_equivalent(qibo_gate, cirq_gate)


@pytest.mark.parametrize(
    ("gate_name", "nqubits", "ndevices"),
    [
        ("H", 3, None),
        ("H", 3, 2),
        ("X", 2, None),
        ("X", 2, 2),
        ("Y", 1, None),
        ("Z", 1, None),
        ("S", 4, None),
        ("T", 4, None),
    ],
)
def test_one_qubit_gates(backend, gate_name, nqubits, ndevices):
    """Check simple one-qubit gates."""
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(
    ("gate_name", "nqubits", "ndevices"),
    [("RX", 3, None), ("RX", 3, 4), ("RY", 2, None), ("RY", 2, 2), ("RZ", 1, None)],
)
def test_one_qubit_parametrized_gates(backend, gate_name, nqubits, ndevices):
    """Check parametrized one-qubit rotations."""
    theta = 0.1234
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets, theta)
    cirq_gate = [(getattr(cirq, gate_name.lower())(theta), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(("nqubits", "ndevices"), [(2, None), (3, 4), (2, 2)])
def test_u1_gate(backend, nqubits, ndevices):
    """Check U1 gate."""
    theta = 0.1234
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = gates.U1(*targets, theta)
    cirq_gate = [(cirq.ZPowGate(exponent=theta / np.pi), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("gate_name", ["CNOT", "SWAP", "CZ"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("ndevices", [None, 2])
def test_two_qubit_gates(backend, gate_name, nqubits, ndevices):
    """Check two-qubit gates."""
    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = getattr(gates, gate_name)(*targets)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(
    ("nqubits", "ndevices"), [(2, None), (6, None), (6, 2), (7, None), (7, 4)]
)
def test_two_qubit_parametrized_gates(backend, nqubits, ndevices):
    """Check ``CU1`` and ``fSim`` gate."""
    theta = 0.1234
    phi = 0.4321

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.CU1(*targets, np.pi * theta)
    cirq_gate = [(cirq.CZPowGate(exponent=theta), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits)

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.fSim(*targets, theta, phi)
    cirq_gate = [(cirq.FSimGate(theta=theta, phi=phi), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(
    ("nqubits", "ndevices"), [(5, None), (6, None), (9, None), (5, 2), (6, 4), (9, 8)]
)
def test_unitary_matrix_gate(backend, nqubits, ndevices):
    """Check arbitrary unitary gate."""
    matrix = random_unitary(2**1, backend=numpy_backend)
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = gates.Unitary(matrix, *targets)
    cirq_gate = [(cirq.MatrixGate(matrix), targets)]
    assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits)

    for _ in range(10):
        matrix = random_unitary(2**2, backend=numpy_backend)
        targets = random_active_qubits(nqubits, nactive=2)
        qibo_gate = gates.Unitary(matrix, *targets)
        cirq_gate = [(cirq.MatrixGate(matrix), targets)]
        assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(
    ("gate_name", "nqubits", "ndevices"),
    [
        ("H", 3, None),
        ("Z", 4, None),
        ("Y", 5, 4),
        ("X", 6, None),
        ("H", 7, 2),
        ("Z", 8, 8),
        ("Y", 12, 16),
        ("S", 13, 4),
        ("T", 9, 2),
    ],
)
def test_one_qubit_gates_controlled_by(backend, gate_name, nqubits, ndevices):
    """Check one-qubit gates controlled on arbitrary number of qubits."""
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nmin=1)
        qibo_gate = getattr(gates, gate_name)(activeq[-1]).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name).controlled(len(activeq) - 1), activeq)]
        assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(
    ("nqubits", "ndevices"),
    [
        (4, None),
        (5, None),
        (8, None),
        (12, None),
        (15, None),
        (17, None),
        (6, 2),
        (9, 2),
        (11, 4),
        (13, 8),
        (14, 16),
    ],
)
def test_two_qubit_gates_controlled_by(backend, nqubits, ndevices):
    """Check ``SWAP`` and ``fSim`` gates controlled on arbitrary number of
    qubits."""
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nmin=2)
        qibo_gate = gates.SWAP(*activeq[-2:]).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.SWAP.controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)

        theta = np.random.random()
        phi = np.random.random()
        qibo_gate = gates.fSim(*activeq[-2:], theta, phi).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.FSimGate(theta, phi).controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("nqubits", [5, 12, 13, 14])
@pytest.mark.parametrize("ntargets", [1, 2])
@pytest.mark.parametrize("ndevices", [None, 2, 8])
def test_unitary_matrix_gate_controlled_by(backend, nqubits, ntargets, ndevices):
    """Check arbitrary unitary gate controlled on arbitrary number of
    qubits."""
    for _ in range(10):
        activeq = random_active_qubits(nqubits, nactive=5)
        matrix = random_unitary(2**ntargets, backend=numpy_backend)
        qibo_gate = gates.Unitary(matrix, *activeq[-ntargets:]).controlled_by(
            *activeq[:-ntargets]
        )
        cirq_gate = [
            (cirq.MatrixGate(matrix).controlled(len(activeq) - ntargets), activeq)
        ]
        assert_gates_equivalent(backend, qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("nqubits", [5, 6, 7, 11, 12])
def test_qft(backend, accelerators, nqubits):
    c = QFT(nqubits, accelerators=accelerators)
    initial_state = random_statevector(2**nqubits, backend=numpy_backend)
    final_state = backend.execute_circuit(c, np.copy(initial_state)).state()
    final_state = backend.cast(final_state, dtype=final_state.dtype)
    cirq_gates = [(cirq.qft, list(range(nqubits)))]
    target_state, _ = execute_cirq(cirq_gates, nqubits, np.copy(initial_state))
    target_state = backend.cast(target_state, dtype=target_state.dtype)
    backend.assert_allclose(target_state, final_state, atol=1e-6)
