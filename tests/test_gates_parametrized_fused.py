"""Numerical and circuit-level contracts for trainable fused blocks."""

import copy
import json

import numpy as np
import pytest

from qibo import Circuit, callbacks, gates


def _unitary(angle):
    indices = np.arange(4)
    fourier = np.exp(0.5j * np.pi * np.outer(indices, indices)) / 2
    return np.diag(np.exp(1j * angle * indices)) @ fourier


def _parameters(theta=0.13, angles=(0.27, -0.31, 0.43), phase=0.22):
    return (theta, *angles, _unitary(phase))


def _leaves(parameters, controlled=False):
    theta, alpha, phi, lam, matrix = parameters
    leaves = [
        gates.RX(0, theta),
        gates.RY(2, -0.37, trainable=False),
        gates.CNOT(0, 2),
        gates.U3(2, alpha, phi, lam),
        gates.Unitary(matrix.copy(), 2, 0),
    ]
    if controlled:
        leaves = [
            leaf.controlled_by(3) if index != 2 else gates.TOFFOLI(3, 0, 2)
            for index, leaf in enumerate(leaves)
        ]
    return leaves


def _block(parameters=None, trainable=True):
    if parameters is None:
        parameters = _parameters()
    block = gates.ParametrizedFusedGate(0, 2, trainable=trainable)
    for leaf in _leaves(parameters):
        block.append(leaf)
    return block


def _circuit(middle, density_matrix=False, outside_angle=0.19):
    circuit = Circuit(4, density_matrix=density_matrix)
    circuit.add([gates.H(1), gates.CNOT(1, 2)])
    circuit.add(middle)
    outside = gates.RY(3, outside_angle)
    circuit.add([outside, gates.CZ(0, 3)])
    return circuit, outside


def _initial_state(nqubits, density_matrix):
    rng = np.random.default_rng(149)
    shape = (2**nqubits, 2**nqubits) if density_matrix else (2**nqubits,)
    state = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if density_matrix:
        state = state @ state.conj().T
        return state / np.trace(state)
    return state / np.linalg.norm(state)


def _assert_execution(backend, circuit, reference):
    initial = _initial_state(circuit.nqubits, circuit.density_matrix)
    actual = backend.execute_circuit(circuit, backend.cast(initial.copy())).state()
    expected = backend.execute_circuit(reference, backend.cast(initial.copy())).state()
    backend.assert_allclose(actual, expected, atol=1e-10)


def _assert_parameters(actual, expected):
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target, atol=1e-12)


def _flat(parameters):
    return [*parameters[:4], *parameters[4].ravel()]


def test_parametrized_fused_matrix_analytical(backend):
    """An explicit matrix product checks order, qubits, and parameter refresh."""
    block = gates.ParametrizedFusedGate(2, 0)
    block.append(gates.RX(0, 0.0))
    block.append(gates.CNOT(0, 2))
    block.append(gates.U3(2, 0.0, 0.0, 0.0))
    assert block.target_qubits == (0, 2)
    assert isinstance(block, gates.ParametrizedGate)
    assert block.nparams == 4

    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    for theta, alpha, phi, lam in [(0.31, -0.7, 0.5, 0.2), (-0.4, 0.9, -0.1, 0.6)]:
        block.parameters = (theta, alpha, phi, lam)
        rx = np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )
        u3 = np.array(
            [
                [np.cos(alpha / 2), -np.exp(1j * lam) * np.sin(alpha / 2)],
                [
                    np.exp(1j * phi) * np.sin(alpha / 2),
                    np.exp(1j * (phi + lam)) * np.cos(alpha / 2),
                ],
            ]
        )
        # Qibo uses the determinant-one convention for U3.
        u3 *= np.exp(-0.5j * (phi + lam))
        expected = np.kron(np.eye(2), u3) @ cx @ np.kron(rx, np.eye(2))
        backend.assert_allclose(block.matrix(backend), expected, atol=1e-12)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("parameter_format", ["list", "dict", "flatlist", "array"])
def test_parametrized_fused_circuit_updates(backend, density_matrix, parameter_format):
    block = _block()
    circuit, outside = _circuit(block, density_matrix)
    reference, _ = _circuit(_leaves(_parameters()), density_matrix)
    _assert_execution(backend, circuit, reference)
    assert circuit.trainable_gates.nparams == 21
    assert set(circuit.get_parameters("dict")) == {block, outside}

    for parameters, outside_angle in [
        (_parameters(0.81, (-0.43, 0.52, -0.21), 0.61), -0.26),
        (_parameters(-0.29, (0.73, -0.81, 0.15), -0.36), 0.47),
    ]:
        if parameter_format == "list":
            update = [parameters, outside_angle]
        elif parameter_format == "dict":
            update = {block: parameters, outside: outside_angle}
        else:
            update = [*_flat(parameters), outside_angle]
            if parameter_format == "array":
                update = np.asarray(update)
        circuit.set_parameters(update)

        # Fresh gates ensure a setter cannot also mutate the reference circuit.
        reference, _ = _circuit(_leaves(parameters), density_matrix, outside_angle)
        _assert_execution(backend, circuit, reference)
        _assert_parameters(block.parameters, parameters)
        np.testing.assert_allclose(
            circuit.get_parameters("flatlist"), [*_flat(parameters), outside_angle]
        )
        assert block.gates[1].parameters == (-0.37,)


def test_parametrized_fused_live_parameters_and_roundtrip(backend):
    block = _block()
    assert block.nparams == 20
    assert len(block.parameters) == 5
    _assert_parameters(block.parameters, _parameters())

    block.gates[0].parameters = -0.91
    block.gates[3].parameters = (0.14, 0.57, -0.63)
    block.gates[4].parameters = _unitary(0.74)
    expected = _parameters(-0.91, (0.14, 0.57, -0.63), 0.74)
    _assert_parameters(block.parameters, expected)

    restored = _block()
    restored.parameters = block.parameters
    backend.assert_allclose(restored.matrix(backend), block.matrix(backend), atol=1e-12)


@pytest.mark.parametrize("parameter_format", ["structured", "flat", "matrix"])
def test_parametrized_fused_single_matrix_parameter(backend, parameter_format):
    block = gates.ParametrizedFusedGate(0, 2)
    block.append(gates.Unitary(np.eye(4), 2, 0))
    expected = _unitary(0.62)
    if parameter_format == "structured":
        block.parameters = (expected,)
    elif parameter_format == "flat":
        block.parameters = expected.ravel()
    else:
        block.parameters = expected
    assert block.nparams == 16
    assert len(block.parameters) == 1
    np.testing.assert_allclose(block.parameters[0], expected)
    circuit = Circuit(3)
    circuit.add(block)
    reference = Circuit(3)
    reference.add(gates.Unitary(expected, 2, 0))
    _assert_execution(backend, circuit, reference)


@pytest.mark.parametrize("inner_trainable", [False, True])
def test_parametrized_fused_nested_masks_and_prepend(backend, inner_trainable):
    inner = gates.ParametrizedFusedGate(2, trainable=inner_trainable)
    inner.append(gates.U3(2, 0.21, -0.32, 0.43))
    inner.append(gates.RZ(2, -0.54, trainable=False))
    inner.append(gates.RY(2, 0.76))
    plain = gates.FusedGate(0, 2)
    plain.append(gates.CNOT(0, 2))
    plain.append(gates.Unitary(_unitary(0.65), 2, 0))

    block = gates.ParametrizedFusedGate(0)
    block.append(gates.RX(0, 0.16))
    block.prepend(inner)
    block.append(plain)
    assert block.nparams == (21 if inner_trainable else 17)
    before = backend.to_numpy(block.matrix(backend)).copy()
    # Ordinary fusion containers are flattened when incorporated in the block.
    plain.append(gates.X(1))
    assert block.target_qubits == (0, 2)
    backend.assert_allclose(block.matrix(backend), before, atol=1e-12)

    # Nesting fixes the child structure, while its parameters remain mutable.
    with pytest.raises(RuntimeError):
        inner.append(gates.RY(1, 0.87))
    angles = (-0.28, 0.39, -0.41) if inner_trainable else (0.21, -0.32, 0.43)
    inner_ry = -0.52 if inner_trainable else 0.76
    parameters = (*angles, inner_ry, 0.63, _unitary(-0.74))
    if not inner_trainable:
        parameters = parameters[-2:]
    block.parameters = parameters

    circuit = Circuit(3)
    circuit.add(block)
    reference = Circuit(3)
    reference.add(
        [
            gates.U3(2, *angles),
            gates.RZ(2, -0.54),
            gates.RY(2, inner_ry),
            gates.RX(0, 0.63),
            gates.CNOT(0, 2),
            gates.Unitary(_unitary(-0.74), 2, 0),
        ]
    )
    _assert_execution(backend, circuit, reference)


@pytest.mark.parametrize("method", ["append", "prepend"])
@pytest.mark.parametrize("container", ["plain", "parametrized"])
def test_parametrized_fused_container_order(backend, method, container):
    constructor = (
        gates.FusedGate if container == "plain" else gates.ParametrizedFusedGate
    )
    inner = constructor(0, 1)
    inner.append(gates.U3(1, 0.23, -0.45, 0.67))
    inner.append(gates.CNOT(1, 0))
    block = gates.ParametrizedFusedGate(0)
    block.append(gates.RX(0, 0.89))
    getattr(block, method)(inner)
    parameters = (0.89, 0.23, -0.45, 0.67)
    if method == "prepend":
        parameters = (*parameters[1:], parameters[0])
    assert block.parameters == parameters

    circuit = Circuit(2)
    circuit.add(block)
    reference = Circuit(2)
    before = [gates.RX(0, 0.89)]
    after = [gates.U3(1, 0.23, -0.45, 0.67), gates.CNOT(1, 0)]
    reference.add(before + after if method == "append" else after + before)
    _assert_execution(backend, circuit, reference)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("copy_circuit", [False, True])
def test_parametrized_fused_deepcopy_independent(backend, trainable, copy_circuit):
    block = _block(trainable=trainable)
    circuit = Circuit(3)
    circuit.add(block)
    if copy_circuit:
        clone_circuit = circuit.copy(deep=True)
        clone = clone_circuit.queue[0]
    else:
        clone = copy.deepcopy(block)
        clone_circuit = Circuit(3)
        clone_circuit.add(clone)
    assert clone.trainable is trainable
    assert clone.gates[1].trainable is False
    for original, duplicate in zip(block.gates, clone.gates):
        assert original is not duplicate

    clone.parameters = _parameters(0.84, (-0.73, 0.62, -0.51), 0.4)
    _assert_parameters(block.parameters, _parameters())
    original_reference = Circuit(3)
    original_reference.add(_leaves(_parameters()))
    _assert_execution(backend, circuit, original_reference)
    clone_reference = Circuit(3)
    clone_reference.add(_leaves(_parameters(0.84, (-0.73, 0.62, -0.51), 0.4)))
    _assert_execution(backend, clone_circuit, clone_reference)


@pytest.mark.parametrize("trainable", [False, True])
def test_parametrized_fused_dagger(backend, trainable):
    block = _block(trainable=trainable)
    inverse = block.dagger()
    assert isinstance(inverse, gates.ParametrizedFusedGate)
    assert inverse.trainable is trainable
    assert inverse.nparams == 20
    assert inverse.gates[-2].trainable is False
    backend.assert_allclose(
        inverse.matrix(backend) @ block.matrix(backend), np.eye(4), atol=1e-12
    )
    _assert_parameters(block.parameters, _parameters())
    inverse.parameters = (_unitary(0.12), 0.23, 0.34, 0.45, 0.56)
    _assert_parameters(block.parameters, _parameters())


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_parametrized_fused_on_qubits(backend, trainable, controlled):
    block = _block(trainable=trainable)
    if controlled:
        block.controlled_by(3)
    qubit_map = {0: 3, 2: 1, 3: 0}
    remapped = block.on_qubits(qubit_map)
    assert remapped.target_qubits == (1, 3)
    assert remapped.control_qubits == ((0,) if controlled else ())
    assert remapped.trainable is trainable
    assert remapped.gates[1].trainable is False
    _assert_parameters(remapped.parameters, block.parameters)
    circuit = Circuit(4)
    circuit.add(remapped)
    reference = Circuit(4)
    reference.add(
        leaf.on_qubits(qubit_map) for leaf in _leaves(_parameters(), controlled)
    )
    _assert_execution(backend, circuit, reference)
    remapped.parameters = _parameters(0.81, (-0.12, 0.23, -0.34), 0.45)
    _assert_parameters(block.parameters, _parameters())


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_parametrized_fused_controlled_block(backend, density_matrix, fuse):
    block = _block().controlled_by(3)
    circuit = Circuit(4, density_matrix=density_matrix)
    circuit.add([gates.H(1), block, gates.CZ(1, 2)])
    if fuse:
        circuit = circuit.fuse()

    for parameters in [_parameters(), _parameters(-0.71, (0.62, -0.53, 0.44), -0.35)]:
        circuit.set_parameters(_flat(parameters))
        reference = Circuit(4, density_matrix=density_matrix)
        reference.add(
            [gates.H(1), *_leaves(parameters, controlled=True), gates.CZ(1, 2)]
        )
        _assert_execution(backend, circuit, reference)
        backend.assert_allclose(
            circuit.unitary(backend), reference.unitary(backend), atol=1e-12
        )


@pytest.mark.parametrize("parameter_format", ["dict", "flatlist"])
def test_parametrized_fused_circuit_fusion_registry(backend, parameter_format):
    block = _block()
    circuit, outside = _circuit(block)
    fused = circuit.fuse()
    parameters = _parameters(-0.76, (0.65, -0.54, 0.43), -0.32)
    outside_angle = 0.21
    if parameter_format == "dict":
        update = {block: parameters, outside: outside_angle}
    else:
        update = [*_flat(parameters), outside_angle]
    fused.set_parameters(update)
    assert fused.trainable_gates.nparams == 21
    assert set(fused.get_parameters("dict")) == {block, outside}
    reference, _ = _circuit(_leaves(parameters), outside_angle=outside_angle)
    _assert_execution(backend, fused, reference)


def test_parametrized_fused_trainable_mask(backend):
    block = _block(trainable=False)
    circuit, outside = _circuit(block)
    assert circuit.get_parameters("dict") == {outside: (0.19,)}
    assert set(circuit.get_parameters("dict", include_not_trainable=True)) == {
        block,
        outside,
    }
    assert len(circuit.get_parameters("flatlist", include_not_trainable=True)) == 21
    circuit.set_parameters([0.72])
    _assert_parameters(block.parameters, _parameters())
    with pytest.raises(KeyError):
        circuit.set_parameters({block: _parameters(0.62)})

    # trainable=False masks circuit updates, not direct gate updates.
    block.parameters = _parameters(0.62)
    reference, _ = _circuit(_leaves(_parameters(0.62)), outside_angle=0.72)
    _assert_execution(backend, circuit, reference)


def test_parametrized_fused_structure_frozen_recursively():
    inner = gates.ParametrizedFusedGate(0)
    inner.append(gates.RX(0, 0.12))
    block = gates.ParametrizedFusedGate(0, 1)
    block.append(inner)
    circuit = Circuit(2)
    circuit.add(block)
    for container in (block, inner):
        for method in (container.append, container.prepend):
            with pytest.raises(RuntimeError):
                method(gates.RY(1, 0.23))
    assert circuit.trainable_gates.nparams == 1
    circuit.set_parameters([0.34])
    assert inner.parameters == (0.34,)


@pytest.mark.parametrize("method", ["append", "prepend"])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: 1,
        lambda: [gates.X(0)],
        lambda: gates.M(0),
        lambda: gates.DepolarizingChannel(0, 0.1),
        lambda: gates.CallbackGate(callbacks.Norm()),
        lambda: gates.Barrier(0),
    ],
)
def test_parametrized_fused_invalid_children(method, factory):
    block = gates.ParametrizedFusedGate(0)
    block.append(gates.RX(0, 0.12))
    with pytest.raises(TypeError):
        getattr(block, method)(factory())
    assert block.parameters == (0.12,)
    assert len(block.gates) == 1


def test_parametrized_fused_rejects_cycles_and_shared_parameters():
    block = gates.ParametrizedFusedGate(0)
    leaf = gates.RX(0, 0.12)
    block.append(leaf)
    with pytest.raises(ValueError):
        block.append(block)
    with pytest.raises(ValueError):
        block.prepend(leaf)

    inner = gates.ParametrizedFusedGate(0)
    inner.append(block)
    with pytest.raises(RuntimeError):
        block.append(inner)
    with pytest.raises(ValueError):
        inner.append(leaf)
    assert block.parameters == (0.12,)
    assert inner.parameters == (0.12,)


def test_parametrized_fused_repeated_frozen_gate(backend):
    leaf = gates.RX(0, 0.32, trainable=False)
    block = gates.ParametrizedFusedGate(0)
    block.append(leaf)
    block.append(leaf)
    assert block.parameters == ()
    assert block.nparams == 0
    backend.assert_allclose(block.matrix(backend), gates.RX(0, 0.64).matrix(backend))


@pytest.mark.parametrize("active_first", [False, True])
def test_parametrized_fused_rejects_active_alias_of_frozen_leaf(active_first):
    leaf = gates.RX(0, 0.32)
    frozen = gates.ParametrizedFusedGate(0, trainable=False)
    frozen.append(leaf)
    block = gates.ParametrizedFusedGate(0)
    block.append(leaf if active_first else frozen)
    with pytest.raises(ValueError):
        block.append(frozen if active_first else leaf)
    assert len(block.gates) == 1
    assert block.parameters == ((0.32,) if active_first else ())


def test_parametrized_fused_repeated_frozen_block(backend):
    frozen = gates.ParametrizedFusedGate(0, trainable=False)
    frozen.append(gates.RX(0, 0.32))
    block = gates.ParametrizedFusedGate(0)
    block.append(frozen)
    block.append(frozen)
    assert block.parameters == ()
    assert block.nparams == 0
    backend.assert_allclose(block.matrix(backend), gates.RX(0, 0.64).matrix(backend))


@pytest.mark.parametrize("parameters", [[], [0.1, 0.2], [0.1] * 5])
def test_parametrized_fused_invalid_parameter_length(parameters):
    block = gates.ParametrizedFusedGate(0)
    block.append(gates.RX(0, 0.12))
    block.append(gates.U3(0, 0.23, 0.34, 0.45))
    with pytest.raises(ValueError):
        block.parameters = parameters
    assert block.parameters == (0.12, 0.23, 0.34, 0.45)


@pytest.mark.parametrize("parameters", [None, "1234", {0: 0.1}, {0.1, 0.2}])
def test_parametrized_fused_invalid_parameter_type(parameters):
    block = gates.ParametrizedFusedGate(0)
    block.append(gates.U3(0, 0.12, 0.23, 0.34))
    with pytest.raises(TypeError):
        block.parameters = parameters
    assert block.parameters == (0.12, 0.23, 0.34)


def test_parametrized_fused_rejects_matrix_shape_before_update():
    block = _block()
    with pytest.raises(ValueError):
        block.parameters = (0.98, 0.87, 0.76, 0.65, np.eye(2))
    _assert_parameters(block.parameters, _parameters())


@pytest.mark.parametrize("method", ["append", "prepend"])
def test_parametrized_fused_invalid_control_overlap_preserves_gate(backend, method):
    block = gates.ParametrizedFusedGate(1)
    block.append(gates.RX(1, 0.23))
    block.controlled_by(0)
    with pytest.raises(ValueError):
        getattr(block, method)(gates.H(0))
    assert block.target_qubits == (1,)
    assert block.control_qubits == (0,)
    assert block.qubit_set == {1}
    assert block.init_args == [1]
    assert len(block.gates) == 1
    assert block.parameters == (0.23,)
    circuit = Circuit(2)
    circuit.add(block)
    reference = Circuit(2)
    reference.add(gates.CRX(0, 1, 0.23))
    _assert_execution(backend, circuit, reference)


def test_parametrized_fused_empty_and_static(backend):
    block = gates.ParametrizedFusedGate(0)
    assert block.parameters == ()
    assert block.nparams == 0
    block.parameters = []
    backend.assert_allclose(block.matrix(backend), np.eye(2))
    block.append(gates.H(0))
    block.append(gates.Z(0))
    circuit = Circuit(1)
    circuit.add(block)
    circuit.set_parameters([])
    assert circuit.get_parameters("flatlist") == []
    reference = Circuit(1)
    reference.add([gates.H(0), gates.Z(0)])
    _assert_execution(backend, circuit, reference)


def test_parametrized_fused_gradient_not_implemented():
    block = gates.ParametrizedFusedGate(0)
    block.append(gates.RX(0, 0.12))
    with pytest.raises(NotImplementedError):
        block.gradient()


@pytest.mark.parametrize("outcome", [0, 1])
def test_parametrized_fused_measurement_parameters(backend, outcome):
    circuit = Circuit(3, density_matrix=True)
    if outcome:
        circuit.add(gates.X(0))
    measurement = circuit.add(gates.M(0, collapse=True))
    inner = gates.ParametrizedFusedGate(1, trainable=False)
    inner.append(gates.RY(1, np.pi * measurement.symbols[0] / 3))
    block = gates.ParametrizedFusedGate(1, 2)
    block.append(inner)
    block.append(gates.RX(2, np.pi * measurement.symbols[0] / 4))
    block.append(gates.CNOT(1, 2))
    circuit.add(block)

    reference = Circuit(3, density_matrix=True)
    if outcome:
        reference.add(gates.X(0))
    reference.add(
        [
            gates.RY(1, np.pi * outcome / 3),
            gates.RX(2, np.pi * outcome / 4),
            gates.CNOT(1, 2),
        ]
    )
    actual = backend.execute_circuit(circuit, nshots=1).state()
    expected = backend.execute_circuit(reference).state()
    backend.assert_allclose(actual, expected, atol=1e-12)


def test_parametrized_fused_serialization_preserves_nested_masks(backend):
    inner = gates.ParametrizedFusedGate(0, trainable=False)
    inner.append(gates.RX(0, 0.12))
    block = gates.ParametrizedFusedGate(0, 1)
    block.append(inner)
    block.append(gates.CNOT(0, 1))
    block.append(gates.RZ(1, 0.23, trainable=False))
    block.append(gates.U3(1, 0.34, 0.45, 0.56))
    circuit = Circuit(2)
    circuit.add(block)
    restored = Circuit.from_dict(json.loads(json.dumps(circuit.raw)))
    restored_block = restored.queue[0]
    assert isinstance(restored_block, gates.ParametrizedFusedGate)
    assert restored_block.nparams == 3
    assert restored_block.gates[0].trainable is False
    assert restored_block.gates[2].trainable is False
    _assert_execution(backend, restored, circuit)
    restored.set_parameters([0.67, 0.78, 0.89])
    reference = Circuit(2)
    reference.add(
        [
            gates.RX(0, 0.12),
            gates.CNOT(0, 1),
            gates.RZ(1, 0.23),
            gates.U3(1, 0.67, 0.78, 0.89),
        ]
    )
    _assert_execution(backend, restored, reference)
    assert block.parameters == (0.34, 0.45, 0.56)


def test_parametrized_fused_parameter_names():
    inner = gates.ParametrizedFusedGate(1)
    inner.append(gates.U3(1, 0.12, 0.23, 0.34))
    inner.append(gates.RZ(1, 0.45, trainable=False))
    frozen = gates.ParametrizedFusedGate(0, trainable=False)
    frozen.append(gates.RY(0, 0.56))
    block = gates.ParametrizedFusedGate(0, 1)
    block.append(frozen)
    block.append(inner)
    block.prepend(gates.RX(0, 0.67))
    block.append(gates.RY(1, 0.78))
    names = block.parameter_names
    assert names == ["0_theta", "1_theta", "1_phi", "1_lam", "2_theta"]
    assert len(set(names)) == len(block.parameters)
    assert block.parameters == (0.67, 0.12, 0.23, 0.34, 0.78)


def test_parametrized_fused_decompose(backend):
    inner = gates.ParametrizedFusedGate(1)
    inner.append(gates.U3(1, 0.12, 0.23, 0.34))
    block = gates.ParametrizedFusedGate(0, 1)
    block.append(gates.RX(0, 0.45))
    block.append(gates.CNOT(0, 1))
    block.append(inner)
    decomposed = block.decompose()
    assert all(not isinstance(gate, gates.FusedGate) for gate in decomposed)
    # Child decompositions may introduce a global phase, so compare density matrices.
    circuit = Circuit(2, density_matrix=True)
    circuit.add(block)
    reference = Circuit(2, density_matrix=True)
    reference.add(decomposed)
    _assert_execution(backend, circuit, reference)
    with pytest.raises(NotImplementedError, match="controlled"):
        copy.deepcopy(block).controlled_by(2).decompose()


@pytest.mark.parametrize("qubit_map", [{0: 1, 2: 2}, {0: 1, 3: 0}])
def test_parametrized_fused_on_qubits_missing_target_or_control(qubit_map):
    block = _block().controlled_by(3)
    with pytest.raises(KeyError, match="qubit_map"):
        block.on_qubits(qubit_map)
    assert block.target_qubits == (0, 2)
    assert block.control_qubits == (3,)


def test_parametrized_fused_rejects_controlled_plain_fused_child():
    child = gates.FusedGate(1)
    child.append(gates.X(1))
    child.controlled_by(0)
    block = gates.ParametrizedFusedGate(1)
    with pytest.raises(NotImplementedError, match="controlled composite"):
        block.append(child)
    assert block.gates == []
    assert block.target_qubits == (1,)


def test_parametrized_fused_rejects_cyclic_plain_fused_child():
    class NestedFusedGate(gates.FusedGate):
        pass

    child = gates.FusedGate(0)
    wrapper = NestedFusedGate(0)
    wrapper.append(child)
    child.append(wrapper)  # FusedGate's subclass-sensitive flattening creates a cycle.
    block = gates.ParametrizedFusedGate(0)
    with pytest.raises(ValueError, match="cycles"):
        block.append(child)
    assert block.gates == []
