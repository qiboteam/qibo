import math

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
import tensorflow as tf

import qibo
from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend
from qibo.derivative import (
    Graph,
    build_graph,
    calculate_circuit_gradients,
    create_hamiltonian,
    finite_differences,
    generate_fubini,
    parameter_shift,
    run_subcircuit_measure,
)
from qibo.gates.gates import Parameter
from qibo.models import Circuit
from qibo.symbols import Z

qibo.set_backend("tensorflow")
tf.get_logger().setLevel("ERROR")
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def ansatz_pdf(layers, params, feature):
    for i in range(layers):
        qml.Hadamard(wires=i)

        qml.RZ(params[12 * i + 0] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 1], wires=i)

        qml.RY(params[12 * i + 2] * feature, wires=i)
        qml.RY(params[12 * i + 3], wires=i)

        qml.RZ(params[12 * i + 4] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 5], wires=i)

        qml.RY(params[12 * i + 6] * feature, wires=i)
        qml.RY(params[12 * i + 7], wires=i)

        qml.RZ(params[12 * i + 8] * math.log(feature), wires=i)
        qml.RZ(params[12 * i + 9], wires=i)

        qml.RY(params[12 * i + 10] * feature, wires=i)
        qml.RY(params[12 * i + 11], wires=i)

    return qml.expval(qml.PauliZ([1]))


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    for qubit in range(nqubits):
        c.add(qibo.gates.H(q=qubit))

        for _ in range(layers):
            c.add(
                qibo.gates.RZ(
                    q=qubit,
                    theta=Parameter(
                        lambda x, th1: th1 * sp.log(x), [0.1], featurep=[0.1]
                    ),
                )
            )
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(
                qibo.gates.RY(
                    q=qubit,
                    theta=Parameter(lambda x, th1: th1 * x, [0.1], featurep=[0.1]),
                )
            )
            c.add(qibo.gates.RY(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))

        c.add(qibo.gates.M(qubit))

    return c


def ansatz_2qubit(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    c.add(qibo.gates.H(q=0))
    c.add(qibo.gates.H(q=1))

    for _ in range(layers):
        c.add(qibo.gates.RY(q=0, theta=0))
        c.add(qibo.gates.RZ(q=1, theta=0))
        c.add(qibo.gates.RY(q=1, theta=0))

    c.add(qibo.gates.CNOT(0, 1))

    c.add(qibo.gates.M(0))

    c.add(qibo.gates.M(1))

    return c


# defining an observable
def hamiltonian(nqubits, backend):
    return hamiltonians.hamiltonians.SymbolicHamiltonian(
        np.prod([Z(i) for i in range(nqubits)]), backend=backend
    )


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
@pytest.mark.parametrize(
    "scale_factor, grads",
    [(1, [-8.51104358e-02, -5.20075970e-01, 0]), (0.5, [-0.02405061, -0.13560379, 0])],
)
def test_standard_parameter_shift(backend, nshots, atol, scale_factor, grads):
    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    test_params *= scale_factor
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad_0 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_1 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_2 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)


@pytest.mark.parametrize("step_size", [10**-i for i in range(5, 10, 1)])
def test_finite_differences(backend, step_size):
    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    test_params = np.linspace(0.1, 1, 3)
    grads = [-8.51104358e-02, -5.20075970e-01, 0]
    atol = 1e-6
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = finite_differences(circuit=c, hamiltonian=c, parameter_index=0)

    # executing all the procedure
    grad_0 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=0, step_size=step_size
    )
    grad_1 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=1, step_size=step_size
    )
    grad_2 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=2, step_size=step_size
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)


def test_parameter():
    # single feature
    param = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        [1.5, 2.0, 3.0],
        featurep=[7.0],
    )

    indices = param.get_indices(10)
    assert indices == [10, 11, 12]

    fixed = param.get_fixed_part(1)
    assert fixed == 73.5

    factor = param.get_scaling_factor(2)
    assert factor == 12.0

    gate_value = param.get_params(trainablep=[15.0, 10.0, 7.0], feature=[5.0])
    assert gate_value == 865

    # multiple features
    param = Parameter(
        lambda x1, x2, th1, th2, th3: x1**2 * th1 + x2 * th2 * th3,
        [1.5, 2.0, 3.0],
        featurep=[7.0, 4.0],
    )

    fixed = param.get_fixed_part(1)
    assert fixed == 73.5

    factor = param.get_scaling_factor(2)
    assert factor == 8.0

    gate_value = param.get_params(trainablep=[15.0, 10.0, 7.0], feature=[5.0, 3.0])
    assert gate_value == 585


def test_run_subcircuit_measure():
    c = circuit(nqubits=1)
    value = run_subcircuit_measure(c, 0, 1, GlobalBackend(), deterministic=True)
    assert value == 1.0


@pytest.mark.parametrize(
    "layer_num, trainable_qubits_correct, affected_params_correct",
    [(1, [0, 1], [[0, 1], [2, 3]]), (2, [0, 1], [[6, 7], [4, 5]]), (6, [1], [16, 17])],
)
def test_graph(layer_num, trainable_qubits_correct, affected_params_correct):
    circuit = ansatz_2qubit(3, 2)

    nqubits = circuit.nqubits
    gates = circuit.queue
    trainable_params = np.linspace(0.1, 1, 18)
    gate_params = [
        trainable_params[i] + trainable_params[i + 1] for i in range(0, 18, 2)
    ]
    trainable_params_index = [[i, i + 1] for i in range(0, 18, 2)]

    graph = Graph(nqubits, gates, trainable_params_index, gate_params)

    graph.build_graph()

    new_circuit, trainable_qubits, affected_params = graph.run_layer(layer_num)

    assert np.allclose(trainable_qubits, trainable_qubits_correct)
    assert np.allclose(affected_params, affected_params_correct)


def graph_improvements(layer_num, trainable_qubits_correct, affected_params_correct):
    circuit = ansatz_2qubit(3, 2)

    nqubits = circuit.nqubits
    gates = circuit.queue
    trainable_params = np.linspace(0.1, 1, 18)
    gate_params = [
        trainable_params[i] + trainable_params[i + 1] for i in range(0, 18, 2)
    ]
    trainable_params_index = [[i, i + 1] for i in range(0, 18, 2)]

    graph = Graph(nqubits, gates, trainable_params_index, gate_params)

    graph.build_graph()

    new_circuit, trainable_qubits, affected_params = graph.run_layer(layer_num)

    assert np.allclose(trainable_qubits, trainable_qubits_correct)
    assert np.allclose(affected_params, affected_params_correct)


def loss_func(ypred, ytrue, other_args=None):
    loss = 0
    for i in range(len(ypred)):
        loss += (ytrue[i] - ypred[i]) ** 2

    return loss


def test_natural_gradient():
    params = qml.numpy.asarray([0.1] * 12)

    # create circuit ansatz for two qubits
    circuit = ansatz(3, 1)

    # initialize optimiser with Parameter objects
    initial_parameters = [0.1] * 12
    optimiser = qibo.optimizers.SGD(
        circuit=circuit, parameters=initial_parameters, loss=loss_func
    )

    _ = optimiser.run_circuit(0.1)

    graph = build_graph(optimiser._circuit, 12, optimiser.nqubits, optimiser.initparams)
    fubini = generate_fubini(
        graph,
        12,
        1,
        optimiser.initparams,
        noise_model=optimiser.options["noise_model"],
        deterministic=True,
    )

    # initialize optimiser with numpy array
    initial_parameters2 = np.full(12, 0.1)
    optimiser2 = qibo.optimizers.SGD(
        circuit=circuit, parameters=initial_parameters2, loss=loss_func
    )

    _ = optimiser2.run_circuit(0.1)

    graph = build_graph(
        optimiser2._circuit, 12, optimiser2.nqubits, optimiser2.initparams
    )
    fubini2 = generate_fubini(
        graph,
        12,
        1,
        optimiser2.initparams,
        noise_model=optimiser2.options["noise_model"],
        deterministic=True,
    )

    assert np.allclose(optimiser.params, params)

    metric_tensor = qml.metric_tensor(ansatz_pdf, approx="diag")(1, params, 0.1)

    assert np.allclose(fubini, metric_tensor)
    assert np.allclose(fubini2, metric_tensor)


def test_multiqubit_natural_gradient():
    # pennylane baseline
    params = qml.numpy.asarray([0.1] * 24)
    metric_tensor = qml.metric_tensor(ansatz_pdf, approx="diag")(2, params, 0.1)

    # local implementation
    nqubits = 2
    circuit = ansatz(
        3, nqubits
    )  # 2 qubits x 3 layers x 2 gates x 2 parameters = 24 params
    initial_parameters = [0.1] * 24

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimiser = qibo.optimizers.SGD(
        circuit=circuit,
        parameters=initial_parameters,
        hamiltonian=hamiltonians,
        loss=loss_func,
    )

    _ = optimiser.run_circuit(0.1)

    graph = build_graph(optimiser._circuit, 24, optimiser.nqubits, optimiser.initparams)
    fubini = generate_fubini(
        graph,
        24,
        nqubits,
        optimiser.initparams,
        noise_model=optimiser.options["noise_model"],
        deterministic=True,
    )

    assert np.allclose(fubini, metric_tensor)
