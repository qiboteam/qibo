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
    create_hamiltonian,
    generate_fubini,
    parameter_shift,
    run_subcircuit_measure,
)
from qibo.gates.gates import Parameter
from qibo.models import Circuit
from qibo.symbols import Z

qibo.set_backend("tensorflow")


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def ansatz_pdf(layers, params, feature):
    for i in range(layers):
        qml.Hadamard(wires=i)

        qml.RZ(params[12 * i + 0] * feature, wires=i)
        qml.RZ(params[12 * i + 1], wires=i)

        qml.RY(params[12 * i + 2] * feature, wires=i)
        qml.RY(params[12 * i + 3], wires=i)

        qml.RZ(params[12 * i + 4] * feature, wires=i)
        qml.RZ(params[12 * i + 5], wires=i)

        qml.RY(params[12 * i + 6] * feature, wires=i)
        qml.RY(params[12 * i + 7], wires=i)

        qml.RZ(params[12 * i + 8] * feature, wires=i)
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
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(qibo.gates.RY(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
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


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.H(q=0))
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


# calculate the exact gradients
def gradient_exact():
    backend = GlobalBackend()

    test_params = tf.Variable(np.linspace(0.1, 1, 3))

    with tf.GradientTape() as tape:
        c = circuit(nqubits=1)
        c.set_parameters(test_params)

        ham = create_hamiltonian(0, 1, GlobalBackend())
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


def test_parameter():
    # single feature
    param = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3,
        [1.5, 2.0, 3.0],
        featurep=[7.0],
    )

    indices = param.get_indices(10)
    assert indices == [10, 11, 12]

    fixed = param.get_fixed_part(1)
    assert fixed == 73.5

    factor = param.get_scaling_factor(2)
    assert factor == 2.0

    gate_value = param.get_params(trainablep=[15.0, 10.0, 7.0], feature=[5.0])
    assert gate_value == 445

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
    value = run_subcircuit_measure(c, 0, 1, GlobalBackend(), stochastic=False)
    assert value == 0.5


def test_psr_commuting_gate():
    # hyperparameters
    scale_factor = 0.5
    nshots = 1024
    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # separating gates
    c = Circuit(1)
    c.add(gates.H(q=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.M(0))

    params = np.array([0.1, 0.2])
    c.set_parameters(params)

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

    # single gate
    c2 = Circuit(1)
    c2.add(gates.H(q=0))
    c2.add(gates.RY(q=0, theta=0))
    c2.add(gates.M(0))

    params = np.array([0.3])
    c2.set_parameters(params)

    grad_2 = parameter_shift(
        circuit=c2,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    assert np.isclose(grad_0, grad_1, atol=1e-5)
    assert np.isclose(grad_1, grad_2, atol=1e-5)


def test_spsr_non_commuting_gates():
    c1 = Circuit(nqubits=1)
    c1.add(gates.RX(q=0, theta=10))
    c1.add(gates.RX(q=0, theta=40))
    c1.add(gates.RY(q=0, theta=40))
    c1.add(gates.M(0))

    print(c1.execute().state())

    c2 = Circuit(nqubits=1)
    c2.add(gates.RX(q=0, theta=10))
    c2.add(gates.RY(q=0, theta=40))
    c2.add(gates.RX(q=0, theta=40))
    c2.add(gates.M(0))

    print(c2.execute().state())

    c3 = Circuit(nqubits=1)
    c3.add(gates.RX(q=0, theta=10))
    c3.add(gates.G(q=0, phi=40))
    c3.add(gates.M(0))

    print(c3.execute().state())

    c4 = Circuit(nqubits=1)
    c4.add(gates.RX(q=0, theta=10))
    c4.add(gates.G(q=0, phi=10))
    c4.add(gates.G(q=0, phi=30))
    c4.add(gates.M(0))

    print(c4.execute().state())

    c5 = Circuit(nqubits=1)
    c5.add(gates.RX(q=0, theta=10))
    c5.add(gates.G(q=0, phi=30))
    c5.add(gates.G(q=0, phi=10))
    c5.add(gates.M(0))

    print(c5.execute().state())


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
def test_psr(backend, nshots, atol):
    grads = gradient_exact()
    scale_factor = 1

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    test_params *= scale_factor
    c.set_parameters(test_params)

    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

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


@pytest.mark.parametrize(
    "layer_num, trainable_qubits_correct, affected_params_correct",
    [(1, [0, 1], [[0, 1], [2, 3]]), (2, [0, 1], [[6, 7], [4, 5]]), (6, [1], [16, 17])],
)
def test_graph(layer_num, trainable_qubits_correct, affected_params_correct):
    circuit = ansatz_2qubit(3, 2)
    print(circuit.draw())
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
    print(circuit.draw())
    nqubits = circuit.nqubits
    gates = circuit.queue
    trainable_params = np.linspace(0.1, 1, 18)
    gate_params = [
        trainable_params[i] + trainable_params[i + 1] for i in range(0, 18, 2)
    ]
    trainable_params_index = [[i, i + 1] for i in range(0, 18, 2)]

    print(gates, trainable_params_index, gate_params)
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

    graph = build_graph(
        optimiser._circuit, 12, optimiser.nqubits, optimiser.paramInputs
    )
    fubini = generate_fubini(
        graph,
        12,
        1,
        optimiser.paramInputs,
        noise_model=optimiser.options["noise_model"],
        stochastic=False,
    )

    # initialize optimiser with numpy array
    initial_parameters2 = np.full(12, 0.1)
    optimiser2 = qibo.optimizers.SGD(
        circuit=circuit, parameters=initial_parameters2, loss=loss_func
    )

    _ = optimiser2.run_circuit(0.1)

    graph = build_graph(
        optimiser2._circuit, 12, optimiser2.nqubits, optimiser2.paramInputs
    )
    fubini2 = generate_fubini(
        graph,
        12,
        1,
        optimiser2.paramInputs,
        noise_model=optimiser2.options["noise_model"],
        stochastic=False,
    )

    assert np.allclose(optimiser.params, params)

    metric_tensor = qml.metric_tensor(ansatz_pdf, approx="diag")(1, params, 1.0)
    print(metric_tensor)
    assert np.allclose(fubini, metric_tensor)
    assert np.allclose(fubini2, metric_tensor)


def test_multiqubit_natural_gradient():
    # pennylane baseline
    params = qml.numpy.asarray([0.1] * 24)
    metric_tensor = qml.metric_tensor(ansatz_pdf, approx="diag")(2, params, 1.0)

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

    graph = build_graph(
        optimiser._circuit, 24, optimiser.nqubits, optimiser.paramInputs
    )
    fubini = generate_fubini(
        graph,
        24,
        nqubits,
        optimiser.paramInputs,
        noise_model=optimiser.options["noise_model"],
        stochastic=False,
    )

    assert np.allclose(fubini, metric_tensor)


if __name__ == "__main__":
    # graph_improvements(1, [0, 1], [[0, 1], [2, 3]])
    # test_multiqubit_natural_gradient()
    # test_parameter()
    # test_psr_commuting_gate()
    test_spsr_non_commuting_gates()
