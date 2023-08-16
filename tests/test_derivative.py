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
    stochastic_parameter_shift,
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
    assert value == 0.5


def test_psr_commuting_gate():
    # hyperparameters
    scale_factor = 0.5
    nshots = None
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


def spsr_circuit_RXRY_decomposed(phi, s, shift):
    ham = create_hamiltonian(0, 1, GlobalBackend())

    c1 = Circuit(nqubits=1)
    c1.add(gates.RXRY(0, phi, s))
    c1.add(gates.RXRY_Variable(0, shift))
    c1.add(gates.RXRY(0, phi, (1 - s)))
    c1.add(gates.M(0))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    return val


def spsr_circuit_RXRY(phi):
    ham = create_hamiltonian(0, 1, GlobalBackend())
    c1 = Circuit(nqubits=1)
    c1.add(gates.RXRY(0, phi, 1.0))
    c1.add(gates.M(0))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([phi + 0.001, 1.0])
    forward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([phi - 0.001, 1.0])
    backward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    diff = (forward - backward) / 0.002

    return val, diff


def test_spsr_RXRY():
    np.random.seed(1430)
    angles = np.linspace(0.1, 2 * np.pi, 50)

    evals = [spsr_circuit_RXRY(theta1) for theta1 in angles]
    fdiff = [res[1] for res in evals]
    evals = [res[0] for res in evals]

    # spsr
    pos_vals = np.array(
        [
            [
                spsr_circuit_RXRY_decomposed(theta1, s=s, shift=np.pi / 4)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    neg_vals = np.array(
        [
            [
                spsr_circuit_RXRY_decomposed(theta1, s=s, shift=-np.pi / 4)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    spsr_vals = (pos_vals - neg_vals).mean(axis=1)

    """
    plt.plot(angles, evals, "b", label="Expectation Value")
    plt.plot(angles, fdiff, "g", label="Finite differences")
    plt.plot(angles, spsr_vals, "r", label="Stochastic parameter-shift rule")
    plt.legend()
    plt.show()
    """

    assert np.allclose(fdiff, spsr_vals, atol=0.05)


def test_spsr_calculate_gradients():
    ham = create_hamiltonian(0, 1, GlobalBackend())
    c1 = Circuit(nqubits=1)
    c1.add(gates.RXRY(0, 0.1, 1.0))
    c1.add(gates.M(0))

    test = stochastic_parameter_shift(c1, ham, 0, 0, gates.RXRY_Variable(q=0, phi=0.0))

    grads = calculate_circuit_gradients(
        c1,
        ham,
        np.array([0.1, 1.0]),
        2,
        "spsr",
        None,
        None,
        True,
        var_gates=[gates.RXRY_Variable(q=0, phi=0.0)],
    )

    assert np.allclose(grads, np.array([0.37, 0.0]), atol=0.02)


def spsr_circuit_crossres_decomposed(theta1, theta2, theta3, s, sign):
    ham = create_hamiltonian(0, 2, GlobalBackend())

    c1 = Circuit(nqubits=2)
    c1.add(gates.CrossRes(0, 1, s, theta1, theta2, theta3))
    c1.add(gates.CrossRes_Variable(0, sign))
    c1.add(gates.CrossRes(0, 1, (1 - s), theta1, theta2, theta3))
    c1.add(gates.M(0))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    return val


def spsr_circuit_crossres(theta1, theta2, theta3):
    ham = create_hamiltonian(0, 2, GlobalBackend())
    c1 = Circuit(nqubits=2)
    c1.add(gates.CrossRes(0, 1, 1.0, theta1, theta2, theta3))
    c1.add(gates.M(0))
    c1.add(gates.M(1))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([1.0, theta1 + 0.001, theta2, theta3])
    forward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([1.0, theta1 - 0.001, theta2, theta3])
    backward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    diff = (forward - backward) / 0.002

    return val, diff


def test_spsr_crossres():
    theta2, theta3 = -0.15, 1.6
    np.random.seed(143)
    angles = np.linspace(0, 2 * np.pi, 50)

    evals = [spsr_circuit_crossres(theta1, theta2, theta3) for theta1 in angles]

    # spsr
    pos_vals = np.array(
        [
            [
                spsr_circuit_crossres_decomposed(theta1, theta2, theta3, s=s, sign=+1)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    neg_vals = np.array(
        [
            [
                spsr_circuit_crossres_decomposed(theta1, theta2, theta3, s=s, sign=-1)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    spsr_vals = (pos_vals - neg_vals).mean(axis=1)

    """
    plt.plot(angles, evals, "b", label="Expectation Value")
    plt.plot(angles, spsr_vals, "r", label="Stochastic parameter-shift rule")
    plt.show()
    """
    res = np.array(
        [
            -3.64291930e-18,
            5.03841782e-01,
            9.73885841e-01,
            1.38185501e00,
            1.69649933e00,
            1.90157392e00,
            1.98116025e00,
            1.93349749e00,
            1.75871260e00,
            1.46393956e00,
            1.07875868e00,
            6.17535241e-01,
            1.22410924e-01,
            -3.85462412e-01,
            -8.68766960e-01,
            -1.29837344e00,
            -1.63259669e00,
            -1.86389153e00,
            -1.97147143e00,
            -1.95591789e00,
            -1.81082722e00,
            -1.54331266e00,
            -1.17736354e00,
            -7.34096811e-01,
            -2.42107926e-01,
            2.68287293e-01,
            7.58880954e-01,
            1.20287446e00,
            1.56840347e00,
            1.83007221e00,
            1.97225039e00,
            1.98603490e00,
            1.86955600e00,
            1.63230981e00,
            1.28616042e00,
            8.57174121e-01,
            3.74380259e-01,
            -1.37062287e-01,
            -6.37496216e-01,
            -1.09616643e00,
            -1.48391894e00,
            -1.77428751e00,
            -1.94997151e00,
            -1.99698582e00,
            -1.91254232e00,
            -1.70513493e00,
            -1.38456757e00,
            -9.74146855e-01,
            -4.99956865e-01,
            7.30461288e-03,
        ]
    )
    assert np.allclose(spsr_vals, res)


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


@pytest.mark.parametrize("nshots, atol", [(None, 1e-1), (100000, 1e-1)])
def test_finite_differences(backend, nshots, atol):
    # exact gradients
    grads = gradient_exact()

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    c.set_parameters(test_params)

    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad_0 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        nshots=nshots,
    )
    grad_1 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        nshots=nshots,
    )
    grad_2 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
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


if __name__ == "__main__":
    # graph_improvements(1, [0, 1], [[0, 1], [2, 3]])
    # test_multiqubit_natural_gradient()
    # test_parameter()
    # test_psr_commuting_gate()
    # rtest_spsr_non_commuting_gates()
    # test_natural_gradient()
    # test_spsr()
    test_spsr_calculate_gradients()
    # test_spsr()
