import math

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
import tensorflow as tf

import qibo
from qibo import gates
from qibo.backends import GlobalBackend
from qibo.derivative import (
    Graph,
    build_graph,
    calculate_circuit_gradients,
    create_hamiltonian,
    error_mitigation,
    execute_circuit,
    finite_differences,
    generate_fubini,
    parameter_shift,
    run_subcircuit_measure,
)
from qibo.models.variational import VariationalCircuit
from qibo.parameter import Parameter

qibo.set_backend("tensorflow")
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def ansatz_pennylane(nqubits, params, feature):
    """
    The circuit's pennylane ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        nqubits (int): number of qubits in circuit
        nqubits (np.ndarray): array of initial parameters
        feature (float): feature used in reuploading strategy
    Returns: abstract pennylane circuit
    """
    for i in range(nqubits):
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


@qml.qnode(dev, interface="autograd")
def ansatz_pennylane_entangled(nqubits, params, feature):
    """
    The circuit's pennylane entangled ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        nqubits (int): number of qubits in circuit
        nqubits (np.ndarray): array of initial parameters
        feature (float): feature used in reuploading strategy
    Returns: abstract pennylane circuit
    """

    j = 0

    for i in range(1):
        for w in range(nqubits):
            qml.RZ(params[j], wires=w)
            qml.RZ(params[j + 1], wires=w)

            qml.RY(params[j + 2], wires=w)
            qml.RY(params[j + 3], wires=w)

            j += 4

        qml.CRZ(params[j], wires=[0, 1])
        j += 1

    return qml.expval(qml.PauliZ([1]))


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
    Returns: abstract qibo circuit
    """

    c = VariationalCircuit(nqubits, density_matrix=True)

    for qubit in range(nqubits):
        c.add(qibo.gates.H(q=qubit))

        for _ in range(layers):
            c.add(
                qibo.gates.RZ(
                    q=qubit,
                    theta=Parameter(
                        lambda x, th1: th1 * sp.log(x), [0.1], feature=[0.1]
                    ),
                )
            )
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(
                qibo.gates.RY(
                    q=qubit,
                    theta=Parameter(lambda x, th1: th1 * x, [0.1], feature=[0.1]),
                )
            )
            c.add(qibo.gates.RY(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))

        c.add(qibo.gates.M(qubit))

    return c


def ansatz_2qubit(layers, nqubits):
    """
    The circuit's 2 qubit ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
    Returns: abstract qibo circuit
    """

    c = VariationalCircuit(nqubits, density_matrix=True)

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


def ansatz_entangled(layers, nqubits):
    """
    The circuit's entangled ansatz: a sequence of RZ and RY with a beginning H gate
    and CRZ gates at the end of each layer.

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
    Returns: abstract qibo circuit
    """

    c = VariationalCircuit(nqubits, density_matrix=True)

    c.add(qibo.gates.H(q=0))
    c.add(qibo.gates.H(q=1))

    for _ in range(layers):
        for qubit in range(nqubits):
            c.add(
                qibo.gates.RZ(
                    q=qubit,
                    theta=Parameter(
                        lambda x, th1: th1 * sp.log(x), [0.1], feature=[0.1]
                    ),
                )
            )
            c.add(qibo.gates.RZ(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))
            c.add(
                qibo.gates.RY(
                    q=qubit,
                    theta=Parameter(lambda x, th1: th1 * x, [0.1], feature=[0.1]),
                )
            )
            c.add(qibo.gates.RY(q=qubit, theta=Parameter(lambda th1: th1, [0.1])))

        c.add(qibo.gates.CRZ(0, 1, theta=Parameter(lambda th1: th1, [0.1])))

    c.add(qibo.gates.M(0))
    c.add(qibo.gates.M(1))

    return c


def circuit(nqubits=1):
    """Small circuit ansatz"""

    c = VariationalCircuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.H(q=0))
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


def gradient_exact():
    """Calculates exact gradient of a circuit"""

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


def test_execute_circuit():
    """Test the `execute_circuit` function"""

    circuit = ansatz(3, 1)
    obs = create_hamiltonian(0, 1, GlobalBackend())
    exp_v = execute_circuit(GlobalBackend(), circuit, obs, deterministic=True)

    assert np.isclose(exp_v, 0.311255)

    exp_v_scaled = execute_circuit(
        GlobalBackend(), circuit, obs, cdr_params=(5, 0.1), nshots=1024
    )

    assert np.isclose(exp_v_scaled, 5 * exp_v + 0.1, atol=0.3)

    # double qubit
    obs = []
    circuit2q = ansatz(3, 2)
    obs.append(create_hamiltonian(1, 2, GlobalBackend()))
    obs.append(create_hamiltonian(0, 2, GlobalBackend()))
    exp_v = execute_circuit(GlobalBackend(), circuit2q, obs, deterministic=True)

    assert np.allclose(exp_v[0], exp_v[1])


one_qubit = tf.constant(
    [[-1.0 - 0.0j, 0.0 - 0.0j], [0.0 - 0.0j, 1.0 + 0.0j]], dtype=tf.complex128
)

two_qubit = tf.constant(
    [
        [-1.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j],
        [0.0 - 0.0j, 1.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j],
        [0.0 - 0.0j, 0.0 - 0.0j, -1.0 - 0.0j, 0.0 - 0.0j],
        [0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j, 1.0 + 0.0j],
    ],
    dtype=tf.complex128,
)


@pytest.mark.parametrize(
    "qubit, nqubits, matrix", [(0, 1, one_qubit), (1, 2, two_qubit)]
)
def test_create_hamiltonian(qubit, nqubits, matrix):
    """Test the `create_hamiltonian` function"""

    ham = create_hamiltonian(qubit, nqubits, GlobalBackend())
    assert np.allclose(ham.matrix, matrix)


def test_run_subcircuit_measure():
    """Test the `run_subcircuit_measure` function"""

    c = circuit(nqubits=1)
    value = run_subcircuit_measure(c, [0], 1, GlobalBackend(), deterministic=True)
    assert value[0] == 0.5


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
def test_psr(backend, nshots, atol):
    """Test PSR gradient calculation

    Args:
        backend (:class:`qibo.backends.abstract.Backend`): simulation backend used to run circuit
        nshots (int): number of shots executed at each circuit run
        atol (float): absolute tolerance allowed on parameter shift rule derivatives compared to ground truth
    """

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
    """Test Finite Difference gradient calculation
    Args:
        backend (:class:`qibo.backends.abstract.Backend`): simulation backend used to run circuit
        nshots (int): number of shots executed at each circuit run
        atol (float): absolute tolerance allowed on parameter shift rule derivatives compared to ground truth
    """

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


def test_psr_commuting_gate():
    """Test PSR with commuting gates, i.e. [X, Y]=0"""

    # hyperparameters
    scale_factor = 0.5
    nshots = None
    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # separating gates
    c = VariationalCircuit(1)
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
    c2 = VariationalCircuit(1)
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
    """RXRY decomposed into its 3 constituents for SPSR"""

    ham = create_hamiltonian(0, 1, GlobalBackend())

    c1 = VariationalCircuit(nqubits=1)
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
    """RXRY gate and circuit used for SPSR"""

    ham = create_hamiltonian(0, 1, GlobalBackend())
    c1 = VariationalCircuit(nqubits=1)
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
    """Test `stochastic_parameter_shift` decomposed into its constituents on RXRY"""

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

    assert np.allclose(fdiff, spsr_vals, atol=0.05)


def test_spsr_calculate_gradients():
    """Test `calculate_circuit_gradients` using SPSR on RXRY"""

    ham = create_hamiltonian(0, 1, GlobalBackend())
    c1 = VariationalCircuit(nqubits=1)
    c1.add(gates.RXRY(0, 0.1, 1.0))
    c1.add(gates.M(0))

    grads = calculate_circuit_gradients(
        c1,
        ham,
        2,
        "spsr",
        None,
        None,
        nshots=1024,
        deterministic=True,
        var_gates=[gates.RXRY_Variable(q=0, phi=0.0)],
    )

    assert np.allclose(grads, np.array([0.37, 0.0]), atol=0.02)


def spsr_circuit_crossres_decomposed(theta1, theta2, theta3, s, sign):
    """RXRY decomposed into its 3 constituents for SPSR"""

    ham = create_hamiltonian(0, 2, GlobalBackend())

    c1 = VariationalCircuit(nqubits=2)
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
    """Crossres gate and circuit used for SPSR"""

    ham = create_hamiltonian(0, 2, GlobalBackend())
    c1 = VariationalCircuit(nqubits=2)
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
    """Test `stochastic_parameter_shift` decomposed into its constituents on Crossres gate"""

    theta2, theta3 = -0.15, 1.6
    np.random.seed(143)
    angles = np.linspace(0, 2 * np.pi, 50)

    evals = [spsr_circuit_crossres(theta1, theta2, theta3) for theta1 in angles]
    assert len(evals) == 50
    yval = []
    fdiff = []
    for y_val, diff in evals:
        yval.append(y_val)
        fdiff.append(diff)

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
