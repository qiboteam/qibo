"""Test IQAE model defined in `qibo/models/iqae.py`."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models.iqae import IQAE


def test_iqae_init(backend):
    A = Circuit(3 + 1)
    Q = Circuit(3 + 1)
    alpha = 0.05
    epsilon = 0.005
    n_shots = 1024
    method = "chernoff"
    iqae = IQAE(A, Q, alpha, epsilon, n_shots, method)
    assert iqae.circuit_a == A
    assert iqae.circuit_q == Q
    assert iqae.alpha == alpha
    assert iqae.epsilon == epsilon
    assert iqae.n_shots == n_shots
    assert iqae.method == method


def test_iqae_init_raising_errors(backend):
    A = Circuit(3 + 1)
    Q = Circuit(4 + 1)
    # incorrect values of:
    alpha = 2
    epsilon = 2
    method = "other"
    # try to initialize passing a `circuit_A` with more qubits than `circuit_Q`
    with pytest.raises(ValueError):
        iqae = IQAE(circuit_a=Q, circuit_q=A)
    # try to initialize with incorrect `alpha`
    with pytest.raises(ValueError):
        iqae = IQAE(A, Q, alpha=alpha)
    # try to initialize with incorrect `epsilon`
    with pytest.raises(ValueError):
        iqae = IQAE(A, Q, epsilon=epsilon)
    # try to initialize with incorrect `method`
    with pytest.raises(ValueError):
        iqae = IQAE(A, Q, method=method)
    # try to initialize with incorrect `n_shots`
    with pytest.raises(ValueError):
        iqae = IQAE(A, Q, n_shots=0.5)
    # testing the line of code when n_shots_i==0
    iqae = IQAE(A, Q, method="beta", n_shots=10, alpha=0.05, epsilon=0.48)
    results = iqae.execute(backend=backend)


def test_iqae_execution(backend):
    # Let's check if we get the correct result for the integral of Sin(x)^2 from 0 to 1
    nbit = 3
    A = A_circ(qx=list(range(nbit)), qx_measure=nbit, nbit=nbit, b_max=1, b_min=0)
    Q = Q_circ(qx=list(range(nbit)), qx_measure=nbit, nbit=nbit, b_max=1, b_min=0)

    iqae = IQAE(A, Q, method="chernoff")
    results = iqae.execute(backend=backend)
    # Check that we run in the lower half-plane
    for i in range(10):
        iqae = IQAE(A, Q, method="chernoff", alpha=0.05, epsilon=0.01, n_shots=100)
        results = iqae.execute(backend=backend)
    iqae = IQAE(A, Q, method="beta")
    results = iqae.execute(backend=backend)
    true_result = 1 / 4 * (2 - np.sin(2))
    # Check that the result lies in the expected interval
    assert results.estimation > true_result - 2 * results.epsilon_estimated
    assert results.estimation < true_result + 2 * results.epsilon_estimated


def reflect_qibo(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
    Computing reflection operator (I - 2|0><0|)
        qc: quantum circuit
        qx: quantum register
        qx_measure: quantum register for measurement
        qx_ancilla: temporal quantum register for decomposing multi controlled NOT gate
        nbit: number of qubits to be used for defining the good and bad state
        b_max: upper limit of integral
        b_min: lower limit of integral
    """
    for i in range(nbit):
        qc.add(gates.X(q=qx[i]))
    qc.add(gates.X(q=qx_measure))
    multi_control_NOT_qibo(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    qc.add(gates.X(q=qx_measure))
    for i in range(nbit):
        qc.add(gates.X(q=qx[i]))


def multi_control_NOT_qibo(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
    Computing multi controlled NOT gate
        qc: quantum circuit
        qx: quantum register
        qx_measure: quantum register for measurement
        qx_ancilla: temporal quantum register for decomposing multi controlled NOT gate
        nbit: number of qubits to be used for defining the good and bad state
        b_max: upper limit of integral
        b_min: lower limit of integral
    """
    if nbit == 1:
        qc.add(gates.CZ(qx[0], qx_measure))
    elif nbit == 2:
        qc.add(gates.H(qx_measure))
        qc.add(gates.TOFFOLI(qx[0], qx[1], qx_measure))
        qc.add(gates.H(qx_measure))
    elif nbit > 2.0:
        qc.add(gates.TOFFOLI(qx[0], qx[1], qx_ancilla[0]))
        for i in range(nbit - 3):
            qc.add(gates.TOFFOLI(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1]))
        qc.add(gates.H(qx_measure))
        qc.add(gates.TOFFOLI(qx[nbit - 1], qx_ancilla[nbit - 3], qx_measure))
        qc.add(gates.H(qx_measure))
        for i in range(nbit - 3)[::-1]:
            qc.add(gates.TOFFOLI(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1]))
        qc.add(gates.TOFFOLI(qx[0], qx[1], qx_ancilla[0]))


def P_qibo(qc, qx):
    """
    Generating the uniform probability distribution p(x)=1/2^n
        qc: quantum circuit
        qx: quantum register

    The inverse of P = P
    """
    for q in qx:
        qc.add(gates.H(q=q))


def R_qibo(qc, qx, qx_measure, nbit, b_max, b_min):
    """
    Encoding the function f(x)=sin(x)^2 to be integrated in the interval [b_min,b_max]
        qc: quantum circuit
        qx: quantum register
        qx_measure: quantum register for measurement
        nbit: number of qubits
        b_max: upper limit of integral
        b_min: lower limit of integral
    """
    qc.add(
        gates.RY(q=qx_measure, theta=(b_max - b_min) / 2**nbit * 2 * 0.5 + 2 * b_min)
    )
    for i in range(nbit):
        qc.add(gates.CU3(qx[i], qx_measure, 2**i * (b_max - b_min) / 2**nbit * 2, 0, 0))


def Rinv_qibo(qc, qx, qx_measure, nbit, b_max, b_min):
    """
    The inverse of R
        qc: quantum circuit
        qx: quantum register
        qx_measure : quantum register for measurement
        nbit: number of qubits
        b_max: upper limit of integral
        b_min: lower limit of integral
    """
    for i in range(nbit)[::-1]:
        qc.add(
            gates.CU3(qx[i], qx_measure, -(2**i) * (b_max - b_min) / 2**nbit * 2, 0, 0)
        )
    qc.add(
        gates.RY(q=qx_measure, theta=-(b_max - b_min) / 2**nbit * 2 * 0.5 - 2 * b_min)
    )


def A_circ(qx, qx_measure, nbit, b_max, b_min):
    """
    The initialization operator
        A: quantum circuit
        qx: quantum register
        qx_measure: quantum register for measurement
        nbit: number of qubits to be used for defining the good and bad state
        b_max: upper limit of integral
        b_max: lower limit of integral
    """
    circ = Circuit(nbit + 1)
    # The operator P encodes the probability function p(x) into the state |0⟩_n
    P_qibo(circ, qx)
    # The operator R encodes the f (x) function into an ancillary qubit that is added to the circuit
    R_qibo(circ, qx, qx_measure, nbit, b_max, b_min)
    return circ


def Q_circ(qx, qx_measure, nbit, b_max, b_min):
    """
    The Grover/Amplification operator: R P (I - 2|0><0|) P^+ R^+ U_psi_0
        q: quantum circuit
        qx: quantum register
        qx_measure: quantum register for measurement
        nbit: number of qubits to be used for defining the good and bad state
        b_max: upper limit of integral
        b_max: lower limit of integral
    """
    if nbit > 2:
        circ = Circuit(2 * nbit - 1)
        qx_ancilla = list(range(nbit + 1, nbit + 1 + nbit - 2))
    else:
        circ = Circuit(nbit + 1)
        qx_ancilla = 0

    circ.add(gates.Z(q=qx_measure))
    Rinv_qibo(circ, qx, qx_measure, nbit, b_max, b_min)
    P_qibo(circ, qx)
    reflect_qibo(circ, qx, qx_measure, qx_ancilla, nbit, b_max)
    P_qibo(circ, qx)
    R_qibo(circ, qx, qx_measure, nbit, b_max, b_min)
    return circ
