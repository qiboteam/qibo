"""Tests for Qibo matplotlib drawer"""

import matplotlib.pyplot
import numpy as np
import pytest

from qibo import Circuit, callbacks, gates
from qibo.models import QFT
from qibo.ui.drawer_utils import FusedEndGateBarrier, FusedStartGateBarrier
from qibo.ui.mpldrawer import (
    plot_circuit,
    _plot_params,
    _plot_quantum_circuit,
    _process_gates,
)


# defining a dummy circuit
def circuit(nqubits=2):
    c = Circuit(nqubits)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.M(0))
    c.add(gates.M(1))
    return c


@pytest.mark.parametrize("nqubits", [2, 3])
def test_plot_circuit(nqubits):
    """Test for main plot function"""
    circ = circuit(nqubits)
    ax, _ = plot_circuit(circ)
    assert ax.title == ax.title


def test_empty_gates():
    "Empty gates test"
    assert _process_gates([], 2) == []


@pytest.mark.parametrize("qubits", [1, 2, 3])
def test_circuit_measure(qubits):
    """Measure circuit"""
    c = Circuit(qubits)
    c.add(gates.M(qubit) for qubit in range(qubits - 1))
    ax, _ = plot_circuit(c)
    assert ax.title == ax.title


def test_plot_circuit_error_style():
    """Test for style error function"""
    style = _plot_params(style="test")
    assert style["facecolor"] == "w"


@pytest.mark.parametrize("qubits", [3, 4, 5, 6])
def test_bigger_circuit_gates(qubits):
    """Test for a bigger circuit"""
    c = Circuit(qubits)
    c.add(gates.H(1))
    c.add(gates.X(1))
    c.add(gates.SX(2))
    c.add(gates.CSX(0, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.SWAP(1, 2))
    c.add(gates.SiSWAP(1, 2))
    c.add(gates.FSWAP(1, 2))
    c.add(gates.DEUTSCH(1, 0, 2, np.pi))
    c.add(gates.X(1))
    c.add(gates.X(0))
    c.add(gates.M(qubit) for qubit in range(2))
    ax, _ = plot_circuit(c)
    assert ax.title == ax.title


@pytest.mark.parametrize("clustered", [False, True])
def test_complex_circuit(clustered):
    """Complex circuits for several cases"""
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.H(2))
    c.add(gates.X(1))
    c.add(gates.Z(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(0, 1))
    c.add(gates.CRX(0, 1, np.pi))
    c.add(gates.Y(1))
    c.add(gates.RY(1, np.pi))
    c.add(gates.CRY(1, 2, np.pi))
    c.add(gates.Z(1))
    c.add(gates.SX(2))
    c.add(gates.CSX(0, 2))
    c.add(gates.X(0))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.X(0))
    c.add(gates.CNOT(1, 2))
    c.add(gates.SWAP(1, 2))
    c.add(gates.SWAP(1, 2).dagger())
    c.add(gates.SX(1).dagger())
    c.add(gates.X(0))
    c.add(gates.X(2))
    c.add(gates.H(0))
    c.add(gates.SiSWAP(1, 2).dagger())
    c.add(gates.FSWAP(1, 2).dagger())
    c.add(gates.DEUTSCH(1, 0, 2, np.pi))
    c.add(gates.X(0))
    c.add(gates.M(qubit) for qubit in range(2))
    ax1, _ = plot_circuit(c.invert(), cluster_gates=clustered, scale=0.70)
    ax2, _ = plot_circuit(c, cluster_gates=clustered, scale=0.70)
    assert ax1.title == ax1.title
    assert ax2.title == ax2.title


def test_align_gate():
    """Test for Align gate"""
    c = Circuit(3)
    c.add(gates.M(qubit) for qubit in range(2))
    c.add(gates.Align(0))
    assert c != None


def test_fused_gates():
    """Test for FusedStartGateBarrier and FusedEndGateBarrier"""
    min_q = 0
    max_q = 1
    l_gates = 1
    equal_qbits = True
    start_barrier = FusedStartGateBarrier(min_q, max_q, l_gates, equal_qbits)
    end_barrier = FusedEndGateBarrier(min_q, max_q)
    assert start_barrier.unitary == start_barrier.unitary
    assert end_barrier.unitary == end_barrier.unitary


def test_circuit_fused_gates():
    """Test for FusedStartGateBarrier and FusedEndGateBarrier"""
    c = QFT(5)
    c.add(gates.M(qubit) for qubit in range(2))
    ax, _ = plot_circuit(c.fuse(), scale=0.8, cluster_gates=True, style="quantumspain")
    assert ax.title == ax.title


def test_empty_circuit():
    """Test for printing empty circuit"""
    c = Circuit(2)
    ax, _ = plot_circuit(c)
    assert ax.title == ax.title


def test_circuit_entangled_entropy():
    """Circuit test for printing entanglement entropy circuit"""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    ax, _ = plot_circuit(c)
    assert ax.title == ax.title


def test_layered_circuit():
    """Layered Circuit test"""
    nqubits = 4
    nlayers = 3

    # Create variational ansatz circuit Twolocal
    ansatz = Circuit(nqubits)
    for l in range(nlayers):

        ansatz.add(gates.RY(q, theta=0) for q in range(nqubits))

        for i in range(nqubits - 3):
            ansatz.add(gates.CNOT(i, i + 1))
            ansatz.add(gates.CNOT(i, i + 2))
            ansatz.add(gates.CNOT(i + 1, i + 2))
            ansatz.add(gates.CNOT(i, i + 3))
            ansatz.add(gates.CNOT(i + 1, i + 3))
            ansatz.add(gates.CNOT(i + 2, i + 3))

    ansatz.add(gates.RY(q, theta=0) for q in range(nqubits))
    ansatz.add(gates.M(qubit) for qubit in range(2))
    ax, _ = plot_circuit(ansatz)
    assert ax.title == ax.title


def test_plot_circuit():
    """Test for circuit plotting"""
    gates_plot = [
        ("H", "q_0"),
        ("U1", "q_0", "q_1"),
        ("U1", "q_0", "q_2"),
        ("U1", "q_0", "q_3"),
        ("U1", "q_0", "q_4"),
        ("H", "q_1"),
        ("U1", "q_1", "q_2"),
        ("U1", "q_1", "q_3"),
        ("U1", "q_1", "q_4"),
        ("H", "q_2"),
        ("U1", "q_2", "q_3"),
        ("U1", "q_2", "q_4"),
        ("H", "q_3"),
        ("U1", "q_3", "q_4"),
        ("H", "q_4"),
        ("SWAP", "q_0", "q_4"),
        ("SWAP", "q_1", "q_3"),
        ("MEASURE", "q_0"),
        ("MEASURE", "q_1"),
    ]

    inits = [0, 1, 2, 3, 4]

    params = {
        "scale": 1.0,
        "fontsize": 14.0,
        "linewidth": 1.0,
        "control_radius": 0.05,
        "not_radius": 0.15,
        "swap_delta": 0.08,
        "label_buffer": 0.0,
        "facecolor": "#d55e00",
        "edgecolor": "#f0e442",
        "fillcolor": "#cc79a7",
        "linecolor": "#f0e442",
        "textcolor": "#f0e442",
        "gatecolor": "#d55e00",
        "controlcolor": "#f0e442",
    }

    labels = ["q_0", "q_1", "q_2", "q_3", "q_4"]

    ax1 = _plot_quantum_circuit(gates_plot, inits, params, labels, scale=0.7)
    ax2 = _plot_quantum_circuit(gates_plot, inits, params, [], scale=0.7)
    assert ax1.title == ax1.title
    assert ax2.title == ax2.title
