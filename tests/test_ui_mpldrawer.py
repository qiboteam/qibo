"""Tests for Qibo matplotlib drawer"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest

from qibo import Circuit, callbacks, gates
from qibo.models import QFT
from qibo.ui.drawer_utils import FusedEndGateBarrier, FusedStartGateBarrier
from qibo.ui.mpldrawer import (
    _make_cluster_gates,
    _plot_params,
    _plot_quantum_circuit,
    _process_gates,
    _render_label,
    plot_circuit,
)

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_ui_array_images")


# convert image file to numpy array
def fig2array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return data


# match a matplotlib figure with a image file
def match_figure_image(fig, arr_path):
    return np.all(fig2array(fig) == np.load(arr_path))


@pytest.mark.parametrize("nqubits", [2, 3])
def test_plot_circuit(nqubits):
    """Test for main plot function"""
    c = Circuit(nqubits)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.M(0))
    c.add(gates.M(1))
    _, fig = plot_circuit(c)
    assert (
        match_figure_image(
            fig, BASEPATH + "/test_plot_circuit_" + str(nqubits) + ".npy"
        )
        == True
    )


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_circuit_measure(nqubits):
    """Measure circuit"""
    c = Circuit(nqubits)
    c.add(gates.M(qubit) for qubit in range(nqubits - 1))
    _, fig = plot_circuit(c)
    assert (
        match_figure_image(
            fig, BASEPATH + "/test_circuit_measure_" + str(nqubits) + ".npy"
        )
        == True
    )


@pytest.mark.parametrize("nqubits", [3, 4, 5, 6])
def test_bigger_circuit_gates(nqubits):
    """Test for a bigger circuit"""
    c = Circuit(nqubits)
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
    _, fig = plot_circuit(c)
    assert (
        match_figure_image(
            fig, BASEPATH + "/test_bigger_circuit_gates_" + str(nqubits) + ".npy"
        )
        == True
    )


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
    _, fig1 = plot_circuit(c.invert(), cluster_gates=clustered, scale=0.70)
    _, fig2 = plot_circuit(c, cluster_gates=clustered, scale=0.70)
    assert (
        match_figure_image(
            fig1,
            BASEPATH
            + "/test_complex_circuit_fig1_"
            + ("true" if clustered else "false")
            + ".npy",
        )
        == True
    )
    assert (
        match_figure_image(
            fig2,
            BASEPATH
            + "/test_complex_circuit_fig2_"
            + ("true" if clustered else "false")
            + ".npy",
        )
        == True
    )


def test_align_gate():
    """Test for Align gate"""
    c = Circuit(3)
    c.add(gates.Align(0))
    _, fig = plot_circuit(c)
    assert match_figure_image(fig, BASEPATH + "/test_align_gate.npy") == True


@pytest.mark.parametrize("clustered", [False, True])
def test_circuit_fused_gates(clustered):
    """Test for FusedStartGateBarrier and FusedEndGateBarrier"""
    c = QFT(5)
    c.add(gates.M(qubit) for qubit in range(2))
    _, fig = plot_circuit(
        c.fuse(), scale=0.8, cluster_gates=clustered, style="quantumspain"
    )
    assert (
        match_figure_image(
            fig,
            BASEPATH
            + "/test_circuit_fused_gates_"
            + ("true" if clustered else "false")
            + ".npy",
        )
        == True
    )


def test_empty_circuit():
    """Test for printing empty circuit"""
    c = Circuit(2)
    _, fig = plot_circuit(c)
    assert match_figure_image(fig, BASEPATH + "/test_empty_circuit.npy") == True


@pytest.mark.parametrize("clustered", [False, True])
def test_circuit_entangled_entropy(clustered):
    """Circuit test for printing entanglement entropy circuit"""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    _, fig = plot_circuit(c, scale=0.8, cluster_gates=clustered)
    assert (
        match_figure_image(
            fig,
            BASEPATH
            + "/test_circuit_entangled_entropy_"
            + ("true" if clustered else "false")
            + ".npy",
        )
        == True
    )


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
    _, fig = plot_circuit(ansatz)
    assert match_figure_image(fig, BASEPATH + "/test_layered_circuit.npy") == True


def test_fused_gates():
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.X(0))
    c.add(gates.H(0))
    c.add(gates.X(1))
    c.add(gates.H(1))
    _, fig = plot_circuit(c.fuse(), scale=0.8, cluster_gates=False)
    assert match_figure_image(fig, BASEPATH + "/test_fused_gates.npy") == True


def test_fuse_cluster():
    """Test for clustering gates"""
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.M(qubit) for qubit in range(2))
    _, fig = plot_circuit(c.fuse())
    assert match_figure_image(fig, BASEPATH + "/test_fuse_cluster.npy") == True


def test_plot_circuit_internal():
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
        "dpi": 100,
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
    assert (
        match_figure_image(ax1.figure, BASEPATH + "/test_plot_circuit_internal_ax1.npy")
        == True
    )
    assert (
        match_figure_image(ax2.figure, BASEPATH + "/test_plot_circuit_internal_ax2.npy")
        == True
    )


def test_empty_gates():
    "Empty gates test"
    assert _process_gates([], 2) == []


def test_plot_circuit_error_style():
    """Test for style error function"""
    style1 = _plot_params(style="test")
    style2 = _plot_params(style="fardelejo")
    custom_style = {
        "facecolor": "#6497bf",
        "edgecolor": "#01016f",
        "linecolor": "#01016f",
        "textcolor": "#01016f",
        "fillcolor": "#ffb9b9",
        "gatecolor": "#d8031c",
        "controlcolor": "#360000",
    }
    style3 = _plot_params(style=custom_style)
    assert style1["facecolor"] == "w"
    assert style2["facecolor"] == "#e17a02"
    assert style3["facecolor"] == "#6497bf"


def test_fused_gates():
    """Test for FusedStartGateBarrier and FusedEndGateBarrier"""
    min_q = 0
    max_q = 1
    l_gates = 1
    equal_qbits = True
    start_barrier = FusedStartGateBarrier(min_q, max_q, l_gates, equal_qbits)
    end_barrier = FusedEndGateBarrier(min_q, max_q)
    assert start_barrier != None
    assert end_barrier != None


def test_render_label():
    """Test render labels"""
    inits = [0]
    assert _render_label("q_0", inits) != ""
    assert _render_label("q_8", inits) != ""


def test_render_label_empty():
    inits = {"q_0": None}
    assert _render_label("q_0", inits) == ""


def test_render_label_not_empty():
    inits = {"q_0": r"\psi"}
    assert _render_label("q_0", inits) != ""


def test_cluster_gates():
    """Test clustering gates"""
    pgates = [
        ("MEASURE", "q_0"),
        ("GFF", "q_0", "q_1"),
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
    assert _make_cluster_gates(pgates) != ""


def test_target_control_qubts():
    """Very dummy test to check the target and control qubits from gates"""
    c = Circuit(3)
    c.add(gates.CSX(0, 2))
    c.queue[0]._target_qubits = ((0, 1), (0, 2))
    c.queue[0]._control_qubits = ((0,), (0,))
    assert _process_gates(c.queue, 3) != ""
