"""Tests for Qibo matplotlib drawer"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from qibo import Circuit, callbacks, construct_backend, gates
from qibo.models import QFT
from qibo.quantum_info import random_unitary
from qibo.ui.drawing_utils import FusedEndGateBarrier, FusedStartGateBarrier
from qibo.ui.mpldrawer import (
    _build_folded_gate_layout,
    _fold_coords,
    _make_cluster_gates,
    _plot_params,
    _plot_quantum_circuit,
    _process_gates,
    _render_label,
    plot_circuit,
)

from .utils import match_figure_image

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_ui_array_images")


@pytest.mark.parametrize("nqubits", [2, 3])
def test_plot_circuit(nqubits):
    """Test for main plot function"""
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    _, fig = plot_circuit(circuit)
    assert (
        match_figure_image(
            fig, BASEPATH + "/test_plot_circuit_" + str(nqubits) + ".npy"
        )
        == True
    )


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_circuit_measure(nqubits):
    """Measure circuit"""
    circuit = Circuit(nqubits)
    circuit.add(gates.M(qubit) for qubit in range(nqubits - 1))
    _, fig = plot_circuit(circuit)
    assert (
        match_figure_image(
            fig, BASEPATH + "/test_circuit_measure_" + str(nqubits) + ".npy"
        )
        == True
    )


@pytest.mark.parametrize("nqubits", [3, 4, 5, 6])
@pytest.mark.parametrize("fold", [-1, 3, 4, 10, 12])
def test_bigger_circuit_gates(nqubits, fold):
    """Test for a bigger circuit with optional folding"""
    circuit = Circuit(nqubits)
    circuit.add(gates.H(1))
    circuit.add(gates.X(1))
    circuit.add(gates.SX(2))
    circuit.add(gates.CSX(0, 2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.SWAP(1, 2))
    circuit.add(gates.SiSWAP(1, 2))
    circuit.add(gates.FSWAP(1, 2))
    circuit.add(gates.DEUTSCH(1, 0, 2, np.pi))
    circuit.add(gates.X(1))
    circuit.add(gates.X(0))
    circuit.add(gates.M(qubit) for qubit in range(2))

    test_file_prefix = f"{BASEPATH}/test_bigger_circuit_gates_{nqubits}"
    fold_suffix = f"_fold_{fold}" if fold > 0 else ""
    extension = ".npy"

    test_file = f"{test_file_prefix}{fold_suffix}{extension}"

    _, fig = plot_circuit(circuit, fold=fold)

    assert match_figure_image(fig, test_file)


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("fold", [-1, 5, 8, 15])
def test_complex_circuit(clustered, fold):
    """Complex circuits for several cases"""
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.H(2))
    circuit.add(gates.X(1))
    circuit.add(gates.Z(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CRX(0, 1, np.pi))
    circuit.add(gates.Y(1))
    circuit.add(gates.RY(1, np.pi))
    circuit.add(gates.CRY(1, 2, np.pi))
    circuit.add(gates.Z(1))
    circuit.add(gates.SX(2))
    circuit.add(gates.CSX(0, 2))
    circuit.add(gates.X(0))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.X(0))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.SWAP(1, 2))
    circuit.add(gates.SWAP(1, 2).dagger())
    circuit.add(gates.SX(1).dagger())
    circuit.add(gates.X(0))
    circuit.add(gates.X(2))
    circuit.add(gates.H(0))
    circuit.add(gates.SiSWAP(1, 2).dagger())
    circuit.add(gates.FSWAP(1, 2).dagger())
    circuit.add(gates.DEUTSCH(1, 0, 2, np.pi))
    circuit.add(gates.X(0))
    circuit.add(gates.M(qubit) for qubit in range(2))
    _, fig1 = plot_circuit(
        circuit.invert(), cluster_gates=clustered, scale=0.70, fold=fold
    )
    _, fig2 = plot_circuit(circuit, cluster_gates=clustered, scale=0.70, fold=fold)

    test_fig1_prefix = f"{BASEPATH}/test_complex_circuit_fig1_"
    test_fig2_prefix = f"{BASEPATH}/test_complex_circuit_fig2_"
    clustered_status = "true" if clustered else "false"
    fold_suffix = f"_fold_{fold}" if fold > 0 else ""
    extension = ".npy"

    test_file_fig1 = f"{test_fig1_prefix}{clustered_status}{fold_suffix}{extension}"
    test_file_fig2 = f"{test_fig2_prefix}{clustered_status}{fold_suffix}{extension}"

    assert match_figure_image(fig1, test_file_fig1)
    assert match_figure_image(fig2, test_file_fig2)


def test_align_gate():
    """Test for Align gate"""
    circuit = Circuit(3)
    circuit.add(gates.Align(0))
    _, fig = plot_circuit(circuit)
    assert match_figure_image(fig, BASEPATH + "/test_align_gate.npy") == True


@pytest.mark.parametrize("fold", [2, 3])
def test_align_gate_with_folds(fold):
    circuit = Circuit(3)
    for _ in range(4):
        circuit.add(gates.Align(0))
        circuit.add(gates.Align(1))
        circuit.add(gates.Align(2))
    _, fig = plot_circuit(circuit, fold=fold)

    test_file = f"{BASEPATH}/test_align_gate_fold_{fold}.npy"

    assert match_figure_image(fig, test_file)


@pytest.mark.parametrize("clustered", [False, True])
def test_circuit_fused_gates(clustered):
    """Test for FusedStartGateBarrier and FusedEndGateBarrier"""
    circuit = QFT(5)
    circuit.add(gates.M(qubit) for qubit in range(2))
    _, fig = plot_circuit(
        circuit.fuse(), scale=0.8, cluster_gates=clustered, style="quantumspain"
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
    circuit = Circuit(2)
    _, fig = plot_circuit(circuit)
    assert match_figure_image(fig, BASEPATH + "/test_empty_circuit.npy") == True


@pytest.mark.parametrize("clustered", [False, True])
def test_circuit_entangled_entropy(clustered):
    """Circuit test for printing entanglement entropy circuit"""
    entropy = callbacks.EntanglementEntropy([0])
    circuit = Circuit(2)
    circuit.add(gates.CallbackGate(entropy))
    circuit.add(gates.H(0))
    circuit.add(gates.CallbackGate(entropy))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CallbackGate(entropy))
    _, fig = plot_circuit(circuit, scale=0.8, cluster_gates=clustered)
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


@pytest.mark.parametrize("fold", [-1, 1])
def test_fused_gates(fold):
    """Test for gates fusion"""
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.X(0))
    circuit.add(gates.H(0))
    circuit.add(gates.X(1))
    circuit.add(gates.H(1))
    _, fig = plot_circuit(circuit.fuse(), scale=0.8, cluster_gates=False, fold=fold)

    test_file_prefix = f"{BASEPATH}/test_fused_gates"
    fold_suffix = f"_fold_{fold}" if fold > 0 else ""
    extension = ".npy"

    test_file = f"{test_file_prefix}{fold_suffix}{extension}"

    assert match_figure_image(fig, test_file)


def test_fused_gates_fold_layout_is_atomic():
    """Folded layout should keep fused boxes as a single fold unit."""
    all_gates = [
        FusedStartGateBarrier(0, 1, 3, True),
        gates.H(0),
        gates.X(0),
        gates.H(0),
        FusedEndGateBarrier(0, 1),
        FusedStartGateBarrier(1, 2, 2, True),
        gates.X(1),
        gates.H(1),
        FusedEndGateBarrier(1, 2),
    ]
    gates_plot = _process_gates(all_gates, 3)

    _, num_groups, num_folds, cols = _build_folded_gate_layout(gates_plot, fold=2)

    assert num_groups == 2
    assert num_folds == 1
    assert cols == len(gates_plot)


def test_fold_coords_without_layout():
    """Fold coordinates should fall back to index math if no layout is given."""
    col, yoff = _fold_coords(
        i=3, fold=2, num_qubits=3, num_folds=2, direction="down", folded_layout=None
    )

    assert col == 1
    assert yoff == 0


def test_fuse_cluster():
    """Test for clustering gates"""
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.M(qubit) for qubit in range(2))
    _, fig = plot_circuit(circuit.fuse())
    assert match_figure_image(fig, BASEPATH + "/test_fuse_cluster.npy") == True


@pytest.mark.parametrize("fold", [-1, 2, 3])
def test_plot_unitaries(fold):
    """Test for plotting unitaries"""
    backend = construct_backend("numpy")

    circuit = Circuit(6)
    circuit.add(gates.Unitary(random_unitary(8, backend=backend, seed=42), 1, 2, 3))
    circuit.add(gates.Unitary(random_unitary(8, backend=backend, seed=42), 0, 2, 4))
    circuit.add(gates.Unitary(random_unitary(2, backend=backend, seed=42), 2))
    circuit.add(gates.Unitary(random_unitary(2, backend=backend, seed=42), 4))
    circuit.add(gates.Unitary(random_unitary(8, backend=backend, seed=42), 0, 2, 5))
    _, fig = plot_circuit(circuit, fold=fold)

    test_file_prefix = f"{BASEPATH}/test_plot_unitaries"
    fold_suffix = f"_fold_{fold}" if fold > 0 else ""
    extension = ".npy"

    test_file = f"{test_file_prefix}{fold_suffix}{extension}"

    assert match_figure_image(fig, test_file)


def test_plot_unitaries_same_init():
    """Test for plotting unitaries with same initial parameters"""
    backend = construct_backend("numpy")

    circuit = Circuit(4)

    array = random_unitary(8, backend=backend, seed=42)

    circuit.add(gates.Unitary(array, 0, 1, 3))
    circuit.add(gates.Unitary(array, 2, 1, 3))

    _, fig = plot_circuit(circuit)
    assert (
        match_figure_image(fig, BASEPATH + "/test_plot_unitaries_same_init.npy") == True
    )


def test_plot_unitaries_different_init():
    """Test for plotting unitaries with different initial parameters"""
    backend = construct_backend("numpy")

    circuit = Circuit(4)

    circuit.add(gates.Unitary(random_unitary(8, backend=backend, seed=42), 0, 1, 3))
    circuit.add(gates.Unitary(random_unitary(8, backend=backend, seed=44), 2, 1, 3))

    _, fig = plot_circuit(circuit)
    assert (
        match_figure_image(fig, BASEPATH + "/test_plot_unitaries_different_init.npy")
        == True
    )


@pytest.mark.parametrize("fold", [-1, 3, 4, 5])
def test_plot_global_unitaries_as_circuit(fold):
    """Test for plotting global unitaries built from circuit"""
    backend = construct_backend("numpy")

    rnd_arr1 = random_unitary(8, backend=backend, seed=42)
    rnd_arr2 = random_unitary(8, backend=backend, seed=24)

    def U(n, a):
        a_bin = format(a, "0" + str(n) + "b")
        circuit = Circuit(n + 1)
        for qubit in range(n):
            if a_bin[qubit] == "1":
                circuit.add(gates.CNOT(qubit + 1, 0))
        return gates.Unitary(
            (circuit.fuse()).unitary(backend), *range(n + 1), name="U_a"
        )

    def qfgate(qbits, name):
        circuit = QFT(qbits)
        return gates.Unitary(circuit.unitary(backend), *range(qbits + 1), name=name)

    circuit = Circuit(5)
    circuit.add(U(4, 7))
    circuit.add(qfgate(2, "U_{fft}"))
    circuit.add(gates.Unitary(QFT(2).unitary(backend), *range(2), name=""))
    circuit.add(gates.Unitary(QFT(3).unitary(backend), 2, 3, 4, name="U_{FT}"))
    circuit.add(gates.Unitary(QFT(2).unitary(backend), 1, 2, name="U_K"))
    circuit.add(gates.X(1))
    circuit.add(gates.Unitary(rnd_arr1, 1, 0, 4, name=None))
    circuit.add(gates.Z(1))
    circuit.add(gates.Unitary(rnd_arr1, 2, 0, 3))
    circuit.add(gates.Y(1))
    circuit.add(gates.Unitary(rnd_arr2, 3, 0, 4))
    circuit.add(gates.Y(0))
    circuit.wire_names = ["q_a", "q_b", "q_c", "q_d", "q_e"]

    _, fig = plot_circuit(circuit, fold=fold)

    test_file_prefix = f"{BASEPATH}/test_plot_global_unitaries_as_circuit"
    fold_suffix = f"_fold_{fold}" if fold > 0 else ""
    extension = ".npy"

    test_file = f"{test_file_prefix}{fold_suffix}{extension}"

    assert match_figure_image(fig, test_file)


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
        "dpi": 200,
    }

    labels = ["q_0", "q_1", "q_2", "q_3", "q_4"]

    ax1 = _plot_quantum_circuit(gates_plot, params, labels, scale=0.7)
    ax2 = _plot_quantum_circuit(gates_plot, params, [], scale=0.7)
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


def test_render_label():
    """Test render labels"""
    assert _render_label("q_0") != ""
    assert _render_label("q_8") != ""
    assert _render_label(r"\psi") != ""


def test_render_label_empty():
    assert _render_label("") == ""


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
    circuit = Circuit(3)
    circuit.add(gates.CSX(0, 2))
    circuit.queue[0]._target_qubits = ((0, 1), (0, 2))
    circuit.queue[0]._control_qubits = ((0,), (0,))
    assert _process_gates(circuit.queue, 3) != ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
