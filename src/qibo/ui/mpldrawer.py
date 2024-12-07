# Some functions in MPLDrawer are from code provided by Rick Muller
# Simplified Plotting Routines for Quantum Circuits
# https://github.com/rpmuller/PlotQCircuit
#
import json
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qibo import gates

from .drawer_utils import FusedEndGateBarrier, FusedStartGateBarrier

UI = Path(__file__).parent
STYLE = json.loads((UI / "styles.json").read_text())
SYMBOLS = json.loads((UI / "symbols.json").read_text())

PLOT_PARAMS = {
    "scale": 1.0,
    "fontsize": 14.0,
    "linewidth": 1.0,
    "control_radius": 0.05,
    "not_radius": 0.15,
    "swap_delta": 0.08,
    "label_buffer": 0.0,
    "dpi": 100,
    "facecolor": "w",
    "edgecolor": "#000000",
    "fillcolor": "#000000",
    "linecolor": "k",
    "textcolor": "k",
    "gatecolor": "w",
    "controlcolor": "#000000",
}


def _plot_quantum_schedule(
    schedule, inits, plot_params, labels=[], plot_labels=True, **kwargs
):
    """Use Matplotlib to plot a queue of quantum circuit.

    Args:
        schedule (list):  List of time steps, each containing a sequence of gates during that step.
        Each gate is a tuple containing (name,target,control1,control2...). Targets and controls initially defined in terms of labels.

        inits (list): Initialization list of gates.

        plot_params (list): Style plot configuration.

        labels (list): List of qubit labels, optional.

        kwargs (list): Variadic list that can override plot parameters.

    Returns:
        matplotlib.axes.Axes: An Axes object encapsulates all the plt elements of a plot in a figure.
    """

    return _plot_quantum_circuit(
        schedule,
        inits,
        plot_params,
        labels=labels,
        plot_labels=plot_labels,
        schedule=True,
        **kwargs
    )


def _plot_quantum_circuit(
    gates, inits, plot_params, labels=[], plot_labels=True, schedule=False, **kwargs
):
    """Use Matplotlib to plot a quantum circuit.

    Args:
        gates (list): List of tuples for each gate in the quantum circuit. (name,target,control1,control2...).
        Targets and controls initially defined in terms of labels.

        inits (list): Initialization list of gates.

        plot_params (list): Style plot configuration.

        labels (list): List of qubit labels. (optional).

        kwargs (list): Variadic list that can override plot parameters.

    Returns:
        matplotlib.axes.Axes: An Axes object encapsulates all the plt elements of a plot in a figure.
    """

    plot_params.update(kwargs)
    scale = plot_params["scale"]

    # Create labels from gates. This will become slow if there are a lot
    #  of gates, in which case move to an ordered dictionary
    if not labels:
        labels = []
        for i, gate in _enumerate_gates(gates, schedule=schedule):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)

    nq = len(labels)
    ng = len(gates)

    wire_grid = np.arange(0.0, nq * scale, scale, dtype=float)

    gate_grid = np.arange(0.0, (nq if ng == 0 else ng) * scale, scale, dtype=float)
    ax, _ = _setup_figure(
        nq, (nq if ng == 0 else ng), gate_grid, wire_grid, plot_params
    )

    measured = None if ng == 0 else _measured_wires(gates, labels, schedule=schedule)
    _draw_wires(ax, nq, gate_grid, wire_grid, plot_params, measured)

    if plot_labels:
        _draw_labels(ax, labels, inits, gate_grid, wire_grid, plot_params)

    if ng > 0:
        _draw_gates(
            ax,
            gates,
            labels,
            gate_grid,
            wire_grid,
            plot_params,
            measured,
            schedule=schedule,
        )

    return ax


def _enumerate_gates(gates_plot, schedule=False):
    """Enumerate the gates in a way that can take l as either a list of gates or a schedule

    Args:
        gates_plot (list): List of gates to plot.

        schedule (bool): Check whether process single gate or array of gates at a time.

    Returns:
        int: Index of gate or list of gates.

        list: Processed list of gates ready to plot.
    """

    if schedule:
        for i, gates in enumerate(gates_plot):
            for gate in gates:
                yield i, gate
    else:
        for i, gate in enumerate(gates_plot):
            yield i, gate


def _measured_wires(gates_plot, labels, schedule=False):
    measured = {}
    for i, gate in _enumerate_gates(gates_plot, schedule=schedule):
        name, target = gate[:2]
        j = _get_flipped_index(target, labels)
        if name.startswith("M"):
            measured[j] = i
    return measured


def _draw_gates(
    ax,
    gates_plot,
    labels,
    gate_grid,
    wire_grid,
    plot_params,
    measured={},
    schedule=False,
):
    for i, gate in _enumerate_gates(gates_plot, schedule=schedule):
        _draw_target(ax, i, gate, labels, gate_grid, wire_grid, plot_params)
        if len(gate) > 2:  # Controlled
            _draw_controls(
                ax, i, gate, labels, gate_grid, wire_grid, plot_params, measured
            )


def _draw_controls(ax, i, gate, labels, gate_grid, wire_grid, plot_params, measured={}):

    name, target = gate[:2]

    if "FUSEDENDGATEBARRIER" in name:
        return

    linewidth = plot_params["linewidth"]
    scale = plot_params["scale"]
    control_radius = plot_params["control_radius"]

    target_index = _get_flipped_index(target, labels)
    controls = gate[2:]
    control_indices = _get_flipped_indices(controls, labels)
    gate_indices = control_indices + [target_index]
    min_wire = min(gate_indices)
    max_wire = max(gate_indices)

    if "FUSEDSTARTGATEBARRIER" in name:
        equal_qbits = False
        if "@EQUAL" in name:
            name = name.replace("@EQUAL", "")
            equal_qbits = True
        nfused = int(name.replace("FUSEDSTARTGATEBARRIER", ""))
        dx_right = 0.30
        dx_left = 0.30
        dy = 0.25
        _rectangle(
            ax,
            gate_grid[i + 1] - dx_left,
            gate_grid[i + nfused] + dx_right,
            wire_grid[min_wire] - dy - (0 if not equal_qbits else -0.9 * scale),
            wire_grid[max_wire] + dy,
            plot_params,
        )
    else:

        _line(
            ax,
            gate_grid[i],
            gate_grid[i],
            wire_grid[min_wire],
            wire_grid[max_wire],
            plot_params,
        )

        for ci in control_indices:
            x = gate_grid[i]
            y = wire_grid[ci]

            is_dagger = False
            if name[-2:] == "DG":
                name = name.replace("DG", "")
                is_dagger = True

            if name == "SWAP":
                _swapx(ax, x, y, plot_params)
            elif name in [
                "ISWAP",
                "SISWAP",
                "FSWAP",
                "FSIM",
                "SYC",
                "GENERALIZEDFSIM",
                "RXX",
                "RYY",
                "RZZ",
                "RZX",
                "RXXYY",
                "G",
                "RBS",
                "ECR",
                "MS",
            ]:

                symbol = SYMBOLS.get(name, name)

                if is_dagger:
                    symbol += r"$\rm{^{\dagger}}$"

                _text(ax, x, y, symbol, plot_params, box=True)

            else:
                _cdot(ax, x, y, plot_params)


def _draw_target(ax, i, gate, labels, gate_grid, wire_grid, plot_params):
    name, target = gate[:2]

    if "FUSEDSTARTGATEBARRIER" in name or "FUSEDENDGATEBARRIER" in name:
        return

    is_dagger = False
    if name[-2:] == "DG":
        name = name.replace("DG", "")
        is_dagger = True

    symbol = SYMBOLS.get(name, name)  # override name with symbols

    if is_dagger:
        symbol += r"$\rm{^{\dagger}}$"

    x = gate_grid[i]
    target_index = _get_flipped_index(target, labels)
    y = wire_grid[target_index]
    if name in ["CNOT", "TOFFOLI"]:
        _oplus(ax, x, y, plot_params)
    elif name == "SWAP":
        _swapx(ax, x, y, plot_params)
    else:
        if name == "ALIGN":
            symbol = "A({})".format(target[2:])
        _text(ax, x, y, symbol, plot_params, box=True)


def _line(ax, x1, x2, y1, y2, plot_params):
    Line2D = matplotlib.lines.Line2D
    line = Line2D(
        (x1, x2), (y1, y2), color=plot_params["linecolor"], lw=plot_params["linewidth"]
    )
    ax.add_line(line)


def _text(ax, x, y, textstr, plot_params, box=False):
    linewidth = plot_params["linewidth"]
    fontsize = (
        12.0
        if _check_list_str(["dagger", "sqrt"], textstr)
        else plot_params["fontsize"]
    )

    if box:
        bbox = dict(
            ec=plot_params["edgecolor"],
            fc=plot_params["gatecolor"],
            fill=True,
            lw=linewidth,
        )
    else:
        bbox = dict(fill=False, lw=0)
    ax.text(
        x,
        y,
        textstr,
        color=plot_params["textcolor"],
        ha="center",
        va="center",
        bbox=bbox,
        size=fontsize,
    )


def _oplus(ax, x, y, plot_params):
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle
    not_radius = plot_params["not_radius"]
    linewidth = plot_params["linewidth"]
    c = Circle(
        (x, y),
        not_radius,
        ec=plot_params["edgecolor"],
        fc=plot_params["gatecolor"],
        fill=True,
        lw=linewidth,
    )
    ax.add_patch(c)
    _line(ax, x, x, y - not_radius, y + not_radius, plot_params)


def _cdot(ax, x, y, plot_params):
    Circle = matplotlib.patches.Circle
    control_radius = plot_params["control_radius"]
    scale = plot_params["scale"]
    linewidth = plot_params["linewidth"]
    c = Circle(
        (x, y),
        control_radius * scale,
        ec=plot_params["edgecolor"],
        fc=plot_params["controlcolor"],
        fill=True,
        lw=linewidth,
    )
    ax.add_patch(c)


def _swapx(ax, x, y, plot_params):
    d = plot_params["swap_delta"]
    linewidth = plot_params["linewidth"]
    _line(ax, x - d, x + d, y - d, y + d, plot_params)
    _line(ax, x - d, x + d, y + d, y - d, plot_params)


def _setup_figure(nq, ng, gate_grid, wire_grid, plot_params):
    scale = plot_params["scale"]
    fig = plt.figure(
        figsize=(ng * scale, nq * scale),
        facecolor=plot_params["facecolor"],
        edgecolor=plot_params["edgecolor"],
        dpi=plot_params["dpi"],
    )
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    ax.set_axis_off()
    offset = 0.5 * scale
    ax.set_xlim(gate_grid[0] - offset, gate_grid[-1] + offset)
    ax.set_ylim(wire_grid[0] - offset, wire_grid[-1] + offset)
    ax.set_aspect("equal")
    return ax, fig


def _draw_wires(ax, nq, gate_grid, wire_grid, plot_params, measured={}):
    scale = plot_params["scale"]
    linewidth = plot_params["linewidth"]
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        _line(
            ax,
            gate_grid[0] - scale,
            gate_grid[-1] + scale,
            wire_grid[i],
            wire_grid[i],
            plot_params,
        )


def _draw_labels(ax, labels, inits, gate_grid, wire_grid, plot_params):
    scale = plot_params["scale"]
    label_buffer = plot_params["label_buffer"]
    fontsize = plot_params["fontsize"]
    nq = len(labels)
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        j = _get_flipped_index(labels[i], labels)
        _text(
            ax,
            xdata[0] - label_buffer,
            wire_grid[j],
            _render_label(labels[i], inits),
            plot_params,
        )


def _get_min_max_qbits(gates):
    def _get_all_tuple_items(iterable):
        t = []
        for each in iterable:
            t.extend(list(each) if isinstance(each, tuple) else [each])
        return tuple(t)

    all_qbits = []
    c_qbits = [t._control_qubits for t in gates.gates]
    t_qbits = [t._target_qubits for t in gates.gates]
    c_qbits = _get_all_tuple_items(c_qbits)
    t_qbits = _get_all_tuple_items(t_qbits)
    all_qbits.append(c_qbits + t_qbits)

    flatten_arr = _get_all_tuple_items(all_qbits)
    return min(flatten_arr), max(flatten_arr)


def _get_flipped_index(target, labels):
    nq = len(labels)
    i = labels.index(target)
    return nq - i - 1


def _rectangle(ax, x1, x2, y1, y2, plot_style):
    Rectangle = matplotlib.patches.Rectangle
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    xm = x + w / 2.0
    ym = y + h / 2.0

    rect = Rectangle(
        (x, y),
        w,
        h,
        ec=plot_style["edgecolor"],
        fc=plot_style["fillcolor"],
        fill=False,
        lw=plot_style["linewidth"],
        label="",
    )
    ax.add_patch(rect)


def _get_flipped_indices(targets, labels):
    return [_get_flipped_index(t, labels) for t in targets]


def _render_label(label, inits={}):
    if label in inits:
        s = inits[label]
        if s is None:
            return ""
        else:
            return r"$|%s\rangle$" % inits[label]
    return r"$|%s\rangle$" % label


def _check_list_str(substrings, string):
    return any(item in string for item in substrings)


def _make_cluster_gates(gates_items):
    """
    Given a list of gates from a Qibo circuit, this fucntion gathers all gates to reduce the depth of the circuit making the circuit more user-friendly to avoid very large circuits printed on screen.

    Args:
        gates_items (list): List of gates to gather for circuit depth reduction.

    Returns:
        list: List of gathered gates.
    """

    temp_gates = []
    temp_mgates = []
    cluster_gates = []

    for item in gates_items:
        if len(item) == 2:  # single qubit gates
            if item[0] == "MEASURE":
                temp_mgates.append(item)
            else:
                if len(temp_gates) > 0:
                    if item[1] in [tup[1] for tup in temp_gates]:
                        cluster_gates.append(temp_gates)
                        temp_gates = []
                        temp_gates.append(item)
                    else:
                        temp_gates.append(item)
                else:
                    temp_gates.append(item)
        else:
            if len(temp_gates) > 0:
                cluster_gates.append(temp_gates)
                temp_gates = []

            if len(temp_mgates) > 0:
                cluster_gates.append(temp_mgates)
                temp_mgates = []

            cluster_gates.append([item])

    if len(temp_gates) > 0:
        cluster_gates.append(temp_gates)

    if len(temp_mgates) > 0:
        cluster_gates.append(temp_mgates)

    return cluster_gates


def _process_gates(array_gates, nqubits):
    """
    Transforms the list of gates given by the Qibo circuit into a list of gates with a suitable structre to print on screen with matplotlib.

    Args:
        array_gates (list): List of gates provided by the Qibo circuit.
        nqubits (int): Number of circuit qubits

    Returns:
        list: List of suitable gates to plot with matplotlib.
    """

    if len(array_gates) == 0:
        return []

    gates_plot = []

    for gate in array_gates:
        init_label = gate.name.upper()

        if init_label == "CCX":
            init_label = "TOFFOLI"
        elif init_label == "CX":
            init_label = "CNOT"
        elif _check_list_str(["SX", "CSX"], init_label):
            is_dagger = init_label[-2:] == "DG"
            init_label = (
                r"$\rm{\sqrt{X}}^{\dagger}$" if is_dagger else r"$\rm{\sqrt{X}}$"
            )
        elif (
            len(gate._control_qubits) > 0
            and "C" in init_label[0]
            and "CNOT" != init_label
        ):
            init_label = gate.draw_label.upper()

        if init_label in [
            "ID",
            "MEASURE",
            "KRAUSCHANNEL",
            "UNITARYCHANNEL",
            "DEPOLARIZINGCHANNEL",
            "READOUTERRORCHANNEL",
        ]:
            for qbit in gate._target_qubits:
                item = (init_label,)
                item += ("q_" + str(qbit),)
                gates_plot.append(item)
        elif init_label == "ENTANGLEMENTENTROPY":
            for qbit in list(range(nqubits)):
                item = (init_label,)
                item += ("q_" + str(qbit),)
                gates_plot.append(item)
        else:
            item = ()
            item += (init_label,)

            for qbit in gate._target_qubits:
                if type(qbit) is tuple:
                    item += ("q_" + str(qbit[0]),)
                else:
                    item += ("q_" + str(qbit),)

            for qbit in gate._control_qubits:
                if type(qbit) is tuple:
                    item += ("q_" + str(qbit[0]),)
                else:
                    item += ("q_" + str(qbit),)

            gates_plot.append(item)

    return gates_plot


def _plot_params(style: Union[dict, str, None]) -> dict:
    """
    Given a style name, the function gets the style configuration, if the style is not available, it return the default style. It is allowed to give a custom dictionary to give the circuit a style.

    Args:
        style (Union[dict, str, None]): Name of the style.

    Returns:
        dict: Style configuration.
    """
    if not isinstance(style, dict):
        style = (
            STYLE.get(style)
            if (style is not None and style in STYLE.keys())
            else STYLE["default"]
        )

    return style


def plot_circuit(circuit, scale=0.6, cluster_gates=True, style=None):
    """Main matplotlib plot function for Qibo circuit

    Args:
        circuit (qibo.models.circuit.Circuit): A Qibo circuit to plot.
        scale (float): Scaling factor for matplotlib output drawing.
        cluster_gates (boolean): Group (or not) circuit gates on drawing.
        style (Union[dict, str, None]): Style applied to the circuit, it can a built-in sytle or custom
        (built-in styles: garnacha, fardelejo, quantumspain, color-blind, cachirulo or custom dictionary).

    Returns:
        matplotlib.axes.Axes: Axes object that encapsulates all the elements of an individual plot in a figure.

        matplotlib.figure.Figure: A matplotlib figure object.

    Example:

        .. testcode::

            import matplotlib.pyplot as plt
            import qibo
            from qibo import gates, models
            from qibo.models import QFT

            # new plot function based on matplotlib
            from qibo.ui import plot_circuit

            %matplotlib inline

            # create a 5-qubits QFT circuit
            c = QFT(5)
            c.add(gates.M(qubit) for qubit in range(2))

            # print circuit with default options (default black & white style, scale factor of 0.6 and clustered gates)
            plot_circuit(c);

            # print the circuit with built-int style "garnacha", clustering gates and a custom scale factor
            # built-in styles: "garnacha", "fardelejo", "quantumspain", "color-blind", "cachirulo" or custom dictionary
            plot_circuit(c, scale = 0.8, cluster_gates = True, style="garnacha");

            # plot the Qibo circuit with a custom style
            custom_style = {
                "facecolor" : "#6497bf",
                "edgecolor" : "#01016f",
                "linecolor" : "#01016f",
                "textcolor" : "#01016f",
                "fillcolor" : "#ffb9b9",
                "gatecolor" : "#d8031c",
                "controlcolor" : "#360000"
            }

            plot_circuit(c, scale = 0.8, cluster_gates = True, style=custom_style);
    """

    params = PLOT_PARAMS.copy()
    params.update(_plot_params(style))

    inits = list(range(circuit.nqubits))

    labels = []
    for i in range(circuit.nqubits):
        labels.append("q_" + str(i))

    all_gates = []
    for gate in circuit.queue:
        if isinstance(gate, gates.FusedGate):
            min_q, max_q = _get_min_max_qbits(gate)

            fgates = None

            if cluster_gates:
                fgates = _make_cluster_gates(
                    _process_gates(gate.gates, circuit.nqubits)
                )
            else:
                fgates = _process_gates(gate.gates, circuit.nqubits)

            l_gates = len(gate.gates)
            equal_qbits = False
            if min_q != max_q:
                l_gates = len(fgates)
            else:
                max_q += 1
                equal_qbits = True

            all_gates.append(FusedStartGateBarrier(min_q, max_q, l_gates, equal_qbits))
            all_gates += gate.gates
            all_gates.append(FusedEndGateBarrier(min_q, max_q))
        else:
            all_gates.append(gate)

    gates_plot = _process_gates(all_gates, circuit.nqubits)

    if cluster_gates and len(gates_plot) > 0:
        gates_cluster = _make_cluster_gates(gates_plot)
        ax = _plot_quantum_schedule(gates_cluster, inits, params, labels, scale=scale)
        return ax, ax.figure

    ax = _plot_quantum_circuit(gates_plot, inits, params, labels, scale=scale)
    return ax, ax.figure
