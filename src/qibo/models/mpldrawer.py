# Simplified Plotting Routines for Quantum Circuits by Rick Muller
# https://github.com/rpmuller/PlotQCircuit
import matplotlib
import numpy as np

def _plot_quantum_circuit(gates,inits={},labels=[],plot_labels=True,**kwargs):
    """Use Matplotlib to plot a quantum circuit.
    gates     List of tuples for each gate in the quantum circuit.
              (name,target,control1,control2...). Targets and controls initially
              defined in terms of labels.
    inits     Initialization list of gates, optional

    kwargs    Can override plot_parameters
    """
    plot_params = dict(scale = 1.0,fontsize = 14.0, linewidth = 1.0,
                         control_radius = 0.05, not_radius = 0.15,
                         swap_delta = 0.08, label_buffer = 0.0)
    plot_params.update(kwargs)
    scale = plot_params['scale']

    # Create labels from gates. This will become slow if there are a lot
    #  of gates, in which case move to an ordered dictionary
    if not labels:
        labels = []
        for i,gate in _enumerate_gates(gates):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)

    nq = len(labels)
    ng = len(gates)
    wire_grid = np.arange(0.0, nq*scale, scale, dtype=float)
    gate_grid = np.arange(0.0, ng*scale, scale, dtype=float)

    fig,ax = _setup_figure(nq,ng,gate_grid,wire_grid,plot_params)

    measured = _measured_wires(gates,labels)
    _draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured)

    if plot_labels:
        _draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params)

    _draw_gates(ax,gates,labels,gate_grid,wire_grid,plot_params,measured)
    return ax

def _enumerate_gates(l,schedule=False):
    "Enumerate the gates in a way that can take l as either a list of gates or a schedule"
    if schedule:
        for i,gates in enumerate(l):
            for gate in gates:
                yield i,gate
    else:
        for i,gate in enumerate(l):
            yield i,gate
    return

def _measured_wires(l,labels,schedule=False):
    "measured[i] = j means wire i is measured at step j"
    measured = {}
    for i,gate in _enumerate_gates(l,schedule=schedule):
        name,target = gate[:2]
        j = _get_flipped_index(target,labels)
        if name.startswith('M'):
            measured[j] = i
    return measured

def _draw_gates(ax,l,labels,gate_grid,wire_grid,plot_params,measured={},schedule=False):
    for i,gate in _enumerate_gates(l,schedule=schedule):
        _draw_target(ax,i,gate,labels,gate_grid,wire_grid,plot_params)
        if len(gate) > 2: # Controlled
            _draw_controls(ax,i,gate,labels,gate_grid,wire_grid,plot_params,measured)
    return

def _draw_controls(ax,i,gate,labels,gate_grid,wire_grid,plot_params,measured={}):
    linewidth = plot_params['linewidth']
    scale = plot_params['scale']
    control_radius = plot_params['control_radius']

    name,target = gate[:2]
    target_index = _get_flipped_index(target,labels)
    controls = gate[2:]
    control_indices = _get_flipped_indices(controls,labels)
    gate_indices = control_indices + [target_index]
    min_wire = min(gate_indices)
    max_wire = max(gate_indices)
    _line(ax,gate_grid[i],gate_grid[i],wire_grid[min_wire],wire_grid[max_wire],plot_params)
    ismeasured = False
    for index in control_indices:
        if measured.get(index,1000) < i:
            ismeasured = True
    if ismeasured:
        dy = 0.04 # TODO: put in plot_params
        _line(ax,gate_grid[i]+dy,gate_grid[i]+dy,wire_grid[min_wire],wire_grid[max_wire],plot_params)

    for ci in control_indices:
        x = gate_grid[i]
        y = wire_grid[ci]
        if name in ['SWAP']:
            _swapx(ax,x,y,plot_params)
        else:
            _cdot(ax,x,y,plot_params)
    return

def _draw_target(ax,i,gate,labels,gate_grid,wire_grid,plot_params):
    target_symbols = dict(CNOT='X',CPHASE='Z',NOP='',CX='X',CZ='Z')
    name,target = gate[:2]
    symbol = target_symbols.get(name,name) # override name with target_symbols
    x = gate_grid[i]
    target_index = _get_flipped_index(target,labels)
    y = wire_grid[target_index]
    if not symbol: return
    if name in ['CNOT','TOFFOLI']:
        _oplus(ax,x,y,plot_params)
    elif name in ['CPHASE']:
        _cdot(ax,x,y,plot_params)
    elif name in ['SWAP']:
        _swapx(ax,x,y,plot_params)
    else:
        _text(ax,x,y,symbol,plot_params,box=True)
    return

def _line(ax,x1,x2,y1,y2,plot_params):
    Line2D = matplotlib.lines.Line2D
    line = Line2D((x1,x2), (y1,y2),
        color='k',lw=plot_params['linewidth'])
    ax.add_line(line)

def _text(ax,x,y,textstr,plot_params,box=False):
    linewidth = plot_params['linewidth']
    fontsize = plot_params['fontsize']
    if box:
        bbox = dict(ec='k',fc='w',fill=True,lw=linewidth)
    else:
        bbox= dict(fill=False,lw=0)
    ax.text(x,y,textstr,color='k',ha='center',va='center',bbox=bbox,size=fontsize)
    return

def _oplus(ax,x,y,plot_params):
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle
    not_radius = plot_params['not_radius']
    linewidth = plot_params['linewidth']
    c = Circle((x, y),not_radius,ec='k',
               fc='w',fill=False,lw=linewidth)
    ax.add_patch(c)
    _line(ax,x,x,y-not_radius,y+not_radius,plot_params)
    return

def _cdot(ax,x,y,plot_params):
    Circle = matplotlib.patches.Circle
    control_radius = plot_params['control_radius']
    scale = plot_params['scale']
    linewidth = plot_params['linewidth']
    c = Circle((x, y),control_radius*scale,
        ec='k',fc='k',fill=True,lw=linewidth)
    ax.add_patch(c)
    return

def _swapx(ax,x,y,plot_params):
    d = plot_params['swap_delta']
    linewidth = plot_params['linewidth']
    _line(ax,x-d,x+d,y-d,y+d,plot_params)
    _line(ax,x-d,x+d,y+d,y-d,plot_params)
    return

def _setup_figure(nq,ng,gate_grid,wire_grid,plot_params):
    scale = plot_params['scale']
    fig = matplotlib.pyplot.figure(
        figsize=(ng*scale, nq*scale),
        facecolor='w',
        edgecolor='w'
    )
    ax = fig.add_subplot(1, 1, 1,frameon=True)
    ax.set_axis_off()
    offset = 0.5*scale
    ax.set_xlim(gate_grid[0] - offset, gate_grid[-1] + offset)
    ax.set_ylim(wire_grid[0] - offset, wire_grid[-1] + offset)
    ax.set_aspect('equal')
    return fig,ax

def _draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured={}):
    scale = plot_params['scale']
    linewidth = plot_params['linewidth']
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        _line(ax,gate_grid[0]-scale,gate_grid[-1]+scale,wire_grid[i],wire_grid[i],plot_params)

    # Add the doubling for measured wires:
    dy=0.04 # TODO: add to plot_params
    for i in measured:
        j = measured[i]
        _line(ax,gate_grid[j],gate_grid[-1]+scale,wire_grid[i]+dy,wire_grid[i]+dy,plot_params)
    return

def _draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params):
    scale = plot_params['scale']
    label_buffer = plot_params['label_buffer']
    fontsize = plot_params['fontsize']
    nq = len(labels)
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        j = _get_flipped_index(labels[i],labels)
        _text(ax,xdata[0]-label_buffer,wire_grid[j],_render_label(labels[i],inits),plot_params)
    return

def _get_flipped_index(target,labels):
    """Get qubit labels from the rest of the line,and return indices

    >>> _get_flipped_index('q0', ['q0', 'q1'])
    1
    >>> _get_flipped_index('q1', ['q0', 'q1'])
    0
    """
    nq = len(labels)
    i = labels.index(target)
    return nq-i-1

def _get_flipped_indices(targets,labels): return [_get_flipped_index(t,labels) for t in targets]

def _render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> _render_label('q0')
    '$|q0\\\\rangle$'
    >>> _render_label('q0', {'q0':'0'})
    '$|0\\\\rangle$'
    """
    if label in inits:
        s = inits[label]
        if s is None:
            return ''
        else:
            return r'$|%s\rangle$' % inits[label]
    return r'$|%s\rangle$' % label

def plot_qibo_circuit(circuit, scale):
    gates_plot = []
    inits = []

    for gate in circuit.queue:
        init_label = gate.draw_label.upper()
        inits.append(init_label)
        item = ()
        item += (init_label, )

        if len(gate.init_args) == 1:
            item += ("q_" + str(gate.init_args[0]), )
        else:
            for i in reversed(range(len(gate.init_args))):
                item += ("q_" + str(gate.init_args[i]), )

        gates_plot.append(item)

    _plot_quantum_circuit(gates_plot, inits, scale = scale)
