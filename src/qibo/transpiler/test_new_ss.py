import networkx as nx

from qibo import Circuit, gates
from qibo.transpiler.router import ShortestPaths


def star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


circuit = Circuit(5)
circuit.add(gates.CNOT(1, 3))
circuit.add(gates.CNOT(2, 1))
circuit.add(gates.CNOT(4, 1))
initial_layout = {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
transpiler = ShortestPaths(connectivity=star_connectivity())
routed_circ, final_layout = transpiler(circuit, initial_layout)

print(routed_circ.draw())
