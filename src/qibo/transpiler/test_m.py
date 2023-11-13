import networkx as nx

import qibo
from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.placer import Trivial
from qibo.transpiler.router import Sabre

# new_gate = gates.M(*qubits, **gate.init_kwargs)
# new_gate.result = gate.result
# new.add(new_gate)

measurement = gates.M(0, register_name="a")
measurement2 = gates.M(1, register_name="b")

circ = Circuit(3)
circ.add(gates.H(0))
circ.add(measurement)
circ.add(measurement2)

circ2 = Circuit(3)
for gate in circ.queue:
    circ2.add(gate)

connectivity = nx.Graph()
connectivity.add_nodes_from([0, 1, 2])
connectivity.add_edges_from([(0, 1), (1, 2)])
router = Sabre(connectivity=connectivity)
initial_layout = {"q0": 0, "q1": 1, "q2": 2}

routed_circ, final_layout = router(circuit=circ, initial_layout=initial_layout)

result2 = routed_circ.execute(nshots=100)
# result2 = circ2.execute(nshots=100)
print(result2.frequencies())

print(measurement.result.has_samples())
print(measurement.result.frequencies())
