from qibo import Circuit, gates
from qibo.backends import CliffordBackend, NumpyBackend
from qibo.backends.clifford_operations import tableau_to_generators

c = Circuit(2)
c.add(gates.H(0))
c.add(gates.CNOT(0, 1))
# c.add(gates.H(1))
c.add(gates.M(0, 1))

for backend in (CliffordBackend(), NumpyBackend()):
    print(backend.execute_circuit(c))
