
from qibo import gates
from qibo.models.grover import Grover
from qibo.models import Circuit

# Create an oracle. Ex: Oracle that detects state |11111>
superposition = Circuit(5)
superposition.add([gates.H(i) for i in range(5)])


oracle = Circuit(5 + 1)
oracle.add(gates.X(5).controlled_by(*range(5)))
# Create superoposition circuit. Ex: Full superposition over 5 qubits.

# Generate and execute Grover class
grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)

solution, iterations = grover()

print('The solution is', solution)
print('Number of iterations needed:', iterations)