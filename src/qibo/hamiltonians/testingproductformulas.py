def product_formulas(H_odd, H_even, dt, order):
    """
    we compute the results for lattice hamiltonians
    second order product formula e^(-itH_odd)e^(-itH_even)e^(-t/2 H_odd)
    H_odd and H_even are Hamiltonians
    dt is the timestep.

    Currently the hamiltonians are symbolic hamiltonians.

    Perhaps next step, go beyond just using Suzuki trotterirzation?

    We use Trotter decomposition to perform the task.
    """
    circuit_even = H_even.circuit(dt=dt)
    if order == 1:
        circuit_odd = H_odd.circuit(dt = dt)
        return circuit_odd + circuit_even
    elif order == 2:
        circuit_half_odd = H_odd.circuit(dt= dt/2)
        return circuit_half_odd + circuit_even + circuit_half_odd
    else:
        print("Currently we only support order of 1 and 2")
        return None
from qibo.symbols import X, Z
from qibo import hamiltonians

H_odd = sum(Z(i) * Z(i + 1) for i in range(3))
# periodic boundary condition term
H_odd += Z(0) * Z(3)
H_odd = hamiltonians.SymbolicHamiltonian(H_odd)
print(type(H_odd))
# X terms
H_even = sum(X(i) for i in range(4))
H_even = hamiltonians.SymbolicHamiltonian(H_even)
print(type(H_even))
circuit = product_formulas(H_odd, H_odd, 0.2, 1)
print(circuit.draw())
print(circuit)