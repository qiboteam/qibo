from qibo.hamiltonians import SymbolicHamiltonian
from qibo import symbols
from double_bracket_evolution_oracles import *
from group_commutator_iteration_transpiler import *



def test_evolution_oracle_gci_classes():
    return 0


"""Test create evolution oracle"""

h_input = SymbolicHamiltonian( symbols.X(0) + symbols.Z(0) * symbols.X(1) + symbols.Y(2), nqubits = 3 )

#By default this initializes with text_strings oracle type
input_hamiltonian_evolution_oracle = EvolutionOracle(h_input, "ZX")

c1 = input_hamiltonian_evolution_oracle.circuit(0.1)
assert isinstance( c1, str), "Should be a string here"
print( c1 )

input_hamiltonian_evolution_oracle.mode_evolution_oracle = EvolutionOracleType.hamiltonian_simulation
c2 =input_hamiltonian_evolution_oracle.circuit(0.2)
assert isinstance( c2, type(h_input.circuit(0.1)) ), "Should be a qibo.Circuit here"
c2.draw()
d_0 = SymbolicHamiltonian(symbols.Z(0) * symbols.Z(1) + symbols.Z(2), nqubits = 3 )
gci = GroupCommutatorIterationWithEvolutionOracles( input_hamiltonian_evolution_oracle )
#By default this will test the dephasing oracle
gci(0.2, d_0)
