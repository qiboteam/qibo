from qibo.hamiltonians import SymbolicHamiltonian
from qibo import symbols
from double_bracket_evolution_oracles import *
from group_commutator_iteration_transpiler import *



def test_evolution_oracle_gci_classes():
    return 0


"""Test create evolution oracle"""

h_input = SymbolicHamiltonian( symbols.X(0) + symbols.Z(0) * symbols.X(1) + symbols.Y(2) + symbols.Y(1) * symbols.Y(2), nqubits = 3 )

## Test initialization of evolution oracles
#By default EvolutionOracle initializes with text_strings oracle type
input_hamiltonian_evolution_oracle_text_strings = EvolutionOracle(h_input, "ZX", mode_evolution_oracle = EvolutionOracleType.text_strings)

c1 = input_hamiltonian_evolution_oracle_text_strings.circuit(0.1)
if input_hamiltonian_evolution_oracle_text_strings.mode_evolution_oracle is EvolutionOracleType.text_strings:
    assert isinstance( c1, str), "Should be a string here"
print( c1 )

# Initialize with EvolutionOracleType hamiltonian_simulation
input_hamiltonian_evolution_oracle_hamiltonian_simulation = EvolutionOracle(h_input, "ZX",
                                                                mode_evolution_oracle = EvolutionOracleType.hamiltonian_simulation)
input_hamiltonian_evolution_oracle_hamiltonian_simulation.please_be_verbose = True
c2 = input_hamiltonian_evolution_oracle_hamiltonian_simulation.circuit(2)
if input_hamiltonian_evolution_oracle_hamiltonian_simulation.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
    print(type(c2))
    assert isinstance( c2, type(h_input.circuit(0.1)) ), "Should be a qibo.Circuit here"
print(c2.draw())

# Initialize with EvolutionOracleType numerical
input_hamiltonian_evolution_oracle_numerical = EvolutionOracle(h_input, "ZX",
                                                                mode_evolution_oracle = EvolutionOracleType.numerical )

c3 = input_hamiltonian_evolution_oracle_numerical.circuit(2)
if input_hamiltonian_evolution_oracle_numerical.mode_evolution_oracle is EvolutionOracleType.numerical:
    assert isinstance( c3, type(h_input.exp(0.1)) ), "Should be a np.array here"
print(np.linalg.norm( c2.unitary() - c3 ))
U2 = c2.unitary()
input_hamiltonian_evolution_oracle_hamiltonian_simulation.mode_evolution_oracle = EvolutionOracleType.numerical
print(np.linalg.norm( U2 - input_hamiltonian_evolution_oracle_hamiltonian_simulation.circuit(2)))

if 0:

    ## Test more fancy functionalities

    gci = GroupCommutatorIterationWithEvolutionOracles( input_hamiltonian_evolution_oracle_hamiltonian_simulation )
    d_0 = SymbolicHamiltonian(symbols.Z(0) * symbols.Z(1) + symbols.Z(2), nqubits = 3 )

    query_list = gci.group_commutator_query_list( 0.2, d_0, input_hamiltonian_evolution_oracle_hamiltonian_simulation )
    gci(0.2, d_0)


    from functools import reduce
    before_circuit = reduce(Circuit.__add__,  query_list['backwards'])
    after_circuit = reduce(Circuit.__add__,  query_list['forwards'])
    print( before_circuit.draw() )
    print( after_circuit.draw() )


    frame_shifted_input_hamiltonian_evolution_oracle = FrameShiftedEvolutionOracle( 
            input_hamiltonian_evolution_oracle_hamiltonian_simulation, 'Step 1', before_circuit, after_circuit)

    print( np.linalg.norm( gci.h.exp(0.3) - frame_shifted_input_hamiltonian_evolution_oracle.circuit(0.3).unitary()))


