.. _tutorials_transpiler:

How to modify the transpiler?
=============================

Logical quantum circuits for quantum algorithms are hardware agnostic. Usually an all-to-all qubit connectivity
is assumed while most current hardware only allows the execution of two-qubit gates on a restricted subset of qubit
pairs. Moreover, quantum devices are restricted to executing a subset of gates, referred to as native.
This means that, in order to execute circuits on a real quantum chip, they must be transformed into an equivalent,
hardware specific, circuit. The transformation of the circuit is carried out by the transpiler through the resolution
of two key steps: connectivity matching and native gates decomposition.
In order to execute a gate between two qubits that are not directly connected SWAP gates are required. This procedure is called routing.
As on NISQ devices two-qubit gates are a large source of noise, this procedure generates an overall noisier circuit.
Therefore, the goal of an efficient routing algorithm is to minimize the number of SWAP gates introduced.
An important step to ease the connectivity problem, is finding anoptimal initial mapping between logical and physical qubits.
This step is called placement.
The native gates decomposition in the transpiling procedure is performed by the unroller. An optimal decomposition uses the least amount
of two-qubit native gates. It is also possible to reduce the number of gates of the resulting circuit by exploiting
commutation relations, KAK decomposition or machine learning techniques.
Qibo implements a built-in transpiler with customizable options for each step. The main algorithms that can
be used at each transpiler step are reported below with a short description.

The initial placement can be found with one of the following procedures:
- Trivial: logical-physical qubit mapping is an identity.
- Custom: custom logical-physical qubit mapping.
- Random greedy: the best mapping is found within a set of random layouts based on a greedy policy.
- Subgraph isomorphism: the initial mapping is the one that guarantees the execution of most gates at
the beginning of the circuit without introducing any SWAP.
- Reverse traversal: this technique uses one or more reverse routing passes to find an optimal mapping by
starting from a trivial layout.

The routing problem can be solved with the following algorithms:
- Shortest paths: when unconnected logical qubits have to interact, they are moved on the chip on
the shortest path connecting them. When multiple shortest paths are present, the one that also matches
the largest number of the following two-qubit gates is chosen.
- Sabre: this heuristic routing technique uses a customizable cost function to add SWAP gates
that reduce the distance between unconnected qubits involved in two-qubit gates.

Qibolab unroller applies recursively a set of hard-coded gates decompositions in order to translate any gate into
single and two-qubit native gates. Single qubit gates are translated into U3, RX, RZ, X and Z gates. It is possible to
fuse multiple single qubit gates acting on the same qubit into a single U3 gate. For the two-qubit native gates it
is possible to use CZ and/or iSWAP. When both CZ and iSWAP gates are available the chosen decomposition is the
one that minimizes the use of two-qubit gates.

Multiple transpilation steps can be implemented using the :class:`qibo.transpiler.pipeline.Pipeline`:
.. testcode:: python

    import networkx as nx

    from qibo import Gates
    from qibo.models import Circuit
    from qibo.transpiler.pipeline import Passes, assert_transpiling
    from qibo.transpiler.abstract import NativeType
    from qibo.transpiler.optimizer import Preprocessing
    from qibo.transpiler.router import ShortestPaths
    from qibo.transpiler.unroller import NativeGates
    from qibo.transpiler.placer import Random

    # Define connectivity as nx.Graph
    def star_connectivity():
        Q = [i for i in range(5)]
        chip = nx.Graph()
        chip.add_nodes_from(Q)
        graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
        chip.add_edges_from(graph_list)
        return chip

    # Define the circuit
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CZ(0, 1))

    # Define custom passes as a list
    custom_passes = []
    # Preprocessing adds qubits in the original circuit to match the number of qubits in the chip
    custom_passes.append(Preprocessing(connectivity=star_connectivity()))
    # Placement step
    custom_passes.append(Random(connectivity=star_connectivity()))
    # Routing step
    custom_passes.append(ShortestPaths(connectivity=star_connectivity()))
    # Gate decomposition step
    custom_passes.append(NativeGates(two_qubit_natives=NativeType.iSWAP))

    # Define the general pipeline
    custom_pipeline = Passes(custom_passes, connectivity=star_connectivity(), native_gates=NativeType.iSWAP)

    # Call the transpiler pipeline on the circuit
    transpiled_circ, final_layout = custom_pipeline(circuit)

    # Optinally call assert_transpiling to check that the final circuit can be executed on hardware
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=star_connectivity(),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeType.iSWAP,
    )

In this case circuits will first be transpiled to respect the 5-qubit star connectivity, with qubit 2 as the middle qubit. This will potentially add some SWAP gates.
Then all gates will be converted to native. The :class:`qibo.transpiler.unroller.NativeGates` transpiler used in this example assumes Z, RZ, GPI2 or U3 as
the single-qubit native gates, and supports CZ and iSWAP as two-qubit natives. In this case we restricted the two-qubit gate set to CZ only.
The final_layout contains the final logical-physical qubit mapping.
