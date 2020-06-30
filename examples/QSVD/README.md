# Quantum Singular Value Decomposer
## Problem
Much progress has been made towards a better understanding of bipartite and multipartite entanglement of quantum systems in the last decades. Among the many figures of merit that have been put forward to quantify entanglement, the von Neumann entropy stands out as it finely reveals the quantum correlations between subparts of the system. Yet, the explicit computation of this entropy, as well as many other  bipartite measures of entanglement, relies on a clever decomposition of the tensor that describes a two-party system, and in general, large investment of computational resources is required.

## Implement the solution
The code herein aims to reproduce the results of the manuscript ["Quantum Singular Value Decomposer"](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.062310). We are going to implement a quantum circuit that produces the elements of the singular value decomposition of a pure bipartite state, that we call Quantum Singular Value Decomposer. This circuit is made of two unitaries, each acting on a separate subpart of the system, that can be determined in a variational way. The frequencies of the outputs of the final state in the circuit deliver the eigenvalues of the decomposition without further treatment. Also, the eigenvectors of the decomposition can be recreated from direct action of the adjoint of the unitaries that conform the system on the computational-basis states.

The key ingredient of the algorithm is to train the circuit on exact coincidence of outputs. This is a subtle way to force a diagonal form onto the state. It also provides an example of a quantum circuit which is not trained to minimize some energy, but rather to achieve a precise relation between the superposition terms in the state.

<img src="QSVD.png" width="510px">

## How to run an example
To be completed by Diego.

## Results
