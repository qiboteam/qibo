# Quantum Singular Value Decomposer
## Problem
Much progress has been made towards a better understanding of bipartite and multipartite entanglement of quantum systems in the last decades. Among the many figures of merit that have been put forward to quantify entanglement, the von Neumann entropy stands out as it finely reveals the quantum correlations between subparts of the system. Yet, the explicit computation of this entropy, as well as many other  bipartite measures of entanglement, relies on a clever decomposition of the tensor that describes a two-party system, and in general, large investment of computational resources is required.

## Implement the solution
The code herein aims to reproduce the results of the manuscript ["Quantum Singular Value Decomposer (QSVD)"](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.062310). We are going to implement a quantum circuit that produces the elements of the singular value decomposition of a pure bipartite state, that we call Quantum Singular Value Decomposer. This circuit is made of two unitaries, each acting on a separate subpart of the system, that can be determined in a variational way. The frequencies of the outputs of the final state in the circuit deliver the eigenvalues of the decomposition without further treatment. Also, the eigenvectors of the decomposition can be recreated from direct action of the adjoint of the unitaries that conform the system on the computational-basis states.

The key ingredient of the algorithm is to train the circuit on exact coincidence of outputs. This is a subtle way to force a diagonal form onto the state. It also provides an example of a quantum circuit which is not trained to minimize some energy, but rather to achieve a precise relation between the superposition terms in the state.

<img src="QSVD.png" width="510px">

## How to run an example
To be completed by Diego.

## Results
The variational approach to the QSVD can be verified on simulations. We can considered random states such that the amplitudes are *c* = *a* + i*b* where *a* and *b* are random real numbers between -0.5 and 0.5, further restricted by a global normalization. We can start, for instance, with 6 qubit states and natural bipartition, i.e. 3 qubits in each subsystem, disregarding the presence of experimental noise and the impact of finite sampling. We may begin by considering the results for a diferent number of layers in our variational circuit. We can compute the Von Neumann entropy of the random state by simply measuring the probability of each computational-basis state after the variational circuit is trained. The following figure shows the entanglement entropy computed from the trained QSVD circuit vs. the exact entropy:

<img src="Entropy_6qubits.png" width="510px">

We have analyzed 500 random states for the 1 and 2 layers case, and 200 random states for the 3, 4 and 5 layers case. The mean number of optimization steps is of the order of a few hundreds. We can also plot the mean error and standard deviation for the different number of layers:

<img src="error.png" width="510px">

As suggested by the ["Solovay-Kitaev theorem"](https://arxiv.org/abs/quant-ph/0505030), we observe fast convergence of results for every instance we analyze. The variational circuit approaches the exact result as we increase the number of layers, whatever the entanglement is. In this respect, it is worth mentioning that we can also analyzed ["Absolute Maximally Entangled states"](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.022342), for which the convergence of the variational QSVD is fast and faithful.
