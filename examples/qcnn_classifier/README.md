# Quantum Convolutional Neural Network Classifier

Code at: [https://github.com/qiboteam/qibo/tree/master/examples/qcnn_classifier]
(https://github.com/qiboteam/qibo/tree/master/examples/qcnn_classifier).

## Problem overview
This tutorial implements a simplified Quantum Convolutional Neural Network (QCNN), a proposed quantum analogue to a classical convolutional neural network that is also translationally invariant. This example demonstrates how to detect certain properties of a quantum data source, such as a quantum sensor or a complex simulation from a device. The quantum data source being a cluster state that may or may not have an excitation—what the QCNN will learn to detect. 
You will prepare a cluster state and train a quantum classifier to detect if it is "excited" or not. The cluster state is highly entangled but not necessarily difficult for a classical computer. For this classification task you will implement a deep MERA-like QCNN architecture since:
1. Like the QCNN, the cluster state on a ring is translationally invariant.
2. The cluster state is highly entangled.
This architecture should be effective at reducing entanglement, obtaining the classification by reading out a single qubit.

![qcnn_architecture](images/qcnn_architecture.png)

## How to run an example
To run a particular instance of the problem, we have to set up the initial arguments:
- `nqubits` (int): number of quantum bits. It must be larger than 1. 
- `nlayers` (int): number of layers of the varitional ansatz.
- `nclasses` (int): number of classes of the training set (default=2).
- `params`
- `init_theta`: list or numpy.array with the angles to be used in the circuit
- `data`
- `labels`: numpy.array with the quantum state to be Schmidt-decomposed
- `nshots` (int):number of runs of the circuit during the sampling process (default=10000)
- `method` (string): str 'classical optimizer for the minimization'. All methods from scipy.optimize.minmize are suported (default='Powell').

To run an example ..., you should execute the following command:

```bash
python main.py
```

To run an example with different values,  type for example:

```bash
python main.py --nqubits 4 --nlayers 5 --nshots 100000 --training
```

Note that ...

## Results
