# Quantum autoencoder for compression of data

Code at: [https://github.com/Quantum-TII/qibo/tree/master/examples/autoencoder](https://github.com/Quantum-TII/qibo/tree/master/examples/autoencoder).

## Problem overview

The task of an autoencoder given an input *x*, is to map *x* to a lower dimensional point *y* such that *x* can likely be recovered from *y*. Specifically, once we obtain *y*, we have effectively compressed the input *x*.

Given that quantum mechanics is able to generate different patterns compared to classical physics, a quantum encoder should be able to recognize patterns beyond classical capabilities.

Recall that a limiting factor for near applications is the amount of quantum resources that can be realized in an experiment. Therefore, for experiments in the near future, a quantum autoencoder which can reduce the experimental overhead in terms of these resources is especially valuable.


## Implementing the solution

The code herein aims to implement a quantum autoencoder, partly based on the manuscript ["Quantum autoencoders for efficient compression of quantum data"](https://iopscience.iop.org/article/10.1088/2058-9565/aa8072).

The idea is to benchmark the accuracy of the [Variational Quantum Eigensolver
(VQE)](https://www.nature.com/articles/ncomms5213) based on a finite-depth
variational quantum circuit encoding ground states of local Hamiltonians,
namely, the Ising and XXZ models.

## How to run an example?

To run a particular instance of the problem we have to set up the initial
arguments:
- `nqubits` (int): number of quantum bits.
- `layers` (int): number of ansatz layers.
- `compress` (int): number of compressed/discarded qubits.
- `lambdas` (list or array): different λ on the Ising model to consider for the training.

As an example, in order to compress 2 qubits on an initial quantum state with 6 qubits, and using 3 layers,
you should execute the following command:

```python
python main.py --nqubits 6 --layers 3 --compress 2 --lambdas [0.9, 0.95, 1.0, 1.05, 1.10]
```

## Interpreting results

TBC
