# Adiabatic evolution for solving an Exact Cover problem.

Code at: [https://github.com/Quantum-TII/qibo/tree/adiabatic-3SAT/examples/adiabatic-3SAT](https://github.com/Quantum-TII/qibo/tree/adiabatic-3SAT/examples/adiabatic-3SAT)

## Introduction

Adiabatic evolution...

An Exact Cover instance of a 3SAT problem is characterized by a set of clauses containing 3 bits that are considered satisfied if one of them is in position 1, while the other remain at 0. The solution of this instance is bitstring that fulfills all the clauses at the same time.

## Adiabatic evolution

...

## How to run the example?

Run the file `main.py` from the console to perform an adiabatic evolution for an instance of 8 qubits.

The program supports the following arguments:

- `--nqubits` (int) allows for instances with different number of qubits (default=8).
- `--T` (float) set the total time of the adiabatic evolution (default=10).
- `--dt` (float) set the interval of the calculations over the adiabatic evolution (default=1e-2).
- `--solver (str) set the type of solver for the evolution (default='exp').

The program returns:

- The most common solution found after the evolution.
- The probability of the most common solution.
- Plot detailing the evolution of the lowest two eigenvalues.
- Plot detailing the evolution of the gap energy.

Initially supported instances are of [4, 8, 10, 12, 14, 16] qubits.

The functions used in this example, including problem hamiltonian creation are included in `functions.py`.

## Create your own instances

An example of an instance for 4 qubits reads:

```text
 4 3 1
0 1 0 0
 1 2 3
 2 3 4
 1 2 4
```

The first line includes:
- number of qubits
- number of clauses
- number of 1's in the solution

The second line is the solution of the instance.

The following lines correspond to the three qubits present in each clause.

Should the solution not be known, leave an empty line in place of the solution as well as remove the number of 1's in the solution.
