## Grover's Algorithm for solving Satisfiability Problems

# Introduction

Grover's Algorithm is an example of the advantages a quantum computer has over a classical computer in the task of searching databases. This program applies a Grover search on a Satisfiability problem, more precisely an Exact Cover instance of a 3SAT problem.


An Exact Cover instance of a 3SAT problem is characterized by a set of clauses containing 3 qubits that are considered satisfied if one of them is in position 1, while the other remain at 0. The solution of this instance is bitstring that fulfills all the clauses at the same time. 

# Grover's search algorithm

The algorithm proposed by Grover in "A fast quantum mechanical algorithm for database search.", Proceedings of the twenty-eighth annual ACM symposium on Theory of computing (1996) achieves a quadratic speed-up on a brute-force search of this satiability problem.

This program builds the necessary parts of the algorithm in order to simulate this algorithm. Crucially, Grover's algorithm requires an oracle that is problem dependent, which changes the sign of the solution of the problem. 

## Oracle

The Exact Cover oracle is built using the same number of ancillary qubits as clauses are in the system, as to have a way too track that all clauses are satisfied. For each clause in the system, a CNOT gate from each of the qubits to its ancilla, followed by a multi-Toffoli gates controlled by the three of them targeting the ancilla leaves said ancilla active only if the clause is satisfied.

A multi-Toffoli gate on all "clause" ancilla targeting the Grover ancilla (qubit initialized by an X gate followed by a Hadamard gate) changes the sign of the solution of the system. Then, all ancillas must be uncomputed for the oracle to be complete. 

## Diffusion operator

After the sign change of the solution states, an inversion over the average amplifies their amplitude. This is at the core of Grover's search algorithm. This operator, also known as Diffusion operator, can be constructed by a set of Hadamard gates on all qubits, a controlled-Z gate so as to change the sign of the |0000...00> state, and another set of H gates. 

## Building the algorithm

The quantum register is started with Hadamard gates on all qubits encoding the problem, and an X gate followed by a Hadamard gate on the Grover ancilla, used to change the sign of the solution. 

Then the oracle and diffusion operator are applied (π/4)*sqrt(N) times, where N is the total number of possible outcomes, namely 2**(# of qubits).

After this has been applied, measuring the quantum registers outputs the solution of the problem. 

# How to run the example

Run the file main.py from the console to find the solution of an instance of 10 qubits.

Adding the argument --nqubits _ (int) allows for instances of different number of qubits.

Supported instances are of [4, 8, 10, 12, 14, 16] qubits.

# Create your own instances

An example of an instance for 4 qubits reads:

 4 3 1
0 1 0 0
 1 2 3
 2 3 4
 1 2 4

The first line is the number of qubits, clauses, and 1's in the solution.

The second line is the solution of the instance.

The following lines are the qubits present in each instance.

Should the solution not be known, consider 00000... to be a placeholder solution for this type of example. 
