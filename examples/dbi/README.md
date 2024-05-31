# Double-bracket quantum algorithms

Qibo features a model implementing double-bracke quantum algorithms (DBQAs) which are helpful for approximating eigenstates based on the ability to run the evolution under the input Hamiltonian.

More specifically, given a Hamiltonian $H_0$, how can we find a circuit which after applying to the reference state (usually $|0\rangle^{\otimes L}$ for $L$ qubits) will approximate an eigenstate?

A standard way is to run variational quantum circuits. For example, Qibo already features  the `VQE` model [2] which provides the implementation of the variational quantum eigensolver framework.
DBQAs allow to go beyond VQE in that they take a different approach to compiling the quantum circuit approximating the eigenstate.

[1] https://arxiv.org/abs/2206.11772

[2] https://github.com/qiboteam/vqe-sun
