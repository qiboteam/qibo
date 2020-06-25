## Problem
Strongly-correlated many-body systems can give rise to exceptional quantum phenomena. In particular, the Ising model or the XXZ model have fundamental significance in condensed matter physics, and thus the realization of these systems may attract tremendous interest. Indeed, it is then an ambitious goal to prepare the groundstate of those systems, and gain some insight into the physics of the problem.

## Implement the solution
The code herein aims to reproduce the results of the manuscript ["Scaling of variational quantum circuit depth for condensed matter systems"](https://quantum-journal.org/papers/q-2020-05-28-272/). The idea is to benchmark the accuracy of the [Variational Quantum Eigensolver (VQE)](https://www.nature.com/articles/ncomms5213) based on a finite-depth variational quantum circuit encoding ground states of local Hamiltonian, namely, the Ising and XXZ models. 

Notice that any potential advantage of the VQE could be lost without practical approaches to perform the parameter optimization due to the optimization in the high-dimensional parameter landscape. A particular proposal to try to solve this optimization problem is the [Adiabatically Assisted Variational Quantum Eigensolver (AAVQE)](https://arxiv.org/abs/1806.02287). The AAVQE is a strategy circumventing the convergence issue, inspired by the adiabatic theorem. The AAVQE method consists of parametrizing a Hamiltonian as $H = (1-s)H_0 + sH_P$ where $H_0$ is a Hamiltonian which ground state can be easily prepared, $H_P$ is the problem Hamiltonian, and $s\in [0,1]$ is the interpolation parameter. The interpolation parameter is used to adjust the Hamiltonian from one VQE run to the next, and the state preparation parameters at each step are initialized by the optimized parameters of the previous step.

## How to run an example
To run a particular instance of the problem we have to set up the initial arguments:
- nqubits (int): number of quantum bits.
- layers (float): number of ansatz layers.
- maxsteps (int): number of maximum steps on each adiabatic path.
- T_max (int): number of maximum adiabatic paths.
- initial_parameters (array): values of the initial parameters.
- easy_hamiltonian (qibo.hamiltonians): initial hamiltonian object, defined as sz_hamiltonian.
- problem_hamiltonian (qibo.hamiltonians): problem hamiltonian object, namely, the Ising or XXZ hamiltonians.

## Results
We can now compute different instances of the problem by varying the number of qubits and the number of layers of our ansatz. Once we have the final energies on each instance, we can compute the logarithm of the difference between the AAVQE result and the exact groundstate energy as $\log(1/(E_{AAVQE} - E_0))$. These are the results that we may obtain:

<img src="ising.png" width="500px"> <img src="XXZ.png" width="470px">

If you do not get those results, do not despair. Sometimes, fine-tuning of the arguments, i.e., maxsteps and T_max, is required, especially for a large number of qubits.
